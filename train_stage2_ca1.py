import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision import models

# 專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4                 # 每張卡 batch
ACCUMULATION_STEPS = 4         # 等效 batch: BATCH_SIZE * num_gpus * ACCUMULATION_STEPS
EPOCHS = 200
PURE_EPOCHS = 50               # 前 50 epoch 不跑 VGG
BASE_LR = 2e-5

SAVE_DIR = "checkpoints_stage2_v14_ultimate"
SAMPLE_DIR = "samples_stage2_v14_ultimate"
LOG_DIR = "runs/stage2_v14_ultimate"

G1_CHECKPOINT = "./checkpoints_sa/G1_latest.pth"
LOAD_G2_PATH = f"{SAVE_DIR}/G2_v14_ultimate_latest.pth"

START_EPOCH = 0
best_psnr = 0.0


# =========================
# SCHEDULE: 動態 loss 權重 + LR
# =========================
def get_sa_config(epoch: int):
    """
    return: w_recon, w_vgg, w_mse, w_adv, lr

    0-49   : 純 pixel (不跑 VGG)
    50-149 : pixel + VGG 漸進 (不跑 GAN)
    150+   : 加 GAN 拋光
    """
    # 預設值
    w_recon, w_vgg, w_mse, w_adv = 1.0, 0.05, 1.0, 0.0
    lr = BASE_LR

    if PURE_EPOCHS <= epoch < 150:
        eta = (epoch - PURE_EPOCHS) / 100.0
        alpha = 0.5 * (1 - math.cos(math.pi * eta))

        # 權重漸進：MSE 變小，recon/VGG 變大
        w_mse = 1.0 * (1 - alpha) + 0.1 * alpha
        w_recon = 0.1 * (1 - alpha) + 1.0 * alpha
        w_vgg = 0.05 * (1 - alpha) + 0.2 * alpha

        # cosine lr
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * eta))
        w_adv = 0.0

    elif epoch >= 150:
        w_mse, w_recon, w_vgg = 0.1, 1.0, 0.2
        lr = BASE_LR * 0.1
        w_adv = 0.05

    # epoch < PURE_EPOCHS 時維持預設，且你外面會用 w_vgg>0 判斷
    return w_recon, w_vgg, w_mse, w_adv, lr


# =========================
# MODELS / LOSSES / METRICS
# =========================
class Discriminator(nn.Module):
    """PatchGAN discriminator（Hinge loss），只在 150 epoch 後用來拋光細節。"""
    def __init__(self):
        super().__init__()

        def conv_block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )

        self.model = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


def manual_ssim(img1, img2, window_size=11):
    """簡單 SSIM（避免額外依賴）。輸入建議為 [0,1]。"""
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - (mu1 * mu2)
    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size // 2) - mu1**2
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size // 2) - mu2**2
    ssim = (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
            ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))
    return ssim.mean()


class VGGLoss(nn.Module):
    """
    多層 feature L1：用多個 content layer 做 perceptual loss。
    注意：輸入 pred/target 為 [-1,1]，內部轉到 [0,1]
    """
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.content_layers = {4, 9, 18, 27}

    def forward(self, pred, target):
        p = (pred + 1) / 2
        t = (target + 1) / 2

        loss = 0.0
        for i, layer in enumerate(self.vgg):
            p = layer(p)
            t = layer(t)
            if i in self.content_layers:
                loss = loss + F.l1_loss(p, t)
        return loss


def calc_psnr(img_a_01, img_b_01):
    """PSNR for [0,1] images."""
    mse = F.mse_loss(img_a_01, img_b_01)
    return 20 * math.log10(1.0 / (torch.sqrt(mse) + 1e-8))


def save_preview_image(imgs, masks, edges, results, psnr_val, epoch, save_dir):
    """輸出 1x5 預覽圖：Original/Mask/Edge/MaskedInput/Result"""
    os.makedirs(save_dir, exist_ok=True)

    imgs_np = (imgs.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
    masks_np = masks.cpu().permute(0, 2, 3, 1).numpy()
    edges_np = edges.cpu().permute(0, 2, 3, 1).numpy()
    res_np = (results.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2

    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    titles = ["Original", "Mask", "Edge Guide", "Masked Input", f"Res (PSNR:{psnr_val:.2f})"]
    data = [
        imgs_np[0],
        masks_np[0, :, :, 0],
        edges_np[0, :, :, 0],
        imgs_np[0] * (1 - masks_np[0]),
        res_np[0]
    ]

    for ax, d, t in zip(axes, data, titles):
        ax.imshow(np.clip(d, 0, 1), cmap='gray' if d.ndim == 2 else None)
        ax.set_title(t, fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"v13_ep{epoch:03d}.png"))
    plt.close()


# =========================
# TRAIN
# =========================
def train():
    global best_psnr

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    writer = SummaryWriter(LOG_DIR)

    train_loader = DataLoader(
        InpaintingDataset("./datasets/img", mode="train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )

    # --- G1 (edge) ---
    G1 = EdgeGenerator().to(DEVICE).eval()
    G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))

    # --- G2 (diffusion unet) + D ---
    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    D = Discriminator().to(DEVICE)

    # load G2
    if LOAD_G2_PATH and os.path.exists(LOAD_G2_PATH):
        print(f"[*] Load Stage2 checkpoint: {LOAD_G2_PATH}")
        G2.load_state_dict(torch.load(LOAD_G2_PATH, map_location=DEVICE))

    # resume best metric (你的原版是手動設 20.0，這裡保留但不強制)
    if START_EPOCH > 0 and best_psnr <= 0:
        best_psnr = 20.0

    # multi-gpu
    if torch.cuda.device_count() > 1:
        print(f"[*] DataParallel enabled: {torch.cuda.device_count()} GPUs")
        G2 = nn.DataParallel(G2)
        D = nn.DataParallel(D)

    opt_G = optim.AdamW(G2.parameters(), lr=BASE_LR, weight_decay=1e-4)
    opt_D = optim.AdamW(D.parameters(), lr=1e-4, weight_decay=1e-4)

    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)

    for epoch in range(START_EPOCH, EPOCHS):
        w_recon, w_vgg, w_mse, w_adv, curr_lr = get_sa_config(epoch)
        for pg in opt_G.param_groups:
            pg["lr"] = curr_lr

        G2.train()
        D.train()

        epoch_l1, epoch_vgg, epoch_mse = 0.0, 0.0, 0.0
        epoch_advG, epoch_total = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} {'[GAN]' if w_adv > 0 else ''}")

        opt_G.zero_grad(set_to_none=True)

        for i, (imgs, _, masks) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)
            curr_dev = imgs.device

            # ---- edge guide (no grad) ----
            with torch.no_grad():
                masked = imgs * (1 - masks)
                pred_edges = G1(torch.cat([masked, masks], dim=1))

            # ---- diffusion prepare ----
            condition = torch.cat([masked, masks, pred_edges], dim=1)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(curr_dev)
            x_t, noise = diffusion.noise_images(imgs, t)

            # ---- forward G2 ----
            with torch.amp.autocast("cuda"):
                pred_noise = G2(x_t, t, condition)
                l_mse = F.mse_loss(pred_noise, noise)

                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * pred_noise) / (torch.sqrt(input=alpha_hat) + 1e-6)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                l_pixel = F.l1_loss(pred_x0, imgs)
                l_vgg = vgg_criterion(pred_x0, imgs) if (epoch >= PURE_EPOCHS and w_vgg > 0) else torch.tensor(0.0, device=curr_dev)

                # ---- GAN part (optional) ----
                l_adv_G = torch.tensor(0.0, device=curr_dev)
                if w_adv > 0:
                    # update D
                    opt_D.zero_grad(set_to_none=True)
                    real_res = D(imgs)
                    fake_res = D(pred_x0.detach())
                    loss_D = (F.relu(1.0 - real_res).mean() + F.relu(1.0 + fake_res).mean()) * 0.5

                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(opt_D)
                    scaler_D.update()

                    # G adversarial loss
                    l_adv_G = -D(pred_x0).mean()

                # total loss (divide by accumulation)
                loss_total = (l_mse * w_mse + l_vgg * w_vgg + l_pixel * w_recon + l_adv_G * w_adv) / ACCUMULATION_STEPS

            if torch.isnan(loss_total):
                continue

            scaler_G.scale(loss_total).backward()

            # grad accumulation step
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G2.parameters(), 1.0)
                scaler_G.step(opt_G)
                scaler_G.update()
                opt_G.zero_grad(set_to_none=True)

            # stats
            epoch_l1 += float(l_pixel.item())
            epoch_mse += float(l_mse.item())
            epoch_vgg += float(l_vgg.item())
            epoch_advG += float(l_adv_G.item())
            epoch_total += float(loss_total.item()) * ACCUMULATION_STEPS  # 還原成「未除以 accumulation」的總量

            if i % 10 == 0:
                with torch.no_grad():
                    psnr_cur = calc_psnr((pred_x0.detach() + 1) / 2, (imgs + 1) / 2)
                pbar.set_postfix({"L1": f"{l_pixel.item():.4f}", "PSNR": f"{psnr_cur:.2f}"})

        # ---- epoch logging ----
        denom = max(1, len(train_loader))  # 避免除 0
        writer.add_scalar("Losses/L1_Loss", epoch_l1 / denom, epoch)
        writer.add_scalar("Losses/VGG_Loss", epoch_vgg / denom, epoch)
        writer.add_scalar("Losses/MSE_Loss", epoch_mse / denom, epoch)
        writer.add_scalar("Losses/Adv_G", epoch_advG / denom, epoch)
        writer.add_scalar("Losses/Total", epoch_total / denom, epoch)

        writer.add_scalar("Weights/w_recon", w_recon, epoch)
        writer.add_scalar("Weights/w_vgg", w_vgg, epoch)
        writer.add_scalar("Weights/w_mse", w_mse, epoch)
        writer.add_scalar("Weights/w_adv", w_adv, epoch)
        writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)

        # ---- quick eval & sample ----
        G2.eval()
        with torch.no_grad():
            m_sam = G2.module if hasattr(G2, "module") else G2
            # 用最後一個 batch 的 condition/imgs/masks 做 sample（跟你原版一致）
            samples = diffusion.sample(m_sam, condition[0:1], n=1, steps=50)
            res_img = (imgs[0:1] * (1 - masks[0:1])) + (samples * masks[0:1])

            v_psnr = calc_psnr((res_img + 1) / 2, (imgs[0:1] + 1) / 2)
            v_ssim = manual_ssim((res_img + 1) / 2, (imgs[0:1] + 1) / 2).item()

        writer.add_scalar("Metrics/PSNR", v_psnr, epoch)
        writer.add_scalar("Metrics/SSIM", v_ssim, epoch)
        writer.add_image("Preview/Epoch_Res", (res_img[0] + 1) / 2, epoch)

        # best PSNR checkpoint
        if v_psnr > best_psnr:
            best_psnr = float(v_psnr)
            torch.save(m_sam.state_dict(), os.path.join(SAVE_DIR, f"G2_BEST_PSNR_ep{epoch}.pth"))
            torch.save(m_sam.state_dict(), os.path.join(SAVE_DIR, "G2_BEST_PSNR_latest.pth"))
            print(f"\n[🏆] New best! Epoch {epoch} PSNR: {best_psnr:.2f} (saved)")

        # preview image & latest
        save_preview_image(imgs, masks, pred_edges, res_img, float(v_psnr), epoch, SAMPLE_DIR)
        torch.save(m_sam.state_dict(), os.path.join(SAVE_DIR, "G2_ultimate_polish.pth"))

    writer.close()


if __name__ == "__main__":
    train()
