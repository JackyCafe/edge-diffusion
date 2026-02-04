import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet, Discriminator, VGGLoss
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.misc import manual_ssim, save_preview_image

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4
EPOCHS = 200
PURE_EPOCHS = 50
BASE_LR = 2e-5

SAVE_DIR = "checkpoints_stage2_v14_ultimate"
SAMPLE_DIR = "samples_stage2_v14_ultimate"
LOG_DIR = "runs/stage2_v14_ultimate"

G1_CHECKPOINT = "./checkpoints_sa/G1_latest.pth"
LOAD_G2_PATH = f"{SAVE_DIR}/checkpoint_latest.pth"
W_ADV_FINAL = 0.005

START_EPOCH = 0
best_psnr = 0.0


# ================= Loss Helper (重要：避免 mask 被全圖平均稀釋) =================
def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    pred/target: [B,C,H,W], mask: [B,1,H,W] with {0,1}
    只在 mask=1 的區域計算，並以 mask 像素數做正規化（不會被全圖稀釋）
    """
    diff = (pred - target).abs() * mask
    denom = mask.sum() * pred.shape[1] + eps  # *C 讓通道一起平均
    return diff.sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (pred - target) ** 2 * mask
    denom = mask.sum() * pred.shape[1] + eps
    return diff.sum() / denom


# ================= Schedules =================
def get_sa_config(epoch):
    """
    return: w_recon, w_vgg, w_mse, w_adv, lr
    """
    if epoch < PURE_EPOCHS:
        ratio = epoch / PURE_EPOCHS
        lr = BASE_LR
        w_recon = 1.0 + 0.5 * (1 - math.cos(math.pi * ratio))  # 1.0 -> 2.0
        w_mse   = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))  # 1.0 -> 0.8
        w_vgg   = 0.01
        w_adv   = 0.0

    elif epoch < 150:
        eta = (epoch - PURE_EPOCHS) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta))

        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * eta))

        w_mse   = 0.8  * (1 - alpha) + 0.2  * alpha
        w_recon = 2.0  * (1 - alpha) + 1.0  * alpha
        w_vgg   = 0.01 * (1 - alpha) + 0.15 * alpha
        w_adv   = 0.0

    else:
        lr = BASE_LR * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.2, 0.1
        w_adv = W_ADV_FINAL

    return w_recon, w_vgg, w_mse, w_adv, lr


def gray_w_schedule(epoch: int) -> float:
    """
    灰階一致性不要太早啟用（早期會讓輸出變平均灰塊）
    """
    if epoch < 40:
        return 0.0
    elif epoch < 80:
        return 0.02
    else:
        return 0.05


def gray_consistency_loss(x: torch.Tensor, masks: torch.Tensor, w: float) -> torch.Tensor:
    """
    只在洞內做通道一致性（灰階化），且用 mask 像素正規化（不稀釋）
    """
    if w <= 0:
        return torch.tensor(0.0, device=x.device)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    loss = (masked_l1(r, g, masks) + masked_l1(r, b, masks) + masked_l1(g, b, masks)) / 3.0
    return loss * w


# ================= Metrics =================
def calculate_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.clamp((pred + 1) / 2, 0, 1)
        t = torch.clamp((target + 1) / 2, 0, 1)
        mse = torch.mean((p - t) ** 2)
        if mse.item() == 0:
            return 100.0
        return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()


def calculate_psnr_mask(pred: torch.Tensor, target: torch.Tensor, masks: torch.Tensor) -> float:
    with torch.no_grad():
        p = torch.clamp((pred + 1) / 2, 0, 1)
        t = torch.clamp((target + 1) / 2, 0, 1)
        m = masks
        mse_m = ((p - t) ** 2 * m).sum() / m.sum().clamp(min=1.0)
        return (10.0 * torch.log10(1.0 / (mse_m + 1e-12))).item()


def train():
    global best_psnr
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    train_loader = DataLoader(
        InpaintingDataset("./datasets/img", mode='train'),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    # --- G1 固定 ---
    G1 = EdgeGenerator().to(DEVICE).eval()
    G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))

    # --- 建立模型（先建好，再決定是否 DataParallel）---
    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    D  = Discriminator().to(DEVICE)

    use_dp = (torch.cuda.device_count() > 1)
    if use_dp:
        G2 = nn.DataParallel(G2)
        D  = nn.DataParallel(D)

    opt_G = optim.AdamW(G2.parameters(), lr=BASE_LR, weight_decay=1e-4)
    opt_D = optim.AdamW(D.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    # --- 續訓載入（自動從 checkpoint epoch+1 開始）---
    start_from = START_EPOCH
    if LOAD_G2_PATH and os.path.exists(LOAD_G2_PATH):
        ckpt = torch.load(LOAD_G2_PATH, map_location=DEVICE)

        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            if 'optimizer_state_dict' in ckpt:
                opt_G.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler_G.load_state_dict(ckpt['scaler_state_dict'])
            if 'best_psnr' in ckpt:
                best_psnr = float(ckpt['best_psnr'])
            if 'epoch' in ckpt:
                start_from = int(ckpt['epoch']) + 1
        else:
            state = ckpt

        try:
            G2.load_state_dict(state, strict=True)
        except RuntimeError:
            new_state = {}
            if use_dp and not any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state["module." + k] = v
                G2.load_state_dict(new_state, strict=False)
            elif (not use_dp) and any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state[k.replace("module.", "", 1)] = v
                G2.load_state_dict(new_state, strict=False)
            else:
                raise

        print(f"[*] 成功載入 G2 檢查點，續訓從 Epoch {start_from} 開始")

    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)

    for epoch in range(start_from, EPOCHS):
        # epoch stats
        epoch_l1 = 0.0
        epoch_vgg = 0.0
        epoch_mse = 0.0
        epoch_adv_g = 0.0
        epoch_gray = 0.0
        epoch_color = 0.0
        epoch_total = 0.0

        epoch_psnr_x0 = 0.0
        epoch_psnr_x0_mask = 0.0

        w_recon, w_vgg, w_mse, w_adv, curr_lr = get_sa_config(epoch)
        for g in opt_G.param_groups:
            g['lr'] = curr_lr

        w_gray = gray_w_schedule(epoch)

        G2.train()
        D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        opt_G.zero_grad(set_to_none=True)
        opt_D.zero_grad(set_to_none=True)

        for i, (imgs, _, masks) in enumerate(pbar):
            imgs = imgs.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            # 1) G1 edge（固定）
            with torch.no_grad():
                masked_imgs = imgs * (1 - masks)
                pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)

            # 2) Diffusion：加噪
            t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(imgs, t)

            with torch.amp.autocast('cuda'):
                pred_noise = G2(x_t, t, condition)
                pred_noise = torch.clamp(torch.nan_to_num(pred_noise), -1.0, 1.0)

                # MSE on noise
                l_mse = F.mse_loss(pred_noise, noise)

                # 還原 x0（你的 debug 已證實這個公式與 noise_images 一致）
                alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]
                denom = torch.sqrt(alpha_hat_t + 1e-7)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / denom
                pred_x0 = torch.clamp(torch.nan_to_num(pred_x0), -1.0, 1.0)

                # ✅ 正確 masked L1（洞內，不稀釋）
                l_pixel = masked_l1(pred_x0, imgs, masks)

                # VGG（非 PURE_EPOCHS 才啟用）
                if (epoch >= PURE_EPOCHS) and (w_vgg > 0):
                    l_vgg = vgg_criterion(pred_x0, imgs)
                else:
                    l_vgg = torch.tensor(0.0, device=DEVICE)

                # ✅ 灰階一致性（延遲啟用 + 洞內正規化）
                l_gray = gray_consistency_loss(pred_x0, masks, w=w_gray)

                # ✅ 色彩/亮度一致性：改成洞內 masked MSE（穩、直覺）
                l_color = masked_mse(pred_x0, imgs, masks)

                # GAN（150 後開）
                l_adv_G = torch.tensor(0.0, device=DEVICE)
                if w_adv > 0:
                    opt_D.zero_grad(set_to_none=True)
                    real_res = D(imgs)
                    fake_res = D(pred_x0.detach())
                    loss_D = (F.relu(1.0 - real_res).mean() + F.relu(1.0 + fake_res).mean()) * 0.5
                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(opt_D)
                    scaler_D.update()

                    l_adv_G = -D(pred_x0).mean()

                loss_total = (
                    l_mse   * w_mse +
                    l_vgg   * w_vgg +
                    l_pixel * w_recon +
                    l_adv_G * w_adv +
                    l_color * 0.1 +
                    l_gray
                ) / ACCUMULATION_STEPS

            if torch.isnan(loss_total):
                print(f"[⚠️] NaN loss at epoch {epoch}, step {i}, skip.")
                continue

            scaler_G.scale(loss_total).backward()

            # teacher-forced PSNR（穩）
            with torch.no_grad():
                psnr_x0 = calculate_psnr(pred_x0.detach(), imgs)
                psnr_x0_mask = calculate_psnr_mask(pred_x0.detach(), imgs, masks)

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G2.parameters(), 0.3)
                scaler_G.step(opt_G)
                scaler_G.update()
                opt_G.zero_grad(set_to_none=True)

                epoch_l1 += float(l_pixel.item())
                epoch_mse += float(l_mse.item())
                epoch_vgg += float(l_vgg.item()) if torch.is_tensor(l_vgg) else float(l_vgg)
                epoch_adv_g += float(l_adv_G.item())
                epoch_gray += float(l_gray.item())
                epoch_color += float(l_color.item())
                epoch_total += float(loss_total.item()) * ACCUMULATION_STEPS

                epoch_psnr_x0 += float(psnr_x0)
                epoch_psnr_x0_mask += float(psnr_x0_mask)

            if i % 10 == 0:
                pbar.set_postfix({
                    "L1_mask": f"{l_pixel.item():.4f}",
                    "MSE": f"{l_mse.item():.4f}",
                    "PSNRm_x0": f"{psnr_x0_mask:.2f}",
                    "PSNR_x0": f"{psnr_x0:.2f}",
                    "w_gray": f"{w_gray:.3f}",
                    "LR": f"{curr_lr:.2e}"
                })

        # ===== epoch end logging =====
        avg_steps = max(1, len(train_loader) // ACCUMULATION_STEPS)

        writer.add_scalar("Losses/L1_Loss", epoch_l1 / avg_steps, epoch)
        writer.add_scalar("Losses/VGG_Loss", epoch_vgg / avg_steps, epoch)
        writer.add_scalar("Losses/MSE_Loss", epoch_mse / avg_steps, epoch)
        writer.add_scalar("Losses/Adv_Loss", epoch_adv_g / avg_steps, epoch)
        writer.add_scalar("Losses/Total_Loss", epoch_total / avg_steps, epoch)

        writer.add_scalar("Weights/w_recon", w_recon, epoch)
        writer.add_scalar("Weights/w_vgg", w_vgg, epoch)
        writer.add_scalar("Weights/w_mse", w_mse, epoch)
        writer.add_scalar("Weights/w_adv", w_adv, epoch)
        writer.add_scalar("Weights/w_gray", w_gray, epoch)
        writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)

        writer.add_scalar("Metrics/PSNR_x0", epoch_psnr_x0 / avg_steps, epoch)
        writer.add_scalar("Metrics/PSNR_x0_mask", epoch_psnr_x0_mask / avg_steps, epoch)

        # ===== Sampling / Preview（抖，但代表最終生成能力） =====
        G2.eval()
        with torch.no_grad():
            m_sam = G2.module if hasattr(G2, 'module') else G2

            if epoch == 149:
                print("[*] 生成 No-GAN Baseline...")
                os.makedirs("baseline_samples", exist_ok=True)
                baseline_res = diffusion.sample(m_sam, condition[0:4], n=4, steps=100)
                save_preview_image(imgs[0:4], masks[0:4], pred_edges[0:4], baseline_res, 0.0, epoch, "baseline_samples")

            # 固定 seed → sampling 指標比較可比
            torch.manual_seed(0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(0)

            samples = diffusion.sample(m_sam, condition[0:1], n=1, steps=100)
            samples = torch.clamp(samples, -1.0, 1.0)
            res_img = imgs[0:1] * (1 - masks[0:1]) + samples * masks[0:1]

            v_psnr = calculate_psnr(res_img, imgs[0:1])
            v_psnr_mask = calculate_psnr_mask(res_img, imgs[0:1], masks[0:1])
            v_ssim = manual_ssim((res_img + 1) / 2, (imgs[0:1] + 1) / 2).item()

            # best checkpoint（你也可改成用 v_psnr_mask）
            if v_psnr > best_psnr:
                best_psnr = v_psnr
                torch.save(m_sam.state_dict(), f"{SAVE_DIR}/G2_BEST_PSNR_ep{epoch}.pth")
                torch.save(m_sam.state_dict(), f"{SAVE_DIR}/G2_BEST_PSNR_latest.pth")
                print(f"\n[🏆] New BEST PSNR_sample! Epoch {epoch} PSNR: {best_psnr:.2f}")

            writer.add_scalar("Metrics/PSNR_sample", v_psnr, epoch)
            writer.add_scalar("Metrics/PSNR_sample_mask", v_psnr_mask, epoch)
            writer.add_scalar("Metrics/SSIM_sample", v_ssim, epoch)

            writer.add_image("Preview/Epoch_Res", (res_img[0] + 1) / 2, epoch)
            save_preview_image(imgs, masks, pred_edges, res_img, v_psnr, epoch, SAMPLE_DIR)

        # ===== save checkpoint =====
        torch.save({
            'epoch': epoch,
            'model_state_dict': (G2.module.state_dict() if hasattr(G2, 'module') else G2.state_dict()),
            'optimizer_state_dict': opt_G.state_dict(),
            'scaler_state_dict': scaler_G.state_dict(),
            'best_psnr': best_psnr,
        }, f"{SAVE_DIR}/checkpoint_latest.pth")

    writer.close()


if __name__ == "__main__":
    train()
