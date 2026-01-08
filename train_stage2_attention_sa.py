import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import math
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from tqdm import tqdm

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# ================= CONFIG (終極收斂版) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_LR = 2e-5
EPOCHS = 200
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

SAVE_DIR = "checkpoints_stage2_sa"
SAMPLE_DIR = "samples_stage2_results"
LOG_DIR = "runs/stage2_sa_psnr_final"

G1_CHECKPOINT = "./checkpoints_sa/G1_SA_epoch_81.pth"
# =====================================================

def get_sa_params(epoch):
    """ 模擬退火與自動學習率排程 """
    lr = BASE_LR
    w_mse, w_recon, w_vgg = 1.0, 0.1, 0.05
    if 50 <= epoch < 150:
        # 轉型期：從結構去噪轉向像素重建
        eta = (epoch - 50) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        w_mse = 1.0 * (1 - alpha) + 0.1 * alpha
        w_recon = 0.1 * (1 - alpha) + 1.0 * alpha
        w_vgg = 0.05 * (1 - alpha) + 0.2 * alpha
    elif epoch >= 150:
        # 收斂期：自動降低學習率精修細節
        w_mse, w_recon, w_vgg = 0.1, 1.0, 0.2
        lr = BASE_LR * 0.1
    return w_mse, w_recon, w_vgg, lr

def calculate_psnr(pred, target):
    """ 修正版 PSNR：防禦 NaN 並確保範圍正確 """
    with torch.no_grad():
        p = torch.clamp((pred.detach() + 1) / 2, 0, 1)
        t = torch.clamp((target.detach() + 1) / 2, 0, 1)
        mse = torch.mean((p - t) ** 2)
        if mse == 0: return torch.tensor(100.0).to(pred.device)
        return 20 * torch.log10(1.0 / torch.sqrt(mse))

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.content_layers = [4, 9, 18, 27]

    def forward(self, pred, target):
        p, t = (pred + 1) / 2, (target + 1) / 2
        loss = 0
        for i, layer in enumerate(self.vgg):
            p, t = layer(p), layer(t)
            if i in self.content_layers:
                loss += nn.functional.l1_loss(p, t)
        return loss

def train():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)

    writer = SummaryWriter(log_dir=LOG_DIR)
    train_loader = DataLoader(InpaintingDataset("./datasets/img", mode='train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    G1 = EdgeGenerator().to(DEVICE).eval()
    if os.path.exists(G1_CHECKPOINT):
        G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)
    mse_loss = nn.MSELoss()

    opt = torch.optim.AdamW(G2.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    # === 續訓邏輯修正 ===
    start_epoch = 0 # 您目前指定的起點
    resume_path = os.path.join(SAVE_DIR, "G2_latest.pth")
    if os.path.exists(resume_path):
        print(f"[*] 載入權重並從 Epoch {start_epoch} 續跑：{resume_path}")
        G2.load_state_dict(torch.load(resume_path, map_location=DEVICE))

    for epoch in range(start_epoch, EPOCHS):
        # 自動調整 LR 與 SA 權重
        w_mse, w_recon, w_vgg, curr_lr = get_sa_params(epoch)
        for g in opt.param_groups: g['lr'] = curr_lr

        G2.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for i, (imgs, _, masks) in enumerate(pbar):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad():
                masked_imgs = imgs * (1 - masks)
                pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(imgs, t)

            with torch.amp.autocast('cuda'):
                predicted_noise = G2(x_t, t, condition)
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                l_mse = mse_loss(predicted_noise, noise)
                l_recon = nn.functional.l1_loss(pred_x0 * masks, imgs * masks)
                l_vgg = vgg_criterion(imgs * (1 - masks) + pred_x0 * masks, imgs)
                total_loss = (w_mse * l_mse + w_recon * l_recon + w_vgg * l_vgg) / ACCUMULATION_STEPS

            scaler.scale(total_loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad()

            if i % 20 == 0:
                step = epoch * len(train_loader) + i
                psnr_val = calculate_psnr(pred_x0, imgs)
                writer.add_scalar('Metrics/PSNR', psnr_val.item(), step)
                writer.add_scalar('Status/LearningRate', curr_lr, step)
                pbar.set_postfix({"PSNR": f"{psnr_val.item():.2f}dB", "LR": f"{curr_lr:.2e}"})

        # === 視覺化渲染 (修正為 Original / Input / Edge / Inpaint) ===
        G2.eval()
        with torch.no_grad():
            samples = diffusion.sample(G2, condition[0:1], n=1)
            samples = torch.clamp(samples, -1.0, 1.0)
            final_res = imgs[0:1] * (1 - masks[0:1]) + samples * masks[0:1]

            final_psnr = calculate_psnr(final_res, imgs[0:1]).item()
            res_np = np.clip((final_res[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0, 0, 1)
            gt_np = (imgs[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
            input_np = (masked_imgs[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
            edge_np = pred_edges[0, 0].cpu().numpy() # 提取 G1 邊緣圖

            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            axes[0].imshow(gt_np); axes[0].set_title("Original Ground Truth")
            axes[1].imshow(input_np); axes[1].set_title("Input (Masked)")
            axes[2].imshow(edge_np, cmap='gray'); axes[2].set_title("Detect Edge (G1)")
            axes[3].imshow(res_np); axes[3].set_title(f"Inpainting ({final_psnr:.2f}dB)")

            plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch}.png"); plt.close()
            writer.add_image('Visual/Comparison', res_np.transpose(2, 0, 1), epoch)

        torch.save(G2.state_dict(), f"{SAVE_DIR}/G2_latest.pth")
    writer.close()

if __name__ == "__main__": train()