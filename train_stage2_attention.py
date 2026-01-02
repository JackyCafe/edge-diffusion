import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
from tqdm import tqdm

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# ================= CONFIG (最終收斂版) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-5
EPOCHS = 100
BATCH_SIZE = 4
ACCUMULATION_STEPS = 4

SAVE_DIR = "checkpoints_stage2"
SAMPLE_DIR = "samples_stage2_results"
LOG_DIR = "runs/stage2_attention_final"

# 使用您最新的 IoU 峰值權重 (Epoch 75/80)
G1_CHECKPOINT = "./checkpoints/G1_epoch_80.pth"

W_RECON = 1.0
W_VGG = 0.1
# =====================================================

def calculate_psnr(pred, target):
    """ 修正版 PSNR：確保數值在 [0, 1] 範圍計算 """
    p = (pred.detach() + 1) / 2
    t = (target.detach() + 1) / 2
    mse = torch.mean((p - t) ** 2)
    if mse == 0: return 100.0
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
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        loss = 0
        x, y = pred, target
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.content_layers:
                loss += nn.functional.l1_loss(x, y)
        return loss

def train():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)

    writer = SummaryWriter(log_dir=LOG_DIR)
    train_dataset = InpaintingDataset("./datasets/img", mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # 初始化 G1 與 G2
    G1 = EdgeGenerator().to(DEVICE)
    if os.path.exists(G1_CHECKPOINT):
        print(f"[*] Loading High-Quality G1 Guide from {G1_CHECKPOINT}")
        G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))
    G1.eval()

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)
    mse_loss = nn.MSELoss()

    opt = torch.optim.AdamW(G2.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(EPOCHS):
        G2.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_psnr = []

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
                loss_mse_val = mse_loss(predicted_noise, noise)

                # 推導 x0 計算損失與 PSNR
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                loss_recon = nn.functional.l1_loss(pred_x0 * masks, imgs * masks)
                loss_vgg = vgg_criterion(imgs * (1 - masks) + pred_x0 * masks, imgs)
                total_loss = (loss_mse_val + W_RECON * loss_recon + W_VGG * loss_vgg) / ACCUMULATION_STEPS

            scaler.scale(total_loss).backward()
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            current_psnr = calculate_psnr(pred_x0, imgs).item()
            epoch_psnr.append(current_psnr)

            if i % 20 == 0:
                global_step = epoch * len(train_loader) + i
                writer.add_scalar('Loss/MSE', loss_mse_val.item(), global_step)
                writer.add_scalar('Metrics/PSNR', current_psnr, global_step)
                pbar.set_postfix({"PSNR": f"{current_psnr:.2f}dB"})

        # === 驗證與繪圖 (解決黑塊關鍵) ===
        G2.eval()
        with torch.no_grad():
            sample_condition = condition[0:1]
            samples = diffusion.sample(G2, sample_condition, n=1)
            final_res = imgs[0:1] * (1 - masks[0:1]) + samples * masks[0:1]

            # 數值轉換與裁切
            res_np = (final_res[0].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
            res_np = np.clip(res_np, 0, 1)

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow((imgs[0].cpu().permute(1, 2, 0) + 1) / 2); axes[0].set_title("Original")
            axes[1].imshow((masked_imgs[0].cpu().permute(1, 2, 0) + 1) / 2); axes[1].set_title("Input")
            axes[2].imshow(pred_edges[0, 0].cpu(), cmap='gray'); axes[2].set_title("G1 Edge")
            axes[3].imshow(res_np); axes[3].set_title(f"Result ({current_psnr:.2f}dB)")

            plt.savefig(f"{SAMPLE_DIR}/epoch_{epoch}.png")
            writer.add_image('Visual/Result', res_np.transpose(2, 0, 1), epoch)
            plt.close()

        torch.save(G2.state_dict(), f"{SAVE_DIR}/G2_latest.pth")
    writer.close()

if __name__ == "__main__":
    train()