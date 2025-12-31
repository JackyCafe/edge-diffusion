import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.utils import compute_psnr

# ================= CONFIG (優化風格連貫性版本) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 5e-5
EPOCHS = 200
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0

SAVE_DIR = "checkpoints"
SAMPLE_DIR = "samples_stage2_vggloss"
G1_CHECKPOINT = f"{SAVE_DIR}/G1_epoch_19.pth"

# --- 損失權重優化 ---
W_RECON = 1.0   # 提高像素重建權重
W_VGG = 0.1     # 提高感知損失權重，改善風格差太多的問題
# =====================================================

class VGGLoss(nn.Module):
    """感知損失：計算全局特徵空間距離"""
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        # 提取不同層級的特徵 (線條、紋理、結構)
        self.content_layers = [4, 9, 18, 27]

    def forward(self, pred, target):
        # 將數據從 [-1, 1] 縮放至 [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        loss = 0
        x, y = pred, target
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.content_layers:
                loss += nn.functional.l1_loss(x, y)
        return loss

def save_diffusion_samples(v_imgs, v_masks, v_edges, v_final, save_path):
    """隨機 Demo 視覺化"""
    try:
        batch_size = v_imgs.shape[0]
        idx = np.random.randint(0, batch_size)

        img_orig = (v_imgs[idx].cpu().permute(1, 2, 0).numpy() + 1) / 2
        img_mask = v_masks[idx].cpu().squeeze().numpy()
        img_edge = v_edges[idx].cpu().squeeze().numpy()
        # 最終結果取修復後的完整圖
        img_out = (v_final[idx].detach().cpu().permute(1, 2, 0).numpy() + 1) / 2
        img_masked = img_orig * (1 - img_mask[:, :, None])

        plt.figure(figsize=(20, 5))
        images = [img_orig, img_masked, img_edge, img_out]
        titles = [f'Original (idx:{idx})', 'Masked Input', 'G1 Edge Guide', 'Diffusion Result']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 4, i + 1)
            plt.title(title)
            # 確保數值在 0-1 內，防止渲染異常
            plt.imshow(np.clip(img, 0, 1) if title != 'G1 Edge Guide' else img,
                       cmap='gray' if title == 'G1 Edge Guide' else None)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f" [!] Visualization failed: {e}")

def train():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)

    train_dataset = InpaintingDataset("./datasets/img", mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = InpaintingDataset("./datasets/img", mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    fixed_batch = next(iter(val_loader))

    G1 = EdgeGenerator().to(DEVICE)
    if os.path.exists(G1_CHECKPOINT):
        G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))
    G1.eval()

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    if torch.cuda.device_count() > 1:
        G2 = nn.DataParallel(G2)

    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)
    opt = torch.optim.AdamW(G2.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()
    mse_loss = nn.MSELoss()

    print(f"--- Stage 2: Enhanced Style Consistency Training ---")

    for epoch in range(EPOCHS):
        G2.train()
        for i, (imgs, _, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with torch.no_grad():
                masked_imgs = imgs * (1 - masks)
                pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(imgs, t)

            with autocast():
                # A. 雜訊預測 (基礎 Diffusion Loss)
                predicted_noise = G2(x_t, t, condition)
                loss_mse = mse_loss(predicted_noise, noise)

                # B. 推導 pred_x0 並進行數值限制 (重要：防數值噴掉)
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * predicted_noise) / torch.sqrt(alpha_hat)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # C. 複合損失：Recon 專注修復區，VGG 負責全局風格
                loss_recon = nn.functional.l1_loss(pred_x0 * masks, imgs * masks)
                # [關鍵改進] 改為全局 VGG，讓模型參考非遮罩區域的風格
                loss_vgg = vgg_criterion(pred_x0, imgs)

                total_loss = (loss_mse + W_RECON * loss_recon + W_VGG * loss_vgg) / ACCUMULATION_STEPS

            scaler.scale(total_loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(G2.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            if i % 20 == 0:
                print(f"Epoch [{epoch+1}] Step [{i}/{len(train_loader)}] "
                      f"Total: {total_loss.item()*ACCUMULATION_STEPS:.4f} "
                      f"(MSE: {loss_mse.item():.4f}, VGG: {loss_vgg.item():.4f})")

        # 驗證與視覺化
        G2.eval()
        v_imgs, _, v_masks = fixed_batch
        v_imgs, v_masks = v_imgs.to(DEVICE), v_masks.to(DEVICE)
        with torch.no_grad():
            v_masked = v_imgs * (1 - v_masks)
            v_edges = G1(torch.cat([v_masked, v_masks], dim=1))
            v_cond = torch.cat([v_masked, v_masks, v_edges], dim=1)

            model_for_sampling = G2.module if hasattr(G2, 'module') else G2
            v_sampled = diffusion.sample(model_for_sampling, v_cond, n=v_imgs.shape[0])
            v_final = v_masked + v_sampled * v_masks

            save_diffusion_samples(v_imgs, v_masks, v_edges, v_final, f"{SAMPLE_DIR}/epoch_{epoch+1}.png")
            print(f"==> Epoch {epoch+1} PSNR: {compute_psnr((v_final+1)/2, (v_imgs+1)/2):.2f} dB")

if __name__ == "__main__":
    train()