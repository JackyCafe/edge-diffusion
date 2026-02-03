import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import models
import numpy as np

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# ================= CONFIG (純淨開局 + 模擬退火) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
ACCUMULATION_STEPS = 12
EPOCHS = 200

BASE_LR = 5e-5           # 初期使用略高 LR 快速收斂結構
SAVE_DIR = "checkpoints_stage2_v13_sa"
SAMPLE_DIR = "samples_stage2_v13_sa"
LOG_DIR = "runs/stage2_v13_sa"
G1_CHECKPOINT = "./checkpoints_sa/G1_latest.pth"

def get_sa_config(epoch):
    """ 使用餘弦函數優化權重與學習率排程 """
    if epoch < PURE_EPOCHS:
        # 第一階段：純淨開局 (No VGG)
        return 10.0, 0.0, 1.0, BASE_LR  # Recon, VGG, MSE, LR

    else:
        # 第二、三階段：餘弦退火 (Cosine Annealing)
        # 計算相對進度 (0.0 到 1.0)
        progress = (epoch - PURE_EPOCHS) / (EPOCHS - PURE_EPOCHS)

        # 餘弦係數 (從 0 變到 1)
        # 1 - cos(progress * pi) / 2 會產生一條平滑的 S 型曲線
        cos_val = 0.5 * (1 - math.cos(progress * math.pi))

        # 1. 學習率：從 BASE_LR 平滑降至 5e-6
        target_lr = 5e-6
        lr = target_lr + (BASE_LR - target_lr) * (1 - cos_val)

        # 2. VGG 權重：從 0.0 平滑升至 0.05
        # 這樣細節會「緩慢地」浮現，不會突然衝擊權重導致網格
        vgg_w = 0.05 * cos_val

        # 3. Recon 權重：從 5.0 平滑升至 15.0，強化像素對齊
        recon_w = 5.0 + (10.0 * cos_val)

        return recon_w, vgg_w, 1.0, lr
# ===============================================================

class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg[:19].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.content_layers = [4, 9, 18] # 捨棄 27 層，防止網格感

    def forward(self, pred, target):
        # 歸一化到 VGG 預期範圍 [0, 1]
        p = (pred + 1) / 2
        t = (target + 1) / 2
        loss = 0
        px, tx = p, t
        for i, layer in enumerate(self.vgg):
            px, tx = layer(px), layer(tx)
            if i in self.content_layers:
                loss += F.l1_loss(px, tx)
        return loss

def train():
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)
    writer = SummaryWriter(LOG_DIR)

    train_loader = DataLoader(InpaintingDataset("./datasets/img", mode='train'),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    G1 = EdgeGenerator().to(DEVICE).eval()
    G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    if torch.cuda.device_count() > 1: G2 = nn.DataParallel(G2)

    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)

    # 初始化優化器
    opt = optim.AdamW(G2.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda')

    print(f"--- 啟動 v13 純淨開局 + 模擬退火模式 ---")

    for epoch in range(EPOCHS):
        w_recon, w_vgg, w_mse, curr_lr = get_sa_config(epoch)

        # 手動更新學習率 (模擬退火)
        for param_group in opt.param_groups:
            param_group['lr'] = curr_lr

        G2.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [LR: {curr_lr:.2e}]")

        for i, (imgs, _, masks) in enumerate(pbar):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            with torch.no_grad():
                pred_edges = G1(torch.cat([imgs * (1 - masks), masks], dim=1))

            condition = torch.cat([imgs * (1 - masks), masks, pred_edges], dim=1)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(imgs, t)

            with torch.amp.autocast('cuda'):
                pred_noise = G2(x_t, t, condition)
                l_mse = F.mse_loss(pred_noise, noise) * w_mse

                # 反推 x0 並 Clamp
                alpha_hat = diffusion.alpha_hat[t][:, None, None, None]
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * pred_noise) / torch.sqrt(alpha_hat)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # 計算感知與重建損失
                loss_recon = F.l1_loss(pred_x0, imgs) * w_recon
                loss_vgg = vgg_criterion(pred_x0, imgs) * w_vgg if w_vgg > 0 else torch.tensor(0.0).to(DEVICE)

                loss_total = (l_mse + loss_vgg + loss_recon) / ACCUMULATION_STEPS

            scaler.scale(loss_total).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(G2.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()

            if i % 20 == 0:
                pbar.set_postfix({"PSNR_Est": f"{20 * np.log10(1.0 / np.sqrt(F.mse_loss(pred_x0, imgs).item())):.2f}"})

        # 每 5 Epoch 儲存並採樣預覽
        if epoch % 1 == 0:
            G2.eval()
            with torch.no_grad():
                model_sam = G2.module if hasattr(G2, 'module') else G2
                samples = diffusion.sample(model_sam, condition[0:1], n=1, steps=50)
                # 這裡 sample 內部已經有 clamp
                res = (imgs[0:1] * (1 - masks[0:1])) + (samples * masks[0:1])
                plt.imsave(f"{SAMPLE_DIR}/v13_sa_ep{epoch}.png", (res[0].cpu().permute(1,2,0).numpy()+1)/2)

            torch.save(model_sam.state_dict(), f"{SAVE_DIR}/G2_v13_sa_latest.pth")

    print("--- 訓練完成 ---")

if __name__ == "__main__":
    train()