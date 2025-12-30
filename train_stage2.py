import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os
import matplotlib.pyplot as plt
import numpy as np

from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.utils import compute_psnr

# ================= CONFIG (雙卡優化版) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 5e-5
EPOCHS = 200

# 雙卡設定：每張卡跑 2，總共 Batch = 4
BATCH_SIZE = 4
ACCUMULATION_STEPS = 2  # Effective Batch = 4 * 2 = 8
MAX_GRAD_NORM = 1.0

SAVE_DIR = "checkpoints"
SAMPLE_DIR = "samples_stage2"
G1_CHECKPOINT = f"{SAVE_DIR}/G1_epoch_19.pth"

RESUME_TRAINING = False
RESUME_CHECKPOINT = f"{SAVE_DIR}/G2_diffusion_latest.pth"

VAL_INTERVAL = 1
SAVE_INTERVAL = 10
# =====================================================

def save_diffusion_samples(v_imgs, v_masks, v_edges, v_final, save_path):
    try:
        idx = 0
        img_orig = (v_imgs[idx].cpu().permute(1, 2, 0).numpy() + 1) / 2
        img_mask = v_masks[idx].cpu().squeeze().numpy()
        img_edge = v_edges[idx].cpu().squeeze().numpy()

        raw_out = v_final[idx].detach().cpu().permute(1, 2, 0).numpy()
        img_out = (raw_out + 1) / 2
        img_masked = img_orig * (1 - img_mask[:, :, None])

        plt.figure(figsize=(20, 5))
        images = [img_orig, img_masked, img_edge, img_out]
        titles = ['Original', 'Masked Input', 'G1 Edge Guide', 'Diffusion Result']

        for i, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 4, i + 1)
            plt.title(title)
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

    # 注意：num_workers 建議設為 4 或 8 (前提是 shm-size 已加載)
    train_dataset = InpaintingDataset("./datasets/img", mode='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = InpaintingDataset("./datasets/img", mode='valid')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    fixed_batch = next(iter(val_loader))

    # --- 模型初始化 ---
    G1 = EdgeGenerator().to(DEVICE)
    if os.path.exists(G1_CHECKPOINT):
        G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))
    G1.eval()

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)

    # [雙卡關鍵] 檢查並使用 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"[*] Using {torch.cuda.device_count()} GPUs!")
        G2 = nn.DataParallel(G2)

    diffusion = DiffusionManager(device=DEVICE)
    opt = torch.optim.AdamW(G2.parameters(), lr=LR, weight_decay=1e-4)
    scaler = GradScaler()
    mse_loss = nn.MSELoss()

    start_epoch = 0
    if RESUME_TRAINING and os.path.exists(RESUME_CHECKPOINT):
        ckpt = torch.load(RESUME_CHECKPOINT)
        # 如果儲存時用了 DataParallel，這裡讀取要小心
        msg = G2.module.load_state_dict(ckpt['model_state_dict']) if hasattr(G2, 'module') else G2.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch']
        print(f"[*] Resumed: {msg}")

    print(f"--- Stage 2: Dual-GPU Training Start (Effective Batch: {BATCH_SIZE * ACCUMULATION_STEPS}) ---")

    for epoch in range(start_epoch, EPOCHS):
        G2.train()
        total_loss = 0

        for i, (imgs, _, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with torch.no_grad():
                masked_imgs = imgs * (1 - masks)
                pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)
            t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
            x_t, noise = diffusion.noise_images(imgs, t)

            with autocast():
                predicted_noise = G2(x_t, t, condition)
                loss = mse_loss(predicted_noise, noise) / ACCUMULATION_STEPS

            if torch.isnan(loss):
                opt.zero_grad()
                continue

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(G2.parameters(), MAX_GRAD_NORM)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(train_loader)}] Loss: {loss.item()*ACCUMULATION_STEPS:.4f}")

        # 視覺化與採樣
        if (epoch + 1) % VAL_INTERVAL == 0:
            G2.eval()
            v_imgs, _, v_masks = fixed_batch
            v_imgs, v_masks = v_imgs.to(DEVICE), v_masks.to(DEVICE)
            with torch.no_grad():
                v_masked = v_imgs * (1 - v_masks)
                v_edges = G1(torch.cat([v_masked, v_masks], dim=1))
                v_cond = torch.cat([v_masked, v_masks, v_edges], dim=1)

                # 注意：DataParallel 物件沒有自定義 method，需用 .module
                model_for_sampling = G2.module if hasattr(G2, 'module') else G2
                v_sampled = diffusion.sample(model_for_sampling, v_cond, n=v_imgs.shape[0])
                v_final = v_masked + v_sampled * v_masks

                save_diffusion_samples(v_imgs, v_masks, v_edges, v_final, f"{SAMPLE_DIR}/epoch_{epoch+1}.png")
                print(f"==> Epoch {epoch+1} PSNR: {compute_psnr((v_final+1)/2, (v_imgs+1)/2):.2f} dB")
            G2.train()

        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': G2.module.state_dict() if hasattr(G2, 'module') else G2.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
            }
            torch.save(checkpoint, f"{SAVE_DIR}/G2_diffusion_latest.pth")

if __name__ == "__main__":
    train()