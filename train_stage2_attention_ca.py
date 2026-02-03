import os
import math
import re
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
from src.networks import EdgeGenerator, DiffusionUNet, Discriminator, VGGLoss
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.utils import manual_ssim, save_preview_image

# ================= CONFIG (多卡並行與穩定性優化) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4           # 每張顯卡的 Batch Size
ACCUMULATION_STEPS = 4  # 實質等效 Batch Size = 4 * 2(GPUs) * 12 = 96
EPOCHS = 200
PURE_EPOCHS = 50         # 前 50 Epoch 純像素對齊，不跑 VGG
BASE_LR = 1e-5           # 降低 LR 以防止 Epoch 30 崩潰
SAVE_DIR = "checkpoints_stage2_v13_ultimate"
SAMPLE_DIR = "samples_stage2_v13_ultimate"
LOG_DIR = "runs/stage2_v13_ultimate"
G1_CHECKPOINT = "./checkpoints_sa/G1_latest.pth"
W_ADV_FINAL = 0.005      # 150 Epoch 後的對抗損失權重


# [續練與最佳化紀錄]
START_EPOCH = 0
LOAD_G2_PATH = None      # 從頭開始建議設為 None，或指定之前的穩定權重
best_psnr = 0.0          # 初始化最佳 PSNR 紀錄

LOAD_G2_PATH = f"checkpoints_stage2_v13_sa/G2_v13_sa_latest.pth"


def get_sa_config(epoch):
    if epoch < PURE_EPOCHS:
        # [階段一：結構穩固期]
        # 讓 lr 保持穩定，w_recon 線性從 1.0 爬升到 2.0，w_mse 微降至 0.8
        ratio = epoch / PURE_EPOCHS
        lr = BASE_LR
        w_recon = 1.0 + 0.5 *(1 - math.cos(math.pi * ratio))  # 平滑增加
        w_mse = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))    # 平滑微降
        w_vgg = 0.01                   # 保持低 VGG 壓制彩噪
        w_adv = 0.0

    elif epoch < 150:
        # [階段二：深度感知強化期]
        # 這裡的起點必須對接階段一的終點 (w_recon=2.0, w_mse=0.8, w_vgg=0.01)
        eta = (epoch - PURE_EPOCHS) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta)) # 餘弦平滑係數 (0 -> 1)

        # 學習率退火
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * eta))

        # 權重平滑過渡
        w_mse = 0.8 * (1 - alpha) + 0.2 * alpha    # 0.8 降至 0.2
        w_recon = 2.0 * (1 - alpha) + 1.0 * alpha  # 2.0 降至 1.0 (釋放自由度給 VGG)
        w_vgg = 0.01 * (1 - alpha) + 0.15 * alpha  # 0.01 升至 0.15
        w_adv = 0.0

    else:
        # [階段三：GAN 拋光期]
        lr = BASE_LR * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.2, 0.1
        w_adv = W_ADV_FINAL # 0.005

    return w_recon, w_vgg, w_mse, w_adv, lr
"""
def get_sa_config(epoch):

    get_sa_config 的 Docstring

    :param epoch: 說明
    return: recon_w, vgg_w, mse_w, adv_w, lr
    1-50 Epoch: 純像素對齊期
    51-200 Epoch: 深度感知強化期 (不跑 GAN)
    201-250 Epoch: 終極 GAN 拋光階段
    透過餘弦退火調整學習率與 Loss 權

    lr = BASE_LR
    # 預設值

    if epoch < PURE_EPOCHS:
        w_recon, w_vgg, w_mse, w_adv = 10.0, 0.05, 1.0, 0.0

    elif PURE_EPOCHS <= epoch < 150:
        eta = (epoch - PURE_EPOCHS) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        w_mse = 1.0 * (1 - alpha) + 0.1 * alpha
        w_recon = 0.1 * (1 - alpha) + 1.0 * alpha
        w_vgg = 0.05 * (1 - alpha) + 0.2 * alpha
        lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * eta))
        w_adv = 0.0
    # else:
    #     # Epoch 150 之後進入拋光階段
    #     # w_mse, w_recon, w_vgg = 0.1, 1.0, 0.2
    #     # lr = BASE_LR * 0.1
    #     eta = (epoch - PURE_EPOCHS) / 100
    #     alpha = 0.5 * (1 - math.cos(math.pi * eta))
    #     w_mse = 1.0 * (1 - alpha) + 0.1 * alpha
    #     w_recon = 0.1 * (1 - alpha) + 1.0 * alpha
    #     w_vgg = 0.05 * (1 - alpha) + 0.2 * alpha
    #     lr = BASE_LR * 0.5 * (1 + math.cos(math.pi * eta))
    #     # w_adv = 0.0
    #     w_adv = 0.005

    return w_recon, w_vgg, w_mse, w_adv, lr
"""

"""
def get_sa_config(epoch):
    #動態調整各階段 Loss 權重與學習率
    if epoch < PURE_EPOCHS:
        return 10.0, 0.0, 1.0, 0.0, BASE_LR # Recon, VGG, MSE, Adv, LR
    elif epoch < 150:
        progress = (epoch - PURE_EPOCHS) / (150 - PURE_EPOCHS)
        cos_val = 0.5 * (1 - math.cos(progress * math.pi))
        lr = 5e-6 + (BASE_LR - 5e-6) * (1 - cos_val)
        vgg_w = 0.05 * cos_val
        recon_w = 5.0 + (10.0 * cos_val)
        return recon_w, vgg_w, 1.0, 0.0, lr
    else:
        # 150 Epoch 後進入終極 GAN 拋光階段
        return 15.0, 0.005, 0.5, W_ADV_FINAL, 5e-6
    """







# ================= 主訓練邏輯 =================
def train():
    global best_psnr
    os.makedirs(SAVE_DIR, exist_ok=True); os.makedirs(SAMPLE_DIR, exist_ok=True)
    writer = SummaryWriter(LOG_DIR)

    train_loader = DataLoader(InpaintingDataset("./datasets/img", mode='train'),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)

    G1 = EdgeGenerator().to(DEVICE).eval()
    G1.load_state_dict(torch.load(G1_CHECKPOINT, map_location=DEVICE))

    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = optim.AdamW(G2.parameters(), lr=BASE_LR, weight_decay=1e-4)
    opt_D = optim.AdamW(D.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler_G = torch.amp.GradScaler('cuda')
    scaler_D = torch.amp.GradScaler('cuda')

    # 加載邏輯
    if LOAD_G2_PATH and os.path.exists(LOAD_G2_PATH):
        checkpoint = torch.load(LOAD_G2_PATH, map_location=DEVICE)
        G2.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            opt_G.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[*] 成功載入檢查點")

    if torch.cuda.device_count() > 1:
        G2 = nn.DataParallel(G2); D = nn.DataParallel(D)

    diffusion = DiffusionManager(device=DEVICE)
    vgg_criterion = VGGLoss(DEVICE)

    for epoch in range(START_EPOCH, EPOCHS):
        epoch_l1, epoch_vgg, epoch_mse, epoch_adv = 0.0, 0.0, 0.0, 0.0
        w_recon, w_vgg, w_mse, w_adv, curr_lr = get_sa_config(epoch)
        for param_group in opt_G.param_groups: param_group['lr'] = curr_lr

        G2.train(); D.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for i, (imgs, _, masks) in enumerate(pbar):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            with torch.amp.autocast('cuda'):
                with torch.no_grad():
                    pred_edges = G1(torch.cat([imgs * (1 - masks), masks], dim=1))

                condition = torch.cat([imgs * (1 - masks), masks, pred_edges], dim=1)
                t = diffusion.sample_timesteps(imgs.shape[0]).to(DEVICE)
                x_t, noise = diffusion.noise_images(imgs, t)

                # [🚀 防禦 1] 限制預測噪聲範圍
                pred_noise = G2(x_t, t, condition)
                pred_noise = torch.clamp(torch.nan_to_num(pred_noise), -1, 1.0)
                pred_noise = pred_noise.mean(1, keepdim=True).repeat(1, 3, 1, 1)  # 強制平均通道，防止色偏

                l_mse = F.mse_loss(pred_noise, noise)

                # [🚀 防禦 2] 穩定分母，徹底解決開局彩點
                alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]

                # denom = torch.sqrt(alpha_hat_t).clamp(min=0.35)+ 1e-6
                denom = torch.sqrt(alpha_hat_t + 1e-7)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / denom


                # [🚀 防禦 3] 嚴格截斷像素範圍
                pred_x0 = torch.clamp(torch.nan_to_num(pred_x0), -1.0, 1.0)

                l_pixel = F.l1_loss(pred_x0, imgs)
                l_vgg = vgg_criterion(pred_x0, imgs) if (epoch >= PURE_EPOCHS and w_vgg > 0) else torch.tensor(0.0, device=DEVICE)

                # 判別器博弈邏輯
                l_adv_G = torch.tensor(0.0, device=DEVICE)
                if w_adv > 0:
                    opt_D.zero_grad()
                    real_res = D(imgs)
                    fake_res = D(pred_x0.detach())
                    loss_D = (F.relu(1.0 - real_res).mean() + F.relu(1.0 + fake_res).mean()) * 0.5
                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(opt_D); scaler_D.update()
                    l_adv_G = -D(pred_x0).mean()
                safe_pred = torch.clamp(pred_x0, -1.0, 1.0)
                with torch.no_grad():
                    mask_pixel_count = masks.sum(dim=[1, 2, 3]).clamp(min=1.0)

                pred_x0: torch.Tensor = pred_x0.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
                mask_safe = masks.sum(dim=[1, 2, 3]).clamp(min=1.0).view(-1, 1)
                pred_mask_mean = (pred_x0 * masks).sum(dim=[2, 3]) /mask_safe
                gt_mask_mean = (imgs * masks).sum(dim=[2, 3]) / mask_safe
# 計算局部 Color Loss
                l_color = F.mse_loss(pred_mask_mean, gt_mask_mean)
                loss_total = (l_mse * w_mse + l_vgg * w_vgg + l_pixel * w_recon + l_adv_G * w_adv + l_color * 0.1)/ ACCUMULATION_STEPS

            if torch.isnan(loss_total):
                print(f"[⚠️] NaN loss detected at epoch {epoch}, step {i}")
                continue
            scaler_G.scale(loss_total).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler_G.unscale_(opt_G)
                torch.nn.utils.clip_grad_norm_(G2.parameters(),0.3)
                scaler_G.step(opt_G); scaler_G.update(); opt_G.zero_grad()
                epoch_l1 += l_pixel.item(); epoch_mse += l_mse.item(); epoch_vgg += l_vgg.item()


            if i % 10 == 0:
                with torch.no_grad():
                    psnr_cur = 20 * math.log10(1.0 / (torch.sqrt(F.mse_loss((pred_x0.detach()+1)/2, (imgs+1)/2)) + 1e-9))
                pbar.set_postfix({"L1": f"{l_pixel.item():.4f}", "PSNR": f"{psnr_cur:.2f}"})



        # --- 保存與預覽 ---
        # --- Epoch 結尾紀錄 TensorBoard ---
        avg_steps = len(train_loader) // ACCUMULATION_STEPS
        writer.add_scalar("Loss_Detail/Pixel_L1", epoch_l1 / avg_steps, epoch)
        writer.add_scalar("Loss_Detail/VGG_Perceptual", epoch_vgg / avg_steps, epoch)
        writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)

        G2.eval()
        with torch.no_grad():
            m_sam = G2.module if hasattr(G2, 'module') else G2

            # [🚀 核心功能] 自動生成 Baseline (Epoch 149)
            if epoch == 149:
                print("[*] 生成 No-GAN Baseline...")
                os.makedirs("baseline_samples", exist_ok=True)
                baseline_res = diffusion.sample(m_sam, condition[0:4], n=4, steps=100)
                save_preview_image(imgs[0:4], masks[0:4], pred_edges[0:4], baseline_res, 0.0, epoch, "baseline_samples")

            samples = diffusion.sample(m_sam, condition[0:1], n=1, steps=50)
            res_img = (imgs[0:1] * (1 - masks[0:1])) + (samples * masks[0:1])
            v_psnr = 20 * math.log10(1.0 / (torch.sqrt(F.mse_loss((res_img+1)/2, (imgs[0:1]+1)/2)) + 1e-8))
            v_ssim = manual_ssim((res_img+1)/2, (imgs[0:1]+1)/2).item()

             # [⭐] 最佳化紀錄點
            if v_psnr > best_psnr:
                best_psnr = v_psnr
                torch.save(m_sam.state_dict(), f"{SAVE_DIR}/G2_BEST_PSNR_ep{epoch}.pth")
                # 同時覆蓋一個不帶 epoch 標籤的方便測試
                torch.save(m_sam.state_dict(), f"{SAVE_DIR}/G2_BEST_PSNR_latest.pth")
                print(f"\n[🏆] 突破紀錄! Epoch {epoch} PSNR: {best_psnr:.2f}. 最佳權重已備份。")

            # 紀錄指標
            writer.add_scalar("Metrics/PSNR", v_psnr, epoch)
            writer.add_scalar("Metrics/SSIM", v_ssim, epoch)
            writer.add_image("Preview/Epoch_Res", (res_img[0]+1)/2, epoch)

            # 紀錄損失 (確保變數存在且名稱正確)
          # 紀錄損失 (確保變數存在且名稱正確)
            avg_steps = len(train_loader) // ACCUMULATION_STEPS
            if avg_steps > 0:
                writer.add_scalar("Losses/L1_Loss", epoch_l1 / avg_steps, epoch)
                writer.add_scalar("Losses/VGG_Loss", epoch_vgg / avg_steps, epoch)
                writer.add_scalar("Losses/MSE_Loss", epoch_mse / avg_steps, epoch)
                writer.add_scalar("Losses/Adv_Loss", float(l_adv_G.item()), epoch)

                writer.add_scalar("Losses/Total_Loss", float(loss_total.item()), epoch)
                writer.add_scalar("Weights/w_recon", w_recon, epoch)
                writer.add_scalar("Weights/w_vgg", w_vgg, epoch)
                writer.add_scalar("Weights/w_mse", w_mse, epoch)
                writer.add_scalar("Weights/w_adv", w_adv, epoch)
                writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)
            # 儲存 PNG 預覽圖
            save_preview_image(imgs, masks, pred_edges, res_img, v_psnr, epoch, SAMPLE_DIR)

        # 保存完整字典
        torch.save({
            'epoch': epoch,
            'model_state_dict': m_sam.state_dict(),
            'optimizer_state_dict': opt_G.state_dict(),
            'scaler_state_dict': scaler_G.state_dict(),
            'best_psnr': best_psnr,
        }, f"{SAVE_DIR}/checkpoint_latest.pth")

    writer.close()

if __name__ == "__main__": train()
