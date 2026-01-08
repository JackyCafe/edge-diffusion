import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import math
from tqdm import tqdm

# 引入專案模組 (請確保路徑正確)
from src.networks import EdgeGenerator, EdgeDiscriminator
from src.dataset import InpaintingDataset
from src.utils import save_sample_images

# =========================================
#                CONFIG 設定
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_LR = 2e-5
EPOCHS = 200
BATCH_SIZE = 16
LOG_DIR = "runs/edge_sa_v1"        # TensorBoard 紀錄資料夾
SAVE_DIR = "checkpoints_sa"        # 模型權重儲存資料夾
SAMPLE_DIR = "images_stage1_sa"    # 訓練預覽圖儲存資料夾

# 模擬退火權重範圍 (SA Weights)
# 初期：高 FM (探索結構), 低 Pixel (允許誤差)
# 後期：低 FM (穩定特徵), 高 Pixel (精確對齊與去鬼影)
W_FM_START, W_FM_END = 50.0, 20.0
W_PIXEL_START, W_PIXEL_END = 10.0, 45.0
W_ADV = 1.0
RAMP_UP_EPOCHS = 10  # 前 10 Epoch 進行平滑過渡

def get_sa_weights(epoch, total_epochs):
    # 1. 基礎模擬退火 (SA) 曲線：使用餘弦冷卻 (1.0 -> 0.0)
    # 在總進度的 80% 處完成退火過渡
    fraction = min(epoch / (total_epochs * 0.8), 1.0)
    cooling_factor = 0.5 * (1 + math.cos(math.pi * fraction))

    target_fm = W_FM_END + (W_FM_START - W_FM_END) * cooling_factor
    target_pixel = W_PIXEL_END + (W_PIXEL_START - W_PIXEL_END) * cooling_factor

    # 2. 線性平滑 (Ramping)：防止第一週就因為 W_FM=50 產生柵欄紋
    if epoch < RAMP_UP_EPOCHS:
        ramp_factor = epoch / RAMP_UP_EPOCHS
        # 從較溫和的權重慢慢升到 SA 目標值
        curr_fm = target_fm * ramp_factor + 5.0 * (1 - ramp_factor)
        curr_pixel = target_pixel * ramp_factor + 10.0 * (1 - ramp_factor)
    else:
        curr_fm = target_fm
        curr_pixel = target_pixel

    return curr_fm, curr_pixel

def train():
    # 建立目錄
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)

    # 初始化 TensorBoard
    writer = SummaryWriter(LOG_DIR)

    print("Loading Datasets (20% Mask Ratio)...")
    # 強制設定為 20% 遮罩比例進行挑戰
    dataset = InpaintingDataset("./datasets/img", mode='train')
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # 初始化模型
    G1 = EdgeGenerator().to(DEVICE)
    D1 = EdgeDiscriminator().to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"[*] 使用 {torch.cuda.device_count()} 顆 GPU 進行平行訓練")
        G1 = nn.DataParallel(G1)
        D1 = nn.DataParallel(D1)

    # 優化器設定 (TTUR 策略: D 的學習率是 G 的 4 倍)
    opt_G = optim.Adam(G1.parameters(), lr=BASE_LR, betas=(0.0, 0.9))
    opt_D = optim.Adam(D1.parameters(), lr=BASE_LR * 4, betas=(0.0, 0.9))

    # 學習率排程：前 100 epoch 持平，後 100 epoch 線性衰減至 0
    lr_lambda = lambda epoch: 1.0 if epoch < 100 else 1.0 - (epoch - 100) / (EPOCHS - 100)
    sch_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=lr_lambda)
    sch_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda=lr_lambda)

    scaler = GradScaler()
    l1_loss = nn.L1Loss()

    print(f"--- Stage 1: Edge Generation [Simulated Annealing Mode] ---")

    for epoch in range(EPOCHS):
        # 獲取本輪退火權重
        curr_fm, curr_pixel = get_sa_weights(epoch, EPOCHS)
        G1.train()

        running_psnr = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for i, (imgs, edges, masks) in enumerate(pbar):
            imgs, edges, masks = imgs.to(DEVICE), edges.to(DEVICE), masks.to(DEVICE)

            # 準備輸入：轉灰階用於 D，Masked 影像用於 G
            imgs_gray = (0.299*imgs[:,0] + 0.587*imgs[:,1] + 0.114*imgs[:,2]).unsqueeze(1)
            masked_imgs = imgs * (1 - masks)
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # ===============================
            #       1. 訓練判別器 (D)
            # ===============================
            opt_D.zero_grad()
            with autocast():
                fake_edges = G1(g_input).detach()
                pred_real, _ = D1(torch.cat([imgs_gray, edges, masks], dim=1))
                pred_fake, _ = D1(torch.cat([imgs_gray, fake_edges, masks], dim=1))

                loss_d_real = torch.mean(torch.relu(1.0 - pred_real))
                loss_d_fake = torch.mean(torch.relu(1.0 + pred_fake))
                loss_d = (loss_d_real + loss_d_fake) / 2

            scaler.scale(loss_d).backward()
            scaler.step(opt_D)

            # ===============================
            #       2. 訓練生成器 (G)
            # ===============================
            opt_G.zero_grad()
            with autocast():
                fake_edges_g = G1(g_input)
                # D1 回傳預測結果與特徵圖以計算 FM Loss
                pred_fake_g, feat_fake = D1(torch.cat([imgs_gray, fake_edges_g, masks], dim=1))
                _, feat_real = D1(torch.cat([imgs_gray, edges, masks], dim=1))

                # 對抗損失
                loss_g_adv = -torch.mean(pred_fake_g) * W_ADV

                # Feature Matching Loss (退火引導)
                loss_g_fm = sum(l1_loss(f_f, f_r) for f_f, f_r in zip(feat_fake, feat_real)) * curr_fm

                # 像素級重建損失 (退火約束)
                loss_g_pixel = l1_loss(fake_edges_g, edges) * curr_pixel

                loss_g = loss_g_adv + loss_g_fm + loss_g_pixel

            scaler.scale(loss_g).backward()
            scaler.step(opt_G)
            scaler.update()

            # 計算即時 PSNR
            with torch.no_grad():
                mse = torch.mean((fake_edges_g - edges) ** 2)
                psnr = 10 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0
                running_psnr += psnr

            # ===============================
            #       Console & TensorBoard
            # ===============================
            if i % 10 == 0:
                # 更新 Console 進度條後方資訊
                pbar.set_postfix({
                    "PSNR": f"{psnr:.2f}",
                    "FM_Loss": f"{loss_g_fm.item():.3f}",
                    "Pix_Loss": f"{loss_g_pixel.item():.3f}",
                    # "W_FM": f"{curr_fm:.1f}",
                    # "W_Pix": f"{curr_pixel:.1f}"
                })

                # 寫入 TensorBoard
                global_step = epoch * len(train_loader) + i
                writer.add_scalar("Loss/G_Total", loss_g.item(), global_step)
                writer.add_scalar("Loss/G_FM", loss_g_fm.item(), global_step)
                writer.add_scalar("Loss/G_Pixel", loss_g_pixel.item(), global_step)
                writer.add_scalar("Loss/D", loss_d.item(), global_step)
                writer.add_scalar("Metrics/PSNR", psnr, global_step)
                writer.add_scalar("Weights/W_FM", curr_fm, global_step)
                writer.add_scalar("Weights/W_Pixel", curr_pixel, global_step)

        # 更新學習率
        sch_G.step()
        sch_D.step()

        # 每 5 個 Epoch 儲存預覽圖與模型
        if epoch % 1 == 0:
            save_path = f"{SAMPLE_DIR}/epoch_{epoch}.png"
            save_sample_images(imgs, masks, edges, fake_edges_g, save_path)

            # 處理 DataParallel 存檔問題
            save_model = G1.module if hasattr(G1, 'module') else G1
            torch.save(save_model.state_dict(), f"{SAVE_DIR}/G1_SA_epoch_{epoch}.pth")
            print(f"\n[!] 預覽圖已儲存至 {save_path}, 模型已存檔。")

    writer.close()
    print("訓練完成！")

if __name__ == "__main__":
    train()