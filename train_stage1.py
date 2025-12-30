import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import os

# 引入專案模組
from src.networks import EdgeGenerator, EdgeDiscriminator
from src.dataset import InpaintingDataset
from src.utils import compute_psnr, save_sample_images

# =========================================
#                CONFIG 設定
# =========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
EPOCHS = 20
# 雙卡建議：如果您有兩張 24GB，BATCH_SIZE 可以設為 16 或 32 (每張卡跑 8-16)
BATCH_SIZE = 16
SAVE_DIR = "checkpoints"
SAMPLE_DIR = "images_stage1"

def validate(model, loader):
    """驗證集評估函數"""
    model.eval()
    total_psnr = 0.0
    count = 0
    with torch.no_grad():
        for imgs, edges, masks in loader:
            imgs, edges, masks = imgs.to(DEVICE), edges.to(DEVICE), masks.to(DEVICE)
            masked_imgs = imgs * (1 - masks)
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # 使用 DataParallel 時呼叫方式不變
            pred_edges = model(g_input)

            mse = torch.mean((pred_edges - edges) ** 2)
            psnr = 10 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0
            total_psnr += psnr
            count += 1
    return total_psnr / count if count > 0 else 0

def train():
    # 0. 準備資料夾
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    if not os.path.exists(SAMPLE_DIR): os.makedirs(SAMPLE_DIR)

    # 1. 準備資料
    print("Loading Datasets...")
    train_dataset = InpaintingDataset("./datasets/img", mode='train')
    # 雙卡模式下，num_workers 可以設高一點 (例如 4 或 8) 以避免資料讀取成為瓶頸
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    try:
        val_dataset = InpaintingDataset("./datasets", mode='valid')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        print("Validation set loaded.")
    except:
        val_loader = None
        print("Warning: No validation set found.")

    # 2. 初始化模型
    G1 = EdgeGenerator().to(DEVICE)
    D1 = EdgeDiscriminator().to(DEVICE)

    # [雙卡關鍵] 使用 DataParallel 包裝模型
    if torch.cuda.device_count() > 1:
        print(f"[*] Detected {torch.cuda.device_count()} GPUs. Enabling DataParallel.")
        G1 = nn.DataParallel(G1)
        D1 = nn.DataParallel(D1)

    # 3. 優化器與 Loss
    opt_G = torch.optim.Adam(G1.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D1.parameters(), lr=LR*4, betas=(0.0, 0.9))

    scaler = GradScaler()
    l1_loss = nn.L1Loss()

    print(f"--- Stage 1 Dual-GPU Training Start on {DEVICE} ---")

    for epoch in range(EPOCHS):
        # 如果是 DataParallel，呼叫 .train() 會傳遞給內部的 module
        G1.train()
        for i, (imgs, edges, masks) in enumerate(train_loader):
            imgs, edges, masks = imgs.to(DEVICE), edges.to(DEVICE), masks.to(DEVICE)

            # 準備輸入
            imgs_gray = 0.299*imgs[:,0] + 0.587*imgs[:,1] + 0.114*imgs[:,2]
            imgs_gray = imgs_gray.unsqueeze(1)
            masked_imgs = imgs * (1 - masks)
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # ===============================
            #       Train Discriminator
            # ===============================
            opt_D.zero_grad()
            with autocast():
                fake_edges = G1(g_input)

                d_input_real = torch.cat([imgs_gray, edges, masks], dim=1)
                d_input_fake = torch.cat([imgs_gray, fake_edges.detach(), masks], dim=1)

                pred_real = D1(d_input_real)
                pred_fake = D1(d_input_fake)

                loss_d_real = torch.mean(torch.relu(1.0 - pred_real))
                loss_d_fake = torch.mean(torch.relu(1.0 + pred_fake))
                loss_d = (loss_d_real + loss_d_fake) / 2

            scaler.scale(loss_d).backward()
            scaler.step(opt_D)
            scaler.update()

            # ===============================
            #         Train Generator
            # ===============================
            opt_G.zero_grad()
            with autocast():
                fake_edges_g = G1(g_input)
                d_input_fake_g = torch.cat([imgs_gray, fake_edges_g, masks], dim=1)
                pred_fake_g = D1(d_input_fake_g)

                loss_g_adv = -torch.mean(pred_fake_g)
                loss_g_fm = l1_loss(fake_edges_g, edges) * 10
                loss_g = loss_g_adv + loss_g_fm

            scaler.scale(loss_g).backward()
            scaler.step(opt_G)
            scaler.update()

            # ===============================
            #       Log & Visualization
            # ===============================
            if i % 50 == 0:
                with torch.no_grad():
                    mse = torch.mean((fake_edges_g - edges) ** 2)
                    current_psnr = 10 * torch.log10(1.0 / mse).item() if mse > 0 else 100.0

                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"D: {loss_d.item():.4f} G: {loss_g.item():.4f} PSNR: {current_psnr:.2f}")

                save_path = f"{SAMPLE_DIR}/epoch_{epoch}_batch_{i}.png"
                # DataParallel 不影響 save_sample_images 的邏輯
                save_sample_images(imgs, masks, edges, fake_edges_g, save_path)
        # Epoch 結束驗證
        if val_loader:
            v_psnr = validate(G1, val_loader)
            print(f"==> Epoch {epoch} Complete Valid PSNR: {v_psnr:.2f} dB")

        # [關鍵] 存權重時，需判斷是否為 DataParallel，存取 .module.state_dict()
        save_model = G1.module if hasattr(G1, 'module') else G1
        torch.save(save_model.state_dict(), f"{SAVE_DIR}/G1_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()