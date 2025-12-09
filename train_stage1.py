import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.networks import EdgeGenerator, EdgeDiscriminator
from src.dataset import InpaintingDataset
from matplotlib import pyplot as plt

# --- 忽略 NVML 警告 (針對 Docker/WSL2 環境) ---
warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4           # 學習率

# [修改 1] 增加訓練回合數
# 2500張圖 / batch 16 = 156 iters/epoch
# 400 epochs * 156 = 62,400 iters (達到及格訓練量)
EPOCHS = 100

BATCH_SIZE = 16     # 雙 GPU 建議開大一點 (原本是 8)

def save_sample(imgs, masks, edges, fake_edges, epoch, step):
    """
    視覺化並儲存訓練過程中的圖片
    """
    os.makedirs("samples", exist_ok=True)

    # 將數據移回 CPU
    imgs = imgs.detach().cpu()
    masks = masks.detach().cpu()
    edges = edges.detach().cpu()
    fake_edges = fake_edges.detach().cpu()

    # 計算灰階圖
    imgs_gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
    imgs_gray = imgs_gray.unsqueeze(1)

    masked_imgs = imgs * (1 - masks)

    # 取 Batch 中的第一張圖來畫
    k = 0

    img_g = imgs_gray[k].squeeze()
    img_m = masks[k].squeeze()
    img_masked = masked_imgs[k].permute(1, 2, 0)
    img_edge_gt = edges[k].squeeze()
    img_edge_fake = fake_edges[k].squeeze()

    # 正規化還原 [-1, 1] -> [0, 1]
    img_g = (img_g + 1) / 2
    img_masked = (img_masked + 1) / 2

    plt.figure(figsize=(20, 4))

    # 畫圖部分省略修改，保持原樣
    titles = ["Original Gray", "Mask", "Masked Input", "GT Edge", "Generated Edge"]
    images = [img_g, img_m, img_masked, img_edge_gt, img_edge_fake]

    for idx, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 5, idx+1)
        plt.title(title)
        if title == "Masked Input":
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"samples/epoch_{epoch}_step_{step}.png")
    plt.close()

def train():
    # Data
    dataset = InpaintingDataset("./datasets/img", mode='train')

    # [建議] drop_last=True 可以避免最後一個 batch 只有 1,2 張圖導致 BatchNorm 報錯
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f'Size of training set: {len(dataset)}')

    # Models
    G1 = EdgeGenerator(in_ch=4).to(DEVICE)
    D1 = EdgeDiscriminator(in_ch=3).to(DEVICE)

    # --- 多 GPU 設定 ---
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        G1 = nn.DataParallel(G1)
        D1 = nn.DataParallel(D1)

    # Optimizers
    opt_G = torch.optim.Adam(G1.parameters(), lr=LR, betas=(0.0, 0.9)) # [建議] GAN 常用的 beta 設定
    opt_D = torch.optim.Adam(D1.parameters(), lr=LR, betas=(0.0, 0.9))

    l1_loss = nn.L1Loss()

    print("Start Training...")

    for epoch in range(EPOCHS):
        for i, (imgs, edges, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            edges = edges.to(DEVICE)
            masks = masks.to(DEVICE)

            # Grayscale for D
            imgs_gray = 0.299*imgs[:,0] + 0.587*imgs[:,1] + 0.114*imgs[:,2]
            imgs_gray = imgs_gray.unsqueeze(1)

            masked_imgs = imgs * (1 - masks)
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            fake_edges = G1(g_input)

            d_input_real = torch.cat([imgs_gray, edges, masks], dim=1)
            d_input_fake = torch.cat([imgs_gray, fake_edges.detach(), masks], dim=1)

            pred_real = D1(d_input_real)
            pred_fake = D1(d_input_fake)

            loss_d = torch.mean(torch.relu(1.0 - pred_real)) + torch.mean(torch.relu(1.0 + pred_fake))
            loss_d.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()

            d_input_fake_g = torch.cat([imgs_gray, fake_edges, masks], dim=1)
            pred_fake_g = D1(d_input_fake_g)

            loss_g_adv = -torch.mean(pred_fake_g)
            loss_g_fm = l1_loss(fake_edges, edges) * 10

            loss_g = loss_g_adv + loss_g_fm
            loss_g.backward()
            opt_G.step()

            # Log & Sample
            if i % 50 == 0:
                print(f"Epoch {epoch}/{EPOCHS} [{i}/{len(loader)}] D: {loss_d.item():.4f} G: {loss_g.item():.4f}")
                save_sample(imgs, masks, edges, fake_edges, epoch, i)

        # [修正 1] 確保 checkpoints 資料夾存在 (加在 train 函式最前面也可以，或放在這裡)
        os.makedirs("checkpoints", exist_ok=True)

        # [修正 2] 修正頻率
        # 既然總共 100 Epochs，建議每 10 或 20 個 Epoch 存一次
        # (epoch + 1) % 10 == 0 代表：第 10, 20, 30... 才會存
        if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
            save_name = f"checkpoints/G1_epoch_{epoch+1}.pth"
            print(f"Saving model: {save_name}")

            if isinstance(G1, nn.DataParallel):
                torch.save(G1.module.state_dict(), save_name)
            else:
                torch.save(G1.state_dict(), save_name)

if __name__ == "__main__":
    train()