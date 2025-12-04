import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.networks import EdgeGenerator, EdgeDiscriminator
from src.dataset import InpaintingDataset
from matplotlib import pyplot as plt

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-4
EPOCHS = 20
BATCH_SIZE = 8

def train():
    # Data
    dataset = InpaintingDataset("./datasets/img", mode='train')
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Models
    G1 = EdgeGenerator(in_ch=4).to(DEVICE)
    D1 = EdgeDiscriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G1.parameters(), lr=LR)
    opt_D = torch.optim.Adam(D1.parameters(), lr=LR * 4)

    l1_loss = nn.L1Loss()
    print(f'size of training set: {len(dataset)}')

    for epoch in range(EPOCHS):
        for i, (imgs, edges, masks) in enumerate(loader):
            imgs = imgs.to(DEVICE)   # RGB [-1, 1]
            edges = edges.to(DEVICE) # Edge GT [0, 1]
            masks = masks.to(DEVICE) # Mask [0, 1]

            # Grayscale image for discriminator
            imgs_gray = 0.299*imgs[:,0] + 0.587*imgs[:,1] + 0.114*imgs[:,2]
            imgs_gray = imgs_gray.unsqueeze(1)

            masked_imgs = imgs * (1 - masks)
            # G Input: Masked RGB (3) + Mask (1) = 4 channels
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # --- Train Discriminator ---
            opt_D.zero_grad()

            fake_edges = G1(g_input)

            # D Input: Gray + Edge + Mask = 3 channels
            d_input_real = torch.cat([imgs_gray, edges, masks], dim=1)
            d_input_fake = torch.cat([imgs_gray, fake_edges.detach(), masks], dim=1)

            pred_real = D1(d_input_real)
            pred_fake = D1(d_input_fake)

            # Hinge Loss for D
            loss_d = torch.mean(torch.relu(1.0 - pred_real)) + torch.mean(torch.relu(1.0 + pred_fake))
            loss_d.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()

            # Re-run D on fake edges (for gradients)
            d_input_fake_g = torch.cat([imgs_gray, fake_edges, masks], dim=1)
            pred_fake_g = D1(d_input_fake_g)

            # Adv Loss + Feature Matching (Simplified to L1 here)
            loss_g_adv = -torch.mean(pred_fake_g)
            loss_g_fm = l1_loss(fake_edges, edges) * 10

            loss_g = loss_g_adv + loss_g_fm
            loss_g.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"Epoch {epoch} [{i}/{len(loader)}] D: {loss_d.item():.4f} G: {loss_g.item():.4f}")

        # Save checkpoint
        torch.save(G1.state_dict(), f"G1_epoch_{epoch}.pth")



def show_images(imgs, masks):
        imgs_gray = imgs_gray.unsqueeze(1)
        masked_imgs = imgs * (1 - masks)

        # 取得目前這個 batch 有幾張圖 (通常等於 BATCH_SIZE，但在最後一個 batch 可能較少)
        current_batch_size = imgs.size(0)

        # 設定畫布大小：高度隨 batch 數量變動 (每一列高 4 inch)
        plt.figure(figsize=(15, 4 * current_batch_size))

        # --- 遍歷 Batch 中的每一張圖 ---
        for k in range(current_batch_size):
            # 取出第 k 張圖
            img_g = imgs_gray[k].detach().cpu().squeeze()
            img_m = masks[k].detach().cpu().squeeze()
            img_masked = masked_imgs[k].detach().cpu().permute(1, 2, 0)

            # 正規化: [-1, 1] -> [0, 1]
            img_g = (img_g + 1) / 2
            img_masked = (img_masked + 1) / 2

            # --- 繪圖邏輯 ---
            # subplot index 計算方式: (總列數, 總欄數, 當前第幾張圖)
            # 每一列有 3 張圖 (Gray, Mask, Masked)

            # 1. Gray Input
            plt.subplot(current_batch_size, 3, k * 3 + 1)
            plt.title(f"Batch[{k}] Grayscale")
            plt.imshow(img_g, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')

            # 2. Mask
            plt.subplot(current_batch_size, 3, k * 3 + 2)
            plt.title(f"Batch[{k}] Mask")
            plt.imshow(img_m, cmap='gray')
            plt.axis('off')

            # 3. Masked Image (RGB)
            plt.subplot(current_batch_size, 3, k * 3 + 3)
            plt.title(f"Batch[{k}] Masked")
            plt.imshow(img_masked)
            plt.axis('off')

        plt.tight_layout() # 自動調整間距避免標題重疊
        plt.savefig('batch_demo.png')
        plt.close()

        print(f"Saved batch_demo.png with {current_batch_size} rows.")


if __name__ == "__main__":
    train()