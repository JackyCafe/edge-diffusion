import torch
import os
import matplotlib.pyplot as plt
from src.networks import EdgeGenerator
from src.dataset import InpaintingDataset
from torch.utils.data import DataLoader

# 設定
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G1_PATH = "./checkpoints/G1_epoch_85.pth"
IMG_PATH = "./datasets/img" # 您的圖片路徑

def test_g1():
    # 1. 載入模型
    model = EdgeGenerator().to(DEVICE)
    if not os.path.exists(G1_PATH):
        print(f"Error: 找不到權重檔案 {G1_PATH}")
        return

    model.load_state_dict(torch.load(G1_PATH, map_location=DEVICE))
    model.eval()
    print(f"[*] G1 權重載入成功: {G1_PATH}")

    # 2. 準備測試資料
    dataset = InpaintingDataset(IMG_PATH, mode='test')
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    imgs, _, masks = next(iter(loader))
    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

    # 3. 測試生成
    with torch.no_grad():
        masked_imgs = imgs * (1 - masks)
        # G1 輸入為 4 通道 (Masked Image 3 + Mask 1)
        pred_edges = model(torch.cat([masked_imgs, masks], dim=1))

    # 4. 視覺化結果
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow((imgs[0].cpu().permute(1, 2, 0) + 1) / 2)
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(masks[0, 0].cpu(), cmap='gray')
    plt.title("Mask")

    plt.subplot(1, 3, 3)
    # 關鍵：檢查這裡是否有線條
    plt.imshow(pred_edges[0, 0].cpu(), cmap='gray')
    plt.title("G1 Output")

    plt.savefig("debug_g1_test.png")
    print("[*] 測試圖片已儲存至 debug_g1_test.png")

    # 數值檢查
    edge_max = torch.max(pred_edges).item()
    print(f"[*] 邊緣圖最大像素值: {edge_max:.4f}")
    if edge_max <= 0:
        print("警告: G1 輸出為全黑！請檢查權重是否損壞或輸入是否正確。")

if __name__ == "__main__":
    test_g1()