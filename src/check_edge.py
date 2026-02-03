import torch
import torchvision.utils as vutils
import torch.nn as nn
from src.networks import EdgeGenerator
from src.dataset import InpaintingDataset
from torch.utils.data import DataLoader
import os

def check_g1():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 載入模型
    G1 = EdgeGenerator().to(device).eval()
    checkpoint_path = "./checkpoints_sa/G1_SA_epoch_84.pth"

    if not os.path.exists(checkpoint_path):
        print(f"[!] 找不到權重文件: {checkpoint_path}")
        return

    G1.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[*] 已載入權重: {checkpoint_path}")

    # 2. 準備測試數據
    dataset = InpaintingDataset("./datasets/img", mode='test')
    loader = DataLoader(dataset, batch_size=4)
    imgs, edges, masks = next(iter(loader))

    imgs = imgs.to(device)
    edges = edges.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        # G1 輸入：[Masked Img, Mask]
        masked_imgs = imgs * (1 - masks)
        pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

    # 3. [關鍵修正] 統一通道數以便拼接
    # 將 1 通道的 edges 轉換成 3 通道 (重複三次)
    edges_3ch = edges.repeat(1, 3, 1, 1)
    pred_edges_3ch = pred_edges.repeat(1, 3, 1, 1)
    masked_imgs_3ch = masked_imgs # 本身就是 3 通道

    # 拼接順序：遮罩輸入 | 真值邊緣 (GT) | 模型預測邊緣
    vis = torch.cat([masked_imgs_3ch, edges_3ch, pred_edges_3ch], dim=3)

    # 4. 存檔
    vutils.save_image(vis.cpu(), "check_edge_v2.png", normalize=True)
    print("[*] 檢查圖已存至 check_edge_v2.png")
    print("[!] 請打開圖片確認第三欄：如果是全白或模糊一片，代表 Stage 1 必須重跑。")

if __name__ == "__main__":
    check_g1()