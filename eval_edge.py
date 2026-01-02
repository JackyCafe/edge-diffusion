import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入您的專案模組
from src.networks import EdgeGenerator
from src.dataset import InpaintingDataset

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "checkpoints"
IMAGE_SIZE = 512
BATCH_SIZE = 8
# 這裡設定您想評估的 Epoch 列表，例如每 5 個 Epoch 評估一次
EPOCHS_TO_EVAL = range(0, 100, 5)
# ==========================================

def calculate_iou(pred, target, threshold=0.5):
    """計算邊緣二值化後的 IoU"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = (pred_bin + target_bin).clamp(0, 1).sum(dim=(1, 2, 3))

    # 避免除以 0
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

@torch.no_grad()
def evaluate_g1():
    # 1. 準備驗證資料
    print("[*] Loading validation dataset...")
    val_dataset = InpaintingDataset("./datasets/img", mode='valid', img_size=IMAGE_SIZE)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []

    # 2. 遍歷指定 Epoch
    for epoch in EPOCHS_TO_EVAL:
        ckpt_path = f"{CHECKPOINT_DIR}/G1_epoch_{epoch}.pth"
        if not os.path.exists(ckpt_path):
            print(f"[!] Skip Epoch {epoch}: Checkpoint not found.")
            continue

        # 載入模型 (自動適應加深後的 G1)
        model = EdgeGenerator().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        total_mse = 0
        total_iou = 0
        count = 0

        print(f"[*] Evaluating Epoch {epoch}...")
        for imgs, edges, masks in tqdm(val_loader, leave=False):
            imgs, edges, masks = imgs.to(DEVICE), edges.to(DEVICE), masks.to(DEVICE)

            # 準備 G1 輸入
            masked_imgs = imgs * (1 - masks)
            g_input = torch.cat([masked_imgs, masks], dim=1)

            # 推論
            pred_edges = model(g_input)

            # 計算指標
            mse = nn.functional.mse_loss(pred_edges, edges).item()
            iou = calculate_iou(pred_edges, edges)

            total_mse += mse
            total_iou += iou
            count += 1

        avg_mse = total_mse / count
        avg_iou = total_iou / count
        results.append({"epoch": epoch, "mse": avg_mse, "iou": avg_iou})
        print(f" >> Result: MSE={avg_mse:.6f}, IoU={avg_iou:.4f}")

    # 3. 儲存與繪圖
    df = pd.DataFrame(results)
    df.to_csv("edge_eval_results.csv", index=False)

    plot_results(df)

def plot_results(df):
    plt.figure(figsize=(12, 5))

    # Plot MSE
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'], df['mse'], marker='o', color='red', label='MSE (Lower is better)')
    plt.title('G1 Edge MSE over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.legend()

    # Plot IoU
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'], df['iou'], marker='s', color='blue', label='IoU (Higher is better)')
    plt.title('G1 Edge IoU over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('edge_evaluation_curve.png')
    print("[+] Evaluation curve saved to edge_evaluation_curve.png")
    plt.show()

if __name__ == "__main__":
    evaluate_g1()