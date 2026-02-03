import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# ================= CONFIG (優化配置) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G1_PATH = "./checkpoints_sa/G1_SA_epoch_81.pth"
G2_PATH = "./checkpoints_stage2/G2_latest.pth"
TEST_DIR = "./datasets/img"
RESULT_DIR = "./diffusion_result_ca"
IMG_SIZE = 512
BATCH_SIZE = 8  # 增加 Batch Size 以提升 GPU 利用率
SAMPLING_STEPS = 50  # 使用 50 步 DDIM
# =====================================================

def load_weights_safe(model, path):
    """ 安全載入權重：自動處理 DataParallel 產生的 'module.' 前綴差異 """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到權重檔案: {path}")

    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = {}

    is_model_parallel = isinstance(model, nn.DataParallel)
    is_ckpt_parallel = list(state_dict.keys())[0].startswith('module.')

    if is_model_parallel and not is_ckpt_parallel:
        new_state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not is_model_parallel and is_ckpt_parallel:
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        new_state_dict = state_dict

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def run_inference():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("[*] Initializing Models...")
    G1 = EdgeGenerator().to(DEVICE)
    G2 = DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE)

    # 多顯卡支援
    if torch.cuda.device_count() > 1:
        print(f"[*] Detected {torch.cuda.device_count()} GPUs. Using DataParallel.")
        G1 = nn.DataParallel(G1)
        G2 = nn.DataParallel(G2)

    G1 = load_weights_safe(G1, G1_PATH)
    G2 = load_weights_safe(G2, G2_PATH)

    diffusion = DiffusionManager(device=DEVICE)

    # 準備測試集
    dataset = InpaintingDataset(TEST_DIR, mode='test', img_size=IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4, # 增加讀取線程防止 CPU 瓶頸
        pin_memory=True
    )

    psnr_list, ssim_list = [], []
    print(f"[*] Starting Inference ({SAMPLING_STEPS} steps) on {len(dataset)} images...")

    # 使用 tqdm 並顯示即時 PSNR
    pbar = tqdm(test_loader, desc="[DDIM Sampling]")

    with torch.no_grad():
        for i, (imgs, _, masks) in enumerate(pbar):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Step A: G1 生成邊緣
            masked_imgs = imgs * (1 - masks)
            pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            # Step B: G2 執行快速採樣
            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)
            model_for_sampling = G2.module if hasattr(G2, 'module') else G2

            # 執行 DDIM 採樣並進行數值防護
            samples = diffusion.sample(model_for_sampling, condition, n=imgs.shape[0], steps=SAMPLING_STEPS)
            samples = torch.clamp(samples, -1.0, 1.0) # 防止白紙現象

            # Step C: 合成結果 (Blending)
            final_res = masked_imgs + samples * masks

            # 指標計算與結果儲存
            for j in range(imgs.shape[0]):
                img_gt = ((imgs[j].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                img_out = ((final_res[j].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255).astype(np.uint8)

                # 計算 PSNR/SSIM
                cur_psnr = psnr_func(img_gt, img_out)
                cur_ssim = ssim_func(img_gt, img_out, channel_axis=2)
                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)

                # 儲存對比圖 (Original vs Inpainted)
                save_idx = i * BATCH_SIZE + j
                combined = np.hstack([img_gt, img_out])
                cv2.imwrite(
                    f"{RESULT_DIR}/test_{save_idx:03d}_psnr_{cur_psnr:.2f}.png",
                    cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                )

            # 即時更新進度條資訊
            pbar.set_postfix({"Avg_PSNR": f"{np.mean(psnr_list):.2f}dB"})

    print("\n" + "="*40)
    print(f"Final Report:")
    print(f" - Avg PSNR: {np.mean(psnr_list):.2f} dB")
    print(f" - Avg SSIM: {np.mean(ssim_list):.4f}")
    print(f" - Total Images: {len(psnr_list)}")
    print(f" - Results Saved in: {RESULT_DIR}")
    print("="*40)

if __name__ == "__main__":
    run_inference()