import torch
import torch.nn as nn
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy import linalg
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torchvision.models import inception_v3, Inception_V3_Weights

# 引入專案模組
from src.networks import EdgeGenerator, DiffusionUNet
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset

# ================= CONFIG (優化配置) =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
G1_PATH = "./checkpoints_sa/G1_SA_epoch_81.pth"
G2_PATH = "./checkpoints_stage2_sa/G2_latest.pth"
TEST_DIR = "./datasets/img"
RESULT_DIR = "./diffusion_results_fid"
IMG_SIZE = 512
BATCH_SIZE = 4      # 包含 FID 計算時，建議 Batch Size 設小以防顯存溢出
SAMPLING_STEPS = 50 # 50 步 DDIM 快速採樣
# =====================================================

class FIDCalculator:
    """ 使用 InceptionV3 提取 2048 維特徵並計算 FID """
    def __init__(self, device):
        self.device = device
        # 載入預訓練 InceptionV3
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
        self.model.fc = nn.Identity() # 移除分類層，保留特徵提取
        self.model.eval()

    def get_features(self, batch):
        """ 將影像縮放至 299x299 並提取特徵 """
        with torch.no_grad():
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            features = self.model(batch)
        return features.cpu().numpy()

    def calculate_fid(self, mu1, sigma1, mu2, sigma2):
        """ FID 數學公式計算 """
        ssdiff = np.sum((mu1 - mu2)**2.0)
        covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)[0]
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)

def load_weights_safe(model, path):
    """ 安全載入權重：自動處理 DataParallel 差異 """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到權重檔案: {path}")
    state_dict = torch.load(path, map_location=DEVICE)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def run_inference():
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print(f"[*] Initializing Models on {DEVICE}...")
    fid_calc = FIDCalculator(DEVICE)
    G1 = load_weights_safe(EdgeGenerator().to(DEVICE), G1_PATH)
    G2 = load_weights_safe(DiffusionUNet(in_channels=8, out_channels=3).to(DEVICE), G2_PATH)
    diffusion = DiffusionManager(device=DEVICE)

    # 準備測試集
    dataset = InpaintingDataset(TEST_DIR, mode='test', img_size=IMG_SIZE)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    psnr_list, ssim_list = [], []
    real_features, fake_features = [], []

    print(f"[*] Starting DDIM Inference + FID Evaluation on {len(dataset)} images...")
    pbar = tqdm(test_loader, desc="[Processing]")

    with torch.no_grad():
        for i, (imgs, _, masks) in enumerate(pbar):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            # Step A: G1 生成邊緣
            masked_imgs = imgs * (1 - masks)
            pred_edges = G1(torch.cat([masked_imgs, masks], dim=1))

            # Step B: G2 Diffusion 快速採樣
            condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)
            samples = diffusion.sample(G2, condition, n=imgs.shape[0], steps=SAMPLING_STEPS)
            samples = torch.clamp(samples, -1.0, 1.0)

            # Step C: 合成結果
            final_res = masked_imgs + samples * masks

            # 收集 FID 特徵 (影像需從 [-1, 1] 轉為 [0, 1])
            real_features.append(fid_calc.get_features((imgs + 1.0) / 2.0))
            fake_features.append(fid_calc.get_features((final_res + 1.0) / 2.0))

            # 計算像素指標與儲存
            for j in range(imgs.shape[0]):
                img_gt = ((imgs[j].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255).astype(np.uint8)
                img_out = ((final_res[j].cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0 * 255).astype(np.uint8)

                cur_psnr = psnr_func(img_gt, img_out)
                cur_ssim = ssim_func(img_gt, img_out, channel_axis=2)
                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)

                # 儲存對比結果
                save_idx = i * BATCH_SIZE + j
                cv2.imwrite(f"{RESULT_DIR}/test_{save_idx:03d}_psnr_{cur_psnr:.2f}.png",
                            cv2.cvtColor(np.hstack([img_gt, img_out]), cv2.COLOR_RGB2BGR))

            pbar.set_postfix({"Avg_PSNR": f"{np.mean(psnr_list):.2f}dB"})

    # --- 最終 FID 計算 ---
    print("[*] Calculating final FID score...")
    real_features = np.concatenate(real_features, axis=0)
    fake_features = np.concatenate(fake_features, axis=0)
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    final_fid = fid_calc.calculate_fid(mu1, sigma1, mu2, sigma2)

    print("\n" + "="*40)
    print(f"Final Report (Epoch 140 Progress):")
    print(f" - Avg PSNR: {np.mean(psnr_list):.2f} dB")
    print(f" - Avg SSIM: {np.mean(ssim_list):.4f}")
    print(f" - FID Score: {final_fid:.4f} (Lower is better)")
    print("="*40)

if __name__ == "__main__":
    run_inference()