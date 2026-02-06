import os
import math
import time
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.networks import EdgeGenerator, DiffusionUNet, Discriminator, VGGLoss
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.misc.utils import save_preview_image
from src.metrics.image_metrics import psnr_torch, ssim_torch
from monitor_v14 import render_dashboard_png_epoch_only,plot_v14_2_three_axis

from src.misc.train_tools import (
    chw01_to_hwc_uint8, to_01, safe_metric_value,
    sanitize_tensor, gray_consistency_loss, masked_l1,
    mask_to_hwc_uint8
)

# =========================================================
# 工具函數
# =========================================================
def gray_w_schedule(epoch: int) -> float:
    if epoch < 50: return 0.2    # 提高開局約束，防止彩色偽影定型
    if epoch < 100: return 0.5   # 強力修正期
    return 0.3                   # 持續穩定輸出

def safe_moving_average(values, window=5):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)] 
    if len(values) == 0: return None
    w = int(min(window, len(values)))
    if w <= 1: return values
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='valid')

def get_sa_config(epoch: int, pure_epochs: int, base_lr: float, w_adv_final: float, steps: int):
    ratio = epoch / max(1, pure_epochs)
    max_steps = 100
    min_steps = steps  # 這是從外部傳入的初始步數 (例如 30)

    if epoch < pure_epochs:
        # --- 階段 1: 基礎重建與平滑 Warm-up ---
        # 修正：lr 應該用乘法，否則 base_lr + 1 會直接讓模型炸掉
        lr_factor = 0.5 * (1 - math.cos(math.pi * ratio))
        lr = base_lr * (0.1 + 0.9 * lr_factor) 
        
        w_recon = 1.0 + 0.5 * (1 - math.cos(math.pi * ratio)) 
        w_mse   = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))
        w_vgg   = 0.01
        w_adv   = 0.0
        
        # Steps 餘弦遞增：從 min_steps 到 max_steps
        steps_curve = 0.5 * (1 - math.cos(math.pi * ratio))
        curr_steps = int(min_steps + (max_steps - min_steps) * steps_curve)
                          
    elif epoch < 150:
        # --- 階段 2: 感官強化 ---
        eta = (epoch - pure_epochs) / (150 - pure_epochs + 1e-6)
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * eta))
        
        w_mse   = 0.6 * (1 - alpha) + 0.1 * alpha
        w_recon = 2.0 * (1 - alpha) + 1.0 * alpha
        w_vgg   = 0.1 * (1 - alpha) + 0.8 * alpha 
        w_adv   = w_adv_final * alpha
        
        # 階段 2 建議直接鎖定在最高步數，以確保 PSNR 評估的穩定性
        curr_steps = max_steps
    else:
        # --- 階段 3: 精細微調 ---
        lr = base_lr * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.8, 0.05
        w_adv = w_adv_final
        curr_steps = max_steps
        
    return w_recon, w_vgg, w_mse, w_adv, lr, curr_steps
""""
崩壞的vgg error1
def get_sa_config(epoch: int, pure_epochs: int, base_lr: float, w_adv_final: float):
    if epoch < pure_epochs:
        lr = 1e-5
        ratio = epoch / max(1, pure_epochs)
        lr =  base_lr + 1 * (1 + math.cos(math.pi * eta))
        w_recon = 1.0 + 0.5 * (1 - math.cos(math.pi * ratio)) 
        w_mse   = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))
        w_vgg   = 0.01
        w_adv   = 0.0
    elif epoch < 150:
        eta = (epoch - pure_epochs) / (150 - pure_epochs + 1e-6)
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * eta))
        w_mse   = 0.8  * (1 - alpha) + 0.2  * alpha
        w_recon = 2.0  * (1 - alpha) + 1.0  * alpha
        w_vgg   = 2.0* (1 - alpha) + 0.15 * alpha
        w_adv   = w_adv_final * alpha
    else:
        lr = base_lr * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.2, 0.1
        w_adv = w_adv_final
    return w_recon, w_vgg, w_mse, w_adv, lr
"""

# =========================================================
# Dashboard 組件 (白底風格)
# =========================================================
@dataclass
class DashEpochBuffer:
    maxlen: int = 5000
    x_epoch: list = field(default_factory=list)
    l1: list = field(default_factory=list)
    mse: list = field(default_factory=list)
    vgg: list = field(default_factory=list)
    advg: list = field(default_factory=list)
    total: list = field(default_factory=list)
    lr: list = field(default_factory=list)
    w_recon: list = field(default_factory=list)
    w_vgg: list = field(default_factory=list)
    w_mse: list = field(default_factory=list)
    w_adv: list = field(default_factory=list)
    psnr_full: list = field(default_factory=list)
    ssim_full: list = field(default_factory=list)

    def append_epoch(self, epoch: float, stats: Dict[str, float]):
        self.x_epoch.append(float(epoch))
        for key in ["l1", "mse", "vgg", "advg", "total", "lr", "w_recon", "w_vgg", "w_mse", "w_adv", "psnr_full", "ssim_full"]:
            val = stats.get(key, np.nan)
            getattr(self, key).append(safe_metric_value(val))
        if len(self.x_epoch) > self.maxlen:
            for attr in self.__dict__.values():
                if isinstance(attr, list): del attr[0]

# def render_dashboard_png_epoch_only(png_path: str, buf: DashEpochBuffer, meta: Dict[str, Any]):
#     plt.style.use('default') 
#     plt.clf()
#     x = np.array(buf.x_epoch)
#     if len(x) < 1: return

#     fig = plt.figure(figsize=(24, 16), dpi=100)
#     gs = fig.add_gridspec(6, 4, height_ratios=[1, 1, 0.8, 2.5, 0.1, 2.0], hspace=0.6, wspace=0.3)
    
#     fig.suptitle(f"Edge-Diffusion stage2 Monitor | {meta.get('update_str')}\nEpoch: {meta['latest_epoch']} | Step: {meta['latest_step']}", 
#                  fontsize=22, fontweight="bold", y=0.97, color="#333333")

#     def plot_sub(r, c, data_list, title, color, is_log=False):
#         ax = fig.add_subplot(gs[r, c])
#         data = np.array(data_list)
#         mask = np.isfinite(data)
#         if np.any(mask):
#             v_d, v_x = data[mask], x[mask]
#             ax.plot(v_x, v_d, alpha=0.3, color=color, linewidth=1)
#             ma = safe_moving_average(v_d, window=5)
#             if ma is not None: ax.plot(v_x[-len(ma):], ma, color=color, linewidth=2.5)
#             y_min, y_max = np.min(v_d), np.max(v_d)
#             margin = max((y_max - y_min) * 0.1, 1e-4)
#             ax.set_ylim(y_min - margin, y_max + margin)
#         else:
#             ax.text(0.5, 0.5, "Waiting for data...", ha='center', va='center', color='gray')
#         ax.set_title(title, fontsize=12, color="#555555", fontweight='bold')
#         ax.grid(True, linestyle="--", alpha=0.5)
#         if is_log and np.any(mask) and np.all(data[mask] > 0): ax.set_yscale('log')

#     plot_sub(0, 0, buf.l1, "L1 Loss", "teal")
#     plot_sub(0, 1, buf.mse, "MSE Noise", "royalblue")
#     plot_sub(0, 2, buf.vgg, "VGG Loss", "darkorange")
#     plot_sub(0, 3, buf.advg, "Adv Loss (G)", "forestgreen")
#     plot_sub(1, 0, buf.total, "Total Loss", "crimson")
#     plot_sub(1, 1, buf.psnr_full, "PSNR (dB)", "deeppink")
#     plot_sub(1, 2, buf.ssim_full, "SSIM", "darkorchid")
#     plot_sub(1, 3, buf.lr, "Learning Rate", "olive", is_log=True)

#     ax_w = fig.add_subplot(gs[2, :])
#     for l, (v, c) in {"Pixel": (buf.w_recon, "teal"), "VGG": (buf.w_vgg, "darkorange"), "MSE": (buf.w_mse, "royalblue"), "Adv": (buf.w_adv, "crimson")}.items():
#         ay = np.array(v); am = np.isfinite(ay)
#         ax_w.plot(x[am], ay[am], label=l, color=c, linewidth=2)
#     ax_w.set_title("Strategy Schedule", color="#555555", fontweight='bold'); ax_w.legend(loc='upper right', ncol=4); ax_w.grid(True, alpha=0.3)

#     ax_main = fig.add_subplot(gs[3, :])
#     vt = np.array(buf.total); m = np.isfinite(vt)
#     if np.any(m):
#         ax_main.plot(x[m], vt[m], color="crimson", linewidth=3, label="Loss")
#         ax_main.set_ylabel("Loss", color="crimson")
#         ax_p_axis = ax_main.twinx(); ax_p_axis.plot(x, buf.psnr_full, color="deeppink", linewidth=3)
#         ax_p_axis.set_ylabel("PSNR (dB)", color="deeppink")
#         ax_s_axis = ax_main.twinx(); ax_s_axis.spines["right"].set_position(("axes", 1.06))
#         ax_s_axis.plot(x, buf.ssim_full, color="darkorchid", linewidth=3); ax_s_axis.set_ylabel("SSIM", color="darkorchid")
#     ax_main.set_title("Performance Triple-Axis Analysis", fontsize=18, fontweight='bold', pad=20)

#     ax_img = fig.add_subplot(gs[5, :])
#     if meta.get("sample_img_path") and os.path.exists(meta["sample_img_path"]):
#         ax_img.imshow(np.asarray(Image.open(meta["sample_img_path"])))
#     ax_img.axis("off")
#     plt.savefig(png_path, bbox_inches="tight", facecolor="white"); plt.close()

# =========================================================
# Trainer
# =========================================================
class Stage2Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        p = cfg["paths"]
        self.save_dir, self.sample_dir, self.log_dir = p["save_dir"], p["sample_dir"], p["log_dir"]
        for d in [self.save_dir, self.sample_dir, self.log_dir]: os.makedirs(d, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.dashboard_png = os.path.join(self.log_dir, "dashboard_latest.png")
        self.buf = DashEpochBuffer()
        self.best_psnr = 0.0
        self._last_sample_img_path = ""
        
        self._build_data(); self._build_models(); self._build_optim()
        self.diffusion = DiffusionManager(device=self.device)
        self.vgg_criterion = VGGLoss(self.device)
        self.start_epoch = 0; self._try_resume(); self.global_step = 0

    def _build_data(self):
        d = self.cfg["data"]
        self.train_loader = DataLoader(InpaintingDataset(d["train_root"], mode="train"), batch_size=d["batch_size"], shuffle=True, num_workers=4, pin_memory=True)

    def _build_models(self):
        # 初始化模型
        self.G1 = EdgeGenerator().to(self.device).eval()
        self.G2 = DiffusionUNet(in_channels=8, out_channels=3).to(self.device)
        self.D = Discriminator().to(self.device)

        # 載入 G1 (固定不變)
        g1_ckpt = torch.load(self.cfg["paths"]["g1_checkpoint"], map_location=self.device)
        self.G1.load_state_dict(self._load_weights_safe(g1_ckpt))

        # --- 多顯卡設定 ---
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training.")
            self.G2 = nn.DataParallel(self.G2)
            self.D = nn.DataParallel(self.D)

    def _load_weights_safe(self, state_dict):
        """處理 DataParallel 與單卡權重不匹配的問題"""
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        
        new_state_dict = {}
        # 檢查當前模型是否為 DataParallel
        is_dp = hasattr(self.G2, "module") 
        
        for k, v in state_dict.items():
            if is_dp and not k.startswith('module.'):
                new_state_dict[f'module.{k}'] = v
            elif not is_dp and k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def _build_optim(self):
        t = self.cfg["train"]
        self.opt_G = optim.AdamW(self.G2.parameters(), lr=float(t["base_lr"]), weight_decay=1e-4)
        self.opt_D = optim.AdamW(self.D.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scaler_G = torch.amp.GradScaler('cuda') if "cuda" in str(self.device) else None
        self.scaler_D = torch.amp.GradScaler('cuda') if "cuda" in str(self.device) else None

    def _try_resume(self):
        path = self.cfg["paths"].get("load_g2_path", "")
        if path and os.path.exists(path):
            print(f"Resuming from checkpoint: {path}")
            ckpt = torch.load(path, map_location=self.device)
            
            # 1. 模型權重 (使用剛才定義的 safe 函數)
            self.G2.load_state_dict(self._load_weights_safe(ckpt))
            
            # 2. 優化器狀態 (必須在 build_optim 之後調用)
            if "optimizer_state_dict" in ckpt:
                self.opt_G.load_state_dict(ckpt["optimizer_state_dict"])
            if "optimizer_D_state_dict" in ckpt:
                self.opt_D.load_state_dict(ckpt["optimizer_D_state_dict"])
                
            # 3. 混合精度 Scaler
            if "scaler_state_dict" in ckpt and self.scaler_G is not None:
                self.scaler_G.load_state_dict(ckpt["scaler_state_dict"])
            
            # 4. 訓練進度
            self.start_epoch = ckpt.get("epoch", 0) + 1
            self.best_psnr = ckpt.get("best_psnr", 0.0)
            print(f"Checkpoint loaded. Start from epoch {self.start_epoch}, best PSNR: {self.best_psnr:.4f}")
    
    def train(self):
        tcfg = self.cfg["train"]; epochs = int(tcfg["epochs"]); pure_epochs = int(tcfg["pure_epochs"])
        init_steps = int(tcfg.get("steps", 30)) # 保持初始值不變
      
        for epoch in range(self.start_epoch, epochs):
            stats_accum = {"l1": 0.0, "mse": 0.0, "vgg": 0.0, "advg": 0.0, "total": 0.0}; n_iter = 0
            w_recon, w_vgg, w_mse, w_adv, curr_lr, cur_steps = get_sa_config(
                epoch, pure_epochs, float(tcfg["base_lr"]), 0.005, init_steps)
            for g in self.opt_G.param_groups: g["lr"] = curr_lr
            w_gray = gray_w_schedule(epoch)

            self.G2.train(); self.D.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

            for i, (imgs, _, masks) in enumerate(pbar):
                self.global_step += 1
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                with torch.no_grad(): edges = self.G1(torch.cat([imgs * (1 - masks), masks], dim=1))
                cond = torch.cat([imgs * (1 - masks), masks, edges], dim=1)

                t = self.diffusion.sample_timesteps(imgs.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(imgs, t)

                with torch.amp.autocast('cuda' if "cuda" in str(self.device) else 'cpu'):
                    pred_noise = self.G2(x_t, t, cond)
                    # --- [防爆 1] 雜訊預測限幅 ---
                    pred_noise = torch.clamp(pred_noise, -3.0, 3.0)
                    l_mse = F.mse_loss(pred_noise, noise)

                    # --- [防爆 2] 重建 x0 增加分母保護 ---
                    alpha_hat_t = self.diffusion.alpha_hat[t][:, None, None, None]
                    pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t.clamp(min=1e-3))
                    # --- [防爆 3] 嚴格限幅 x0 區間，防止 L1 NaN ---
                    pred_x0 = torch.clamp(sanitize_tensor(pred_x0), -1.0, 1.0)

                    l_pixel = masked_l1(pred_x0, imgs, masks)
                    l_vgg = self.vgg_criterion(pred_x0, imgs) if epoch >= pure_epochs else torch.tensor(0.0).to(self.device)
                    l_gray = gray_consistency_loss(pred_x0, masks, w=w_gray)
                    
                    l_adv_G = torch.tensor(0.0).to(self.device)
                    if w_adv > 0:
                        self.opt_D.zero_grad()
                        real_out = self.D(imgs); fake_out = self.D(pred_x0.detach())
                        loss_D = (F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()) * 0.5
                        if torch.isfinite(loss_D):
                            self.scaler_D.scale(loss_D).backward()
                            self.scaler_D.unscale_(self.opt_D)
                            torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                            self.scaler_D.step(self.opt_D); self.scaler_D.update()
                            l_adv_G = -self.D(pred_x0).mean()

                    # --- 關鍵修正：總損失安全性檢查 (NaN 過濾) ---
                    safe_l_mse = torch.nan_to_num(l_mse, 0.0); safe_l_vgg = torch.nan_to_num(l_vgg, 0.0)
                    safe_l_pixel = torch.nan_to_num(l_pixel, 0.0); safe_l_adv = torch.nan_to_num(l_adv_G, 0.0)

                    loss_total = safe_l_mse * w_mse + safe_l_vgg * w_vgg + safe_l_pixel * w_recon + safe_l_adv * w_adv + l_gray

                self.opt_G.zero_grad(); self.scaler_G.scale(loss_total).backward()
                # --- [防爆 4] G2 梯度裁剪 ---
                self.scaler_G.unscale_(self.opt_G)
                torch.nn.utils.clip_grad_norm_(self.G2.parameters(), 1.0)
                self.scaler_G.step(self.opt_G); self.scaler_G.update()

                if torch.isfinite(loss_total):
                    stats_accum["l1"] += safe_l_pixel.item(); stats_accum["mse"] += safe_l_mse.item()
                    stats_accum["vgg"] += safe_l_vgg.item(); stats_accum["advg"] += safe_l_adv.item()
                    stats_accum["total"] += loss_total.item(); n_iter += 1

                if i % 50 == 0: pbar.set_postfix({"L1": f"{safe_l_pixel.item():.4f}", "Total": f"{loss_total.item():.4f}"})

            # === Epoch End Logic (TensorBoard & Dashboard) ===
            if n_iter > 0:
                avg = {k: v / n_iter for k, v in stats_accum.items()}
                
                # 寫入 TensorBoard Scalars (與 Monitor 對接)
                self.writer.add_scalar("Loss_Detail/Pixel_L1", avg["l1"], epoch)
                self.writer.add_scalar("Loss_Detail/MSE_Noise", avg["mse"], epoch)
                self.writer.add_scalar("Loss_Detail/VGG", avg["vgg"], epoch)
                self.writer.add_scalar("Losses/Adv_Loss_G", avg["advg"], epoch)
                self.writer.add_scalar("Losses/Total_Loss", avg["total"], epoch)
                self.writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)
                self.writer.add_scalar("Weights/init_step", cur_steps, epoch)
                self.writer.add_scalar("Weights/w_recon", w_recon, epoch)
                self.writer.add_scalar("Weights/w_vgg", w_vgg, epoch)
                self.writer.add_scalar("Weights/w_mse", w_mse, epoch)
                self.writer.add_scalar("Weights/w_adv", w_adv, epoch)

                self.G2.eval()
                with torch.no_grad():
                    g2 = self.G2.module if hasattr(self.G2, "module") else self.G2
                    sample_x = self.diffusion.sample(g2, cond[0:1], n=1, steps=cur_steps)
                    res = imgs[0:1] * (1 - masks[0:1]) + sample_x * masks[0:1]
                    psnr, ssim = psnr_torch(res, imgs[0:1]).item(), ssim_torch(res, imgs[0:1]).item()
                    
                    self.writer.add_scalar("Metrics/PSNR", psnr, epoch)
                    self.writer.add_scalar("Metrics/SSIM", ssim, epoch)

                    sample_fn = os.path.join(self.sample_dir, f"epoch_{epoch}.png")
                    # self._save_ultimate_img(sample_fn, imgs[0], masks[0], edges[0], res[0], psnr, ssim, epoch)
                    self._last_sample_img_path = sample_fn
                    
                    save_preview_image(imgs[0:1], masks[0:1], edges[0:1], res[0:1], psnr, epoch, os.path.join(self.log_dir, f"epoch_{epoch}.png"))  
                avg.update({"psnr_full": psnr, "ssim_full": ssim, "lr": curr_lr, "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse, "w_adv": w_adv})
                self.buf.append_epoch(epoch, avg)
                plot_v14_2_three_axis()
                # render_dashboard_png_epoch_only(self.dashboard_png, self.buf, {"update_str": time.strftime("%Y-%m-%d %H:%M:%S"), "latest_step": self.global_step, "latest_epoch": epoch, "sample_img_path": self._last_sample_img_path})
                self.writer.flush()
                 # =========================================================
            # Save checkpoint
            # =========================================================
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
    
                    # 建立保存清單
                    save_data = {
                        "epoch": epoch,
                        "best_psnr": self.best_psnr,
                        # 直接存，載入時靠 _load_weights_safe 處理 module. 前綴
                        "model_state_dict": self.G2.state_dict(),
                        "optimizer_state_dict": self.opt_G.state_dict(),
                        "optimizer_D_state_dict": self.opt_D.state_dict(),
                        "scaler_state_dict": self.scaler_G.state_dict() if self.scaler_G else {},
                    }
                    
                    # 保存最新與最優權重
                    torch.save(save_data, os.path.join(self.save_dir, "checkpoint_latest.pth"))
    # 如果想保留 best 模型可以多存一份
                    torch.save(save_data, os.path.join(self.save_dir, f"best_psnr_{psnr:.2f}.pth"))
                    # torch.save({"model_state_dict": g2.state_dict(), "epoch": epoch}, f"{self.save_dir}/best.pth")

    def _save_ultimate_img(self, path, img, mask, edge, res, psnr, ssim, ep):
        p = [chw01_to_hwc_uint8(to_01(x)) for x in [img, mask, edge, img*(1-mask), res]]
        imgs = [Image.fromarray(x) for x in p]; w, h = imgs[0].size
        canvas = Image.new("RGB", (w*5 + 80, h + 100), (255, 255, 255))
        for idx, im in enumerate(imgs): canvas.paste(im, (idx*w + (idx+1)*15, 70))
        draw = ImageDraw.Draw(canvas); draw.text((20, 20), f"Epoch {ep} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}", fill="blue")
        canvas.save(path)