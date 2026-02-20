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
from monitor_v14 import plot_three_axis
from src.strategy.v19_strategy import EdgeDiffusionV19Strategy
from src.misc.train_tools import (
    chw01_to_hwc_uint8, to_01, safe_metric_value,
    sanitize_tensor, gray_consistency_loss, masked_l1,masked_color_loss,color_consistency_tools
    )

#

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
        self.strategy = EdgeDiffusionV19Strategy(
            pure_epochs=int(cfg["train"]["pure_epochs"]),
            base_lr=float(cfg["train"]["base_lr"]),
            w_adv_final=float(cfg["train"].get("w_adv_final", 0.005)),
            min_steps=int(cfg["train"].get("steps", 30)),
            total_epochs=int(cfg["train"].get("total_epochs", 300))
        )



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
        tcfg = self.cfg["train"];
        epochs = int(tcfg["epochs"]);
        # pure_epochs = int(tcfg["pure_epochs"])
        # init_steps = int(tcfg.get("steps", 30)) # 保持初始值不變

        for epoch in range(self.start_epoch, epochs):
            stats_accum = {"l1": 0.0, "mse": 0.0, "vgg": 0.0, "advg": 0.0, "total": 0.0};
            conf = self.strategy.get_config(epoch, metrics_history={"psnr_full": self.buf.psnr_full})
            # 從 conf 中提取參數
            w_recon = conf["w_recon"]
            w_vgg   = conf["w_vgg"]
            w_mse   = conf["w_mse"]
            w_adv   = conf["w_adv"]
            w_gray  = conf["w_gray"]
            curr_lr = conf["lr"]
            cur_steps = conf["steps"]
            n_iter = 0

            for g in self.opt_G.param_groups: g["lr"] = curr_lr
            # w_gray = gray_w_schedule(epoch)

            self.G2.train(); self.D.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
            if conf["is_boosted"]: pbar.set_description(f"Epoch {epoch} [Boosted 150 Steps]")


            for i, (imgs, _, masks) in enumerate(pbar):
                self.global_step += 1
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                with torch.no_grad(): edges = self.G1(torch.cat([imgs * (1 - masks), masks], dim=1))
                cond = torch.cat([imgs * (1 - masks), masks, edges], dim=1)

                t = self.diffusion.sample_timesteps(imgs.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(imgs, t)
                # --- forward G2 ---
                with torch.amp.autocast('cuda'):
                    pred_noise = self.G2(x_t, t, cond)
                    pred_noise = torch.clamp(pred_noise, -3.0, 3.0)
                    l_mse = F.mse_loss(pred_noise, noise)

                    t = t.long()
                    alpha_hat_t = self.diffusion.alpha_hat[t].view(-1,1,1,1)
                    pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t.clamp(min=1e-4))
                    pred_x0 = torch.clamp(sanitize_tensor(pred_x0), -1.0, 1.0)

                # =========================
                # (A) Train D
                # =========================
                l_adv_G = torch.tensor(0.0, device=self.device)
                if w_adv > 0:
                    self.opt_D.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda'):
                        real_out = self.D(imgs)
                        fake_out = self.D(pred_x0.detach())
                        loss_D = (F.relu(1.0 - real_out).mean() + F.relu(1.0 + fake_out).mean()) * 0.5

                    if torch.isfinite(loss_D):
                        self.scaler_D.scale(loss_D).backward()
                        self.scaler_D.unscale_(self.opt_D)
                        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
                        self.scaler_D.step(self.opt_D)
                        self.scaler_D.update()

                # =========================
                # (B) Train G
                # =========================
                with torch.amp.autocast('cuda'):
                    l_pixel = masked_l1(pred_x0, imgs, masks)
                    l_vgg   = self.vgg_criterion(pred_x0, imgs)
                    l_gray  = gray_consistency_loss(pred_x0, masks, w=w_gray)

                    if w_adv > 0:
                        l_adv_G = -self.D(pred_x0).mean()

                    safe_l_mse   = torch.nan_to_num(l_mse, 0.0)
                    safe_l_vgg   = torch.nan_to_num(l_vgg, 0.0)
                    safe_l_pixel = torch.nan_to_num(l_pixel, 0.0)
                    safe_l_adv   = torch.nan_to_num(l_adv_G, 0.0)

                    # ⚠️ color_consistency 先建議降權或只在 early 關掉
                    color_consistency = torch.mean(torch.std(pred_x0, dim=1))
                    l_color_fix = F.mse_loss(pred_x0.mean(dim=(2,3)), imgs.mean(dim=(2,3)))
                    l_color_match = masked_color_loss(pred_x0, imgs, masks)
                    l_stats, l_gray_struct = color_consistency_tools(pred_x0, imgs, masks)
                    # 取得策略權重
                    w_cs = conf.get("w_color_stats", 1.0)
                    w_gs = conf.get("w_gray_struct", 1.0)
                    # loss_total = (safe_l_mse * w_mse +
                    #             safe_l_vgg * w_vgg +
                    #             safe_l_pixel * w_recon +
                    #             safe_l_adv * w_adv +
                    #             l_gray +l_color_match * 1.0 +
                    #             l_color_fix * 0.5 +
                    #             color_consistency * 0.02)  # 先降到 0.02 比較安全
                    # 整合進 Total Loss
                    loss_total = (safe_l_mse * w_mse +
                                safe_l_vgg * w_vgg +
                                safe_l_pixel * w_recon +
                                safe_l_adv * w_adv +
                                l_stats * w_cs +        # 新增：統計約束
                                l_gray_struct * w_gs)   # 新增：灰度結構約束

                self.opt_G.zero_grad(set_to_none=True)
                self.scaler_G.scale(loss_total).backward()
                self.scaler_G.unscale_(self.opt_G)
                torch.nn.utils.clip_grad_norm_(self.G2.parameters(), 1.0)
                self.scaler_G.step(self.opt_G)
                self.scaler_G.update()



                if torch.isfinite(loss_total):
                    stats_accum["l1"] += safe_l_pixel.item(); stats_accum["mse"] += safe_l_mse.item()
                    stats_accum["vgg"] += safe_l_vgg.item(); stats_accum["advg"] += safe_l_adv.item()
                    stats_accum["total"] += loss_total.item(); n_iter += 1

                if i % 50 == 0: pbar.set_postfix({"L1": f"{safe_l_pixel.item():.4f}" ,"Total": f"{loss_total.item():.4f}"})

            # === Epoch End Logic (TensorBoard & Dashboard) ===
            if n_iter > 0:
                avg = {k: v / n_iter for k, v in stats_accum.items()}

                # 寫入 TensorBoard Scalars (與 Monitor 對接)
                self.writer.add_scalar("Loss_Detail/Pixel_L1", avg["l1"], epoch)
                self.writer.add_scalar("Loss_Detail/MSE_Noise", avg["mse"], epoch)
                self.writer.add_scalar("Loss_Detail/VGG", avg["vgg"], epoch)
                self.writer.add_scalar("Loss_Detail/Color_Match", l_color_match.item(), epoch)
                self.writer.add_scalar("Loss_Detail/Color_Fix", l_color_fix.item(), epoch)
                self.writer.add_scalar("Loss_Detail/Color_Stats", l_stats.item(), epoch)
                self.writer.add_scalar("Loss_Detail/Gray_Structural", l_gray_struct.item(), epoch)
                self.writer.add_scalar("Losses/Adv_Loss_G", avg["advg"], epoch)
                self.writer.add_scalar("Losses/Total_Loss", avg["total"], epoch)
                self.writer.add_scalar("Params/Learning_Rate", curr_lr, epoch)
                self.writer.add_scalar("Weights/init_step", cur_steps, epoch)
                self.writer.add_scalar("Weights/w_recon", w_recon, epoch)
                self.writer.add_scalar("Weights/w_vgg", w_vgg, epoch)
                self.writer.add_scalar("Weights/w_mse", w_mse, epoch)
                self.writer.add_scalar("Weights/w_adv", w_adv, epoch)
                self.writer.add_scalar("Weights/w_color_stats", w_cs, epoch)
                self.writer.add_scalar("Weights/w_gray_struct", w_gs, epoch)
                

                self.G2.eval()
                with torch.no_grad():
                    g2 = self.G2.module if hasattr(self.G2, "module") else self.G2
                    sample_x = self.diffusion.sample(g2, cond[0:1], n=1, steps=cur_steps)

                    res = imgs[0:1] * (1 - masks[0:1]) + sample_x * masks[0:1]
                    psnr, ssim = psnr_torch(res, imgs[0:1]).item(), ssim_torch(res, imgs[0:1]).item()
                    pbar.set_postfix({"L1": f"{safe_l_pixel.item():.4f}","PSNR": f"{psnr:.2f}" ,"Total": f"{loss_total.item():.4f}"})
                    self.writer.add_scalar("Metrics/PSNR", psnr, epoch)
                    self.writer.add_scalar("Metrics/SSIM", ssim, epoch)

                    sample_fn = os.path.join(self.sample_dir, f"epoch_{epoch}.png")
                    # self._save_ultimate_img(sample_fn, imgs[0], masks[0], edges[0], res[0], psnr, ssim, epoch)
                    self._last_sample_img_path = sample_fn

                    save_preview_image(imgs[0:1], masks[0:1], edges[0:1], res[0:1], psnr, epoch, os.path.join(self.log_dir, f"epoch_{epoch}.png"))
                avg.update({"psnr_full": psnr, "ssim_full": ssim, "lr": curr_lr, "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse, "w_adv": w_adv})
                self.buf.append_epoch(epoch, avg)
                plot_three_axis(self.log_dir, self.log_dir, "training_reports/v19_axis_report.png")
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

