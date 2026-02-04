# src/trainers/stage2_trainer.py
import os
import math
import time
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

# headless plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from src.networks import EdgeGenerator, DiffusionUNet, Discriminator, VGGLoss
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.misc.utils import save_preview_image
from src.metrics.image_metrics import psnr_torch, ssim_torch

# 你已經把工具抽到這裡（沿用你的 import）
from src.misc.train_tools import (
    chw01_to_hwc_uint8,
    to_01,
    safe_metric_value,
    sanitize_tensor,
    gray_consistency_loss,
    masked_l1,
    gray_w_schedule,
    mask_to_hwc_uint8,
    plot_ieee_dashboard,   # 如果你這個 function 參數不同也沒關係，我底下有 try/except
)


# =========================================================
# SA schedule
# =========================================================
def get_sa_config(epoch: int, pure_epochs: int, base_lr: float, w_adv_final: float):
    """
    return: w_recon, w_vgg, w_mse, w_adv, lr
    """
    if epoch < pure_epochs:
        ratio = epoch / max(1, pure_epochs)
        lr = base_lr
        w_recon = 1.0 + 0.5 * (1 - math.cos(math.pi * ratio))  # 1.0 -> 2.0
        w_mse   = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))  # 1.0 -> 0.8
        w_vgg   = 0.01
        w_adv   = 0.0

    elif epoch < 150:
        eta = (epoch - pure_epochs) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * eta))

        w_mse   = 0.8  * (1 - alpha) + 0.2  * alpha
        w_recon = 2.0  * (1 - alpha) + 1.0  * alpha
        w_vgg   = 0.01 * (1 - alpha) + 0.15 * alpha
        w_adv   = 0.0

    else:
        lr = base_lr * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.2, 0.1
        w_adv = w_adv_final

    return w_recon, w_vgg, w_mse, w_adv, lr


# =========================================================
# samples_stage2_v14_ultimate (內建版)
# - 直接做一張 5-panel 合成圖（給 dashboard 用）
# =========================================================
def samples_stage2_v14_ultimate(
    out_path: str,
    orig_u8: np.ndarray,
    mask_u8: np.ndarray,
    edge_u8: np.ndarray,
    masked_u8: np.ndarray,
    res_u8: np.ndarray,
    psnr_full: float,
    ssim_full: float,
    epoch: int,
) -> Tuple[str, Dict[str, np.ndarray], str]:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    panels = {
        "orig": orig_u8,
        "mask": mask_u8,
        "edge": edge_u8,
        "masked": masked_u8,
        "res": res_u8,
    }
    titles = ["Original", "Mask", "Edge Guide", "Masked input", "Res (Inpaint)"]

    imgs = [Image.fromarray(panels[k]) for k in ["orig", "mask", "edge", "masked", "res"]]

    # 統一高度
    h = min(im.size[1] for im in imgs)
    resized = []
    for im in imgs:
        w = int(im.size[0] * (h / im.size[1]))
        resized.append(im.resize((w, h), resample=Image.BILINEAR))

    gap = 20
    top_pad = 78
    bot_pad = 18
    w_total = sum(im.size[0] for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (w_total, top_pad + h + bot_pad), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    header = f"Epoch {epoch} | PSNR(full)={psnr_full:.2f} | SSIM(full)={ssim_full:.4f}"
    draw.text((10, 10), header, fill=(0, 0, 0))
    draw.text((10, 38), "samples_stage2_v14_ultimate", fill=(60, 60, 60))

    x = 0
    for idx, im in enumerate(resized):
        canvas.paste(im, (x, top_pad))
        draw.text((x + 4, top_pad - 22), titles[idx], fill=(0, 0, 0))
        x += im.size[0] + gap

    canvas.save(out_path)
    return out_path, panels, header


# =========================================================
# Dashboard Buffer (epoch-only points)
# =========================================================
@dataclass
class DashEpochBuffer:
    maxlen: int = 5000

    x_epoch: list = field(default_factory=list)

    # epoch averaged losses (training)
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

    # epoch-end SAMPLE (full image) -> this is the "one point per epoch"
    psnr_full: list = field(default_factory=list)
    ssim_full: list = field(default_factory=list)

    def _trim(self):
        if len(self.x_epoch) <= self.maxlen:
            return
        k = len(self.x_epoch) - self.maxlen
        for _, arr in self.__dict__.items():
            if isinstance(arr, list) and len(arr) == len(self.x_epoch):
                del arr[:k]

    def append_epoch(self, epoch: float, stats: Dict[str, float]):
        self.x_epoch.append(float(epoch))
        self.l1.append(safe_metric_value(stats.get("l1", np.nan)))
        self.mse.append(safe_metric_value(stats.get("mse", np.nan)))
        self.vgg.append(safe_metric_value(stats.get("vgg", np.nan)))
        self.advg.append(safe_metric_value(stats.get("advg", np.nan)))
        self.total.append(safe_metric_value(stats.get("total", np.nan)))

        self.lr.append(safe_metric_value(stats.get("lr", np.nan)))

        self.w_recon.append(safe_metric_value(stats.get("w_recon", np.nan)))
        self.w_vgg.append(safe_metric_value(stats.get("w_vgg", np.nan)))
        self.w_mse.append(safe_metric_value(stats.get("w_mse", np.nan)))
        self.w_adv.append(safe_metric_value(stats.get("w_adv", np.nan)))

        self.psnr_full.append(safe_metric_value(stats.get("psnr_full", np.nan)))
        self.ssim_full.append(safe_metric_value(stats.get("ssim_full", np.nan)))

        self._trim()


def _as_np(a: list) -> np.ndarray:
    return np.asarray(a, dtype=np.float32) if len(a) > 0 else np.asarray([], dtype=np.float32)


# =========================================================
# Dashboard renderer (1 png per epoch)
# - 三軸：只畫 FULL
# - feedback：直接貼 samples_stage2_v14_ultimate 圖
# - 排版：加大間距避免文字疊在一起
# =========================================================
def render_dashboard_png_epoch_only(
    png_path: str,
    buf: DashEpochBuffer,
    meta: Dict[str, Any]
):
    x = _as_np(buf.x_epoch)
    if len(x) < 1:
        return

    l1 = _as_np(buf.l1)
    mse = _as_np(buf.mse)
    vgg = _as_np(buf.vgg)
    advg = _as_np(buf.advg)
    total = _as_np(buf.total)

    lr = _as_np(buf.lr)

    w_recon = _as_np(buf.w_recon)
    w_vgg = _as_np(buf.w_vgg)
    w_mse = _as_np(buf.w_mse)
    w_adv = _as_np(buf.w_adv)

    psnr_full = _as_np(buf.psnr_full)
    ssim_full = _as_np(buf.ssim_full)

    title = meta.get("title", "v14.2 Strategy Dashboard (Built-in PNG)")
    update_str = meta.get("update_str", "")
    latest_step = meta.get("latest_step", "")
    latest_epoch = meta.get("latest_epoch", "")
    sample_img_path = meta.get("sample_img_path", "")

    # 版面：6 rows, 4 cols
    # - row3 大圖
    # - row5 feedback
    fig = plt.figure(figsize=(18, 10.6), dpi=150)
    gs = fig.add_gridspec(
        6, 4,
        height_ratios=[1, 1, 1, 1.75, 0.10, 1.55],
        hspace=0.92,
        wspace=0.42
    )

    fig.suptitle(
        f"{title}\nUpdate: {update_str} | Latest Epoch: {latest_epoch} | Latest Step: {latest_step}",
        fontsize=14, fontweight="bold", y=0.985
    )

    # 全域調整，減少疊字
    fig.subplots_adjust(top=0.90, bottom=0.045, left=0.05, right=0.965)

    def ax_plot(r, c, y, t, ylabel=None):
        ax = fig.add_subplot(gs[r, c])
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.2)
        ax.set_title(t, fontsize=10, pad=8)
        ax.set_xlabel("epoch", labelpad=3)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=3)
        ax.grid(True, linestyle="--", alpha=0.25)
        if len(x) == 1:
            ax.set_xlim(x[0] - 0.5, x[0] + 0.5)
        return ax

    # row0
    ax_plot(0, 0, l1,   "L1 / Masked L1 (epoch avg)", "loss")
    ax_plot(0, 1, mse,  "MSE (noise) (epoch avg)",    "loss")
    ax_plot(0, 2, vgg,  "VGG loss (epoch avg)",       "loss")
    ax_plot(0, 3, advg, "Adv loss (G) (epoch avg)",   "loss")

    # row1
    ax_plot(1, 0, total,     "Total loss (epoch avg)",          "loss")
    ax_plot(1, 1, psnr_full, "PSNR (FULL) [epoch-end sample]",  "dB")
    ax_plot(1, 2, ssim_full, "SSIM (FULL) [epoch-end sample]",  "ssim")
    ax_plot(1, 3, lr,        "Learning rate",                   "lr")

    # row2 weights
    ax_plot(2, 0, w_recon, "Pixel weight (w_recon)", "w")
    ax_plot(2, 1, w_vgg,   "VGG weight (w_vgg)",     "w")
    ax_plot(2, 2, w_mse,   "Noise weight (w_mse)",   "w")
    ax_plot(2, 3, w_adv,   "Adv weight (w_adv)",     "w")

    # row3: SAME plot, different colors (FULL only)
    ax_big = fig.add_subplot(gs[3, :])
    ax_big.set_title("Total Loss vs PSNR(full) vs SSIM(full) (epoch-end sample point)", fontsize=11, pad=10)
    ax_big.grid(True, linestyle="--", alpha=0.25)
    ax_big.set_xlabel("epoch", labelpad=4)

    # Total loss (left axis) - blue
    l_total, = ax_big.plot(x, total, marker="o", markersize=3, linewidth=1.6,
                           label="total loss", color="tab:blue")
    ax_big.set_ylabel("total loss", color="tab:blue", labelpad=6)
    ax_big.tick_params(axis="y", labelcolor="tab:blue")

    # PSNR (right axis) - red
    ax_big2 = ax_big.twinx()
    l_psnr, = ax_big2.plot(x, psnr_full, marker="o", markersize=3, linewidth=1.6,
                           label="psnr(full) dB", color="tab:red")
    ax_big2.set_ylabel("psnr (dB)", color="tab:red", labelpad=6)
    ax_big2.tick_params(axis="y", labelcolor="tab:red")

    # SSIM (third axis) - purple
    ax_big3 = ax_big.twinx()
    ax_big3.spines["right"].set_position(("axes", 1.10))
    l_ssim, = ax_big3.plot(x, ssim_full, marker="o", markersize=3, linewidth=1.6,
                           label="ssim(full)", color="tab:purple")
    ax_big3.set_ylabel("ssim", color="tab:purple", labelpad=10)
    ax_big3.tick_params(axis="y", labelcolor="tab:purple")

    if len(x) == 1:
        ax_big.set_xlim(x[0] - 0.5, x[0] + 0.5)

    ax_big.legend(handles=[l_total, l_psnr, l_ssim], loc="upper right", fontsize=9, framealpha=0.92)

    # row4 spacer
    ax_sp = fig.add_subplot(gs[4, :])
    ax_sp.axis("off")

    # row5 feedback image
    ax_fb = fig.add_subplot(gs[5, :])
    ax_fb.axis("off")
    ax_fb.set_title("LATEST FEEDBACK (samples_stage2_v14_ultimate)", fontsize=11, fontweight="bold", pad=10)

    if sample_img_path and os.path.exists(sample_img_path):
        im = Image.open(sample_img_path).convert("RGB")
        ax_fb.imshow(np.asarray(im), aspect="auto")
    else:
        ax_fb.text(0.5, 0.5, "sample image not available", ha="center", va="center", fontsize=12)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# Trainer
# =========================================================
class Stage2Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        self.save_dir = cfg["paths"]["save_dir"]
        self.sample_dir = cfg["paths"]["sample_dir"]
        self.log_dir = cfg["paths"]["log_dir"]
        self.g1_ckpt = cfg["paths"]["g1_checkpoint"]
        self.load_g2_path = cfg["paths"].get("load_g2_path", "")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        # dashboard
        self.dashboard_png = os.path.join(self.log_dir, "dashboard_latest.png")
        self.buf = DashEpochBuffer(maxlen=5000)
        self._last_sample_img_path: str = ""

        # best
        self.best_psnr = 0.0

        # metrics cfg
        self.assume_range = cfg.get("metrics", {}).get("assume_range", "neg1_1")
        self.eps = float(cfg.get("metrics", {}).get("eps", 1e-8))
        self.ssim_mode_full = "full"

        self._build_data()
        self._build_models()
        self._build_optim()

        self.diffusion = DiffusionManager(device=self.device)
        self.vgg_criterion = VGGLoss(self.device)

        self.start_epoch = 0
        self._try_resume()

        self.global_step = 0

    # -------------------------
    # Build
    # -------------------------
    def _build_data(self):
        dcfg = self.cfg["data"]
        self.train_loader = DataLoader(
            InpaintingDataset(dcfg["train_root"], mode="train"),
            batch_size=int(dcfg["batch_size"]),
            shuffle=bool(dcfg.get("shuffle", True)),
            num_workers=int(dcfg.get("num_workers", 4)),
            pin_memory=bool(dcfg.get("pin_memory", True)),
        )

    def _build_models(self):
        # G1 fixed
        self.G1 = EdgeGenerator().to(self.device).eval()
        self.G1.load_state_dict(torch.load(self.g1_ckpt, map_location=self.device))

        # G2 + D
        self.G2 = DiffusionUNet(in_channels=8, out_channels=3).to(self.device)
        self.D = Discriminator().to(self.device)

        self.use_dp = (torch.cuda.device_count() > 1 and str(self.device).startswith("cuda"))
        if self.use_dp:
            self.G2 = nn.DataParallel(self.G2)
            self.D = nn.DataParallel(self.D)

    def _build_optim(self):
        tcfg = self.cfg["train"]
        base_lr = float(tcfg["base_lr"])
        wd = float(tcfg.get("weight_decay", 1e-4))
        d_lr = float(tcfg.get("d_lr", 1e-4))

        self.opt_G = optim.AdamW(self.G2.parameters(), lr=base_lr, weight_decay=wd)
        self.opt_D = optim.AdamW(self.D.parameters(), lr=d_lr, weight_decay=wd)

        # AMP scaler (only on CUDA)
        self.scaler_G = torch.amp.GradScaler('cuda') if str(self.device).startswith("cuda") else None
        self.scaler_D = torch.amp.GradScaler('cuda') if str(self.device).startswith("cuda") else None

    def _try_resume(self):
        path = self.load_g2_path
        if not path or (not os.path.exists(path)):
            return

        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            if "optimizer_state_dict" in ckpt:
                self.opt_G.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and self.scaler_G is not None:
                try:
                    self.scaler_G.load_state_dict(ckpt["scaler_state_dict"])
                except Exception:
                    pass
            if "best_psnr" in ckpt:
                self.best_psnr = float(ckpt["best_psnr"])
            if "epoch" in ckpt:
                self.start_epoch = int(ckpt["epoch"]) + 1
        else:
            state = ckpt

        try:
            self.G2.load_state_dict(state, strict=True)
        except RuntimeError:
            new_state = {}
            if self.use_dp and not any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state["module." + k] = v
                self.G2.load_state_dict(new_state, strict=False)
            elif (not self.use_dp) and any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state[k.replace("module.", "", 1)] = v
                self.G2.load_state_dict(new_state, strict=False)
            else:
                raise

        print(f"[*] 成功載入 G2 檢查點，續訓從 Epoch {self.start_epoch} 開始")

    # -------------------------
    # AMP helpers (避免 scaler=None 會炸)
    # -------------------------
    @property
    def amp_enabled(self) -> bool:
        return self.scaler_G is not None and str(self.device).startswith("cuda")

    def _backward_G(self, loss: torch.Tensor):
        if self.amp_enabled:
            self.scaler_G.scale(loss).backward()
        else:
            loss.backward()

    def _step_G(self):
        if self.amp_enabled:
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()
        else:
            self.opt_G.step()

    def _unscale_clip_G(self, grad_clip: float):
        if not (grad_clip and grad_clip > 0):
            return
        if self.amp_enabled:
            self.scaler_G.unscale_(self.opt_G)
        torch.nn.utils.clip_grad_norm_(self.G2.parameters(), grad_clip)

    def _backward_D(self, loss: torch.Tensor):
        if self.scaler_D is not None and str(self.device).startswith("cuda"):
            self.scaler_D.scale(loss).backward()
        else:
            loss.backward()

    def _step_D(self):
        if self.scaler_D is not None and str(self.device).startswith("cuda"):
            self.scaler_D.step(self.opt_D)
            self.scaler_D.update()
        else:
            self.opt_D.step()

    # -------------------------
    # Train
    # -------------------------
    def train(self):
        tcfg = self.cfg["train"]
        epochs = int(tcfg["epochs"])
        pure_epochs = int(tcfg["pure_epochs"])
        accumulation_steps = int(tcfg["accumulation_steps"])
        grad_clip = float(tcfg.get("grad_clip", 0.0))
        w_adv_final = float(tcfg.get("w_adv_final", 0.005))

        log_step_every = int(tcfg.get("log_step_every", 50))
        tb_image_every = int(tcfg.get("tb_image_every", 99999999))  # 你要省事就關掉

        scfg = self.cfg.get("sampling", {})
        sample_steps = int(scfg.get("steps", 200))  # ✅ 建議 200 (論文/穩定性比較好)

        for epoch in range(self.start_epoch, epochs):
            # -------------------------
            # epoch accum (loss avg)
            # -------------------------
            sum_l1 = 0.0
            sum_mse = 0.0
            sum_vgg = 0.0
            sum_advg = 0.0
            sum_total = 0.0
            n_update = 0

            w_recon, w_vgg, w_mse, w_adv, curr_lr = get_sa_config(
                epoch, pure_epochs, float(tcfg["base_lr"]), w_adv_final
            )
            for g in self.opt_G.param_groups:
                g["lr"] = curr_lr
            w_gray = gray_w_schedule(epoch)

            self.G2.train()
            self.D.train()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

            self.opt_G.zero_grad(set_to_none=True)
            self.opt_D.zero_grad(set_to_none=True)

            # 這些會在 epoch end 用（你要抓最後那個點）
            last_imgs = None
            last_masks = None
            last_pred_edges = None
            last_condition = None

            for i, (imgs, _, masks) in enumerate(pbar):
                self.global_step += 1
                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # 1) G1 edge (fixed)
                with torch.no_grad():
                    masked_imgs = imgs * (1 - masks)
                    pred_edges = self.G1(torch.cat([masked_imgs, masks], dim=1))

                condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)

                # 存最後一個 batch（epoch end sample 用）
                last_imgs = imgs
                last_masks = masks
                last_pred_edges = pred_edges
                last_condition = condition

                # 2) Diffusion noise
                t = self.diffusion.sample_timesteps(imgs.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(imgs, t)

                autocast_dev = 'cuda' if str(self.device).startswith("cuda") else 'cpu'
                with torch.amp.autocast(autocast_dev):
                    pred_noise = self.G2(x_t, t, condition)
                    pred_noise = torch.clamp(sanitize_tensor(pred_noise, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

                    l_mse = F.mse_loss(pred_noise, noise)

                    alpha_hat_t = self.diffusion.alpha_hat[t][:, None, None, None]
                    denom = torch.sqrt(alpha_hat_t.clamp(min=1e-4))
                    pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / denom
                    pred_x0 = torch.clamp(sanitize_tensor(pred_x0, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

                    l_pixel = masked_l1(pred_x0, imgs, masks)

                    if (epoch >= pure_epochs) and (w_vgg > 0):
                        l_vgg = self.vgg_criterion(pred_x0, imgs)
                    else:
                        l_vgg = torch.tensor(0.0, device=self.device)

                    l_gray = gray_consistency_loss(pred_x0, masks, w=w_gray)

                    l_adv_G = torch.tensor(0.0, device=self.device)
                    if w_adv > 0:
                        # D step
                        self.opt_D.zero_grad(set_to_none=True)
                        real_res = self.D(imgs)
                        fake_res = self.D(pred_x0.detach())
                        loss_D = (F.relu(1.0 - real_res).mean() + F.relu(1.0 + fake_res).mean()) * 0.5
                        self._backward_D(loss_D)
                        self._step_D()

                        l_adv_G = -self.D(pred_x0).mean()

                    loss_total = (
                        l_mse   * w_mse +
                        l_vgg   * w_vgg +
                        l_pixel * w_recon +
                        l_adv_G * w_adv +
                        l_gray
                    ) / accumulation_steps

                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    print(f"[⚠️] NaN/Inf loss at epoch {epoch}, step {i}, skip.")
                    continue

                # backward
                self._backward_G(loss_total)

                # optim step (accum)
                if (i + 1) % accumulation_steps == 0:
                    self._unscale_clip_G(grad_clip)
                    self._step_G()
                    self.opt_G.zero_grad(set_to_none=True)

                    # epoch sums (use true per-step loss_total * accumulation_steps)
                    sum_l1 += float(l_pixel.item())
                    sum_mse += float(l_mse.item())
                    sum_vgg += float(l_vgg.item()) if torch.is_tensor(l_vgg) else float(l_vgg)
                    sum_advg += float(l_adv_G.item())
                    sum_total += float(loss_total.item()) * accumulation_steps
                    n_update += 1

                # TB step scalars (可留)
                if self.global_step % log_step_every == 0:
                    total_step = float(loss_total.item()) * accumulation_steps
                    self.writer.add_scalar("loss/total_step", total_step, self.global_step)
                    self.writer.add_scalar("loss/l1_mask_step", float(l_pixel.item()), self.global_step)
                    self.writer.add_scalar("loss/mse_step", float(l_mse.item()), self.global_step)
                    self.writer.add_scalar("params/lr", curr_lr, self.global_step)
                    self.writer.flush()

                # TB images (你不想每 step，就乾脆關掉)
                if self.global_step % tb_image_every == 0:
                    self.writer.add_image("Preview/pred_x0", to_01(pred_x0[0], self.assume_range), self.global_step)

                if i % 20 == 0:
                    pbar.set_postfix({
                        "L1_mask": f"{l_pixel.item():.4f}",
                        "MSE": f"{l_mse.item():.4f}",
                        "LR": f"{curr_lr:.2e}",
                    })

            # =========================
            # epoch averages (loss)
            # =========================
            denom = max(1, n_update)
            avg_l1 = sum_l1 / denom
            avg_mse = sum_mse / denom
            avg_vgg = sum_vgg / denom
            avg_advg = sum_advg / denom
            avg_total = sum_total / denom

            # log epoch scalars
            self.writer.add_scalar("Epoch/L1_mask_avg", avg_l1, epoch)
            self.writer.add_scalar("Epoch/MSE_avg", avg_mse, epoch)
            self.writer.add_scalar("Epoch/VGG_avg", avg_vgg, epoch)
            self.writer.add_scalar("Epoch/AdvG_avg", avg_advg, epoch)
            self.writer.add_scalar("Epoch/Total_avg", avg_total, epoch)
            self.writer.add_scalar("Epoch/LR", curr_lr, epoch)

            self.writer.add_scalar("Epoch/w_recon", w_recon, epoch)
            self.writer.add_scalar("Epoch/w_vgg", w_vgg, epoch)
            self.writer.add_scalar("Epoch/w_mse", w_mse, epoch)
            self.writer.add_scalar("Epoch/w_adv", w_adv, epoch)
            self.writer.add_scalar("Epoch/w_gray", w_gray, epoch)
            self.writer.flush()

            # =========================================================
            # Epoch-end SAMPLE (唯一採用的 dashboard 點：full psnr/ssim)
            # - 使用「最後一個 batch」的資料，符合你說的：每個 epoch 抓最後那個點
            # =========================================================
            self.G2.eval()
            with torch.no_grad():
                g2 = self.G2.module if hasattr(self.G2, "module") else self.G2

                if last_condition is None:
                    # 理論上不會發生
                    last_condition = condition
                    last_imgs = imgs
                    last_masks = masks
                    last_pred_edges = pred_edges

                # 固定 seed（讓 epoch-end sample 可重現）
                seed = int(self.cfg.get("seed", 0))
                torch.manual_seed(seed + epoch)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + epoch)

                # generate one sample (inpaint result)
                samples = self.diffusion.sample(g2, last_condition[0:1], n=1, steps=sample_steps)
                samples = torch.clamp(sanitize_tensor(samples, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                res_img = last_imgs[0:1] * (1 - last_masks[0:1]) + samples * last_masks[0:1]
                res_img = torch.clamp(sanitize_tensor(res_img, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

                # FULL metrics (mask=None)  <-- ✅ 大圖只用這個
                psnr_full = psnr_torch(res_img, last_imgs[0:1], mask=None,
                                       assume_range=self.assume_range, eps=self.eps).item()
                ssim_full = ssim_torch(res_img, last_imgs[0:1], mask=None,
                                       mode=self.ssim_mode_full,
                                       assume_range=self.assume_range, eps=self.eps).item()
                psnr_full = safe_metric_value(psnr_full, default=np.nan)
                ssim_full = safe_metric_value(ssim_full, default=np.nan)

                # BEST by full PSNR
                if np.isfinite(psnr_full) and psnr_full > self.best_psnr:
                    self.best_psnr = float(psnr_full)
                    torch.save(g2.state_dict(), f"{self.save_dir}/G2_BEST_PSNR_latest.pth")
                    torch.save(g2.state_dict(), f"{self.save_dir}/G2_BEST_PSNR_ep{epoch}.pth")
                    print(f"\n[🏆] New BEST PSNR(full)! Epoch {epoch} PSNR(full): {self.best_psnr:.2f}")

                self.writer.add_scalar("Sample/PSNR_full", psnr_full, epoch)
                self.writer.add_scalar("Sample/SSIM_full", ssim_full, epoch)

                # Build ultimate sample image (for dashboard feedback)
                orig_u = chw01_to_hwc_uint8(to_01(last_imgs[0], self.assume_range))
                mask_u = mask_to_hwc_uint8(torch.clamp(last_masks[0], 0, 1))
                edge_map = torch.clamp(sanitize_tensor(last_pred_edges[0], nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
                edge_u = chw01_to_hwc_uint8((edge_map + 1) * 0.5)
                masked_u = chw01_to_hwc_uint8(to_01(last_imgs[0] * (1 - last_masks[0]), self.assume_range))
                res_u = chw01_to_hwc_uint8(to_01(res_img[0], self.assume_range))

                sample_img_path = os.path.join(self.sample_dir, f"samples_stage2_v14_ultimate_ep{epoch}.png")
                # sample_img_path, _, _ = samples_stage2_v14_ultimate(
                #     out_path=sample_img_path,
                #     orig_u8=orig_u,
                #     mask_u8=mask_u,
                #     edge_u8=edge_u,
                #     masked_u8=masked_u,
                #     res_u8=res_u,
                #     psnr_full=float(psnr_full) if np.isfinite(psnr_full) else float("nan"),
                #     ssim_full=float(ssim_full) if np.isfinite(ssim_full) else float("nan"),
                #     epoch=epoch,
                # )
                self._last_sample_img_path = sample_img_path

                # 你原本的存檔（保留）
                self.writer.add_image("Sample/Res_img", to_01(res_img[0], self.assume_range), epoch)
                self.writer.flush()
                save_preview_image(last_imgs, last_masks, last_pred_edges, res_img,
                                   psnr_full if np.isfinite(psnr_full) else 0.0, epoch, self.sample_dir)

            # =========================================================
            # Dash buffer: append ONLY ONE POINT per epoch (epoch-end sample point)
            # =========================================================
            self.buf.append_epoch(epoch, {
                "l1": avg_l1,
                "mse": avg_mse,
                "vgg": avg_vgg,
                "advg": avg_advg,
                "total": avg_total,
                "lr": curr_lr,
                "w_recon": w_recon,
                "w_vgg": w_vgg,
                "w_mse": w_mse,
                "w_adv": w_adv,
                "psnr_full": psnr_full,
                "ssim_full": ssim_full,
            })

            # =========================================================
            # Write dashboard_latest.png ONCE per epoch  <-- ✅ 你要的
            # =========================================================
            meta = {
                "title": "v14.2 Strategy Dashboard (Built-in PNG)",
                "update_str": time.strftime("%Y-%m-%d %H:%M:%S"),
                "latest_step": self.global_step,
                "latest_epoch": str(epoch),
                "sample_img_path": self._last_sample_img_path,
            }
            render_dashboard_png_epoch_only(self.dashboard_png, self.buf, meta)

            # =========================================================
            # IEEE figure（用 epoch 曲線，不是單點；失敗不會中斷訓練）
            # =========================================================
            try:
                plot_ieee_dashboard(
                    out_path=os.path.join(self.log_dir, "ieee_fig_epoch_curves.png"),
                    epochs=self.buf.x_epoch,
                    total_loss=self.buf.total,
                    psnr_full=self.buf.psnr_full,
                    ssim_full=self.buf.ssim_full,
                    l1=self.buf.l1,
                    mse=self.buf.mse,
                    lr=self.buf.lr,
                    title="Stage-2 Training Curves (FULL-image PSNR/SSIM, epoch-end sample point)"
                )
            except Exception as e:
                print(f"[WARN] plot_ieee_dashboard failed (ignored): {e}")

            # =========================================================
            # Save checkpoint
            # =========================================================
            torch.save({
                "epoch": epoch,
                "model_state_dict": (self.G2.module.state_dict() if hasattr(self.G2, "module") else self.G2.state_dict()),
                "optimizer_state_dict": self.opt_G.state_dict(),
                "scaler_state_dict": (self.scaler_G.state_dict() if self.scaler_G is not None else {}),
                "best_psnr": self.best_psnr,
            }, f"{self.save_dir}/checkpoint_latest.pth")

        self.writer.close()
