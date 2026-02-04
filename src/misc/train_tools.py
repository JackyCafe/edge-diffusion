# src/utils/train_tools.py
from __future__ import annotations
import math
import os
import glob
from typing import Optional, Dict, Any, Tuple
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# =========================
# safe numeric
# =========================
def is_finite_number(x: float) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_float(x, fallback: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else fallback
    except Exception:
        return fallback




# =========================================================
# Tool / Utils
# =========================================================

def sanitize_tensor(x: torch.Tensor,
                    nan: float = 0.0,
                    posinf: float = 1.0,
                    neginf: float = -1.0) -> torch.Tensor:
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def to_01(x: torch.Tensor, assume_range: str = "neg1_1") -> torch.Tensor:
    x = sanitize_tensor(x, nan=0.0, posinf=1.0, neginf=0.0)
    if assume_range == "neg1_1":
        x = (x + 1.0) * 0.5
    x = torch.clamp(x, 0.0, 1.0)
    return x



def chw01_to_hwc_uint8(x_chw01: torch.Tensor) -> np.ndarray:
    x = sanitize_tensor(x_chw01, nan=0.0, posinf=1.0, neginf=0.0)
    x = torch.clamp(x, 0.0, 1.0).detach().cpu()

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Unsupported tensor shape for image: {tuple(x.shape)}")

    x = x.permute(1, 2, 0).numpy()  # HWC

    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    elif x.shape[2] != 3:
        x = np.repeat(x[:, :, :1], 3, axis=2)

    return (x * 255.0).clip(0, 255).astype(np.uint8)


def mask_to_hwc_uint8(mask_1hw: torch.Tensor) -> np.ndarray:
    m = sanitize_tensor(mask_1hw, nan=0.0, posinf=1.0, neginf=0.0)
    m = torch.clamp(m, 0.0, 1.0).detach().cpu()

    if m.ndim == 4:
        m = m[0]
    if m.ndim == 3:
        m = m.squeeze(0)
    if m.ndim != 2:
        raise ValueError(f"Unsupported mask shape: {tuple(m.shape)}")

    u = (m.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return np.stack([u, u, u], axis=-1)

def sanitize_mask(mask: torch.Tensor) -> torch.Tensor:
    # mask should be [B,1,H,W] in {0,1}
    m = torch.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
    return torch.clamp(m, 0.0, 1.0)


def mask_pixel_count(mask: torch.Tensor) -> torch.Tensor:
    # [B,1,H,W] -> [B]
    return mask.flatten(1).sum(dim=1)


def has_small_mask(mask: torch.Tensor, min_pixels: int = 64) -> torch.Tensor:
    # return [B] bool
    return mask_pixel_count(mask) < float(min_pixels)


def safe_metric_value(v: float, default: float = np.nan) -> float:
    try:
        v = float(v)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
              eps: float = 1e-6, min_pixels: float = 64.0) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    msum = torch.clamp(mask.sum(), min=min_pixels)
    denom = msum * pred.shape[1] + eps
    return diff.sum() / denom


def gray_w_schedule(epoch: int) -> float:
    if epoch < 40:
        return 0.0
    elif epoch < 80:
        return 0.02
    else:
        return 0.05


def gray_consistency_loss(x: torch.Tensor, masks: torch.Tensor, w: float) -> torch.Tensor:
    if w <= 0:
        return torch.tensor(0.0, device=x.device)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    loss = (masked_l1(r, g, masks) + masked_l1(r, b, masks) + masked_l1(g, b, masks)) / 3.0
    return loss * w



# =========================
# safe metrics wrapper
# =========================
@torch.no_grad()
def safe_psnr(
    psnr_fn,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    assume_range: str,
    eps: float,
    fallback: float = 0.0,
) -> float:
    try:
        v = psnr_fn(pred, target, mask=mask, assume_range=assume_range, eps=eps).item()
        return v if is_finite_number(v) else fallback
    except Exception:
        return fallback


@torch.no_grad()
def safe_ssim(
    ssim_fn,
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor],
    mode: str,
    assume_range: str,
    eps: float,
    fallback: float = 0.0,
) -> float:
    try:
        v = ssim_fn(pred, target, mask=mask, mode=mode, assume_range=assume_range, eps=eps).item()
        return v if is_finite_number(v) else fallback
    except Exception:
        return fallback


# =========================
# TB image safe conversion
# =========================
def to_tb_image_01(x: torch.Tensor) -> torch.Tensor:
    """
    input: CHW or BCHW in [-1,1] or [0,1]
    output: CHW float in [0,1] (safe for tensorboard)
    """
    if x.dim() == 4:
        x = x[0]
    x = sanitize_tensor(x, nan=0.0, posinf=1.0, neginf=0.0)
    # if looks like [-1,1], map to [0,1]
    if x.min().item() < 0.0:
        x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    return x


# =========================
# file tools
# =========================
def find_latest_event_file(log_dir: str) -> Optional[str]:
    pattern = os.path.join(log_dir, "**", "events.out.tfevents.*")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p))
    return files[-1]

def ieee_mpl_style(fontsize=8):
    """IEEE 論文常見風格：小字、細線、serif。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": fontsize,
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize-1,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.1,
        "savefig.bbox": "tight",
    })

def save_ieee_figure(fig, out_path, dpi=300):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # PDF（向量）建議優先；PNG 做備用
    base, ext = os.path.splitext(out_path)
    fig.savefig(base + ".pdf")               # vector
    fig.savefig(base + ".png", dpi=dpi)      # raster
    plt.close(fig)



import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_ieee_dashboard(
    out_path: str,
    x,
    total_loss,
    psnr_full,
    ssim_full,
    l1=None,
    mse=None,
    lr=None,
    title="Stage-2 Training Curves (FULL-image PSNR/SSIM, epoch-end sample point)"
):
    """
    IEEE-style multi-panel:
      (a) Total loss + PSNR + SSIM (3 y-axes)
      (b) L1(masked)
      (c) MSE(noise)
      (d) LR
      Notes panel

    Robust for very few points (even 1 point).
    """

    def to_np(arr):
        if arr is None:
            return None
        a = np.asarray(arr, dtype=np.float32)
        return a

    x = to_np(x)
    total_loss = to_np(total_loss)
    psnr_full = to_np(psnr_full)
    ssim_full = to_np(ssim_full)
    l1 = to_np(l1)
    mse = to_np(mse)
    lr = to_np(lr)

    if x is None or len(x) == 0:
        return

    # ---- handle 1-point case: expand xlim so lines/markers visible ----
    if len(x) == 1:
        x_min, x_max = float(x[0] - 1.0), float(x[0] + 1.0)
    else:
        pad = max(0.5, 0.03 * (x.max() - x.min()))
        x_min, x_max = float(x.min() - pad), float(x.max() + pad)

    # ---- figure layout (constrained + extra right margin for 3rd axis) ----
    fig = plt.figure(figsize=(7.2, 4.2), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[2.3, 1.5], width_ratios=[1, 1, 1, 1.15])

    # (a) big panel
    ax = fig.add_subplot(gs[0, :])
    ax.set_title(title, fontsize=10, pad=6)

    # total loss (left)
    l_total, = ax.plot(x, total_loss, marker="o", markersize=3, linewidth=1.2,
                       color="tab:blue", label="Total Loss")
    ax.set_xlabel("Epoch", fontsize=9)
    ax.set_ylabel("Total Loss", fontsize=9, color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")
    ax.set_xlim(x_min, x_max)
    ax.grid(True, linestyle="--", alpha=0.25)

    # PSNR (right)
    ax2 = ax.twinx()
    l_psnr, = ax2.plot(x, psnr_full, marker="o", markersize=3, linewidth=1.2,
                       color="tab:red", label="PSNR (dB)")
    ax2.set_ylabel("PSNR (dB)", fontsize=9, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # SSIM (third axis) — IMPORTANT: leave room + do NOT let tight clip it
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))  # push outward
    l_ssim, = ax3.plot(x, ssim_full, marker="o", markersize=3, linewidth=1.2,
                       color="tab:purple", label="SSIM")
    ax3.set_ylabel("SSIM", fontsize=9, color="tab:purple", labelpad=10)
    ax3.tick_params(axis="y", labelcolor="tab:purple")

    # Legend: put BELOW (a), outside plotting area to avoid overlap
    ax.legend(handles=[l_total, l_psnr, l_ssim],
              loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, fontsize=8, framealpha=0.95)

    # (b) L1
    axb = fig.add_subplot(gs[1, 0])
    axb.set_title("(b)", fontsize=10, loc="left", pad=2)
    if l1 is not None:
        axb.plot(x, l1, marker="o", markersize=3, linewidth=1.0)
    axb.set_xlabel("Epoch", fontsize=9)
    axb.set_ylabel("L1 (masked)", fontsize=9)
    axb.set_xlim(x_min, x_max)
    axb.grid(True, linestyle="--", alpha=0.25)

    # (c) MSE
    axc = fig.add_subplot(gs[1, 1])
    axc.set_title("(c)", fontsize=10, loc="left", pad=2)
    if mse is not None:
        axc.plot(x, mse, marker="o", markersize=3, linewidth=1.0)
    axc.set_xlabel("Epoch", fontsize=9)
    axc.set_ylabel("MSE (noise)", fontsize=9)
    axc.set_xlim(x_min, x_max)
    axc.grid(True, linestyle="--", alpha=0.25)

    # (d) LR
    axd = fig.add_subplot(gs[1, 2])
    axd.set_title("(d)", fontsize=10, loc="left", pad=2)
    if lr is not None:
        axd.plot(x, lr, marker="o", markersize=3, linewidth=1.0)
    axd.set_xlabel("Epoch", fontsize=9)
    axd.set_ylabel("LR", fontsize=9)
    axd.set_xlim(x_min, x_max)
    axd.grid(True, linestyle="--", alpha=0.25)

    # Notes
    axn = fig.add_subplot(gs[1, 3])
    axn.axis("off")
    axn.text(0.0, 1.0,
             "Notes:\n"
             "• Metrics computed on FULL image\n"
             "• One point per epoch (epoch-end sample)\n"
             "• 3-axis plot uses: Total / PSNR(full) / SSIM(full)",
             fontsize=8.5, va="top")

    # IMPORTANT: save without bbox_inches="tight" (it clips 3rd axis!)
    # Make sure directory exists
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
