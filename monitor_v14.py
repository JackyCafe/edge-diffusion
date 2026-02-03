import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorboard.backend.event_processing import event_accumulator
from scipy import stats

# ================= 設定區 =================
LOG_DIR = "runs/stage2_v14_ultimate"
SAMPLE_DIR = "samples_stage2_v14_ultimate"
SAVE_REPORT_PATH = "training_reports/v14_2_three_axis_report.png"
WINDOW_SIZE = 10
TOTAL_EPOCHS = 200
SLEEP_SECONDS = 600
# =========================================

# --------- 小工具：tag 相容層 ---------
TAG_ALIASES = {
    # Loss
    "L1": ["Loss_Detail/Masked_L1", "Losses/L1_Loss", "Loss_Detail/Pixel_L1"],
    "MSE": ["Loss_Detail/MSE_Noise", "Losses/MSE_Loss"],
    "VGG": ["Loss_Detail/VGG", "Losses/VGG_Loss", "Loss_Detail/VGG_Perceptual"],
    "ADV": ["Loss_Detail/Adv_G", "Losses/Adv_Loss", "Losses/Adv_Loss_G"],
    "TOTAL": ["Losses/Total_Loss", "Loss_Detail/Total", "Loss_Detail/Total_Loss"],

    # Metrics
    "PSNR": ["Metrics/PSNR"],
    "SSIM": ["Metrics/SSIM"],

    # LR
    "LR": ["Params/Learning_Rate", "Status/LearningRate"],
    # Weights
    "W_RECON": ["Weights/w_recon"],
    "W_VGG": ["Weights/w_vgg"],
    "W_MSE": ["Weights/w_mse"],
    "W_ADV": ["Weights/w_adv"],
}

def pick_first_existing_tag(tags_available, candidates):
    for t in candidates:
        if t in tags_available:
            return t
    return None

def safe_moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return None
    w = int(min(window, len(values)))
    if w <= 1:
        return values  # 沒必要做平滑
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode='valid')

def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def get_latest_event_file(log_dir):
    if not os.path.exists(log_dir):
        return None
    all_events = []
    for r, _, fs in os.walk(log_dir):
        for f in fs:
            if "events.out.tfevents" in f:
                all_events.append(os.path.join(r, f))
    if not all_events:
        return None
    return max(all_events, key=os.path.getmtime)

def extract_epoch_num(filename):
    # 支援 epoch_12.png、epoch12.png、..._ep12.png 這類
    m = re.search(r"(?:epoch[_\-]?|ep[_\-]?)(\d+)", filename, re.IGNORECASE)
    return int(m.group(1)) if m else -1

def get_latest_sample_image(sample_dir):
    if not os.path.exists(sample_dir):
        return None
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".png")]
    if not files:
        return None
    files_sorted = sorted(files, key=lambda x: extract_epoch_num(x))
    # 若都抓不到 epoch，fallback 用 mtime
    if extract_epoch_num(files_sorted[-1]) == -1:
        full = [os.path.join(sample_dir, f) for f in files]
        return max(full, key=os.path.getmtime)
    return os.path.join(sample_dir, files_sorted[-1])

def read_scalar(ea, tag):
    ev = ea.Scalars(tag)
    steps = np.array([e.step for e in ev], dtype=np.int64)
    vals = np.array([e.value for e in ev], dtype=np.float64)
    return steps, vals

def plot_series(ax, steps, vals, title, ylog=False, ma_window=10, color=None):
    if len(vals) == 0:
        ax.set_title(f"{title}\n(no data)", fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2)
        return

    ax.plot(steps, vals, alpha=0.18, color=color)
    ma = safe_moving_average(vals, ma_window)
    if ma is not None and len(ma) >= 2:
        ax.plot(steps[len(vals) - len(ma):], ma, linewidth=3.5, color=color)

    if ylog:
        # 避免 log(0) 爆掉
        ax.set_yscale('log')

    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2)

def plot_v14_2_three_axis():
    event_file = get_latest_event_file(LOG_DIR)
    if not event_file:
        print(f"[{time.strftime('%H:%M:%S')}] 找不到 events 檔：{LOG_DIR}")
        return

    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))

    # 解析核心 tags（有就用、沒有就跳過）
    tag_total = pick_first_existing_tag(tags, TAG_ALIASES["TOTAL"])
    tag_psnr  = pick_first_existing_tag(tags, TAG_ALIASES["PSNR"])
    tag_ssim  = pick_first_existing_tag(tags, TAG_ALIASES["SSIM"])

    # 取目前最新 epoch（用 PSNR step 優先，其次 total loss）
    latest_step = None
    if tag_psnr:
        ps, _ = read_scalar(ea, tag_psnr)
        latest_step = int(ps[-1]) if len(ps) else None
    if latest_step is None and tag_total:
        ls, _ = read_scalar(ea, tag_total)
        latest_step = int(ls[-1]) if len(ls) else None
    if latest_step is None:
        latest_step = 0

    # 設置畫布
    fig = plt.figure(figsize=(50, 80))
    gs = fig.add_gridspec(
        10, 4,
        height_ratios=[1, 1, 1, 1.2, 2.5, 1.5, 2.5, 0.4, 0.1, 0.1]
    )
    plt.suptitle(
        f"v14.2 Three-Axis Strategy Monitor\nUpdate: {time.strftime('%Y-%m-%d %H:%M:%S')} | Latest Step: {latest_step}",
        fontsize=40, fontweight='bold', y=0.98
    )

    # --- 1. 基礎指標區 (Row 0-1) ---
    metric_panels = [
        ("L1",   "L1 / Masked L1",   0, 0, "tab:cyan"),
        ("MSE",  "MSE (Noise)",      0, 1, "tab:blue"),
        ("VGG",  "VGG Loss",         0, 2, "tab:orange"),
        ("ADV",  "Adv Loss (G)",     0, 3, "tab:green"),
        ("TOTAL","Total Loss",       1, 0, "tab:red"),
        ("SSIM", "SSIM",             1, 1, "tab:purple"),
        ("PSNR", "PSNR (dB)",        1, 2, "tab:red"),
        ("LR",   "Learning Rate",    1, 3, "darkred"),
    ]

    for key, label, row, col, color in metric_panels:
        ax = fig.add_subplot(gs[row, col])
        tag = pick_first_existing_tag(tags, TAG_ALIASES[key])
        if not tag:
            ax.set_title(f"{label}\n(tag missing)", fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.2)
            continue
        s, v = read_scalar(ea, tag)
        plot_series(ax, s, v, f"{label}\n({tag})", ylog=("Learning Rate" in label), ma_window=WINDOW_SIZE, color=color)

    # --- 2. 權重區 (Row 2) ---
    weight_panels = [
        ("W_RECON", "Pixel Weight", "tab:blue"),
        ("W_VGG",   "VGG Weight",   "tab:orange"),
        ("W_MSE",   "Noise Weight", "tab:green"),
        ("W_ADV",   "Adv Weight",   "tab:red"),
    ]
    for i, (key, label, color) in enumerate(weight_panels):
        ax_w = fig.add_subplot(gs[2, i])
        tag = pick_first_existing_tag(tags, TAG_ALIASES[key])
        if not tag:
            ax_w.set_title(f"{label}\n(tag missing)", fontsize=16, fontweight='bold')
            ax_w.grid(True, alpha=0.2)
            continue
        s, v = read_scalar(ea, tag)
        ax_w.plot(s, v, color=color, linewidth=4)
        ax_w.fill_between(s, v, color=color, alpha=0.08)
        ax_w.set_title(f"{label}\n({tag})", fontsize=18, fontweight='bold')
        ax_w.grid(True, alpha=0.3)

    # --- 3. 三軸對比 (Row 4) ---
    ax_main = fig.add_subplot(gs[4, :])
    if not (tag_total and tag_psnr and tag_ssim):
        ax_main.set_title(
            "Triple-Axis Analysis\n(need tags: Total Loss + PSNR + SSIM)",
            fontsize=28, fontweight='bold'
        )
        ax_main.grid(True, alpha=0.2)
    else:
        ls, lv = read_scalar(ea, tag_total)
        ps, pv = read_scalar(ea, tag_psnr)
        ss, sv = read_scalar(ea, tag_ssim)

        # Total loss (left)
        ax_main.plot(ls, lv, alpha=0.1, color='tab:blue')
        l_ma = safe_moving_average(lv, WINDOW_SIZE)
        if l_ma is not None and len(l_ma) >= 2:
            ax_main.plot(ls[len(lv)-len(l_ma):], l_ma, color='tab:blue', linewidth=6, label='Loss')
        ax_main.set_ylabel("Total Loss", fontsize=22, color='tab:blue', fontweight='bold')
        ax_main.tick_params(axis='y', labelcolor='tab:blue')
        ax_main.grid(True, alpha=0.2)

        # PSNR (right 1)
        ax_psnr = ax_main.twinx()
        ax_psnr.plot(ps, pv, alpha=0.1, color='tab:red')
        p_ma = safe_moving_average(pv, WINDOW_SIZE)
        if p_ma is not None and len(p_ma) >= 2:
            ax_psnr.plot(ps[len(pv)-len(p_ma):], p_ma, color='tab:red', linewidth=6, label='PSNR')
        ax_psnr.set_ylabel("PSNR (dB)", fontsize=22, color='tab:red', fontweight='bold')
        ax_psnr.tick_params(axis='y', labelcolor='tab:red')

        # SSIM (right 2 offset)
        ax_ssim = ax_main.twinx()
        ax_ssim.spines["right"].set_position(("axes", 1.08))
        ax_ssim.plot(ss, sv, alpha=0.1, color='tab:purple')
        s_ma = safe_moving_average(sv, WINDOW_SIZE)
        if s_ma is not None and len(s_ma) >= 2:
            ax_ssim.plot(ss[len(sv)-len(s_ma):], s_ma, color='tab:purple', linewidth=6, label='SSIM')
        ax_ssim.set_ylabel("SSIM", fontsize=22, color='tab:purple', fontweight='bold')
        ax_ssim.tick_params(axis='y', labelcolor='tab:purple')
        ax_ssim.set_ylim(min(sv)*0.98 if len(sv) else 0.0, 1.0)

        ax_main.set_title(
            f"Triple-Axis Analysis: Loss vs PSNR vs SSIM\nLossTag={tag_total} | PSNRTag={tag_psnr} | SSIMTag={tag_ssim}",
            fontsize=28, fontweight='bold'
        )

    # --- 4. Milestone Prediction (Row 5) ---
    ax_pred = fig.add_subplot(gs[5, :])
    if tag_psnr:
        ps, pv = read_scalar(ea, tag_psnr)
        ax_pred.plot(ps, pv, alpha=0.25, color='tab:red')

        # 用最近 N 點做線性回歸（避免太少點爆掉）
        N = min(20, len(pv))
        if N >= 8:
            slope, intercept, r, _, _ = stats.linregress(ps[-N:], pv[-N:])
            future_x = np.arange(ps[-1], TOTAL_EPOCHS + 1)
            future_y = slope * future_x + intercept
            ax_pred.plot(future_x, future_y, '--', color='darkred', linewidth=3)

            ax_pred.scatter([TOTAL_EPOCHS], [future_y[-1]], s=500, marker='*',
                            label=f"Ep{TOTAL_EPOCHS}≈{future_y[-1]:.2f}dB | r={r:.2f}")
            ax_pred.legend(fontsize=18)

        ax_pred.set_title(f"Future PSNR Projection ({tag_psnr})", fontsize=24, fontweight='bold')
        ax_pred.grid(True, alpha=0.2)
    else:
        ax_pred.set_title("Future PSNR Projection\n(tag missing)", fontsize=24, fontweight='bold')
        ax_pred.grid(True, alpha=0.2)

    # --- 5. 視覺回饋 (Row 6) ---
    sample_img = get_latest_sample_image(SAMPLE_DIR)
    ax_img = fig.add_subplot(gs[6, :])
    if sample_img and os.path.exists(sample_img):
        ax_img.imshow(mpimg.imread(sample_img))
        ax_img.set_title(f"LATEST VISUAL FEEDBACK\n{os.path.basename(sample_img)}", fontsize=30, fontweight='bold')
        ax_img.axis('off')
    else:
        ax_img.set_title("LATEST VISUAL FEEDBACK\n(no sample image found)", fontsize=30, fontweight='bold')
        ax_img.axis('off')

    # --- 6. Footer / Debug info (Row 7) ---
    ax_footer = fig.add_subplot(gs[7, :])
    ax_footer.axis('off')
    shown_tags = {
        "TOTAL": tag_total,
        "PSNR": tag_psnr,
        "SSIM": tag_ssim,
        "LR": pick_first_existing_tag(tags, TAG_ALIASES["LR"]),
        "L1": pick_first_existing_tag(tags, TAG_ALIASES["L1"]),
        "MSE": pick_first_existing_tag(tags, TAG_ALIASES["MSE"]),
        "VGG": pick_first_existing_tag(tags, TAG_ALIASES["VGG"]),
        "ADV": pick_first_existing_tag(tags, TAG_ALIASES["ADV"]),
    }
    ax_footer.text(
        0.01, 0.65,
        "Resolved Tags:\n" + "\n".join([f"- {k}: {v}" for k, v in shown_tags.items()]),
        fontsize=18, family='monospace'
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    ensure_dir(SAVE_REPORT_PATH)
    plt.savefig(SAVE_REPORT_PATH, dpi=100)
    plt.close()

    print(f"[{time.strftime('%H:%M:%S')}] 報表更新成功 → {SAVE_REPORT_PATH}")


if __name__ == "__main__":
    while True:
        try:
            plot_v14_2_three_axis()
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
        time.sleep(SLEEP_SECONDS)
