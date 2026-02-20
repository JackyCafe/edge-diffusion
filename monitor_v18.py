import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from scipy import stats
from PIL import Image

from monitor import plot_v14_2_three_axis

# ================= 設定區 =================
TAG_ALIASES = {
    "L1":      ["Loss_Detail/Pixel_L1"],
    "MSE":     ["Loss_Detail/MSE_Noise"],
    "VGG":     ["Loss_Detail/VGG"],
    "ADV":     ["Losses/Adv_Loss_G"],
    "TOTAL":   ["Losses/Total_Loss"],
    "PSNR":    ["Metrics/PSNR"],
    "SSIM":    ["Metrics/SSIM"],
    "LR":      ["Params/Learning_Rate"],
    "W_RECON": ["Weights/w_recon"],
    "W_MSE":   ["Weights/w_mse"],
    "W_VGG":   ["Weights/w_vgg"],
    "W_CS":    ["Weights/w_color_stats"],
    "W_GS":    ["Weights/w_gray_struct"],
    "COLOR_S": ["Loss_Detail/Color_Stats"],
    "GRAY_S":  ["Loss_Detail/Gray_Structural"],
}

SMOOTHING_WINDOW = 10 
FORECAST_STEPS = 10 

# ================= 工具函數 =================
def moving_average(vals, window=SMOOTHING_WINDOW):
    """修復版滑動平均：避免頭尾掉落"""
    if len(vals) < window: return vals
    # 使用 valid 模式計算，並用原始數據補足頭尾，避免卷積補零導致的垂直掉落
    weights = np.ones(window) / window
    sma = np.convolve(vals, weights, mode='valid')
    
    # 補足因捲積損失的長度 (頭尾各補一點)
    head = np.full(window // 2, sma[0])
    tail = np.full(len(vals) - len(sma) - len(head), sma[-1])
    return np.concatenate([head, sma, tail])

def pick_first_existing_tag(tags_available, candidates):
    for t in candidates:
        if t in tags_available: return t
    return None

def get_latest_event_file(log_dir):
    if not os.path.exists(log_dir): return None
    all_events = []
    for r, _, fs in os.walk(log_dir):
        for f in fs:
            if "events.out.tfevents" in f:
                all_events.append(os.path.join(r, f))
    return max(all_events, key=os.path.getmtime) if all_events else None

def get_latest_n_sample_images(sample_dir, n=4):
    if not os.path.exists(sample_dir): return []
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg")) and "dashboard" not in f.lower()]
    if not files: return []
    def extract_num(f):
        m = re.search(r"epoch_(\d+)", f)
        return int(m.group(1)) if m else 0
    sorted_files = sorted(files, key=extract_num)
    return [os.path.join(sample_dir, f) for f in sorted_files[-n:]]

def read_scalar(ea, tag):
    try:
        ev = ea.Scalars(tag)
        steps = np.array([e.step for e in ev])
        vals = np.array([e.value for e in ev])
        mask = np.isfinite(vals)
        return steps[mask], vals[mask]
    except:
        return np.array([]), np.array([])

# ================= 核心繪圖邏輯 =================
def plot_three_axis(log_dir, sample_dir, save_report_path):
    event_file = get_latest_event_file(log_dir)
    if not event_file:
        print(f"[{time.strftime('%H:%M:%S')}] 等待 Log 檔案於 {log_dir}...")
        return

    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))

    plt.style.use('default')
    fig = plt.figure(figsize=(55, 110), dpi=100)
    
    gs = fig.add_gridspec(10, 3, height_ratios=[1, 1, 1, 1, 0.2, 2.5, 3.0, 3.0, 3.0, 3.0], hspace=0.6)

    tag_total_ref = pick_first_existing_tag(tags, TAG_ALIASES["TOTAL"])
    latest_step = "N/A"
    if tag_total_ref:
        s_tmp, _ = read_scalar(ea, tag_total_ref)
        if len(s_tmp) > 0: latest_step = s_tmp[-1]

    plt.suptitle(f"Edge-Diffusion stage2 Multi-Visual Monitor (V19 Ultimate)\nUpdate: {time.strftime('%Y-%m-%d %H:%M:%S')} | Latest Step: {latest_step}",
                  fontsize=55, fontweight='bold', color='#333333', y=0.985)

    def plot_sub(ax, s, v, label, col, is_log=False, is_weight=False):
        if len(v) > 0:
            ax.plot(s, v, alpha=0.15, color=col, linewidth=1.5)
            v_plot = moving_average(v) if not is_weight else v
            ax.plot(s, v_plot, alpha=1.0, color=col, linewidth=6)

            y_min, y_max = np.min(v), np.max(v)
            margin = (y_max - y_min) * 0.15 if y_max != y_min else 0.1
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlim(min(s), max(s))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        ax.set_title(label, fontsize=28, color="#555555", fontweight='bold', pad=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        if is_log and len(v) > 0 and np.all(v > 0): ax.set_yscale('log')

    # --- 1. 指標面板 (Row 0, 1) ---
    panels = [
        ("L1", "L1 Pixel Loss", 0, 0, "teal"), ("MSE", "MSE Noise Loss", 0, 1, "royalblue"), ("VGG", "VGG Perceptual", 0, 2, "darkorange"),
        ("COLOR_S", "Color Stats Loss", 1, 0, "crimson"), ("GRAY_S", "Gray Struct Loss", 1, 1, "slategray"), ("PSNR", "PSNR (dB)", 1, 2, "deeppink"),
    ]
    for key, label, r, c, col in panels:
        ax = fig.add_subplot(gs[r, c])
        s, v = read_scalar(ea, pick_first_existing_tag(tags, TAG_ALIASES[key]))
        plot_sub(ax, s, v, label, col)

    # --- 2. 權重面板 (Row 2, 3) ---
    weight_panels = [
        ("W_RECON", "Weight: Pixel", 2, 0, "teal"), ("W_MSE", "Weight: Noise", 2, 1, "royalblue"), ("W_VGG", "Weight: VGG", 2, 2, "darkorange"),
        ("W_CS", "Weight: Color Stats", 3, 0, "crimson"), ("W_GS", "Weight: Gray Struct", 3, 1, "slategray"), ("LR", "Learning Rate (log)", 3, 2, "olive"),
    ]
    for key, label, r, c, col in weight_panels:
        ax_w = fig.add_subplot(gs[r, c])
        s, v = read_scalar(ea, pick_first_existing_tag(tags, TAG_ALIASES[key]))
        plot_sub(ax_w, s, v, label, col, is_log=(key == "LR"), is_weight=True)

    # --- 3. 分析大圖 (Row 5) - 三軸並存修復版 ---
    ax_main = fig.add_subplot(gs[5, :])
    t_tag, p_tag, s_tag = [pick_first_existing_tag(tags, TAG_ALIASES[k]) for k in ["TOTAL", "PSNR", "SSIM"]]

    if t_tag and p_tag:
        ts, tv = read_scalar(ea, t_tag)
        ps, pv = read_scalar(ea, p_tag)
        
        # 左軸：Total Loss
        ax_main.plot(ts, moving_average(tv), color='blue', alpha=0.8, linewidth=5, label='Total Loss')
        ax_main.set_ylabel("Total Loss", color='blue', fontsize=28, fontweight='bold')
        ax_main.tick_params(axis='y', labelcolor='blue', labelsize=20)

        # 右軸 1：PSNR
        ax_psnr = ax_main.twinx()
        ax_psnr.plot(ps, moving_average(pv), color='deeppink', alpha=0.9, linewidth=6, label='PSNR')
        ax_psnr.set_ylabel("PSNR (dB)", color='deeppink', fontsize=28, fontweight='bold')
        ax_psnr.tick_params(axis='y', labelcolor='deeppink', labelsize=20)

        # 右軸 2：SSIM (修復不見的問題)
        if s_tag:
            ss, sv = read_scalar(ea, s_tag)
            ax_ssim = ax_main.twinx()
            ax_ssim.spines["right"].set_position(("axes", 1.05)) # 向外偏移避免重疊
            ax_ssim.plot(ss, moving_average(sv), color='darkorchid', alpha=0.7, linewidth=4, label='SSIM')
            ax_ssim.set_ylabel("SSIM", color='darkorchid', fontsize=28, fontweight='bold', labelpad=15)
            ax_ssim.tick_params(axis='y', labelcolor='darkorchid', labelsize=20)

        # 預測線
        if len(ps) > 10:
            fit_n = min(20, len(ps))
            slope, intercept, _, _, _ = stats.linregress(ps[-fit_n:], pv[-fit_n:])
            future_s = np.array([ps[-1], ps[-1] + FORECAST_STEPS])
            predict_v = slope * future_s + intercept
            ax_psnr.plot(future_s, predict_v, color='red', linestyle='--', linewidth=6)
            ax_psnr.text(future_s[-1], predict_v[-1], f'  PSNR Pred: {predict_v[-1]:.2f}\n  Slope: {slope:.4f}',
                         color='red', fontsize=24, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8))
            ax_main.set_xlim(min(ts), max(ts) + FORECAST_STEPS + 5)

    ax_main.set_title("Performance Analysis Track & PSNR Forecast (Loss / PSNR / SSIM)", fontsize=36, fontweight='bold', pad=30)
    ax_main.grid(True, alpha=0.3)

    # --- 4. 視覺回饋 (Row 6-9) ---
    img_paths = get_latest_n_sample_images(sample_dir, n=4)
    for i in range(4):
        ax_img = fig.add_subplot(gs[6 + i, :])
        if i < len(img_paths):
            img_data = Image.open(img_paths[i])
            ax_img.imshow(img_data)
            ax_img.set_title(f"Visual Feedback: {os.path.basename(img_paths[i])}", fontsize=35, pad=20, fontweight='bold')
        ax_img.axis('off')

    plt.savefig(save_report_path, bbox_inches='tight', facecolor="white")
    plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] V19 報表已儲存至 -> {save_report_path}")

if __name__ == "__main__":
    log_dir = "runs/stage2_v19_ultimate"  # 請替換為你的 Log 目錄
    sample_dir = "runs/stage2_v19_ultimate"  # 請替換為你的 Sample 圖片目錄 
    while True:
        # try: plot_three_axis("runs/stage2_v18_ultimate", "runs/stage2_v18_ultimate", "v19_report.png")  
        try: plot_three_axis(log_dir, sample_dir, "v19_axis_report.png")
        except Exception as e: print(f"Error: {e}")
        time.sleep(600)
     