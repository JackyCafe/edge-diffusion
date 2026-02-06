import os
import time
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorboard.backend.event_processing import event_accumulator
from scipy import stats  # 用於回歸預測
from PIL import Image

# ================= 設定區 =================
LOG_DIR = "runs/stage2_v14_ultimate"
SAMPLE_DIR = "runs/stage2_v14_ultimate"
SAVE_REPORT_PATH = "training_reports/v14_2_three_axis_report.png"

# 統計與預測設定
SLEEP_SECONDS = 300 
FORECAST_STEPS = 5  # 預測未來幾個 Epoch
# =========================================

TAG_ALIASES = {
    "L1":    ["Loss_Detail/Pixel_L1"],
    "MSE":   ["Loss_Detail/MSE_Noise"],
    "VGG":   ["Loss_Detail/VGG"],
    "ADV":   ["Losses/Adv_Loss_G"],
    "TOTAL": ["Losses/Total_Loss"],
    "PSNR":  ["Metrics/PSNR"],
    "SSIM":  ["Metrics/SSIM"],
    "LR":    ["Params/Learning_Rate"],
    "W_RECON": ["Weights/w_recon"],
    "W_VGG":   ["Weights/w_vgg"],
    "W_MSE":   ["Weights/w_mse"],
    "W_ADV":   ["Weights/w_adv"],
    "STEPS":   ["Weights/init_step"],
}

def pick_first_existing_tag(tags_available, candidates):
    for t in candidates:
        if t in tags_available: return t
    for t in tags_available:
        for cand in candidates:
            if t.lower() == cand.lower(): return t
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

def plot_v14_2_three_axis():
    event_file = get_latest_event_file(LOG_DIR)
    if not event_file:
        print(f"[{time.strftime('%H:%M:%S')}] 等待 Log 檔案...")
        return

    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = set(ea.Tags().get('scalars', []))

    plt.style.use('default')
    fig = plt.figure(figsize=(55, 110), dpi=100) 
    
    gs = fig.add_gridspec(9, 4, height_ratios=[1, 1, 1, 2.5, 0.2, 3.0, 3.0, 3.0, 3.0], hspace=0.5)
    
    tag_total_ref = pick_first_existing_tag(tags, TAG_ALIASES["TOTAL"])
    latest_step = "N/A"
    if tag_total_ref:
        s_tmp, _ = read_scalar(ea, tag_total_ref)
        if len(s_tmp) > 0: latest_step = s_tmp[-1]

    plt.suptitle(f"Edge-Diffusion stage2 Multi-Visual Monitor\nUpdate: {time.strftime('%Y-%m-%d %H:%M:%S')} | Latest Step: {latest_step}",
                 fontsize=55, fontweight='bold', color='#333333', y=0.985)

    def plot_sub(ax, s, v, label, col, is_log=False, is_weight=False):
        if len(v) > 0:
            # 原始數據線
            ax.plot(s, v, alpha=0.2, color=col, linewidth=2.5)
            
            y_min, y_max = np.min(v), np.max(v)
            margin = max((y_max - y_min) * 0.1, 1e-4)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_xlim(min(s), max(s))
        else:
            ax.text(0.5, 0.5, "TAG NOT FOUND", ha='center', va='center', color='red', fontsize=20, transform=ax.transAxes)
        
        title_fs = 22 if is_weight else 28
        ax.set_title(label, fontsize=title_fs, color="#555555", fontweight='bold', pad=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        if is_log and len(v) > 0 and np.all(v > 0): ax.set_yscale('log')

    # --- 1. 指標繪製 (Row 0, 1) ---
    panels = [("L1", "L1 Pixel Loss", 0, 0, "teal"), ("MSE", "MSE Noise Loss", 0, 1, "royalblue"),
              ("VGG", "VGG Perceptual", 0, 2, "darkorange"), ("ADV", "Adv Loss (G)", 0, 3, "forestgreen"),
              ("TOTAL", "Total Loss", 1, 0, "blue"), ("SSIM", "SSIM Score", 1, 1, "darkorchid"),
              ("PSNR", "PSNR (dB)", 1, 2, "deeppink"),
                ("LR", "Learning Rate", 1, 3, "olive")]
    for key, label, r, c, col in panels:
        ax = fig.add_subplot(gs[r, c])
        tag = pick_first_existing_tag(tags, TAG_ALIASES[key])
        s, v = read_scalar(ea, tag) if tag else (np.array([]), np.array([]))
        plot_sub(ax, s, v, label, col, is_log=(key == "LR"))

    # --- 2. 權重繪製 (Row 2) ---
    weight_panels = [("W_RECON", "Weight: Pixel", 0, "teal"), ("W_VGG", "Weight: VGG", 1, "darkorange"),
                     ("W_MSE", "Weight: Noise", 2, "royalblue"),("STEPS", "Sampling Steps", 3, "deeppink"), 
                    #  ("W_ADV", "Weight: Adv", 4, "crimson")
                     ]
    for key, label, c, col in weight_panels:
        ax_w = fig.add_subplot(gs[2, c])
        tag = pick_first_existing_tag(tags, TAG_ALIASES.get(key, []))
        s, v = read_scalar(ea, tag) if tag else (np.array([]), np.array([]))
        plot_sub(ax_w, s, v, label, col, is_weight=True)

    # --- 3. 分析大圖：整合 PSNR 預測 (Row 3) ---
    ax_main = fig.add_subplot(gs[3, :])
    t_tag, p_tag, s_tag = [pick_first_existing_tag(tags, TAG_ALIASES[k]) for k in ["TOTAL", "PSNR", "SSIM"]]
    
    if t_tag and p_tag:
        ts, tv = read_scalar(ea, t_tag)
        ps, pv = read_scalar(ea, p_tag)
        
        # Total Loss (左軸 - 藍色)
        ax_main.plot(ts, tv, color='blue', alpha=0.4, linewidth=4, label='Total Loss')
        ax_main.set_ylabel("Total Loss", color='blue', fontsize=26, fontweight='bold')
        
        # PSNR (右軸 - 粉紅)
        ax_psnr = ax_main.twinx()
        ax_psnr.plot(ps, pv, color='deeppink', alpha=0.4, linewidth=4, label='PSNR (dB)')
        ax_psnr.set_ylabel("PSNR (dB)", color='deeppink', fontsize=26, fontweight='bold')
        
        # --- PSNR 趨勢預測 ---
        if len(ps) > 5:
            fit_n = min(10, len(ps))
            slope, intercept, r_val, p_val, std_err = stats.linregress(ps[-fit_n:], pv[-fit_n:])
            
            future_s = np.array([ps[-1], ps[-1] + FORECAST_STEPS])
            predict_v = slope * future_s + intercept
            
            # 畫出預測趨勢虛線
            ax_psnr.plot(future_s, predict_v, color='red', linestyle='--', linewidth=5, label='PSNR Forecast')
            
            # 標註預測值與斜率
            pred_final = predict_v[-1]
            ax_psnr.text(future_s[-1], pred_final, f'  PSNR Pred: {pred_final:.2f}\n  Slope: {slope:.4f}', 
                         color='red', fontsize=22, fontweight='bold', verticalalignment='center',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # 為了標註文字，X 軸稍微向右預留
            ax_main.set_xlim(min(ts), max(ts) + FORECAST_STEPS + 2)

        if s_tag:
            ss, sv = read_scalar(ea, s_tag)
            ax_ssim = ax_main.twinx()
            ax_ssim.spines["right"].set_position(("axes", 1.08))
            ax_ssim.plot(ss, sv, color='darkorchid', alpha=0.4, linewidth=4, label='SSIM')
            ax_ssim.set_ylabel("SSIM", color='darkorchid', fontsize=26, fontweight='bold')

    ax_main.set_title("Performance Analysis Track & PSNR Forecast", fontsize=36, fontweight='bold', pad=20)
    ax_main.grid(True, alpha=0.3)

    # --- 4. 視覺回饋 (Row 5-8) ---
    img_paths = get_latest_n_sample_images(SAMPLE_DIR, n=4)
    for i in range(4):
        ax_img = fig.add_subplot(gs[5 + i, :])
        if i < len(img_paths):
            img_data = Image.open(img_paths[i])
            ax_img.imshow(img_data)
            ax_img.set_title(f"Visual Feedback: {os.path.basename(img_paths[i])}", 
                             fontsize=35, color="#333333", fontweight='bold', pad=20)
        ax_img.axis('off')

    plt.savefig(SAVE_REPORT_PATH, bbox_inches='tight', facecolor="white")
    plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] 報表更新成功 -> {SAVE_REPORT_PATH}")

def render_dashboard_png_epoch_only():
    while True:
        try:
            plot_v14_2_three_axis()
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    render_dashboard_png_epoch_only()