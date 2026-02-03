import os
import time
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
# =========================================

def get_latest_event_file(log_dir):
    if not os.path.exists(log_dir): return None
    all_events = [os.path.join(r, f) for r, _, fs in os.walk(log_dir) for f in fs if "events.out.tfevents" in f]
    return sorted(all_events, key=os.path.getmtime, reverse=True)[0] if all_events else None

def get_latest_sample_image(sample_dir):
    if not os.path.exists(sample_dir): return None
    files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    return os.path.join(sample_dir, sorted(files)[-1]) if files else None

def plot_v14_2_three_axis():
    event_file = get_latest_event_file(LOG_DIR)
    if not event_file: return

    ea = event_accumulator.EventAccumulator(event_file, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    tags = ea.Tags()['scalars']

    # 檢查必要標籤是否齊全
    required_tags = ['Losses/Total_Loss', 'Metrics/PSNR', 'Metrics/SSIM']
    if not all(t in tags for t in required_tags):
        print(f"[{time.strftime('%H:%M:%S')}] 等待標籤寫入中...")
        return

    # 設置畫布 (9列佈局)
    fig = plt.figure(figsize=(50, 80))
    gs = fig.add_gridspec(10, 4, height_ratios=[1, 1, 1, 1.2, 2.5, 1.5, 2.5, 0.4, 0.1, 0.1])
    plt.suptitle(f"v14.2 Three-Axis Strategy Monitor\nUpdate: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 fontsize=40, fontweight='bold', y=0.98)

    # --- 1. 基礎指標區 (Row 0-1) ---
    metrics = [
        ("Losses/L1_Loss", "L1 Loss", "tab:cyan", 0, 0), ("Losses/MSE_Loss", "MSE Loss", "tab:blue", 0, 1),
        ("Losses/VGG_Loss", "VGG Loss", "tab:orange", 0, 2), ("Losses/Adv_Loss", "Adv Loss", "tab:green", 0, 3),
        ("Losses/Total_Loss", "Total Loss", "tab:red", 1, 0), ("Metrics/SSIM", "SSIM", "tab:purple", 1, 1),
        ("Metrics/PSNR", "PSNR", "tab:red", 1, 2), ("Params/Learning_Rate", "Learning Rate", "darkred", 1, 3)
    ]
    for tag, label, color, row, col in metrics:
        if tag in tags:
            ev = ea.Scalars(tag)
            s, v = [e.step for e in ev], [e.value for e in ev]
            ax = fig.add_subplot(gs[row, col])
            ax.plot(s, v, alpha=0.2, color=color)
            if len(v) >= WINDOW_SIZE:
                ma = np.convolve(v, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
                ax.plot(s[WINDOW_SIZE-1:], ma, color=color, linewidth=4)
            if "Learning Rate" in label: ax.set_yscale('log')
            ax.set_title(label, fontsize=22, fontweight='bold'); ax.grid(True, alpha=0.2)

    # --- 2. 獨立權重區 (Row 2) ---
    w_map = [("Weights/w_recon", "Pixel Weight", "tab:blue"), ("Weights/w_vgg", "VGG Weight", "tab:orange"),
             ("Weights/w_mse", "Noise Weight", "tab:green"), ("Weights/w_adv", "Adv Weight", "tab:red")]
    for i, (tag, label, color) in enumerate(w_map):
        ax_w = fig.add_subplot(gs[2, i])
        if tag in tags:
            ev = ea.Scalars(tag)
            ax_w.plot([e.step for e in ev], [e.value for e in ev], color=color, linewidth=5)
            ax_w.fill_between([e.step for e in ev], [e.value for e in ev], color=color, alpha=0.1)
        ax_w.set_title(label, fontsize=18, fontweight='bold'); ax_w.grid(True, alpha=0.3)

    # --- 🚀 3. 三軸對比 Overfit Monitor (Row 4: Triple Axis) ---
    ax_main = fig.add_subplot(gs[4, :])
    l_ev, p_ev, s_ev = ea.Scalars('Losses/Total_Loss'), ea.Scalars('Metrics/PSNR'), ea.Scalars('Metrics/SSIM')
    ls, lv = np.array([e.step for e in l_ev]), np.array([e.value for e in l_ev])
    ps, pv = np.array([e.step for e in p_ev]), np.array([e.value for e in p_ev])
    ss, sv = np.array([e.step for e in s_ev]), np.array([e.value for e in s_ev])

    # 軸1: Total Loss (左側)
    ax_main.plot(ls, lv, alpha=0.1, color='tab:blue')
    l_ma = np.convolve(lv, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
    ax_main.plot(ls[WINDOW_SIZE-1:], l_ma, color='tab:blue', linewidth=6, label='Loss')
    ax_main.set_ylabel("Total Loss", fontsize=22, color='tab:blue', fontweight='bold')
    ax_main.tick_params(axis='y', labelcolor='tab:blue')

    # 軸2: PSNR (右側 1)
    ax_psnr = ax_main.twinx()
    ax_psnr.plot(ps, pv, alpha=0.1, color='tab:red')
    p_ma = np.convolve(pv, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
    ax_psnr.plot(ps[WINDOW_SIZE-1:], p_ma, color='tab:red', linewidth=6, label='PSNR')
    ax_psnr.set_ylabel("PSNR (dB)", fontsize=22, color='tab:red', fontweight='bold')
    ax_psnr.tick_params(axis='y', labelcolor='tab:red')

    # 軸3: SSIM (右側 2, 偏移位置)
    ax_ssim = ax_main.twinx()
    ax_ssim.spines["right"].set_position(("axes", 1.08))
    ax_ssim.plot(ss, sv, alpha=0.1, color='tab:purple')
    s_ma = np.convolve(sv, np.ones(WINDOW_SIZE)/WINDOW_SIZE, mode='valid')
    ax_ssim.plot(ss[WINDOW_SIZE-1:], s_ma, color='tab:purple', linewidth=6, label='SSIM')
    ax_ssim.set_ylabel("SSIM", fontsize=22, color='tab:purple', fontweight='bold')
    ax_ssim.tick_params(axis='y', labelcolor='tab:purple')
    ax_ssim.set_ylim(min(sv)*0.98, 1.0)

    ax_main.set_title("Triple-Axis Analysis: Loss (Blue) vs PSNR (Red) vs SSIM (Purple)", fontsize=28, fontweight='bold')
    ax_main.grid(True, alpha=0.2)

    # --- 4. Milestone Prediction (Row 5) ---
    ax_pred = fig.add_subplot(gs[5, :])
    if 'Metrics/PSNR' in tags:
        ax_pred.plot(ps, pv, alpha=0.3, color='tab:red')
        if len(pv) > 10:
            slope, intercept, _, _, _ = stats.linregress(ps[-20:], pv[-20:])
            future = slope * np.arange(ps[-1], TOTAL_EPOCHS + 1) + intercept
            ax_pred.plot(np.arange(ps[-1], TOTAL_EPOCHS+1), future, '--', color='darkred', linewidth=3)
            ax_pred.scatter([TOTAL_EPOCHS], [future[-1]], color='gold', s=500, marker='*', label=f"Target Ep200: {future[-1]:.2f}dB")
        ax_pred.legend(fontsize=20); ax_pred.set_title("Future PSNR Projection", fontsize=24, fontweight='bold')

    # --- 5. 視覺回饋 (Row 6) ---
    sample_img = get_latest_sample_image(SAMPLE_DIR)
    if sample_img:
        ax_img = fig.add_subplot(gs[6, :])
        ax_img.imshow(mpimg.imread(sample_img))
        ax_img.set_title(f"LATEST VISUAL FEEDBACK (Epoch {int(ps[-1])})", fontsize=30, fontweight='bold')
        ax_img.axis('off')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(SAVE_REPORT_PATH, dpi=100); plt.close()
    print(f"[{time.strftime('%H:%M:%S')}] 報表 v14.2 更新成功")

if __name__ == "__main__":
    while True:
        try: plot_v14_2_three_axis()
        except Exception as e: print(f"Error: {e}")
        time.sleep(600)