from src.strategy.schedule import ScheduleStrategy
import numpy as np
import math

class EdgeDiffusionV14Strategy(ScheduleStrategy):
    def __init__(self, pure_epochs: int, base_lr: float, w_adv_final: float, min_steps: int):
        self.pure_epochs = pure_epochs
        self.base_lr = base_lr
        self.w_adv_final = w_adv_final
        self.min_steps = min_steps
        self.max_steps = 100
        self.boosted_steps = 150
        self.is_boosted = False

    def get_config(self, epoch: int, metrics_history: dict = None):
        ratio = epoch / max(1, self.pure_epochs)

        # --- 1. 灰階約束排程 (w_gray) ---
        if epoch < 50: w_gray = 0.2
        elif epoch < 100: w_gray = 0.5
        else: w_gray = 0.3

        # --- 2. 自動增步邏輯 (Epoch 150+ 觸發) ---
        current_max_steps = self.max_steps
        if metrics_history and epoch >= 150:
            psnr_list = metrics_history.get("psnr_full", [])
            if len(psnr_list) >= 10 and not self.is_boosted:
                recent_avg = np.mean(psnr_list[-5:])
                older_avg = np.mean(psnr_list[-10:-5])
                # 若 PSNR 增長停滯 (< 0.05dB)，切換至 150 步進行極致拋光
                if (recent_avg - older_avg) < 0.05:
                    self.is_boosted = True

        if self.is_boosted:
            current_max_steps = self.boosted_steps

        # --- 3. 權重排程排程 ---
        if epoch < self.pure_epochs:
            # 階段 1: 基礎重建與 VGG 預熱
            lr_factor = 0.5 * (1 - math.cos(math.pi * ratio))
            lr = self.base_lr * (0.1 + 0.9 * lr_factor)
            w_recon = 1.2 + 0.8 * lr_factor
            w_mse   = 0.8 - 0.4 * lr_factor
            w_vgg   = 0.05 + 0.1 * lr_factor
            w_adv   = 0.0
            steps_curve = 0.5 * (1 - math.cos(math.pi * ratio))
            curr_steps = int(self.min_steps + (current_max_steps - self.min_steps) * steps_curve)

        elif epoch < 150:
            # 階段 2: 感官強化
            eta = (epoch - self.pure_epochs) / (150 - self.pure_epochs + 1e-6)
            alpha = 0.5 * (1 - math.cos(math.pi * eta))
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * eta))
            w_mse   = 0.4 * (1 - alpha) + 0.1 * alpha
            w_recon = 2.0 * (1 - alpha) + 1.0 * alpha
            w_vgg   = 0.15 * (1 - alpha) + 0.8 * alpha
            w_adv   = self.w_adv_final * alpha
            curr_steps = current_max_steps

        else:
            # 階段 3: 精細微調
            lr = self.base_lr * 0.1
            w_recon, w_vgg, w_mse = 1.0, 0.8, 0.05
            w_adv = self.w_adv_final
            curr_steps = current_max_steps

        return {
            "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse,
            "w_adv": w_adv, "lr": lr, "steps": curr_steps,
            "w_gray": w_gray, "is_boosted": self.is_boosted
        }