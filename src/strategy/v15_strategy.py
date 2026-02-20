from abc import ABC, abstractmethod
import math
import numpy as np
from src.strategy.schedule import ScheduleStrategy


class EdgeDiffusionV15Strategy(ScheduleStrategy):
    def __init__(self, pure_epochs: int, base_lr: float, w_adv_final: float, min_steps: int):
        self.pure_epochs = pure_epochs
        self.base_lr = base_lr
        self.w_adv_final = w_adv_final
        self.min_steps = min_steps
        self.max_steps = 100
        self.is_boosted = False

    def get_config(self, epoch: int, metrics_history: dict = None):
        # 定義 ratio 確保階段 1 邏輯正確
        ratio = epoch / max(1, self.pure_epochs)

        # --- 自動增步偵測 (Epoch 150+ 觸發) ---
        target_max_steps = self.max_steps
        if metrics_history and epoch >= 150:
            psnr_list = metrics_history.get("psnr_full", [])
            if len(psnr_list) >= 10 and not self.is_boosted:
                recent_avg = np.mean(psnr_list[-5:])
                older_avg = np.mean(psnr_list[-10:-5])
                if (recent_avg - older_avg) < 0.05:
                    self.is_boosted = True

        if self.is_boosted: target_max_steps = 150

        # --- 權重排程 ---
        if epoch < self.pure_epochs:
            # 階段 1: 基礎重建與穩定
            # 修正：w_recon 保持 10.0，避免在 Epoch 50 產生斷層
            w_recon = 10.0
            w_vgg = 0.01  # 預熱 VGG 壓制偽影，不要設為 0
            w_mse = 1.0
            w_adv = 0.0
            w_gray = 0.3
            lr = self.base_lr

            steps_curve = 0.5 * (1 - math.cos(math.pi * ratio))
            curr_steps = int(self.min_steps + (target_max_steps - self.min_steps) * steps_curve)

        elif epoch < 150:
            # 階段 2: 感官強化
            eta = (epoch - self.pure_epochs) / 100
            alpha = 0.5 * (1 - math.cos(math.pi * eta))

            # 修正：銜接點 alpha=0 時 w_recon=10.0，維持與階段 1 一致
            w_mse = 1.0 * (1 - alpha) + 0.1 * alpha
            w_recon = 10.0  # 持續維持高強度像素約束
            w_vgg = 0.05 * (1 - alpha) + 0.4 * alpha
            w_adv = self.w_adv_final * alpha
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * eta))
            w_gray = 0.5
            curr_steps = target_max_steps

        else:
            # 階段 3: 精細微調
            w_recon = 15.0
            w_vgg = 0.5
            w_mse = 0.05
            w_adv = self.w_adv_final
            lr = 2e-6
            w_gray = 0.3
            curr_steps = target_max_steps

        return {
            "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse,
            "w_adv": w_adv, "lr": lr, "steps": curr_steps,
            "w_gray": w_gray, "is_boosted": self.is_boosted
        }