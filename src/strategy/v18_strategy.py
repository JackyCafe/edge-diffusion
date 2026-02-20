from abc import ABC, abstractmethod
import math
import numpy as np
from src.strategy.schedule import ScheduleStrategy

class EdgeDiffusionV18_1Strategy(ScheduleStrategy):
    def __init__(self, pure_epochs: int, base_lr: float, w_adv_final: float, total_epochs: int = 300):
        self.pure_epochs = pure_epochs
        self.base_lr = base_lr
        self.w_adv_final = w_adv_final * 0.6  # 1. 降低對抗強度上限，減緩顏色衝擊
        self.total_epochs = total_epochs
        self.min_steps = 30
        self.max_steps = 150
        self.is_boosted = False

    def get_config(self, epoch: int, metrics_history: dict = None):
        total_range = max(1, self.total_epochs - self.pure_epochs)
        progress = min(1.0, max(0.0, (epoch - self.pure_epochs) / total_range))
        cos_factor = 0.5 * (1 - math.cos(math.pi * progress))

        if epoch < self.pure_epochs:
            return {
                "w_recon": 15.0, "w_vgg": 0.01, "w_mse": 1.0, "w_adv": 0.0,
                "w_color_stats": 2.5, "w_gray": 0.3, "lr": self.base_lr,
                "steps": self.min_steps, "is_boosted": False
            }

        # --- 2. 核心修正：鎖定色彩權重底線 ---
        # 即使在後期，也要維持高水位的色彩約束力（從 2.5 降至 2.0，不再降到 1.5 以下）
        w_color_stats = 2.5 * (1 - cos_factor) + 2.0 * cos_factor

        # --- 3. 穩定感知與結構 ---
        w_vgg = 0.01 * (1 - cos_factor) + 0.8 * cos_factor
        w_gray = 0.3 * (1 - cos_factor) + 0.45 * cos_factor # 微幅調降 gray，減少「灰色補丁」感

        # --- 4. 基礎重建 ---
        w_recon = 15.0 * (1 - cos_factor) + 10.0 * cos_factor
        w_mse = 1.0 * (1 - cos_factor) + 0.3 * cos_factor # 提高 MSE 保底，穩定噪點預測

        # --- 5. 對抗損失平滑介入 ---
        adv_start = 150
        if epoch < adv_start:
            w_adv = 0.0
        else:
            adv_prog = min(1.0, (epoch - adv_start) / max(1, (self.total_epochs - adv_start)))
            w_adv = self.w_adv_final * 0.5 * (1 - math.cos(math.pi * adv_prog))

        step = int(self.min_steps + (self.max_steps - self.min_steps) * progress)
        lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        return {
            "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse, "w_adv": w_adv,
            "w_color_stats": w_color_stats, "w_gray": w_gray,
            "lr": lr, "steps": step, "is_boosted": self.is_boosted
        }