from abc import ABC, abstractmethod
import math
import numpy as np
from src.strategy.schedule import ScheduleStrategy
class EdgeDiffusionV19Strategy(ScheduleStrategy):
    """
    優化版策略 V18.2：
    1. 穩定 MSE 權重：不再大幅削減去噪基礎。
    2. 平滑感知過渡：將 VGG 權重上限下調，避免 Total Loss 量級失控。
    3. 色彩強約束：固定 Color Stats 權重在 2.5，防止邊緣生成時色彩漂移。
    4. 延後 GAN 接入：給予模型更多時間適應感知損失。
     pure_epochs=int(cfg["train"]["pure_epochs"]),
            base_lr=float(cfg["train"]["base_lr"]),
            w_adv_final=float(cfg["train"].get("w_adv_final", 0.005)),
            min_steps=int(cfg["train"].get("steps", 30)),
            total_epochs=int(cfg["train"].get("total_epochs", 300))
    """
    def __init__(self, pure_epochs: int, base_lr: float, w_adv_final: float,min_steps: int=30, total_epochs: int = 300):
        self.pure_epochs = pure_epochs
        self.base_lr = base_lr
        self.w_adv_final = w_adv_final * 0.6  # 1. 降低對抗強度上限，減緩顏色衝擊
        self.total_epochs = total_epochs
        self.min_steps = min_steps
        self.max_steps = 150
        self.is_boosted = False

    def get_config(self, epoch: int, metrics_history: dict = None):
        # 計算訓練進度 (0.0 ~ 1.0)
        total_range = max(1, self.total_epochs - self.pure_epochs)
        progress = min(1.0, max(0.0, (epoch - self.pure_epochs) / total_range))
        
        # 使用平滑的 Cosine 因子
        cos_factor = 0.5 * (1 - math.cos(math.pi * progress))

        # --- 第一階段：純重建期 (Warm-up) ---
        if epoch < self.pure_epochs:
            return {
                "w_recon": 15.0, 
                "w_vgg": 0.01, 
                "w_mse": 1.0, 
                "w_adv": 0.0,
                "w_color_stats": 2.5, 
                "w_gray": 0.3, 
                "lr": self.base_lr,
                "steps": self.min_steps, 
                "is_boosted": False
            }

        # --- 第二階段：權重動態演進期 ---

        # 1. 基礎重建與去噪 (MSE 不再降到 0.3，維持在 0.7 確保結構不散)
        w_mse = 1.0 * (1 - cos_factor) + 0.7 * cos_factor 
        w_recon = 15.0 * (1 - cos_factor) + 12.0 * cos_factor

        # 2. 感知損失 (VGG 降至 0.4 以壓制 Total Loss 飆升)
        w_vgg = 0.01 * (1 - cos_factor) + 0.4 * cos_factor 

        # 3. 色彩與灰階穩定 (固定高權重，解決你 Res 圖片偏灰的問題)
        w_color_stats = 2.5 
        w_gray = 0.3 * (1 - cos_factor) + 0.4 * cos_factor

        # 4. 對抗損失 (GAN)：延後到 200 Epoch 才開始極少量介入
        adv_start = 200
        if epoch < adv_start:
            w_adv = 0.0
        else:
            adv_prog = min(1.0, (epoch - adv_start) / max(1, (self.total_epochs - adv_start)))
            w_adv = self.w_adv_final * 0.5 * (1 - math.cos(math.pi * adv_prog))

        # 5. 採樣步數與學習率 (Cosine Annealing)
        step = int(self.min_steps + (self.max_steps - self.min_steps) * progress)
        lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))

        return {
            "w_recon": w_recon, 
            "w_vgg": w_vgg, 
            "w_mse": w_mse, 
            "w_adv": w_adv,
            "w_color_stats": w_color_stats, 
            "w_gray": w_gray,
            "lr": lr, 
            "steps": step, 
            "is_boosted": self.is_boosted
        }