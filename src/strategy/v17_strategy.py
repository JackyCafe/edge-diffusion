from abc import ABC, abstractmethod
import math
import numpy as np
from src.strategy.schedule import ScheduleStrategy


class EdgeDiffusionV17Strategy(ScheduleStrategy):
    def __init__(self, pure_epochs: int, base_lr: float, w_adv_final: float, min_steps: int,total_epochs: int=200):
        self.pure_epochs = pure_epochs
        self.base_lr = base_lr
        self.w_adv_final = w_adv_final
        self.min_steps = min_steps
        self.max_steps = 100
        self.is_boosted = False
        self.total_epochs = total_epochs  # 假設總訓練周期為 200 epochs

    def get_config(self, epoch: int, metrics_history: dict = None):
        # 1. 自動增步偵測 (保持你的邏輯)
        target_max_steps = self.max_steps
        if metrics_history and epoch >= 150:
            psnr_list = metrics_history.get("psnr_full", [])
            if len(psnr_list) >= 10 and not self.is_boosted:
                recent_avg = np.mean(psnr_list[-5:])
                older_avg = np.mean(psnr_list[-10:-5])
                if (recent_avg - older_avg) < 0.05:
                    self.is_boosted = True
        
        if self.is_boosted: target_max_steps = 150
        
        # 2. 階段判定與進度計算
        adv_start_epoch = 150
        ratio = epoch / max(1, self.pure_epochs)
        if epoch < self.pure_epochs:
            # Stage 1: 基礎重建期 (Warm-up)
            w_recon = 15.0
            w_vgg = 0.5
            w_mse = 1.0
            w_adv = 0.0
            
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * ratio))
            steps_curve = 0.5 * (1 - math.cos(math.pi * ratio))
            steps = int(self.min_steps + (target_max_steps - self.min_steps) * steps_curve)
            w_color_stats = 2.0   # 強力鎖定均值，防止初期發灰
            w_gray_struct = 0.5   # 輕微約束結構
            
        else:
            # Stage 2 & 3: 平滑轉移期
            # 使用正確的總進度分母
            total_prog_range = self.total_epochs - self.pure_epochs
            progress = min(1.0, (epoch - self.pure_epochs) / max(1, total_prog_range))
            
            cos_decay = 0.5 * (1 - math.cos(math.pi * progress))
            
            # 權重計算
            w_mse   = 1.0 * (1 - cos_decay) + 0.05 * cos_decay
            w_recon = 15.0 * (1 - cos_decay) + 10.0 * cos_decay
            w_vgg   = 0.01 * (1 - cos_decay) + 0.5 * cos_decay
            
            # 推理步數隨進度線性增加
            steps = int(self.min_steps + (target_max_steps - self.min_steps) * progress)
            
            # LR 餘弦退火
            lr = self.base_lr * 0.5 * (1 + math.cos(math.pi * progress))
            
            # 修正後的 Adv 介入邏輯 (從 150 epoch 到總量結束平滑上升)
            w_adv = 0.0
            w_color_stats = 2.0 * (1 - cos_decay) + 1.0 * cos_decay
            w_gray_struct = 0.5 * (1 - cos_decay) + 3.0 * cos_decay
            if epoch >= adv_start_epoch:
                adv_prog_range = self.total_epochs - adv_start_epoch
                adv_progress = min(1.0, (epoch - adv_start_epoch) / max(1, adv_prog_range))
                w_adv = self.w_adv_final * (0.5 * (1 - math.cos(math.pi * adv_progress)))

        return {
            "w_recon": w_recon, "w_vgg": w_vgg, "w_mse": w_mse,
            "w_adv": w_adv, "lr": lr, "steps": steps,
            "w_gray": 0.3, "is_boosted": self.is_boosted,
              "w_color_stats": w_color_stats, "w_gray_struct": w_gray_struct
        }