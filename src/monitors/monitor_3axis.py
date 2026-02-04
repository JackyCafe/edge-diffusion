# monitors/monitor_3axis.py
from collections import deque
import io
import numpy as np
import matplotlib.pyplot as plt
import torch


class Monitor3Axis:
    def __init__(self, maxlen: int = 3000, plot_every: int = 50, debug: bool = False):
        self.maxlen = int(maxlen)
        self.plot_every = max(1, int(plot_every))
        self.debug = debug

        self.steps = deque(maxlen=self.maxlen)
        self.loss  = deque(maxlen=self.maxlen)
        self.psnr  = deque(maxlen=self.maxlen)
        self.ssim  = deque(maxlen=self.maxlen)

        # 防止 plot_every=1 時每 step 都畫造成 I/O 爆
        self._last_plotted_step = -1

    @staticmethod
    def _safe_float(x, default=np.nan):
        try:
            v = float(x)
            return v if np.isfinite(v) else default
        except Exception:
            return default

    def update(self, step: int, total_loss: float, psnr: float, ssim: float):
        s = int(step)
        self.steps.append(s)
        self.loss.append(self._safe_float(total_loss))
        self.psnr.append(self._safe_float(psnr))
        self.ssim.append(self._safe_float(ssim))

    def maybe_log(self, writer, global_step: int):
        gs = int(global_step)

        # ✅ 也寫一份 scalar（保證你一定看得到）
        if len(self.steps) > 0:
            writer.add_scalar("monitor/total_loss", float(self.loss[-1]) if np.isfinite(self.loss[-1]) else 0.0, gs)
            writer.add_scalar("monitor/psnr", float(self.psnr[-1]) if np.isfinite(self.psnr[-1]) else 0.0, gs)
            writer.add_scalar("monitor/ssim", float(self.ssim[-1]) if np.isfinite(self.ssim[-1]) else 0.0, gs)

        # ✅ 畫圖頻率控制
        if gs % self.plot_every != 0:
            return
        if len(self.steps) < 5:
            return
        if gs == self._last_plotted_step:
            return
        self._last_plotted_step = gs

        fig = self._plot()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        import PIL.Image
        img = PIL.Image.open(buf).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # HWC 0-1
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

        writer.add_image("monitor/triple_axis", tensor, gs)
        writer.flush()

        if self.debug:
            print(f"[Monitor3Axis] wrote image at step={gs}, points={len(self.steps)}")

    def _plot(self):
        steps = np.asarray(self.steps)
        loss = np.asarray(self.loss, dtype=np.float32)
        psnr = np.asarray(self.psnr, dtype=np.float32)
        ssim = np.asarray(self.ssim, dtype=np.float32)

        def ffill(a):
            out = a.copy()
            last = np.nan
            for i in range(len(out)):
                if np.isfinite(out[i]):
                    last = out[i]
                else:
                    out[i] = last
            return out

        loss_p = ffill(loss)
        psnr_p = ffill(psnr)
        ssim_p = ffill(ssim)

        fig = plt.figure(figsize=(10, 4.5))
        ax1 = fig.add_subplot(111)
        ax1.set_title("Triple-Axis Monitor: Total Loss + PSNR + SSIM")

        ax1.plot(steps, loss_p, label="loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("total loss")

        ax2 = ax1.twinx()
        ax2.plot(steps, psnr_p, alpha=0.8, label="psnr")
        ax2.set_ylabel("psnr (dB)")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(steps, ssim_p, alpha=0.8, label="ssim")
        ax3.set_ylabel("ssim")

        ax1.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig
