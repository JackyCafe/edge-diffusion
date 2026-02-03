# monitors/monitor_3axis.py
from collections import deque
import io
import numpy as np
import matplotlib.pyplot as plt
import torch


class Monitor3Axis:
    def __init__(self, maxlen: int = 3000, plot_every: int = 50):
        self.maxlen = maxlen
        self.plot_every = plot_every
        self.steps = deque(maxlen=maxlen)
        self.loss = deque(maxlen=maxlen)
        self.psnr = deque(maxlen=maxlen)
        self.ssim = deque(maxlen=maxlen)

    @staticmethod
    def _safe_float(x, default=np.nan):
        try:
            v = float(x)
            if np.isfinite(v):
                return v
            return default
        except Exception:
            return default

    def update(self, step: int, total_loss: float, psnr: float, ssim: float):
        self.steps.append(int(step))
        self.loss.append(self._safe_float(total_loss))
        self.psnr.append(self._safe_float(psnr))
        self.ssim.append(self._safe_float(ssim))

    def maybe_log(self, writer, global_step: int):
        if global_step % self.plot_every != 0:
            return
        if len(self.steps) < 5:
            return

        fig = self._plot()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        import PIL.Image
        img = PIL.Image.open(buf).convert("RGB")
        arr = np.array(img)  # HWC
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # CHW

        writer.add_image("monitor/triple_axis", tensor, global_step)

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

        ax1.plot(steps, loss_p)
        ax1.set_xlabel("step")
        ax1.set_ylabel("total loss")

        ax2 = ax1.twinx()
        ax2.plot(steps, psnr_p)
        ax2.set_ylabel("psnr (dB)")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(steps, ssim_p)
        ax3.set_ylabel("ssim")

        ax1.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        return fig
