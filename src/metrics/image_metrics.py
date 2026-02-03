# metrics/image_metrics.py
import torch
import torch.nn.functional as F


def to_01(x: torch.Tensor, assume_range: str = "auto") -> torch.Tensor:
    if assume_range == "auto":
        if x.min().item() < 0:
            x = (x.clamp(-1, 1) + 1) / 2
        else:
            x = x.clamp(0, 1)
    elif assume_range == "neg1_1":
        x = (x.clamp(-1, 1) + 1) / 2
    elif assume_range == "0_1":
        x = x.clamp(0, 1)
    else:
        raise ValueError(f"Unknown assume_range: {assume_range}")
    return x


@torch.no_grad()
def psnr_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    assume_range: str = "auto",
    data_range: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred01 = to_01(pred, assume_range=assume_range)
    tgt01 = to_01(target, assume_range=assume_range)

    diff2 = (pred01 - tgt01) ** 2

    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()
        mask_c = mask.expand(-1, pred01.size(1), -1, -1)
        num = mask_c.sum(dim=(1, 2, 3)).clamp_min(1.0)
        mse = (diff2 * mask_c).sum(dim=(1, 2, 3)) / num
    else:
        mse = diff2.mean(dim=(1, 2, 3))

    mse = mse.clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr.mean()


def _gaussian_window(window_size: int, sigma: float, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w1d = g.view(1, 1, 1, -1)
    w2d = w1d.transpose(-1, -2) @ w1d
    return w2d


def _ssim_core(x: torch.Tensor, y: torch.Tensor, data_range: float,
               window_size: int, sigma: float, K1: float, K2: float, eps: float) -> torch.Tensor:
    device, dtype = x.device, x.dtype
    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.expand(x.size(1), 1, window_size, window_size)

    mu_x = F.conv2d(x, window, padding=window_size // 2, groups=x.size(1))
    mu_y = F.conv2d(y, window, padding=window_size // 2, groups=y.size(1))

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, window, padding=window_size // 2, groups=x.size(1)) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=window_size // 2, groups=y.size(1)) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=window_size // 2, groups=y.size(1)) - mu_xy

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den.clamp_min(eps)

    return ssim_map.mean(dim=(1, 2, 3)).mean()


@torch.no_grad()
def ssim_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    mode: str = "full",
    assume_range: str = "auto",
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    K1: float = 0.01,
    K2: float = 0.03,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred01 = to_01(pred, assume_range=assume_range)
    tgt01 = to_01(target, assume_range=assume_range)

    B, C, H, W = pred01.shape

    if (mask is not None) and (mode == "masked_bbox"):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = (mask > 0.5).float()

        vals = []
        pad = max(window_size // 2, 3)

        for b in range(B):
            mb = mask[b, 0]
            ys, xs = torch.where(mb > 0.5)
            if ys.numel() == 0:
                vals.append(_ssim_core(pred01[b:b+1], tgt01[b:b+1], data_range,
                                       window_size, sigma, K1, K2, eps))
                continue
            y0, y1 = ys.min().item(), ys.max().item()
            x0, x1 = xs.min().item(), xs.max().item()
            y0 = max(0, y0 - pad); y1 = min(H - 1, y1 + pad)
            x0 = max(0, x0 - pad); x1 = min(W - 1, x1 + pad)

            vals.append(_ssim_core(pred01[b:b+1, :, y0:y1+1, x0:x1+1],
                                   tgt01[b:b+1, :, y0:y1+1, x0:x1+1],
                                   data_range, window_size, sigma, K1, K2, eps))
        return torch.stack(vals).mean()

    return _ssim_core(pred01, tgt01, data_range, window_size, sigma, K1, K2, eps)
