# trainers/stage2_trainer.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.networks import EdgeGenerator, DiffusionUNet, Discriminator, VGGLoss
from src.diffusion import DiffusionManager
from src.dataset import InpaintingDataset
from src.utils import manual_ssim, save_preview_image

from src.metrics.image_metrics import psnr_torch, ssim_torch
from src.monitors.monitor_3axis import Monitor3Axis


# --- Loss Helper（你原本的 그대로） ---
def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum() * pred.shape[1] + eps
    return diff.sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = (pred - target) ** 2 * mask
    denom = mask.sum() * pred.shape[1] + eps
    return diff.sum() / denom


# --- schedules（你原本的 그대로） ---
def get_sa_config(epoch, pure_epochs, base_lr, w_adv_final):
    if epoch < pure_epochs:
        ratio = epoch / pure_epochs
        lr = base_lr
        w_recon = 1.0 + 0.5 * (1 - math.cos(math.pi * ratio))
        w_mse   = 1.0 - 0.2 * (1 - math.cos(math.pi * ratio))
        w_vgg   = 0.01
        w_adv   = 0.0
    elif epoch < 150:
        eta = (epoch - pure_epochs) / 100
        alpha = 0.5 * (1 - math.cos(math.pi * eta))
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * eta))

        w_mse   = 0.8  * (1 - alpha) + 0.2  * alpha
        w_recon = 2.0  * (1 - alpha) + 1.0  * alpha
        w_vgg   = 0.01 * (1 - alpha) + 0.15 * alpha
        w_adv   = 0.0
    else:
        lr = base_lr * 0.1
        w_recon, w_vgg, w_mse = 1.0, 0.2, 0.1
        w_adv = w_adv_final

    return w_recon, w_vgg, w_mse, w_adv, lr


def gray_w_schedule(epoch: int) -> float:
    if epoch < 40:
        return 0.0
    elif epoch < 80:
        return 0.02
    else:
        return 0.05


def gray_consistency_loss(x: torch.Tensor, masks: torch.Tensor, w: float) -> torch.Tensor:
    if w <= 0:
        return torch.tensor(0.0, device=x.device)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    loss = (masked_l1(r, g, masks) + masked_l1(r, b, masks) + masked_l1(g, b, masks)) / 3.0
    return loss * w


class Stage2Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

        self.save_dir = cfg["paths"]["save_dir"]
        self.sample_dir = cfg["paths"]["sample_dir"]
        self.log_dir = cfg["paths"]["log_dir"]
        self.g1_ckpt = cfg["paths"]["g1_checkpoint"]
        self.load_g2_path = cfg["paths"].get("load_g2_path", "")

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.monitor = Monitor3Axis(
            maxlen=3000,
            plot_every=int(cfg["train"].get("triple_axis_plot_every", 50))
        )

        self.best_psnr = 0.0

        # metrics cfg
        self.assume_range = cfg.get("metrics", {}).get("assume_range", "neg1_1")
        self.ssim_mode = cfg.get("metrics", {}).get("ssim_mode", "masked_bbox")
        self.eps = float(cfg.get("metrics", {}).get("eps", 1e-8))

        # build everything
        self._build_data()
        self._build_models()
        self._build_optim()

        self.diffusion = DiffusionManager(device=self.device)
        self.vgg_criterion = VGGLoss(self.device)

        # resume if any
        self.start_epoch = 0
        self._try_resume()

        self.global_step = 0

    def _build_data(self):
        dcfg = self.cfg["data"]
        self.train_loader = DataLoader(
            InpaintingDataset(dcfg["train_root"], mode="train"),
            batch_size=int(dcfg["batch_size"]),
            shuffle=bool(dcfg.get("shuffle", True)),
            num_workers=int(dcfg.get("num_workers", 4)),
            pin_memory=bool(dcfg.get("pin_memory", True)),
        )

    def _build_models(self):
        # G1 fixed
        self.G1 = EdgeGenerator().to(self.device).eval()
        self.G1.load_state_dict(torch.load(self.g1_ckpt, map_location=self.device))

        # G2 + D
        self.G2 = DiffusionUNet(in_channels=8, out_channels=3).to(self.device)
        self.D  = Discriminator().to(self.device)

        self.use_dp = (torch.cuda.device_count() > 1 and self.device.startswith("cuda"))
        if self.use_dp:
            self.G2 = nn.DataParallel(self.G2)
            self.D  = nn.DataParallel(self.D)

    def _build_optim(self):
        tcfg = self.cfg["train"]
        base_lr = float(tcfg["base_lr"])
        wd = float(tcfg.get("weight_decay", 1e-4))
        d_lr = float(tcfg.get("d_lr", 1e-4))

        self.opt_G = optim.AdamW(self.G2.parameters(), lr=base_lr, weight_decay=wd)
        self.opt_D = optim.AdamW(self.D.parameters(),  lr=d_lr,  weight_decay=wd)

        # AMP scaler（你的寫法一致）
        self.scaler_G = torch.amp.GradScaler('cuda') if self.device.startswith("cuda") else None
        self.scaler_D = torch.amp.GradScaler('cuda') if self.device.startswith("cuda") else None

    def _try_resume(self):
        path = self.load_g2_path
        if not path or not os.path.exists(path):
            return

        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
            if "optimizer_state_dict" in ckpt:
                self.opt_G.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and self.scaler_G is not None:
                self.scaler_G.load_state_dict(ckpt["scaler_state_dict"])
            if "best_psnr" in ckpt:
                self.best_psnr = float(ckpt["best_psnr"])
            if "epoch" in ckpt:
                self.start_epoch = int(ckpt["epoch"]) + 1
        else:
            state = ckpt

        try:
            self.G2.load_state_dict(state, strict=True)
        except RuntimeError:
            new_state = {}
            if self.use_dp and not any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state["module." + k] = v
                self.G2.load_state_dict(new_state, strict=False)
            elif (not self.use_dp) and any(k.startswith("module.") for k in state.keys()):
                for k, v in state.items():
                    new_state[k.replace("module.", "", 1)] = v
                self.G2.load_state_dict(new_state, strict=False)
            else:
                raise

        print(f"[*] 成功載入 G2 檢查點，續訓從 Epoch {self.start_epoch} 開始")

    def train(self):
        tcfg = self.cfg["train"]
        epochs = int(tcfg["epochs"])
        pure_epochs = int(tcfg["pure_epochs"])
        accumulation_steps = int(tcfg["accumulation_steps"])
        grad_clip = float(tcfg.get("grad_clip", 0.0))
        w_adv_final = float(tcfg.get("w_adv_final", 0.005))

        log_step_every = int(tcfg.get("log_step_every", 10))
        tb_image_every = int(tcfg.get("tb_image_every", 300))

        for epoch in range(self.start_epoch, epochs):
            # epoch stats
            epoch_l1 = epoch_vgg = epoch_mse = epoch_adv_g = 0.0
            epoch_gray = epoch_color = epoch_total = 0.0
            epoch_psnr_x0 = epoch_psnr_x0_mask = 0.0
            epoch_ssim_x0 = epoch_ssim_x0_mask = 0.0

            w_recon, w_vgg, w_mse, w_adv, curr_lr = get_sa_config(
                epoch, pure_epochs, float(tcfg["base_lr"]), w_adv_final
            )
            for g in self.opt_G.param_groups:
                g["lr"] = curr_lr

            w_gray = gray_w_schedule(epoch)

            self.G2.train()
            self.D.train()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

            self.opt_G.zero_grad(set_to_none=True)
            self.opt_D.zero_grad(set_to_none=True)

            for i, (imgs, _, masks) in enumerate(pbar):
                self.global_step += 1

                imgs = imgs.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                # 1) G1 edge (fixed)
                with torch.no_grad():
                    masked_imgs = imgs * (1 - masks)
                    pred_edges = self.G1(torch.cat([masked_imgs, masks], dim=1))

                condition = torch.cat([masked_imgs, masks, pred_edges], dim=1)

                # 2) Diffusion: noise
                t = self.diffusion.sample_timesteps(imgs.shape[0]).to(self.device)
                x_t, noise = self.diffusion.noise_images(imgs, t)

                with torch.amp.autocast('cuda' if self.device.startswith("cuda") else 'cpu'):
                    pred_noise = self.G2(x_t, t, condition)
                    pred_noise = torch.clamp(torch.nan_to_num(pred_noise), -1.0, 1.0)

                    l_mse = F.mse_loss(pred_noise, noise)

                    alpha_hat_t = self.diffusion.alpha_hat[t][:, None, None, None]
                    denom = torch.sqrt(alpha_hat_t + 1e-7)
                    pred_x0 = (x_t - torch.sqrt(1 - alpha_hat_t) * pred_noise) / denom
                    pred_x0 = torch.clamp(torch.nan_to_num(pred_x0), -1.0, 1.0)

                    l_pixel = masked_l1(pred_x0, imgs, masks)

                    if (epoch >= pure_epochs) and (w_vgg > 0):
                        l_vgg = self.vgg_criterion(pred_x0, imgs)
                    else:
                        l_vgg = torch.tensor(0.0, device=self.device)

                    l_gray = gray_consistency_loss(pred_x0, masks, w=w_gray)
                    l_color = masked_mse(pred_x0, imgs, masks)

                    l_adv_G = torch.tensor(0.0, device=self.device)
                    if w_adv > 0:
                        # D update
                        self.opt_D.zero_grad(set_to_none=True)
                        real_res = self.D(imgs)
                        fake_res = self.D(pred_x0.detach())
                        loss_D = (F.relu(1.0 - real_res).mean() + F.relu(1.0 + fake_res).mean()) * 0.5
                        self.scaler_D.scale(loss_D).backward()
                        self.scaler_D.step(self.opt_D)
                        self.scaler_D.update()

                        l_adv_G = -self.D(pred_x0).mean()

                    loss_total = (
                        l_mse   * w_mse +
                        l_vgg   * w_vgg +
                        l_pixel * w_recon +
                        l_adv_G * w_adv +
                        l_color * 0.1 +
                        l_gray
                    ) / accumulation_steps

                if torch.isnan(loss_total) or torch.isinf(loss_total):
                    print(f"[⚠️] NaN/Inf loss at epoch {epoch}, step {i}, skip.")
                    continue

                # backward G
                self.scaler_G.scale(loss_total).backward()

                # --- metrics (正確版 PSNR/SSIM) ---
                with torch.no_grad():
                    psnr_x0 = psnr_torch(pred_x0, imgs, mask=None, assume_range=self.assume_range, eps=self.eps).item()
                    psnr_x0_mask = psnr_torch(pred_x0, imgs, mask=masks, assume_range=self.assume_range, eps=self.eps).item()
                    ssim_x0 = ssim_torch(pred_x0, imgs, mask=None, mode="full", assume_range=self.assume_range, eps=self.eps).item()
                    ssim_x0_mask = ssim_torch(pred_x0, imgs, mask=masks, mode=self.ssim_mode, assume_range=self.assume_range, eps=self.eps).item()

                # optimizer step (accum)
                if (i + 1) % accumulation_steps == 0:
                    self.scaler_G.unscale_(self.opt_G)
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.G2.parameters(), grad_clip)

                    self.scaler_G.step(self.opt_G)
                    self.scaler_G.update()
                    self.opt_G.zero_grad(set_to_none=True)

                    # epoch stats（你原本只在 step 時累積，維持一致）
                    epoch_l1 += float(l_pixel.item())
                    epoch_mse += float(l_mse.item())
                    epoch_vgg += float(l_vgg.item()) if torch.is_tensor(l_vgg) else float(l_vgg)
                    epoch_adv_g += float(l_adv_G.item())
                    epoch_gray += float(l_gray.item())
                    epoch_color += float(l_color.item())
                    epoch_total += float(loss_total.item()) * accumulation_steps

                    epoch_psnr_x0 += float(psnr_x0)
                    epoch_psnr_x0_mask += float(psnr_x0_mask)
                    epoch_ssim_x0 += float(ssim_x0)
                    epoch_ssim_x0_mask += float(ssim_x0_mask)

                # --- TB step scalars（固定 tag，避免你監控讀不到）---
                if self.global_step % log_step_every == 0:
                    # loss
                    self.writer.add_scalar("loss/total_step", float(loss_total.item()) * accumulation_steps, self.global_step)
                    self.writer.add_scalar("loss/l1_mask_step", float(l_pixel.item()), self.global_step)
                    self.writer.add_scalar("loss/mse_step", float(l_mse.item()), self.global_step)
                    self.writer.add_scalar("loss/vgg_step", float(l_vgg.item()) if torch.is_tensor(l_vgg) else float(l_vgg), self.global_step)
                    self.writer.add_scalar("loss/advG_step", float(l_adv_G.item()), self.global_step)

                    # metrics
                    self.writer.add_scalar("metrics/psnr_x0_step", psnr_x0, self.global_step)
                    self.writer.add_scalar("metrics/psnr_x0_mask_step", psnr_x0_mask, self.global_step)
                    self.writer.add_scalar("metrics/ssim_x0_step", ssim_x0, self.global_step)
                    self.writer.add_scalar("metrics/ssim_x0_mask_step", ssim_x0_mask, self.global_step)

                    # weights + lr
                    self.writer.add_scalar("weights/w_recon", w_recon, self.global_step)
                    self.writer.add_scalar("weights/w_vgg", w_vgg, self.global_step)
                    self.writer.add_scalar("weights/w_mse", w_mse, self.global_step)
                    self.writer.add_scalar("weights/w_adv", w_adv, self.global_step)
                    self.writer.add_scalar("weights/w_gray", w_gray, self.global_step)
                    self.writer.add_scalar("params/lr", curr_lr, self.global_step)

                # --- 三軸監控：永遠不消失（image）---
                self.monitor.update(
                    step=self.global_step,
                    total_loss=float(loss_total.item()) * accumulation_steps,
                    psnr=float(psnr_x0_mask),
                    ssim=float(ssim_x0_mask),
                )
                self.monitor.maybe_log(self.writer, self.global_step)

                # --- Preview images ---
                if self.global_step % tb_image_every == 0:
                    # 這裡用 tensorboard 直接記錄一張（你也保留 save_preview_image 在 epoch end）
                    self.writer.add_image("Preview/pred_x0", (pred_x0[0].detach() + 1) / 2, self.global_step)
                    self.writer.add_image("Preview/gt", (imgs[0].detach() + 1) / 2, self.global_step)
                    self.writer.add_image("Preview/mask", masks[0].detach(), self.global_step)
                    self.writer.add_image("Preview/masked_input", ((imgs[0]*(1-masks[0])).detach() + 1) / 2, self.global_step)

                if i % 10 == 0:
                    pbar.set_postfix({
                        "L1_mask": f"{l_pixel.item():.4f}",
                        "MSE": f"{l_mse.item():.4f}",
                        "PSNRm": f"{psnr_x0_mask:.2f}",
                        "SSIMm": f"{ssim_x0_mask:.4f}",
                        "w_gray": f"{w_gray:.3f}",
                        "LR": f"{curr_lr:.2e}",
                    })

            # ===== epoch end logging =====
            avg_steps = max(1, len(self.train_loader) // accumulation_steps)

            self.writer.add_scalar("Losses/L1_Loss", epoch_l1 / avg_steps, epoch)
            self.writer.add_scalar("Losses/VGG_Loss", epoch_vgg / avg_steps, epoch)
            self.writer.add_scalar("Losses/MSE_Loss", epoch_mse / avg_steps, epoch)
            self.writer.add_scalar("Losses/Adv_Loss", epoch_adv_g / avg_steps, epoch)
            self.writer.add_scalar("Losses/Total_Loss", epoch_total / avg_steps, epoch)

            self.writer.add_scalar("Weights/w_recon_epoch", w_recon, epoch)
            self.writer.add_scalar("Weights/w_vgg_epoch", w_vgg, epoch)
            self.writer.add_scalar("Weights/w_mse_epoch", w_mse, epoch)
            self.writer.add_scalar("Weights/w_adv_epoch", w_adv, epoch)
            self.writer.add_scalar("Weights/w_gray_epoch", w_gray, epoch)
            self.writer.add_scalar("Params/Learning_Rate_epoch", curr_lr, epoch)

            self.writer.add_scalar("Metrics/PSNR_x0", epoch_psnr_x0 / avg_steps, epoch)
            self.writer.add_scalar("Metrics/PSNR_x0_mask", epoch_psnr_x0_mask / avg_steps, epoch)
            self.writer.add_scalar("Metrics/SSIM_x0", epoch_ssim_x0 / avg_steps, epoch)
            self.writer.add_scalar("Metrics/SSIM_x0_mask", epoch_ssim_x0_mask / avg_steps, epoch)

            # ===== Sampling / Preview（你原本的保留） =====
            self.G2.eval()
            with torch.no_grad():
                m_sam = self.G2.module if hasattr(self.G2, "module") else self.G2

                scfg = self.cfg.get("sampling", {})
                steps = int(scfg.get("steps", 100))
                baseline_epoch = int(scfg.get("baseline_epoch", 149))
                baseline_dir = scfg.get("baseline_dir", "baseline_samples")

                if epoch == baseline_epoch:
                    print("[*] 生成 No-GAN Baseline...")
                    os.makedirs(baseline_dir, exist_ok=True)
                    baseline_res = self.diffusion.sample(m_sam, condition[0:4], n=4, steps=steps)
                    save_preview_image(imgs[0:4], masks[0:4], pred_edges[0:4], baseline_res, 0.0, epoch, baseline_dir)

                # 固定 seed
                torch.manual_seed(int(self.cfg.get("seed", 0)))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(self.cfg.get("seed", 0)))

                samples = self.diffusion.sample(m_sam, condition[0:1], n=1, steps=steps)
                samples = torch.clamp(samples, -1.0, 1.0)
                res_img = imgs[0:1] * (1 - masks[0:1]) + samples * masks[0:1]

                # sample 指標：PSNR 用正確版；SSIM 你原本 manual_ssim 也可以留
                v_psnr = psnr_torch(res_img, imgs[0:1], mask=None, assume_range=self.assume_range, eps=self.eps).item()
                v_psnr_mask = psnr_torch(res_img, imgs[0:1], mask=masks[0:1], assume_range=self.assume_range, eps=self.eps).item()
                v_ssim = manual_ssim((res_img + 1) / 2, (imgs[0:1] + 1) / 2).item()

                if v_psnr > self.best_psnr:
                    self.best_psnr = v_psnr
                    torch.save(m_sam.state_dict(), f"{self.save_dir}/G2_BEST_PSNR_ep{epoch}.pth")
                    torch.save(m_sam.state_dict(), f"{self.save_dir}/G2_BEST_PSNR_latest.pth")
                    print(f"\n[🏆] New BEST PSNR_sample! Epoch {epoch} PSNR: {self.best_psnr:.2f}")

                self.writer.add_scalar("Metrics/PSNR_sample", v_psnr, epoch)
                self.writer.add_scalar("Metrics/PSNR_sample_mask", v_psnr_mask, epoch)
                self.writer.add_scalar("Metrics/SSIM_sample", v_ssim, epoch)

                self.writer.add_image("Preview/Epoch_Res", (res_img[0] + 1) / 2, epoch)
                save_preview_image(imgs, masks, pred_edges, res_img, v_psnr, epoch, self.sample_dir)

            # ===== save checkpoint =====
            torch.save({
                "epoch": epoch,
                "model_state_dict": (self.G2.module.state_dict() if hasattr(self.G2, "module") else self.G2.state_dict()),
                "optimizer_state_dict": self.opt_G.state_dict(),
                "scaler_state_dict": (self.scaler_G.state_dict() if self.scaler_G is not None else {}),
                "best_psnr": self.best_psnr,
            }, f"{self.save_dir}/checkpoint_latest.pth")

        self.writer.close()
