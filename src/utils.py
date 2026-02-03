from curses import raw
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def compute_psnr(img1, img2):
    """
    計算 Tensor PSNR
    img1, img2: [B, C, H, W]
    支援範圍 [-1, 1] (RGB) 或 [0, 1] (Edge)
    """
    with torch.no_grad():
        # 如果是 RGB [-1, 1]，轉回 [0, 1]
        if img1.min() < 0: img1 = (img1 + 1) / 2.0
        if img2.min() < 0: img2 = (img2 + 1) / 2.0

        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)

        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100.0

        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()




"""

第一階段的視覺化

"""
def save_sample_images(imgs, masks, edges, fake_edges, save_path):
    # Move to CPU and detach
    imgs = imgs.detach().cpu()
    masks = masks.detach().cpu()
    edges = edges.detach().cpu()
    fake_edges = fake_edges.detach().cpu()

    # Calculate Grayscale
    imgs_gray = 0.299 * imgs[:, 0] + 0.587 * imgs[:, 1] + 0.114 * imgs[:, 2]
    imgs_gray = imgs_gray.unsqueeze(1)

    masked_imgs = imgs * (1 - masks)

    # Select the first image in the batch (index 0)
    k = 0

    # Squeeze to remove batch dimension if necessary, but keep spatial dims
    img_g = imgs_gray[k].squeeze()       # Shape: (H, W)
    img_m = masks[k].squeeze()           # Shape: (H, W)
    img_edge_gt = edges[k].squeeze()     # Shape: (H, W)
    img_edge_fake = fake_edges[k].squeeze() # Shape: (H, W)

    # Fix the RGB image: Permute from (C, H, W) to (H, W, C)
    img_masked = masked_imgs[k].permute(1, 2, 0) # Shape: (H, W, 3)

    # Normalize [-1, 1] -> [0, 1] for visualization
    img_g = (img_g + 1) / 2
    img_masked = (img_masked + 1) / 2

    plt.figure(figsize=(25, 4))
    titles = ["Original Gray", "Mask", "Masked Input", "GT Edge", "Generated Edge"]
    images = [img_g, img_m, img_masked, img_edge_gt, img_edge_fake]

    for idx, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 5, idx+1)
        plt.title(title)

        # Ensure image is numpy for matplotlib
        display_img = img.numpy()

        if title == "Masked Input":
            # Matplotlib handles (H, W, 3) automatically
            plt.imshow(display_img)
        else:
            # For grayscale, ensure shape is (H, W)
            plt.imshow(display_img, cmap='gray', vmin=0, vmax=1)

        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()




"""
第二階段的視覺化

"""

def save_diffusion_samples(v_imgs, v_masks, v_edges, v_final, save_path):
    """
    v_imgs: 原始圖 [B, 3, 512, 512] (-1 to 1)
    v_masks: 遮罩 [B, 1, 512, 512] (0 or 1)
    v_edges: G1 生成的邊緣 [B, 1, 512, 512] (0 to 1)
    v_final: Diffusion 生成結果 [B, 3, 512, 512] (-1 to 1)
    """
    # 取第一張圖並轉為 numpy
    idx = 0
    img_orig = (v_imgs[idx].cpu().permute(1, 2, 0).numpy() + 1) / 2
    img_mask = v_masks[idx].cpu().squeeze().numpy()
    img_edge = v_edges[idx].cpu().squeeze().numpy()
    img_out = (v_final[idx].detach().cpu().permute(1, 2, 0).numpy() + 1) / 2

    # 製作 Masked Input 用於視覺化
    img_masked = img_orig * (1 - img_mask[:, :, None])

    plt.figure(figsize=(20, 5))
    titles = ['Original', 'Masked Input', 'G1 Edge Guide', 'Diffusion Result']
    images = [img_orig, img_masked, img_edge, img_out]

    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, 4, i + 1)
        plt.title(title)
        if title == 'G1 Edge Guide':
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample saved: {save_path}")




def save_preview_image(imgs, masks, edges, results, psnr_val, epoch, save_dir):
    imgs_np = (imgs.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
    masks_np = masks.cpu().permute(0, 2, 3, 1).numpy()
    edges_np = edges.cpu().permute(0, 2, 3, 1).numpy()
    res_np = (results.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2
    fig, axes = plt.subplots(1, 5, figsize=(30, 5))
    titles = ["Original", "Mask", "Edge Guide", "Masked Input", f"Res (PSNR:{psnr_val:.2f})"]
    data = [imgs_np[0], masks_np[0,:,:,0], edges_np[0,:,:,0], imgs_np[0]*(1-masks_np[0]), res_np[0]]
    for ax, d, t in zip(axes, data, titles):
        ax.imshow(np.clip(d, 0, 1), cmap='gray' if len(d.shape)==2 else None)
        ax.set_title(t, fontsize=10); ax.axis('off')
    plt.tight_layout(); plt.savefig(f"{save_dir}/v13_ep{epoch:03d}.png"); plt.close()

def manual_ssim(img1, img2, window_size=11):
    """ 手動實現 SSIM 避免依賴問題 """
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - (mu1 * mu2)
    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1**2
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2**2
    return (((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))).mean()
