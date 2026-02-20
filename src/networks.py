import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import math
from torchvision import models




class AttentionBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (channels // heads) ** -0.5
        self.to_qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.to_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        # 轉為 [B, Heads, H*W, Dim_head] 進行矩陣運算
        q, k, v = map(lambda t: t.view(b, self.heads, c // self.heads, h * w).transpose(-1, -2), (q, k, v))

        # 計算全域熱力圖 (Attention Map)
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        # 特徵加權與還原尺寸
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.to_out(out)
# ==========================================
#        共用組件
# ==========================================
class BaseConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=4, stride=2, padding=1, activ='lrelu', norm='instance'):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False)
        self.norm = nn.InstanceNorm2d(out_ch) if norm == 'instance' else nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True) if activ == 'lrelu' else nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            nn.Conv2d(dim, dim, 3, 1, 0, dilation=dilation),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, dilation=1),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

# ==========================================
#  Stage 1: G1 (Edge Generator)
# ==========================================
class EdgeGenerator(nn.Module):
    def __init__(self, input_channels=4, residual_blocks=12): # 增加殘差塊到 12 個
        super().__init__()

        # Encoder: 使用 Spectral Norm 增加穩定性
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=0)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Middle: 增加感受野 (Dilation 混合 2 與 4)
        blocks = []
        for i in range(residual_blocks):
            # 讓一半的層級使用更大的 Dilation (4)，以擴大感受野
            d_rate = 4 if i % 2 == 0 else 2
            blocks.append(ResBlockWithSN(256, dilation=d_rate))
        self.middle = nn.Sequential(*blocks)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2),
            spectral_norm(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        return self.decoder(x)

# 配合 Spectral Norm 的殘差塊實作
class ResBlockWithSN(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(dim, dim, 3, 1, 0, dilation=dilation)),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(dim, dim, 3, 1, 0, dilation=1)),
            nn.InstanceNorm2d(dim)
        )
    def forward(self, x):
        return x + self.block(x)

# ==========================================
#  Stage 1: D1 (Edge Discriminator)
# ==========================================
class EdgeDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super().__init__()
        ndf = 64
        # 將原本的 Sequential 拆解，以便提取中間層
        self.layer1 = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channels, ndf, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer2 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer3 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer4 = nn.Sequential(
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.layer5 = nn.Conv2d(ndf*8, 1, 4, 1, 1)

    def forward(self, x):
        feat = []
        x = self.layer1(x); feat.append(x)
        x = self.layer2(x); feat.append(x)
        x = self.layer3(x); feat.append(x)
        x = self.layer4(x); feat.append(x)
        x = self.layer5(x)
        # 回傳最後結果與中間層特徵列表
        return x, feat


#=============================================
# Time Embedding

#=============================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim//2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # 生成相位 (time * emb):將時間 t 與這些頻率相乘。
        # 這意味著對於同一個 t，低維度索引對應高頻，高維度索引對應低頻。
        emb = time[:, None] * emb[None, :]
        #正餘弦變換 (sin, cos):將結果通過 sin 和 cos 映射。
        # 最終輸出維度為 [batch_size, dim]（即 half_dim * 2）
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb




"""
這是一個典型的 Diffusion Block 實作，它結合了影像特徵提取與時間編碼（Time Embedding）的注入。
這種設計是擴散模型（如 DDPM 或 Stable Diffusion）能夠在不同時間步（$t$）生成不同效果的核心。
"""

import torch
import torch.nn as nn

class DiffBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, mode=None):
        """
        mode:
          - None: 保持尺寸不變 (用於 Middle Block 或 一般層)
          - "down": 下採樣 (尺寸 / 2)
          - "up": 上採樣 (尺寸 * 2)
        """
        super().__init__()

        # 1. 時間嵌入映射
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )

        # 2. 尺寸變換層
        if mode == "down":
            self.resample = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)
        elif mode == "up":
            self.resample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.resample = nn.Identity()

        # 3. 主卷積路徑
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        # 4. 殘差連接 (如果輸入輸出通道不同，需用 1x1 Conv 對齊)
        self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        # A. 先調整空間尺寸 (Up / Down / Keep)
        x = self.resample(x)

        # B. 第一層卷積
        h = self.conv1(x)

        # C. 注入時間編碼 [B, C] -> [B, C, 1, 1]
        time_emb = self.time_mlp(t)[:, :, None, None]
        h = h + time_emb

        # D. 第二層卷積
        h = self.conv2(h)

        # E. 殘差相加
        return h + self.shortcut(x)




#================================================
# DiffusionUNet
#================================================


class DiffusionUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = DiffBlock(64, 128, time_dim, mode="down")
        self.down2 = DiffBlock(128, 256, time_dim, mode="down")
        self.down3 = DiffBlock(256, 512, time_dim, mode="down")
        self.down4 = DiffBlock(512, 512, time_dim, mode="down")

        # Bottleneck (加入 Attention 以優化 FID)
        self.mid = DiffBlock(512, 512, time_dim)
        self.mid_attn = AttentionBlock(512)

        # Decoder
        self.up1 = DiffBlock(1024, 512, time_dim, mode="up")
        self.up2 = DiffBlock(1024, 256, time_dim, mode="up")
        self.up3 = DiffBlock(512, 128, time_dim, mode="up")
        self.up4 = DiffBlock(256, 64, time_dim, mode="up")

        self.outc = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, t, condition):
        t_emb = self.time_mlp(t)
        # --- 優化建議：淨化邊緣圖 (Edge Cleaning) ---
        # 1. 提高門檻值：只保留信心度高的邊緣 (例如 > 0.7)
        # 2. 或是使用高斯模糊來減少細碎的高頻噪點
        # condition = torch.where(condition > 0.7, condition, torch.zeros_like(condition))
        masked_rgb = condition[:, :3]
        mask = condition[:, 3:4]
        edge= condition[:, 4:5]
        # edge = (edge > 0.7).float() * edge  # 只清 edge
        edge = edge * torch.sigmoid(10 * (edge - 0.7))
        cond = torch.cat([masked_rgb, mask, edge], dim=1)
        x = torch.cat([x, cond], dim=1)

        # 可選：如果邊緣還是太碎，加入一個簡單的池化或模糊
        # condition = F.avg_pool2d(condition, kernel_size=3, stride=1, padding=1)
        
        # x = torch.cat([x, condition], dim=1)

        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        m = self.mid(x5, t_emb)
        m = self.mid_attn(m) # 執行全域特徵對齊

        x = self.up1(torch.cat([m, x5], dim=1), t_emb)
        x = self.up2(torch.cat([x, x4], dim=1), t_emb)
        x = self.up3(torch.cat([x, x3], dim=1), t_emb)
        x = self.up4(torch.cat([x, x2], dim=1), t_emb)
        return self.outc(torch.cat([x, x1], dim=1))


class Discriminator(nn.Module):
    """ PatchGAN 判別器：用於 150 Epoch 後的細節強化 """
    def __init__(self):
        super().__init__()
        def conv_block(in_f, out_f, stride=2):
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_f, out_f, 4, stride, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            )
        self.model = nn.Sequential(
            conv_block(3, 64), conv_block(64, 128), conv_block(128, 256),
            conv_block(256, 512, stride=1), nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, x): return self.model(x)



class VGGLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.vgg = vgg.to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        # 關鍵：ImageNet 標準化常數
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
        self.content_layers = [4, 9, 18, 27]

    def forward(self, pred, target):
        p, t = (pred + 1) / 2, (target + 1) / 2
        p = (p - self.mean) / self.std
        t = (t - self.mean) / self.std
        loss = 0
        max_i = max(self.content_layers)

        for i, layer in enumerate(self.vgg):
            p, t = layer(p), layer(t)
            if i in self.content_layers:
                loss += nn.functional.l1_loss(p, t)
            if i >= max_i:  # 提前結束，節省計算
                break
        return loss

   