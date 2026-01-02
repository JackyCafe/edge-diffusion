import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import math

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

        # 1. 補上缺失的時間嵌入層 (Time MLP)
        self.time_mlp = nn.Sequential(
            TimeEmbedding(dim=time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2. Encoder (Downsampling 路徑)
        # Input: RGB(3) + Masked_Img(3) + Mask(1) + Edge(1) = 8
        self.inc = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.down1 = DiffBlock(64, 128, self.time_dim, mode="down")  # 512 -> 256
        self.down2 = DiffBlock(128, 256, self.time_dim, mode="down") # 256 -> 128
        self.down3 = DiffBlock(256, 512, self.time_dim, mode="down") # 128 -> 64
        self.down4 = DiffBlock(512, 512, self.time_dim, mode="down") # 64 -> 32

        # 3. Middle (Bottleneck)
        self.mid = DiffBlock(512, 512, self.time_dim, mode=None)     # 保持 32x32

        # 4. Decoder (Upsampling 路徑)
        # 注意: in_ch 必須包含 cat 之後的通道數
        self.up1 = DiffBlock(1024, 512, self.time_dim, mode="up")    # 32 -> 64
        self.up2 = DiffBlock(1024, 256, self.time_dim, mode="up")    # 64 -> 128
        self.up3 = DiffBlock(512, 128, self.time_dim, mode="up")     # 128 -> 256
        self.up4 = DiffBlock(256, 64, self.time_dim, mode="up")      # 256 -> 512

        # 5. Final Output
        # 這裡拼接最初的 inc 特徵 (64 + 64 = 128)
        self.outc = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x, t, condition):
        # x: [B, 3, 512, 512] (含有雜訊的影像)
        # t: [B] (時間步數)
        # condition: [B, 5, 512, 512]

        # A. 先計算時間嵌入向量
        t_emb = self.time_mlp(t)

        # B. 將雜訊影像與條件拼接
        x = torch.cat([x, condition], dim=1) # [B, 8, 512, 512]

        # C. Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1, t_emb)
        x3 = self.down2(x2, t_emb)
        x4 = self.down3(x3, t_emb)
        x5 = self.down4(x4, t_emb)

        # D. Middle
        m = self.mid(x5, t_emb)

        # E. Decoder with Skip Connections (拼接對應層級的特徵)
        x = self.up1(torch.cat([m, x5], dim=1), t_emb)
        x = self.up2(torch.cat([x, x4], dim=1), t_emb)
        x = self.up3(torch.cat([x, x3], dim=1), t_emb)
        x = self.up4(torch.cat([x, x2], dim=1), t_emb)

        # F. Final output
        return self.outc(torch.cat([x, x1], dim=1))