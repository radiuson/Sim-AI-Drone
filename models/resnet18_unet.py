# ──────────────────────────────────────────────────────────────────────────────
# FPV Diffusion Policy 模型定义（独立文件）
# ──────────────────────────────────────────────────────────────────────────────
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ──────────────────────────────────────────────────────────────────────────────
# 时间嵌入模块
# ──────────────────────────────────────────────────────────────────────────────
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.fc(emb)

# ──────────────────────────────────────────────────────────────────────────────
# 1D 残差块，用于时间序列动作扩散
# ──────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, ch: int, cond_dim: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.time = nn.Linear(time_dim, ch)
        self.cond = nn.Linear(cond_dim, ch)

    def forward(self, x, cond, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.time(t_emb)[:, :, None] + self.cond(cond)[:, :, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h

# ──────────────────────────────────────────────────────────────────────────────
# UNet1D：时间维度卷积扩散网络
# ──────────────────────────────────────────────────────────────────────────────
class UNet1D(nn.Module):
    def __init__(self, in_ch: int, cond_dim: int, base: int = 64, time_dim: int = 128):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)
        self.b1 = ResBlock1D(base, cond_dim, time_dim)
        self.down = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)
        self.b2 = ResBlock1D(base * 2, cond_dim, time_dim)
        self.up = nn.ConvTranspose1d(base * 2, base, 4, stride=2, padding=1)
        self.b3 = ResBlock1D(base, cond_dim, time_dim)
        self.out = nn.Conv1d(base, in_ch, 3, padding=1)

    def forward(self, x, cond, t: torch.Tensor):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        h = self.b1(h, cond, t_emb)
        if h.shape[-1] >= 4:
            h = self.down(h)
            h = self.b2(h, cond, t_emb)
            h = self.up(h)
        h = self.b3(h, cond, t_emb)
        return self.out(h)



# ──────────────────────────────────────────────────────────────────────────────
# 图像编码器：ResNet18 backbone + 全连接投影
# ──────────────────────────────────────────────────────────────────────────────
class ImageEncoder(nn.Module):
    def __init__(self, out_dim=256, pretrained=False):
        super().__init__()
        backbone = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        self.enc = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, out_dim)

    def forward(self, x):
        h = self.enc(x)
        h = h.flatten(1)
        return self.proj(h)

# ──────────────────────────────────────────────────────────────────────────────
# 状态编码器：13维 VIO 状态 → 128 维特征
# ──────────────────────────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    def __init__(self, in_dim=13, out_dim=128):  # Adjusted in_dim to 13
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.SiLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, s):
        return self.net(s)

# ──────────────────────────────────────────────────────────────────────────────
# 主体：FPV Diffusion Policy 模型
# ──────────────────────────────────────────────────────────────────────────────
class FPVDiffusionPolicy(nn.Module):
    def __init__(self, action_dim=4, horizon=1,
                 img_feat=256, state_feat=128, base=64,
                 pretrained_backbone=False):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        # 编码器
        self.image_enc = ImageEncoder(out_dim=img_feat, pretrained=pretrained_backbone)
        self.state_enc = StateEncoder(in_dim=13, out_dim=state_feat)  # Adjusted in_dim to 13

        # 条件维度（图像 + 状态 + 动作特征）
        self.cond_dim = img_feat + state_feat + action_dim

        # 时间序列 UNet
        self.unet = UNet1D(in_ch=action_dim, cond_dim=self.cond_dim, base=base)

    def forward(self, noisy_action_seq, image, state, t):
        """
        noisy_action_seq : [B,4,L]
        image            : [B,3,H,W]
        state            : [B,13]
        t                : [B] long, diffusion timestep
        """
        # 图像与状态编码
        img_f = self.image_enc(image)             # [B,256]
        st_f = self.state_enc(state)              # [B,128]

        # 动作特征：均值（可换成 flatten 或线性层）
        act_f = noisy_action_seq.mean(dim=-1)     # [B,4]

        # 拼接条件向量
        cond = torch.cat([img_f, st_f, act_f], dim=-1)  # [B,384]

        # 扩散 UNet
        eps = self.unet(noisy_action_seq, cond, t)      # [B,4,L]
        return eps

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor, state_tensor: torch.Tensor,
                num_diffusion_steps: int = 10) -> torch.Tensor:
        """
        统一推理接口：DDPM采样生成动作序列

        Args:
            image_tensor: [B, 3, H, W] 图像张量
            state_tensor: [B, 13] 状态张量
            num_diffusion_steps: 扩散去噪步数

        Returns:
            [B, horizon, 4] 预测的动作序列 (Throttle, Yaw, Pitch, Roll)
        """
        device = image_tensor.device
        B = image_tensor.shape[0]

        # 从纯噪声开始
        action_seq = torch.randn(B, self.action_dim, self.horizon, device=device)

        # DDPM逆向去噪
        for t in reversed(range(num_diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            eps = self.forward(action_seq, image_tensor, state_tensor, t_tensor)

            # 简化的DDPM更新（可替换为更精确的调度器）
            alpha_t = 1.0 - t / num_diffusion_steps
            action_seq = action_seq - eps * (1 - alpha_t) * 0.1

        # 返回 [B, horizon, 4]
        return action_seq.permute(0, 2, 1)