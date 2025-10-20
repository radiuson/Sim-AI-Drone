#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
 _____ ___   ____ _____
|_   _/ _ \ / ___| ____|
  | || | | | |  _|  _|
  | || |_| | |_| | |___
  |_| \___/ \____|_____|

TOGE: Temporal Observation Guided Executor
时序观测引导执行器 - FPV 无人机扩散策略模型

主要特性：
1. 多种强大的视觉编码器：EfficientNet-B3 / ConvNeXt / ResNet50 / ViT
2. 状态编码器（带自注意力）：捕获时序依赖
3. 交叉模态注意力融合：视觉-状态深度融合
4. 深层扩散 UNet：3-4 层下采样，Bottleneck 注意力
5. FiLM 条件化：Feature-wise Linear Modulation
6. 多尺度残差连接：更稳定的训练
"""

import math
from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ──────────────────────────────────────────────────────────────────────────────
# 时间嵌入
# ──────────────────────────────────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """正弦位置编码的时间嵌入"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t: torch.Tensor):
        """t: [B] 或 [B,1]"""
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)


# ──────────────────────────────────────────────────────────────────────────────
# 视觉编码器（多种 backbone 可选）
# ──────────────────────────────────────────────────────────────────────────────
class TOGEImageEncoder(nn.Module):
    """
    支持多种预训练 backbone:
    - efficientnet_b3: 1536 特征维度，更好的性能/效率平衡
    - convnext_tiny: 768 特征维度，现代卷积架构
    - resnet50: 2048 特征维度，经典强力模型
    - vit_b_16: 768 特征维度，Vision Transformer
    """
    def __init__(
        self,
        backbone: Literal["efficientnet_b3", "convnext_tiny", "resnet50", "vit_b_16"] = "efficientnet_b3",
        out_dim: int = 512,
        pretrained: bool = True,
        freeze_bn: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone

        # 加载 backbone
        if backbone == "efficientnet_b3":
            from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
            weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = efficientnet_b3(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-1])  # 去掉分类头
            feat_dim = 1536
        elif backbone == "convnext_tiny":
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            model = convnext_tiny(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            feat_dim = 768
        elif backbone == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            model = resnet50(weights=weights)
            self.encoder = nn.Sequential(*list(model.children())[:-1])
            feat_dim = 2048
        elif backbone == "vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            model = vit_b_16(weights=weights)
            # ViT 需要特殊处理
            self.encoder = model
            self.encoder.heads = nn.Identity()  # 移除分类头
            feat_dim = 768
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 冻结 BatchNorm
        if freeze_bn:
            self._freeze_bn()

        # 投影到统一维度
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Mish()
        )

    def _freeze_bn(self):
        """冻结所有 BatchNorm 层"""
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        返回: [B, out_dim]
        """
        h = self.encoder(x)
        if h.dim() > 2:
            h = h.flatten(1)
        return self.proj(h)


# ──────────────────────────────────────────────────────────────────────────────
# 状态编码器（带注意力）
# ──────────────────────────────────────────────────────────────────────────────
class StateEncoder(nn.Module):
    """
    状态编码器，支持历史状态的时序建模
    """
    def __init__(
        self,
        state_dim: int = 13,  # 更新：匹配新的 liftoff_bridge (无位置数据)
        out_dim: int = 256,
        hidden_dim: int = 128,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        if use_attention:
            # 简单的自注意力层
            self.self_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: [B, state_dim] 或 [B, T, state_dim]
        返回: [B, out_dim]
        """
        if state.dim() == 2:
            # 单个状态
            return self.mlp(state)
        else:
            # 状态序列
            B, T, D = state.shape
            state_flat = state.view(B * T, D)
            feat = self.mlp(state_flat).view(B, T, -1)

            if self.use_attention:
                # 自注意力
                feat, _ = self.self_attn(feat, feat, feat)

            # 取最后一个时间步或平均
            return feat[:, -1, :]  # 或 feat.mean(dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# 动作历史编码器
# ──────────────────────────────────────────────────────────────────────────────
class ActionHistoryEncoder(nn.Module):
    """
    动作历史编码器，用于编码过去的动作序列
    """
    def __init__(
        self,
        action_dim: int = 4,
        history_len: int = 4,
        out_dim: int = 128,
        hidden_dim: int = 64,
        use_attention: bool = True
    ):
        super().__init__()
        self.action_dim = action_dim
        self.history_len = history_len
        self.use_attention = use_attention

        # 动作序列编码
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

        if use_attention:
            # 自注意力捕获时序依赖
            self.self_attn = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )

    def forward(self, action_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_history: [B, history_len, action_dim] 历史动作序列

        Returns:
            [B, out_dim] 编码后的动作历史特征
        """
        B, T, D = action_history.shape

        # 编码每个时间步的动作
        action_flat = action_history.view(B * T, D)
        feat = self.mlp(action_flat).view(B, T, -1)  # [B, T, out_dim]

        if self.use_attention:
            # 自注意力
            feat, _ = self.self_attn(feat, feat, feat)

        # 聚合：取最后一个时间步（最近的动作最重要）
        return feat[:, -1, :]  # [B, out_dim]


# ──────────────────────────────────────────────────────────────────────────────
# 交叉注意力融合模块
# ──────────────────────────────────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    """视觉-状态交叉注意力融合"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, visual: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        visual: [B, dim]
        state: [B, dim]
        返回: [B, dim]
        """
        # 扩展为序列格式
        visual = visual.unsqueeze(1)  # [B, 1, dim]
        state = state.unsqueeze(1)    # [B, 1, dim]

        # 交叉注意力：visual 作为 query，state 作为 key/value
        attn_out, _ = self.attn(visual, state, state)
        visual = self.norm1(visual + attn_out)

        # FFN
        ffn_out = self.ffn(visual)
        visual = self.norm2(visual + ffn_out)

        return visual.squeeze(1)  # [B, dim]


# ──────────────────────────────────────────────────────────────────────────────
# FiLM 条件化模块
# ──────────────────────────────────────────────────────────────────────────────
class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, L]
        cond: [B, cond_dim]
        """
        gamma = self.gamma(cond).unsqueeze(-1)  # [B, C, 1]
        beta = self.beta(cond).unsqueeze(-1)    # [B, C, 1]
        return gamma * x + beta


# ──────────────────────────────────────────────────────────────────────────────
# 改进的 1D 残差块（带 FiLM 条件化）
# ──────────────────────────────────────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        time_dim: int,
        dropout: float = 0.1,
        use_film: bool = True
    ):
        super().__init__()
        self.use_film = use_film

        self.norm1 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)

        self.norm2 = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, out_channels * 2)
        )

        if use_film:
            self.film = FiLM(cond_dim, out_channels)
        else:
            self.cond_mlp = nn.Linear(cond_dim, out_channels)

        self.dropout = nn.Dropout(dropout)

        # 如果输入输出通道数不同，需要一个投影层
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_C, L]
        cond: [B, cond_dim]
        t_emb: [B, time_dim]
        返回: [B, out_C, L]
        """
        h = self.norm1(x)
        h = F.mish(h)
        h = self.conv1(h)

        # 时间调制
        t_out = self.time_mlp(t_emb)
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (scale.unsqueeze(-1) + 1) + shift.unsqueeze(-1)

        # 条件调制
        if self.use_film:
            h = self.film(h, cond)
        else:
            h = h + self.cond_mlp(cond).unsqueeze(-1)

        h = self.norm2(h)
        h = F.mish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return self.skip_conv(x) + h


# ──────────────────────────────────────────────────────────────────────────────
# 注意力块（用于 bottleneck）
# ──────────────────────────────────────────────────────────────────────────────
class AttentionBlock1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels // 4), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, L]"""
        B, C, L = x.shape
        h = self.norm(x)
        h = h.permute(0, 2, 1)  # [B, L, C]
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1)  # [B, C, L]
        return x + h


# ──────────────────────────────────────────────────────────────────────────────
# 增强版 UNet1D（3层下采样 + 注意力）
# ──────────────────────────────────────────────────────────────────────────────
class TOGEUNet1D(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        cond_dim: int = 768,
        time_dim: int = 256,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        use_film: bool = True
    ):
        super().__init__()
        self.action_dim = action_dim
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks

        # 时间嵌入
        self.time_mlp = SinusoidalTimeEmbedding(time_dim)

        # 输入投影
        self.input_proj = nn.Conv1d(action_dim, base_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mult[:-1]):
            out_ch = base_channels * mult
            # ResBlocks
            for j in range(num_res_blocks):
                # 第一个 block 输入通道是 ch，输出是 out_ch
                # 后续 block 输入输出都是 out_ch
                in_ch = ch if j == 0 else out_ch
                self.down_blocks.append(
                    ResBlock1D(in_ch, out_ch, cond_dim, time_dim, dropout, use_film)
                )
            # 下采样后通道数变为下一层的通道数
            next_ch = base_channels * channel_mult[i+1] if i+1 < len(channel_mult) else out_ch
            self.down_samples.append(
                nn.Conv1d(out_ch, next_ch, 4, stride=2, padding=1)
            )
            ch = next_ch

        # Bottleneck
        mid_ch = base_channels * channel_mult[-1]
        self.mid_blocks = nn.ModuleList([
            ResBlock1D(mid_ch, mid_ch, cond_dim, time_dim, dropout, use_film),
            AttentionBlock1D(mid_ch) if use_attention else nn.Identity(),
            ResBlock1D(mid_ch, mid_ch, cond_dim, time_dim, dropout, use_film)
        ])

        # 上采样路径
        self.up_samples = nn.ModuleList()
        self.up_channel_adjust = nn.ModuleList()  # 通道调整层
        self.up_blocks = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mult[:-1]))):
            out_ch = base_channels * mult
            # 上采样（空间上采样）
            self.up_samples.append(
                nn.ConvTranspose1d(mid_ch, out_ch, 4, stride=2, padding=1)
            )
            # 通道调整（用于长度不需要上采样时）
            self.up_channel_adjust.append(
                nn.Conv1d(mid_ch, out_ch, 1) if mid_ch != out_ch else nn.Identity()
            )
            for _ in range(num_res_blocks):
                self.up_blocks.append(
                    ResBlock1D(out_ch, out_ch, cond_dim, time_dim, dropout, use_film)
                )
            mid_ch = out_ch

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(32, base_channels // 4), base_channels),
            nn.Mish(),
            nn.Conv1d(base_channels, action_dim, 3, padding=1)
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, action_dim, L] - noisy action sequence
        cond: [B, cond_dim] - condition vector
        t: [B] - timestep
        """
        # 时间嵌入
        t_emb = self.time_mlp(t)

        # 输入投影
        h = self.input_proj(x)

        # 下采样
        block_idx = 0
        skips = []  # 存储跳跃连接
        for i, downsample in enumerate(self.down_samples):
            # 每个downsample前有 num_res_blocks 个 ResBlocks
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, cond, t_emb)
                block_idx += 1
            skips.append(h)
            # 只在序列长度足够时下采样（至少要能除以2）
            if h.shape[-1] >= 2:
                h = downsample(h)

        # Bottleneck
        for block in self.mid_blocks:
            if isinstance(block, ResBlock1D):
                h = block(h, cond, t_emb)
            else:
                h = block(h)

        # 上采样
        block_idx = 0
        for i, (upsample, channel_adjust) in enumerate(zip(self.up_samples, self.up_channel_adjust)):
            # 根据长度决定是否需要空间上采样
            if h.shape[-1] < x.shape[-1]:  # 如果当前长度小于输入长度，进行空间上采样
                h = upsample(h)
            else:  # 否则只调整通道数
                h = channel_adjust(h)

            # 每个upsample后有 num_res_blocks 个 ResBlocks
            for _ in range(self.num_res_blocks):
                h = self.up_blocks[block_idx](h, cond, t_emb)
                block_idx += 1

        # 输出
        return self.output_proj(h)


# ──────────────────────────────────────────────────────────────────────────────
# 主模型：TOGE (Temporal Observation Guided Executor)
# ──────────────────────────────────────────────────────────────────────────────
class TOGEPolicy(nn.Module):
    """
    TOGE: Temporal Observation Guided Executor
    时序观测引导执行器

    这是一个用于FPV无人机控制的高性能扩散策略模型，结合了：
    - Temporal: 深度时序建模（Transformer + 自注意力）
    - Observation: 多模态观测融合（视觉 + 状态）
    - Guided: 条件引导的扩散生成（FiLM + 交叉注意力）
    - Executor: 高精度动作执行（深层 UNet）

    核心特性：
    - 强大的视觉编码器（EfficientNet/ConvNeXt/ResNet50/ViT）
    - 状态编码器（带自注意力）
    - 交叉模态注意力融合
    - 深层 UNet（3-4层下采样 + Bottleneck 注意力）
    - FiLM 条件化机制
    - 多尺度残差连接
    """

    def __init__(
        self,
        # 任务参数
        action_dim: int = 4,
        state_dim: int = 13,  # 更新：匹配新的 liftoff_bridge (无位置数据)
        horizon: int = 4,

        # 编码器参数
        visual_backbone: str = "efficientnet_b3",
        visual_dim: int = 512,
        state_dim_out: int = 256,
        pretrained_backbone: bool = True,

        # 动作历史参数
        action_history_len: int = 4,
        action_history_dim: int = 128,
        use_action_history: bool = True,

        # 融合参数
        fusion_dim: int = 768,
        use_cross_attention: bool = True,

        # UNet 参数
        unet_base_channels: int = 128,
        unet_channel_mult: tuple = (1, 2, 4, 8),
        unet_num_res_blocks: int = 2,
        time_dim: int = 256,
        dropout: float = 0.1,
        use_film: bool = True,
        use_attention: bool = True
    ):
        super().__init__()

        self.action_dim = action_dim
        self.horizon = horizon
        self.use_cross_attention = use_cross_attention
        self.use_action_history = use_action_history
        self.action_history_len = action_history_len

        # 编码器
        self.visual_encoder = TOGEImageEncoder(
            backbone=visual_backbone,
            out_dim=visual_dim,
            pretrained=pretrained_backbone
        )

        self.state_encoder = StateEncoder(
            state_dim=state_dim,
            out_dim=state_dim_out,
            use_attention=True
        )

        # 动作历史编码器（可选）
        if use_action_history:
            self.action_history_encoder = ActionHistoryEncoder(
                action_dim=action_dim,
                history_len=action_history_len,
                out_dim=action_history_dim,
                use_attention=True
            )
        else:
            self.action_history_encoder = None
            action_history_dim = 0  # 不使用时维度为0

        # 融合
        if use_cross_attention:
            # 先投影到相同维度
            # 如果使用动作历史，需要调整融合维度
            base_dim = fusion_dim // 2 if not use_action_history else fusion_dim // 3

            self.visual_proj = nn.Linear(visual_dim, base_dim)
            self.state_proj = nn.Linear(state_dim_out, base_dim)

            if use_action_history:
                self.action_history_proj = nn.Linear(action_history_dim, base_dim)

            self.cross_attn = CrossModalAttention(
                dim=base_dim,
                num_heads=8,
                dropout=dropout
            )

            cond_dim = base_dim * (3 if use_action_history else 2)
        else:
            # 简单拼接
            self.visual_proj = nn.Linear(visual_dim, fusion_dim // 2)
            self.state_proj = nn.Linear(state_dim_out, fusion_dim // 2)

            if use_action_history:
                self.action_history_proj = nn.Linear(action_history_dim, fusion_dim // 4)
                cond_dim = fusion_dim + fusion_dim // 4
            else:
                cond_dim = fusion_dim

        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish()
        )

        # 扩散 UNet
        self.unet = TOGEUNet1D(
            action_dim=action_dim,
            cond_dim=fusion_dim,
            time_dim=time_dim,
            base_channels=unet_base_channels,
            channel_mult=unet_channel_mult,
            num_res_blocks=unet_num_res_blocks,
            dropout=dropout,
            use_attention=use_attention,
            use_film=use_film
        )

    def encode_visual(self, image: torch.Tensor) -> torch.Tensor:
        """
        编码视觉特征（慢速路径，30Hz）

        Args:
            image: [B, 3, H, W]

        Returns:
            visual_feat: [B, base_dim] 投影后的视觉特征
        """
        visual_feat = self.visual_encoder(image)  # [B, visual_dim]
        visual_feat = self.visual_proj(visual_feat)  # [B, base_dim]
        return visual_feat

    def encode_and_fuse(
        self,
        visual_feat: torch.Tensor,
        state: torch.Tensor,
        action_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合视觉特征、状态和动作历史（快速路径，100Hz）

        Args:
            visual_feat: [B, base_dim] 预计算的视觉特征
            state: [B, state_dim]
            action_history: [B, action_history_len, action_dim] 历史动作（可选）

        Returns:
            condition: [B, fusion_dim]
        """
        # 编码状态
        state_feat = self.state_encoder(state)    # [B, state_dim_out]
        state_feat = self.state_proj(state_feat)  # [B, base_dim]

        # 编码动作历史（如果使用）
        if self.use_action_history and action_history is not None:
            action_hist_feat = self.action_history_encoder(action_history)  # [B, action_history_dim]
            action_hist_feat = self.action_history_proj(action_hist_feat)   # [B, base_dim]

        # 融合
        if self.use_cross_attention:
            # 交叉注意力
            fused = self.cross_attn(visual_feat, state_feat)  # [B, base_dim]

            # 拼接动作历史（如果使用）
            if self.use_action_history and action_history is not None:
                fused = torch.cat([fused, state_feat, action_hist_feat], dim=-1)  # [B, base_dim * 3]
            else:
                fused = torch.cat([fused, state_feat], dim=-1)  # [B, base_dim * 2]
        else:
            # 简单拼接
            if self.use_action_history and action_history is not None:
                fused = torch.cat([visual_feat, state_feat, action_hist_feat], dim=-1)
            else:
                fused = torch.cat([visual_feat, state_feat], dim=-1)

        # 投影到条件空间
        condition = self.cond_proj(fused)  # [B, fusion_dim]

        return condition

    def encode_observation(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码观测并融合（保留用于向后兼容）

        Args:
            image: [B, 3, H, W]
            state: [B, state_dim]
            action_history: [B, action_history_len, action_dim] 历史动作（可选）

        Returns:
            condition: [B, fusion_dim]
        """
        # 先编码视觉
        visual_feat = self.encode_visual(image)
        # 再融合
        return self.encode_and_fuse(visual_feat, state, action_history)

    def forward(
        self,
        noisy_action: torch.Tensor,
        image: torch.Tensor,
        state: torch.Tensor,
        timestep: torch.Tensor,
        action_history: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            noisy_action: [B, action_dim, horizon] - 带噪声的动作序列
            image: [B, 3, H, W] - 图像观测
            state: [B, state_dim] - 状态观测
            timestep: [B] - 扩散时间步
            action_history: [B, action_history_len, action_dim] - 历史动作（可选）

        Returns:
            predicted_noise: [B, action_dim, horizon] - 预测的噪声
        """
        # 编码观测（包含动作历史）
        condition = self.encode_observation(image, state, action_history)

        # 扩散预测
        predicted_noise = self.unet(noisy_action, condition, timestep)

        return predicted_noise

    @torch.no_grad()
    def predict(
        self,
        image_tensor: torch.Tensor,
        state_tensor: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 10
    ) -> torch.Tensor:
        """
        统一推理接口：DDPM采样生成动作序列

        Args:
            image_tensor: [B, 3, H, W] 图像张量
            state_tensor: [B, 13] 状态张量
            action_history: [B, action_history_len, action_dim] 历史动作（可选）
            num_diffusion_steps: 扩散去噪步数

        Returns:
            [B, horizon, 4] 预测的动作序列 (Throttle, Yaw, Pitch, Roll)
        """
        device = image_tensor.device
        B = image_tensor.shape[0]

        # 如果没有提供action_history但模型需要，创建零初始化的历史
        if self.use_action_history and action_history is None:
            action_history = torch.zeros(
                B, self.action_history_len, self.action_dim,
                device=device, dtype=torch.float32
            )

        # 从纯噪声开始
        action_seq = torch.randn(B, self.action_dim, self.horizon, device=device)

        # DDPM逆向去噪
        for t in reversed(range(num_diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # 预测噪声
            eps = self.forward(action_seq, image_tensor, state_tensor, t_tensor, action_history)

            # 简化的DDPM更新（可替换为更精确的调度器）
            alpha_t = 1.0 - t / num_diffusion_steps
            action_seq = action_seq - eps * (1 - alpha_t) * 0.1

        # 返回 [B, horizon, 4]
        return action_seq.permute(0, 2, 1)

    @torch.no_grad()
    def predict_fast(
        self,
        visual_feat: torch.Tensor,
        state_tensor: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 10
    ) -> torch.Tensor:
        """
        快速推理接口：使用预计算的视觉特征（用于双频率推理）

        Args:
            visual_feat: [B, base_dim] 预计算的视觉特征
            state_tensor: [B, 13] 状态张量
            action_history: [B, action_history_len, action_dim] 历史动作（可选）
            num_diffusion_steps: 扩散去噪步数

        Returns:
            [B, horizon, 4] 预测的动作序列 (Throttle, Yaw, Pitch, Roll)
        """
        device = visual_feat.device
        B = visual_feat.shape[0]

        # 如果没有提供action_history但模型需要，创建零初始化的历史
        if self.use_action_history and action_history is None:
            action_history = torch.zeros(
                B, self.action_history_len, self.action_dim,
                device=device, dtype=torch.float32
            )

        # 融合特征（快速路径）
        condition = self.encode_and_fuse(visual_feat, state_tensor, action_history)

        # 从纯噪声开始
        action_seq = torch.randn(B, self.action_dim, self.horizon, device=device)

        # DDPM逆向去噪
        for t in reversed(range(num_diffusion_steps)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # 直接使用UNet，跳过重复的observation编码
            predicted_noise = self.unet(action_seq, condition, t_tensor)

            # 简化的DDPM更新
            alpha_t = 1.0 - t / num_diffusion_steps
            action_seq = action_seq - predicted_noise * (1 - alpha_t) * 0.1

        # 返回 [B, horizon, 4]
        return action_seq.permute(0, 2, 1)


# ──────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ──────────────────────────────────────────────────────────────────────────────
def create_toge_policy(
    model_size: Literal["small", "medium", "large", "xlarge"] = "medium",
    **kwargs
) -> TOGEPolicy:
    """
    创建不同大小的模型

    small: ~15M params, 快速训练
    medium: ~35M params, 平衡性能
    large: ~70M params, 高性能
    xlarge: ~120M params, 最强性能
    """
    configs = {
        "small": {
            "visual_backbone": "efficientnet_b3",
            "visual_dim": 256,
            "state_dim_out": 128,
            "fusion_dim": 384,
            "unet_base_channels": 64,
            "unet_channel_mult": (1, 2, 4),
            "unet_num_res_blocks": 1,
        },
        "medium": {
            "visual_backbone": "efficientnet_b3",
            "visual_dim": 512,
            "state_dim_out": 256,
            "fusion_dim": 768,
            "unet_base_channels": 128,
            "unet_channel_mult": (1, 2, 4, 8),
            "unet_num_res_blocks": 2,
        },
        "large": {
            "visual_backbone": "convnext_tiny",
            "visual_dim": 768,
            "state_dim_out": 384,
            "fusion_dim": 1152,
            "unet_base_channels": 192,
            "unet_channel_mult": (1, 2, 4, 8),
            "unet_num_res_blocks": 3,
        },
        "xlarge": {
            "visual_backbone": "resnet50",
            "visual_dim": 1024,
            "state_dim_out": 512,
            "fusion_dim": 1536,
            "unet_base_channels": 256,
            "unet_channel_mult": (1, 2, 4, 8),
            "unet_num_res_blocks": 3,
        }
    }

    config = configs[model_size]
    config.update(kwargs)

    return TOGEPolicy(**config)


# ──────────────────────────────────────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 测试模型
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for size in ["small", "medium", "large"]:
        print(f"\n{'='*70}")
        print(f"Testing {size.upper()} model")
        print(f"{'='*70}")

        model = create_toge_policy(
            model_size=size,
            action_dim=4,
            state_dim=13,  # 更新：匹配新的录制器数据格式
            horizon=4,
            pretrained_backbone=False  # 测试时不加载预训练权重
        ).to(device)

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"Total parameters: {total_params / 1e6:.2f}M")
        print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

        # 测试前向传播
        B = 4
        image = torch.randn(B, 3, 224, 224).to(device)
        state = torch.randn(B, 13).to(device)  # 更新：13维状态
        noisy_action = torch.randn(B, 4, 4).to(device)
        timestep = torch.randint(0, 100, (B,)).to(device)

        with torch.no_grad():
            output = model(noisy_action, image, state, timestep)

        print(f"Input shape: {noisy_action.shape}")
        print(f"Output shape: {output.shape}")
        print(f"✓ Forward pass successful!")

    print(f"\n{'='*70}")
    print("All tests passed!")
    print(f"{'='*70}\n")
