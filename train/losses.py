"""
损失函数定义
作用：
    提供训练用的各种损失函数
主要函数：
    - diffusion_loss: 扩散模型的MSE损失
    - action_mse_loss: 动作预测MSE损失
依赖：
    torch
注意：
    所有损失函数返回标量张量
示例：
    loss = diffusion_loss(predicted_noise, target_noise)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def diffusion_loss(predicted_noise: torch.Tensor, target_noise: torch.Tensor) -> torch.Tensor:
    """
    扩散模型损失（简单MSE）

    Args:
        predicted_noise: [B, action_dim, horizon] 预测的噪声
        target_noise: [B, action_dim, horizon] 目标噪声

    Returns:
        标量损失
    """
    return F.mse_loss(predicted_noise, target_noise)


def action_mse_loss(predicted_action: torch.Tensor, target_action: torch.Tensor) -> torch.Tensor:
    """
    动作MSE损失

    Args:
        predicted_action: [B, action_dim] 或 [B, horizon, action_dim]
        target_action: [B, action_dim] 或 [B, horizon, action_dim]

    Returns:
        标量损失
    """
    return F.mse_loss(predicted_action, target_action)


def weighted_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor
) -> torch.Tensor:
    """
    加权MSE损失

    Args:
        predicted: 预测值
        target: 目标值
        weights: 权重（与predicted/target形状相同或可广播）

    Returns:
        标量损失
    """
    sq_error = (predicted - target) ** 2
    weighted_error = sq_error * weights
    return weighted_error.mean()


class DiffusionTrainingLoss(nn.Module):
    """
    扩散训练损失包装类
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        model_output: torch.Tensor,
        target_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        计算扩散损失

        Args:
            model_output: 模型输出的噪声预测
            target_noise: 真实添加的噪声

        Returns:
            损失值
        """
        return diffusion_loss(model_output, target_noise)
