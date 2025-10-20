"""
训练引擎
作用：
    提供通用的训练循环、验证和检查点管理
主要功能：
    - TrainingEngine: 训练引擎类
    - train_one_epoch: 单轮训练
    - validate: 验证
依赖：
    torch, numpy
注意：
    支持自动混合精度训练（AMP）
示例：
    engine = TrainingEngine(model, optimizer, criterion, device)
    engine.train_one_epoch(train_loader, epoch)
"""

import time
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler


class TrainingEngine:
    """
    训练引擎
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        use_amp: bool = True,
        grad_clip: float = 1.0
    ):
        """
        初始化训练引擎

        Args:
            model: 模型
            optimizer: 优化器
            criterion: 损失函数
            device: 设备
            scheduler: 学习率调度器
            use_amp: 是否使用自动混合精度
            grad_clip: 梯度裁剪值
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # 统计
        self.train_losses = []
        self.val_losses = []

    def train_one_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        print_freq: int = 100
    ) -> float:
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            print_freq: 打印频率

        Returns:
            平均损失
        """
        self.model.train()

        total_loss = 0.0
        num_batches = len(train_loader)
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # 数据移到设备
            images = batch['image'].to(self.device)  # [B, 3, H, W]
            states = batch['state'].to(self.device)  # [B, 13]
            actions = batch['action'].to(self.device)  # [B, 4]

            # 获取动作历史（如果dataset提供）
            action_history = batch.get('action_history', None)
            if action_history is not None:
                action_history = action_history.to(self.device)  # [B, action_history_len, 4]

            # 前向传播（扩散训练）
            # 1. 随机采样时间步
            B = images.shape[0]
            t = torch.randint(0, 100, (B,), device=self.device)

            # 2. 为动作添加噪声
            horizon = self.model.horizon
            action_seq = actions.unsqueeze(1).repeat(1, horizon, 1)  # [B, horizon, 4]
            action_seq = action_seq.permute(0, 2, 1)  # [B, 4, horizon]

            noise = torch.randn_like(action_seq)
            # 简化的前向扩散
            alpha_t = 1.0 - t.float().unsqueeze(-1).unsqueeze(-1) / 100.0
            noisy_action = alpha_t * action_seq + (1 - alpha_t) * noise

            # 前向传播
            if self.use_amp:
                with autocast():
                    predicted_noise = self.model(noisy_action, images, states, t, action_history)
                    loss = self.criterion(predicted_noise, noise)

                # 反向传播
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predicted_noise = self.model(noisy_action, images, states, t, action_history)
                loss = self.criterion(predicted_noise, noise)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.optimizer.step()

            # 统计
            total_loss += loss.item()

            # 打印
            if (batch_idx + 1) % print_freq == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx + 1) * train_loader.batch_size / elapsed
                print(
                    f"Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                    f"Speed: {samples_per_sec:.1f} samples/s"
                )

        # Epoch结束
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch}] completed in {elapsed:.1f}s, avg loss: {avg_loss:.4f}")

        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader: DataLoader, epoch: int) -> float:
        """
        验证

        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch

        Returns:
            平均损失
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = len(val_loader)

        for batch in val_loader:
            # 数据移到设备
            images = batch['image'].to(self.device)
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)

            # 获取动作历史（如果dataset提供）
            action_history = batch.get('action_history', None)
            if action_history is not None:
                action_history = action_history.to(self.device)

            # 前向传播（与训练相同的扩散过程）
            B = images.shape[0]
            t = torch.randint(0, 100, (B,), device=self.device)

            horizon = self.model.horizon
            action_seq = actions.unsqueeze(1).repeat(1, horizon, 1)
            action_seq = action_seq.permute(0, 2, 1)

            noise = torch.randn_like(action_seq)
            alpha_t = 1.0 - t.float().unsqueeze(-1).unsqueeze(-1) / 100.0
            noisy_action = alpha_t * action_seq + (1 - alpha_t) * noise

            predicted_noise = self.model(noisy_action, images, states, t, action_history)
            loss = self.criterion(predicted_noise, noise)

            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        print(f"Validation [{epoch}]: avg loss: {avg_loss:.4f}")

        return avg_loss

    def save_checkpoint(
        self,
        save_path: Path,
        epoch: int,
        best_loss: Optional[float] = None,
        **extra_info
    ):
        """
        保存检查点

        Args:
            save_path: 保存路径
            epoch: 当前epoch
            best_loss: 最佳损失
            **extra_info: 额外信息
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_loss': best_loss,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        checkpoint.update(extra_info)

        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: Path):
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'scaler_state_dict' in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"✓ Checkpoint loaded from {checkpoint_path}")
        print(f"  Resuming from epoch {checkpoint.get('epoch', 0)}")

        return checkpoint
