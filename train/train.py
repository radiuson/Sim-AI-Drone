"""
统一训练入口
作用：
    提供统一的模型训练脚本，支持多种模型和配置
主要功能：
    - 命令行参数解析
    - 数据集加载
    - 模型创建
    - 训练循环
    - 检查点管理
依赖：
    torch, numpy, yaml
注意：
    支持从命令行或YAML配置文件加载参数
示例：
    # 使用命令行参数
    python -m train.train --model toge --dataset-root ./data --epochs 100

    # 使用配置文件
    python -m train.train --config configs/train_example.yaml
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
from torch.utils.data import DataLoader, random_split

# 导入本地模块
from models import get_model, list_models
from record.datasets import FPVDataset
from train.engine import TrainingEngine
from train.losses import DiffusionTrainingLoss


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Unified Training Script for FPV Drone Policies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Train TOGE policy
  python -m train.train \\
      --model toge \\
      --dataset-root ./record/lerobot_datasets/liftoff_drone_dataset \\
      --img-size 224 \\
      --batch-size 32 \\
      --epochs 100 \\
      --lr 1e-4

  # Train ResNet-UNet policy
  python -m train.train \\
      --model resnet_unet \\
      --dataset-root ./record/lerobot_datasets/liftoff_drone_dataset \\
      --img-size 224 \\
      --epochs 50

  # Load from config file
  python -m train.train --config configs/train_example.yaml

Available models: {', '.join(list_models())}
        """
    )

    # 基本配置
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--model', type=str, choices=list_models(), help='Model name')
    parser.add_argument('--dataset-root', type=str, help='Dataset root directory')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')

    # 数据配置
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'step', 'none'], default='cosine')

    # 模型配置
    parser.add_argument('--horizon', type=int, default=4, help='Action horizon')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')

    # 系统配置
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 日志配置
    parser.add_argument('--print-freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='Save checkpoint frequency')

    return parser.parse_args()


def load_config(config_path: str, args: argparse.Namespace):
    """从YAML文件加载配置并合并到args"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 合并配置（命令行优先）
    for key, value in config.items():
        if not hasattr(args, key) or getattr(args, key) is None:
            setattr(args, key, value)

    return args


def create_dataloaders(args):
    """创建数据加载器"""
    print("Loading dataset...")

    # 加载完整数据集
    full_dataset = FPVDataset(
        dataset_root=args.dataset_root,
        image_size=args.img_size,
        augment=args.augment,
        normalize=True
    )

    # 划分训练/验证集
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def create_model(args, device):
    """创建模型"""
    print(f"Creating model: {args.model}")

    # 根据模型类型设置参数
    model_kwargs = {
        'action_dim': 4,
        'horizon': args.horizon
    }

    if args.model == 'toge':
        model_kwargs.update({
            'state_dim': 13,
            'visual_backbone': 'efficientnet_b3',
            'pretrained_backbone': args.pretrained
        })
    elif args.model in ['resnet_unet', 'fpv_diffusion']:
        model_kwargs.update({
            'pretrained_backbone': args.pretrained
        })

    model = get_model(args.model, **model_kwargs)
    model = model.to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    return model


def main():
    # 解析参数
    args = parse_args()

    # 加载配置文件
    if args.config:
        args = load_config(args.config, args)

    # 验证必需参数
    if args.model is None or args.dataset_root is None:
        print("❌ Error: --model and --dataset-root are required")
        return

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    config_file = output_dir / 'config.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(vars(args), f)

    print("="*70)
    print("Training Configuration")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print("="*70)
    print()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 创建数据加载器
    train_loader, val_loader = create_dataloaders(args)

    # 创建模型
    model = create_model(args, device)

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.epochs // 3,
            gamma=0.1
        )

    # 创建损失函数
    criterion = DiffusionTrainingLoss()

    # 创建训练引擎
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=args.amp,
        grad_clip=args.grad_clip
    )

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint = engine.load_checkpoint(Path(args.resume))
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_loss', float('inf'))

    # 训练循环
    print()
    print("="*70)
    print("Starting Training")
    print("="*70)
    print()

    try:
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print("-" * 70)

            # 训练
            train_loss = engine.train_one_epoch(train_loader, epoch+1, args.print_freq)

            # 验证
            val_loss = engine.validate(val_loader, epoch+1)

            # 保存检查点
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # 定期保存
            if (epoch + 1) % args.save_freq == 0:
                save_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1:04d}.pt'
                engine.save_checkpoint(
                    save_path=save_path,
                    epoch=epoch,
                    best_loss=best_val_loss
                )

            # 保存最佳模型
            if is_best:
                best_path = checkpoint_dir / 'best.pt'
                engine.save_checkpoint(
                    save_path=best_path,
                    epoch=epoch,
                    best_loss=best_val_loss
                )
                print(f"✓ New best model saved! Val loss: {val_loss:.4f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")

    finally:
        # 保存最终模型
        final_path = checkpoint_dir / 'final.pt'
        engine.save_checkpoint(
            save_path=final_path,
            epoch=args.epochs - 1,
            best_loss=best_val_loss
        )

        print()
        print("="*70)
        print("Training Summary")
        print("="*70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Checkpoints saved to: {checkpoint_dir}")
        print("="*70)
        print("✓ Training completed")


if __name__ == '__main__':
    main()
