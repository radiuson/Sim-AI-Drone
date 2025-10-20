# AI Drone 项目概述

## 项目简介

AI Drone 是一个端到端的FPV（第一人称视角）无人机AI控制系统，使用扩散策略模型从Liftoff模拟器中学习飞行控制。该项目旨在通过深度学习技术实现自主无人机飞行控制。

## 技术栈

- **深度学习框架**: PyTorch
- **计算机视觉**: OpenCV, Pillow
- **屏幕捕获**: mss, OBS虚拟摄像头
- **输入设备**: evdev, pynput
- **数据处理**: pandas, pyarrow (Parquet格式)
- **配置管理**: PyYAML

## 项目架构

```
ai-drone/
├── models/                    # 模型定义
│   ├── __init__.py           # 模型注册表
│   ├── resnet18_unet.py      # ResNet-UNet扩散模型
│   └── toge_policy.py        # TOGE策略模型
│
├── record/                    # 数据录制
│   ├── __init__.py
│   ├── datasets.py           # 数据集加载器
│   └── liftoff_capture.py    # 数据采集工具
│
├── train/                     # 训练模块
│   ├── __init__.py
│   ├── engine.py             # 训练引擎
│   ├── losses.py             # 损失函数
│   └── train.py              # 统一训练入口
│
├── deploy/                    # 部署模块
│   ├── __init__.py
│   ├── screen_capture.py     # 屏幕捕获
│   ├── virtual_joystick.py   # 虚拟遥控器
│   └── run_policy.py         # 统一部署入口
│
├── tools/                     # 工具
│   ├── __init__.py
│   ├── joystick_calibrate.py # 虚拟遥控器校准
│   └── joystick_keyboard.py  # 键盘驾驶测试
│
├── configs/                   # 配置文件
│   └── train_example.yaml    # 训练配置示例
│
├── outputs/                   # 训练输出（自动生成）
├── legacy/                    # 旧代码备份
├── requirements.txt
└── README.md
```

## 核心功能模块

### 1. 模型 (models/)

提供两种主要的扩散策略模型：
- **ResNet-UNet Diffusion Policy**: 轻量级模型，适合快速原型开发和资源受限环境
- **TOGE Policy**: 高性能模型，具有强大的视觉编码能力

### 2. 数据录制 (record/)

从Liftoff模拟器采集图像、状态和遥控器输入，保存为LeRobot格式数据集：
- 支持MSS屏幕捕获和OBS虚拟摄像头捕获
- 通过ROS2 bridge获取真实状态数据
- 支持遥控器输入录制

### 3. 训练 (train/)

提供统一的模型训练入口：
- 支持命令行参数和YAML配置文件
- 包含完整的训练循环和检查点管理
- 支持自动混合精度训练

### 4. 部署 (deploy/)

加载训练好的模型，实时捕获Liftoff画面并控制虚拟遥控器：
- 支持多种策略模型部署
- 提供动作平滑（EMA）与限幅功能
- 支持双频率推理模式（视觉30Hz + 动作100Hz）

## 开发与部署流程

### 1. 环境准备
```bash
# 安装系统依赖
sudo apt-get update
sudo apt-get install -y xdotool
sudo modprobe uinput
sudo usermod -a -G input $USER

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 数据录制
```bash
python -m record.liftoff_capture \
  --output-dir ./dataset/liftoff_data \
  --window-name "Liftoff" \
  --fps 30 \
  --joystick-device /dev/input/js0
```

### 3. 模型训练
```bash
# 使用命令行参数训练
python -m train.train \
  --model toge \
  --dataset-root ./dataset/liftoff_data \
  --img-size 224 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-4

# 使用配置文件训练
python -m train.train --config configs/train_example.yaml
```

### 4. 策略部署
```bash
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_20250101_120000/checkpoints/best.pt \
  --window-name "Liftoff" \
  --image-size 224 \
  --rate 30
```

## 配置说明

### 训练配置 (configs/train_example.yaml)
- `model`: 模型类型 (toge, resnet_unet, fpv_diffusion)
- `horizon`: 动作序列长度
- `dataset_root`: 数据集路径
- `img_size`: 输入图像尺寸
- `batch_size`: 批处理大小
- `epochs`: 训练轮数
- `lr`: 学习率

### 部署配置
- `--policy`: 策略名称
- `--checkpoint`: 模型检查点路径
- `--rate`: 控制频率
- `--ema`: EMA平滑系数
- `--max-action-change`: 单步最大动作变化