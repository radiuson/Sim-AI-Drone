# AI Drone - FPV Drone Control with Diffusion Policies

端到端的FPV无人机AI控制系统，使用扩散策略模型从Liftoff模拟器中学习飞行控制。

## 特性

- **统一模型接口**: 支持多种模型架构（ResNet-UNet, TOGE）
- **LeRobot数据格式**: 标准化数据集格式，易于共享和复现
- **虚拟遥控器**: 通过Linux uinput直接控制Liftoff模拟器
- **完整训练流程**: 数据录制 → 训练 → 部署
- **实时控制**: 30+ FPS的低延迟策略推理

## 目录结构

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

## 安装

### 1. 系统依赖

```bash
# 安装xdotool（用于窗口定位）
sudo apt-get update
sudo apt-get install -y xdotool

# 加载uinput内核模块
sudo modprobe uinput

# 永久加载（可选）
echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf

# 添加用户到input组（避免需要root权限）
sudo usermod -a -G input $USER
# 然后重新登录
```

### 2. Python依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 测试虚拟遥控器
python -m deploy.virtual_joystick --test

# 测试键盘控制
python -m tools.joystick_keyboard
```

## 使用流程

### 1. 数据录制

使用真实遥控器在Liftoff中飞行，录制数据集：

```bash
python -m record.liftoff_capture \
  --output-dir ./record/lerobot_datasets/my_dataset \
  --window-name "Liftoff" \
  --fps 30 \
  --joystick-device /dev/input/js0
```

### 2. 训练模型

#### 训练TOGE模型

```bash
python -m train.train \
  --model toge \
  --dataset-root ./record/lerobot_datasets/my_dataset \
  --img-size 224 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --augment
```

#### 训练ResNet-UNet模型

```bash
python -m train.train \
  --model resnet_unet \
  --dataset-root ./record/lerobot_datasets/my_dataset \
  --img-size 224 \
  --batch-size 64 \
  --epochs 50
```

#### 使用配置文件训练

```bash
python -m train.train --config configs/train_example.yaml
```

### 3. 部署控制

在Liftoff中使用训练好的模型：

```bash
# 首先在Liftoff中配置虚拟遥控器
# Settings → Controls → Add Controller → 选择 "AI-Liftoff-Controller"

# 运行策略控制
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_20250101_120000/checkpoints/best.pt \
  --window-name "Liftoff" \
  --image-size 224 \
  --rate 30 \
  --ema 0.2 \
  --max-action-change 0.3
```

### 4. 校准和测试

#### 校准虚拟遥控器

```bash
python -m tools.joystick_calibrate
```

在Liftoff中：
1. Settings → Controls → Add Controller
2. 选择 "TOGE-AI-Controller"
3. 按键盘按键逐个测试通道（W/S, A/D, I/K, J/L）
4. 分配通道：Throttle, Yaw, Pitch, Roll

#### 键盘驾驶测试

```bash
python -m tools.joystick_keyboard --mode incremental --sensitivity 0.05
```

按键控制：
- W/S: 油门
- A/D: 偏航
- I/K: 俯仰
- J/L: 横滚
- Space: 重置
- T: 解锁
- ESC/Q: 退出

## 模型说明

### ResNet-UNet Diffusion Policy

- **特点**: 轻量级，训练快速
- **参数量**: ~20M
- **Backbone**: ResNet18
- **Horizon**: 1-4帧
- **适用场景**: 快速原型，资源受限环境

### TOGE Policy

- **特点**: 高性能，强大的视觉编码
- **参数量**: 35M-120M（可配置）
- **Backbone**: EfficientNet-B3 / ConvNeXt / ResNet50 / ViT
- **Horizon**: 4-8帧
- **适用场景**: 追求最佳性能，GPU资源充足

## 配置说明

### 训练参数

- `--img-size`: 输入图像尺寸（224推荐）
- `--batch-size`: 批大小（GPU显存允许越大越好）
- `--lr`: 学习率（1e-4推荐）
- `--epochs`: 训练轮数（100-200）
- `--augment`: 数据增强（推荐开启）
- `--pretrained`: 使用预训练backbone（推荐开启）

### 部署参数

- `--rate`: 控制频率（30-60 Hz）
- `--ema`: EMA平滑系数（0.1-0.3，越小越平滑）
- `--max-action-change`: 单步最大动作变化（0.2-0.5）
- `--num-diffusion-steps`: 扩散采样步数（5-20，越大越准确但越慢）

## 故障排除

### 虚拟遥控器无法创建

```bash
# 检查uinput模块
lsmod | grep uinput

# 如果没有，加载模块
sudo modprobe uinput

# 检查权限
groups | grep input
# 如果没有input组，需要重新登录
```

### Liftoff窗口识别失败

```bash
# 手动查找窗口
xdotool search --name "Liftoff"

# 如果xdotool找不到窗口，尝试使用完整窗口名
xdotool search --name "Liftoff: The Game"
```

### 训练显存不足

```bash
# 减小batch size
python -m train.train --batch-size 16

# 使用混合精度训练（默认开启）
python -m train.train --amp

# 使用CPU训练（非常慢）
python -m train.train --device cpu
```

## 性能指标

典型性能（NVIDIA RTX 3080）：

| 模型 | 训练速度 | 推理速度 | GPU显存 |
|------|---------|---------|---------|
| ResNet-UNet | ~500 samples/s | 60+ FPS | 4GB |
| TOGE-Medium | ~200 samples/s | 40+ FPS | 8GB |
| TOGE-Large | ~100 samples/s | 25+ FPS | 12GB |

## 常见问题

**Q: 数据集需要多少数据？**
A: 推荐至少30-50个episode，每个episode 30-60秒，总计1500-3000帧。

**Q: 模型训练需要多久？**
A: ResNet-UNet约1-2小时，TOGE约3-6小时（100 epochs，RTX 3080）。

**Q: 为什么部署时飞机抖动？**
A: 尝试增大`--ema`参数（如0.3）或减小`--max-action-change`（如0.2）。

**Q: 如何提升模型性能？**
A:
1. 收集更多高质量数据
2. 使用数据增强（`--augment`）
3. 增加训练轮数
4. 尝试TOGE模型
5. 调整horizon长度

## 引用

如果这个项目对你有帮助，请引用：

```bibtex
@software{ai_drone_fpv,
  title = {AI Drone: FPV Drone Control with Diffusion Policies},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/ai-drone}
}
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 致谢

- [LeRobot](https://github.com/huggingface/lerobot) - 数据集格式
- [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) - 扩散策略思想
- Liftoff模拟器 - 训练环境
