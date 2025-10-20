# 快速开始指南

## 1. 安装依赖 (5分钟)

### 安装Python依赖
```bash
pip install -r requirements.txt
```

### 安装系统依赖
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y xdotool

# 加载uinput模块
sudo modprobe uinput

# 添加用户到input组（避免需要sudo）
sudo usermod -a -G input $USER
# 重新登录使组权限生效
```

### 验证安装
```bash
python verify_installation.py
```

## 2. 虚拟遥控器配置 (10分钟)

### 步骤1: 校准虚拟遥控器
```bash
python -m tools.joystick_calibrate
```

### 步骤2: 在Liftoff中配置
1. 启动Liftoff
2. 进入: **Settings → Controls → Add Controller**
3. 选择 **"TOGE-AI-Controller"**
4. 按键盘按键，观察哪个通道在移动：
   - 按 W/S → 记住这是哪个通道 → 分配为 **Throttle**
   - 按 A/D → 分配为 **Yaw**
   - 按 I/K → 分配为 **Pitch**
   - 按 J/L → 分配为 **Roll**
5. 保存配置

### 步骤3: 测试键盘驾驶
```bash
python -m tools.joystick_keyboard
```

按键控制：
- **W/S**: 油门 ↑↓
- **A/D**: 偏航 ←→
- **I/K**: 俯仰 ↑↓
- **J/L**: 横滚 ←→
- **Space**: 重置
- **T**: 解锁（ARM）
- **ESC/Q**: 退出

## 3. 训练模型 (30分钟 - 6小时)

### 选项A: 使用示例数据集（如果已有）
```bash
python -m train.train \
  --model toge \
  --dataset-root ./record/lerobot_datasets/liftoff_drone_dataset \
  --img-size 224 \
  --batch-size 32 \
  --epochs 100 \
  --lr 1e-4 \
  --augment \
  --pretrained
```

### 选项B: 使用配置文件
```bash
# 编辑配置文件
nano configs/train_example.yaml

# 运行训练
python -m train.train --config configs/train_example.yaml
```

### 训练监控
训练过程中会显示：
- Epoch进度
- 训练损失
- 验证损失
- 训练速度（samples/s）

检查点保存在：`outputs/toge_YYYYMMDD_HHMMSS/checkpoints/`

## 4. 部署控制 (立即可用)

### 启动AI控制（标准模式）
```bash
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_20250101_120000/checkpoints/best.pt \
  --window-name "Liftoff" \
  --image-size 224 \
  --rate 30 \
  --ema 0.2 \
  --max-action-change 0.3
```

### 启动AI控制（双频率模式 - 推荐TOGE）
```bash
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_20250101_120000/checkpoints/best.pt \
  --window-name "Liftoff" \
  --dual-rate \
  --visual-rate 30 \
  --action-rate 100
```

**双频率模式优势**:
- ✅ 动作更新频率: 30Hz → 100Hz (3.3倍提升)
- ✅ 控制延迟降低: 34ms → 12ms
- ✅ 动作更平滑，抖动更少
- ✅ 仅适用于TOGE策略（需要GPU）

详细说明请查看: [DUAL_RATE_INFERENCE.md](DUAL_RATE_INFERENCE.md)

### 参数调优

**如果飞机抖动严重**:
```bash
--ema 0.3              # 增大平滑系数
--max-action-change 0.2  # 减小最大变化
```

**如果反应太慢**:
```bash
--ema 0.1              # 减小平滑系数
--max-action-change 0.5  # 增大最大变化
--dual-rate            # 启用双频率模式（仅TOGE）
```

**如果推理太慢**:
```bash
--num-diffusion-steps 5  # 减少采样步数（默认10）
--dual-rate            # 启用双频率模式（仅TOGE）
```

## 5. 常见问题

### Q: "Module not found" 错误
```bash
# 确保在项目根目录运行命令
cd /home/ihpc/code/ai-drone
python -m train.train ...
```

### Q: 虚拟遥控器无法创建
```bash
# 检查uinput模块
lsmod | grep uinput

# 如果没有，加载模块
sudo modprobe uinput

# 检查权限
groups | grep input
# 如果没有input组，需要重新登录
```

### Q: Liftoff窗口识别失败
```bash
# 查找窗口
xdotool search --name "Liftoff"

# 如果找不到，尝试使用完整窗口名
python -m deploy.run_policy --window-name "Liftoff: The Game" ...
```

### Q: CUDA out of memory
```bash
# 减小batch size
python -m train.train --batch-size 16 ...

# 或使用CPU（很慢）
python -m train.train --device cpu ...
```

## 6. 推荐工作流程

### 新手流程（第一次使用）
1. ✅ 安装依赖
2. ✅ 校准虚拟遥控器
3. ✅ 测试键盘驾驶（熟悉控制）
4. ✅ 下载或准备数据集
5. ✅ 训练模型（建议先用小数据集快速测试）
6. ✅ 部署控制

### 数据收集流程
1. 用物理遥控器在Liftoff中手动飞行
2. 运行 `python -m record.liftoff_capture` 录制
3. 收集至少30-50个episode（每个30-60秒）
4. 训练模型
5. 测试部署

### 模型改进流程
1. 部署现有模型，观察问题
2. 收集更多针对性数据（如特定机动、场景）
3. 重新训练（可以从检查点继续）
4. 对比新旧模型性能
5. 调整参数和数据

## 7. 性能优化提示

### 训练加速
- 使用 `--amp` (自动混合精度，默认开启)
- 增大 `--batch-size` (在GPU显存允许的情况下)
- 使用 `--num-workers 8` (增加数据加载线程)
- 使用SSD存储数据集

### 推理加速
- 使用GPU: `--device cuda`
- 减少扩散步数: `--num-diffusion-steps 5`
- 降低控制频率: `--rate 20` (如果30Hz太高)

### 控制质量
- 增加训练数据量
- 使用数据增强: `--augment`
- 使用预训练backbone: `--pretrained`
- 增加训练轮数: `--epochs 200`

## 8. 下一步

- 📖 阅读完整文档: [README.md](README.md)
- 🔧 查看配置示例: [configs/train_example.yaml](configs/train_example.yaml)
- 📊 查看迁移报告: [MIGRATION_REPORT.md](MIGRATION_REPORT.md)
- 🐛 报告问题或建议

## 需要帮助？

运行验证脚本查看详细状态：
```bash
python verify_installation.py
```

祝飞行愉快！🚁✨
