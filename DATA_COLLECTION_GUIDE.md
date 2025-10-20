# Liftoff 数据采集完整指南

本指南介绍如何使用 OBS + ROS2 Bridge + RadioMaster 采集 Liftoff 飞行数据。

---

## 🎯 系统架构

```
┌─────────────┐        UDP         ┌──────────────────┐
│   Liftoff   │─────────────────────>│ liftoff_bridge   │
│  (Simulator)│   Port 30001        │   (ROS2 Node)    │
└─────────────┘                     └──────────────────┘
      │                                      │
      │ 画面                                  │ ROS2 Topics:
      ▼                                      │ - /liftoff/rc
┌─────────────┐                             │ - /liftoff/pose
│     OBS     │                             │ - /liftoff/twist
│   Virtual   │                             │ - /liftoff/imu
│   Camera    │                             ▼
└─────────────┘                     ┌──────────────────┐
      │                             │ liftoff_capture  │
      │ /dev/video10                │  (Data Logger)   │
      │                             └──────────────────┘
      │                                      │
      └──────────────────────────────────────┘
                                             ▼
                                    ┌──────────────────┐
                                    │  LeRobot Dataset │
                                    │  (Parquet Files) │
                                    └──────────────────┘
```

---

## 📋 前置准备

### 1. 系统要求

- Ubuntu 24.04 (推荐) 或 22.04
- ROS2 Jazzy (或 Humble)
- Python 3.10+
- OBS Studio 30.0+
- Liftoff 游戏

### 2. 安装依赖

```bash
# ROS2 依赖（如果还没安装）
sudo apt install ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs

# Python 依赖
pip install numpy pandas pillow opencv-python

# OBS 和虚拟摄像头
sudo apt install obs-studio v4l2loopback-dkms
```

### 3. 硬件连接

**RadioMaster 遥控器**：
1. 通过 USB 连接到电脑
2. 设置为 **Joystick 模式**（EdgeTX 设置中）
3. 验证设备：`ls -la /dev/input/js0`

---

## 🚀 快速开始

### 第一步：启动系统组件

#### 1. 加载虚拟摄像头模块

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"
```

#### 2. 启动 OBS

```bash
obs &
```

**在 OBS 中配置**：
- 添加 **"窗口捕获 (Xcomposite)"** 源
- 选择 Liftoff 窗口
- 点击 **"启动虚拟摄像头"**

#### 3. 启动 Liftoff

```bash
# 启动 Liftoff 游戏
# 在游戏设置中启用 UDP 输出：
# Settings → Extras → UDP Output → Enable
# Host: 127.0.0.1
# Port: 30001
```

#### 4. 启动 ROS2 Bridge

```bash
cd /home/ihpc/code/ai-drone

# 方法1：使用启动脚本（推荐）
./start_bridge.sh

# 方法2：直接运行
source /opt/ros/jazzy/setup.bash
python3 liftoff_bridge_ros2.py
```

**验证 bridge 运行**：

打开新终端：
```bash
source /opt/ros/jazzy/setup.bash

# 查看话题
ros2 topic list
# 应该看到：
# /liftoff/rc
# /liftoff/pose
# /liftoff/twist
# /liftoff/imu

# 查看遥控器数据
ros2 topic echo /liftoff/rc
```

---

### 第二步：开始数据采集

#### 启动采集器

```bash
cd /home/ihpc/code/ai-drone

# 使用默认设置（最简单）
python -m record.liftoff_capture \
  --output-dir ./dataset/my_flights
```

**采集界面**：
```
Controls:
  Press 'r' to start recording episode
  Press 's' to stop and save current episode
  Press 'q' to quit
```

#### 录制流程

1. **准备飞行**：
   - 在 Liftoff 中选择地图和无人机
   - 确保 RadioMaster 已连接
   - 确保 bridge 正在接收数据

2. **开始录制**：
   - 在采集器终端输入 `r` 并回车
   - 看到 `📹 Starting episode 0`

3. **飞行操作**：
   - 使用 RadioMaster 手动飞行
   - 尝试多样化的动作（起飞、转弯、穿越、降落等）
   - 建议每个 episode 持续 10-30 秒

4. **停止录制**：
   - 输入 `s` 并回车
   - 看到 `✓ Saved episode 0: XX frames`

5. **重复录制**：
   - 可以继续录制更多 episodes
   - 每次都会自动创建新的 episode

6. **退出**：
   - 输入 `q` 并回车
   - 元数据会自动保存

---

## 📊 数据集结构

采集完成后，数据保存在：

```
dataset/my_flights/
├── videos/                          # 图像帧
│   ├── episode_000000_frame_000000.png
│   ├── episode_000000_frame_000001.png
│   ├── ...
│   ├── episode_000001_frame_000000.png
│   └── ...
├── data/                            # Episode 数据（Parquet 格式）
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
└── meta/                            # 元数据
    └── info.json
```

### 数据格式

每个 episode 的 Parquet 文件包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `episode_index` | int | Episode 编号 |
| `frame_index` | int | 帧编号 |
| `timestamp` | float | 时间戳 (秒) |
| `observation.images.cam_front` | str | 图像文件名 |
| `observation.state` | list[13] | 状态向量 |
| `action` | list[4] | 动作向量 |

**状态向量 (13维)**：
```
[vx, vy, vz,          # 线速度 (m/s)
 qw, qx, qy, qz,      # 四元数姿态
 wx, wy, wz,          # 角速度 (rad/s)
 ax, ay, az]          # 线加速度 (m/s²)
```

**动作向量 (4维)**：
```
[throttle,  # 油门 [-1, 1]
 yaw,       # 偏航 [-1, 1]
 pitch,     # 俯仰 [-1, 1]
 roll]      # 横滚 [-1, 1]
```

---

## ⚙️ 高级配置

### 自定义采集参数

```bash
# 更高帧率
python -m record.liftoff_capture \
  --output-dir ./dataset/high_fps \
  --fps 60

# 更大图像
python -m record.liftoff_capture \
  --output-dir ./dataset/large_images \
  --image-size 640

# 使用 MSS 捕获（不推荐）
python -m record.liftoff_capture \
  --output-dir ./dataset/mss_data \
  --capture-method mss \
  --window-name "Liftoff"

# 禁用 ROS2（使用模拟数据，仅用于测试）
python -m record.liftoff_capture \
  --output-dir ./dataset/mock_data \
  --no-ros2
```

### ROS2 Bridge 参数

编辑 `liftoff_bridge_ros2.py` 或通过 ROS2 参数：

```bash
python3 liftoff_bridge_ros2.py \
  --ros-args \
  -p host:=127.0.0.1 \
  -p port:=30001 \
  -p print_rate_hz:=5.0
```

---

## 🐛 故障排查

### 问题1: "Failed to open OBS virtual camera"

**原因**：虚拟摄像头模块未加载或 OBS 未启动虚拟摄像头

**解决方案**：
```bash
# 1. 加载模块
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 2. 验证设备
ls -l /dev/video10

# 3. 在 OBS 中启动虚拟摄像头
```

### 问题2: "Failed to initialize ROS2 receiver"

**原因**：ROS2 环境未配置或 bridge 未运行

**解决方案**：
```bash
# 1. Source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 2. 启动 bridge
./start_bridge.sh

# 3. 验证话题
ros2 topic list | grep liftoff
```

### 问题3: 遥控器数据全是零

**原因**：Liftoff 未启用 UDP 输出或 bridge 未接收数据

**解决方案**：
1. 在 Liftoff 中启用 UDP 输出：
   - Settings → Extras → UDP Output → Enable
   - Host: 127.0.0.1, Port: 30001

2. 确认 RadioMaster 在 Joystick 模式

3. 在 Liftoff 中移动摇杆，验证 bridge 输出：
   ```bash
   ros2 topic echo /liftoff/rc
   ```

### 问题4: 画面捕获是黑屏

**原因**：OBS 源配置错误或虚拟摄像头未启动

**解决方案**：
1. 确认 OBS 中的窗口捕获源显示 Liftoff 画面
2. 点击 "启动虚拟摄像头" 按钮
3. 测试捕获：
   ```bash
   python3 -m deploy.screen_capture --obs
   ```

### 问题5: 采集频率达不到目标 FPS

**原因**：系统性能不足或其他程序占用资源

**解决方案**：
1. 降低采集帧率：`--fps 30`
2. 降低图像分辨率：`--image-size 224`
3. 关闭其他程序
4. 使用 GPU 加速（OBS 设置）

---

## 📈 最佳实践

### 数据采集建议

1. **多样化场景**：
   - 不同地图（室内、室外、竞速赛道）
   - 不同光照条件
   - 不同飞行高度和速度

2. **动作多样性**：
   - 基础飞行（起飞、降落、悬停）
   - 机动动作（翻滚、倒飞、急转）
   - 穿越障碍
   - 跟踪目标

3. **数据质量**：
   - 避免碰撞或失控
   - 保持流畅的操作
   - 每个 episode 10-30 秒
   - 收集 50-100+ episodes

4. **文件管理**：
   - 按日期或场景分文件夹
   - 定期备份数据
   - 记录采集条件（README）

### 性能优化

1. **OBS 设置**：
   - 输出分辨率：224x224 或 640x480
   - 编码器：使用硬件编码（NVENC）
   - 比特率：中等质量即可

2. **系统设置**：
   - 关闭不必要的后台程序
   - 使用性能模式（笔记本）
   - 确保散热良好

3. **存储**：
   - 使用 SSD 存储数据集
   - 预留足够空间（1小时 ≈ 10-20 GB）

---

## 📚 相关文档

- [OBS_SETUP_GUIDE.md](OBS_SETUP_GUIDE.md) - OBS 虚拟摄像头设置
- [QUICKSTART.md](QUICKSTART.md) - 项目快速开始
- [liftoff_bridge_ros2.py](liftoff_bridge_ros2.py) - ROS2 Bridge 源码

---

## 🎓 训练模型

数据采集完成后，可以训练模型：

```bash
# 预处理数据集
python -m train.preprocess_dataset \
  --input-dir ./dataset/my_flights \
  --output-dir ./dataset/processed

# 训练模型
python -m train.train_policy \
  --dataset ./dataset/processed \
  --policy toge \
  --epochs 100 \
  --batch-size 32
```

详见训练文档。

---

**最后更新**：2025-10-20
**版本**：v2.0
**作者**：AI Drone Team
