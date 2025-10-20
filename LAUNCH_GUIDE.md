# 一键启动指南

三种启动方式，从简单到完全自动化。

---

## 🚀 方式 1：完全自动启动（推荐）

**一条命令启动所有组件**（包括 OBS）

```bash
cd /home/ihpc/code/ai-drone

./start_full_system.sh [输出目录] [帧率]
```

### 示例

```bash
# 使用默认设置
./start_full_system.sh

# 自定义输出目录
./start_full_system.sh ./dataset/my_flights

# 自定义输出目录和帧率
./start_full_system.sh ./dataset/high_fps 60
```

### 自动完成的任务

- ✅ 加载 v4l2loopback 虚拟摄像头
- ✅ 检查 ROS2 环境
- ✅ 启动 OBS Studio
- ✅ 启动 ROS2 Bridge
- ✅ 启动数据采集器
- ✅ 自动清理（Ctrl+C 时）

### 手动步骤

启动后你需要：
1. 在 OBS 中添加"窗口捕获"源（选择 Liftoff）
2. 点击"启动虚拟摄像头"
3. 按 Enter 继续

---

## 🎯 方式 2：自动启动（不含 OBS）

**适合已经手动启动 OBS 的情况**

```bash
cd /home/ihpc/code/ai-drone

./start_data_collection.sh [输出目录] [帧率] [图像大小] [捕获方法]
```

### 示例

```bash
# 使用默认设置（OBS 捕获）
./start_data_collection.sh

# 自定义参数
./start_data_collection.sh ./dataset/flights 30 224 obs

# 使用 MSS 捕获
./start_data_collection.sh ./dataset/flights 30 224 mss
```

### 自动完成的任务

- ✅ 检查虚拟摄像头
- ✅ 检查 ROS2 环境
- ✅ 检查 RadioMaster
- ✅ 检查 OBS 状态
- ✅ 启动 ROS2 Bridge
- ✅ 启动数据采集器
- ✅ 保存日志到 `logs/` 目录

---

## 📋 方式 3：ROS2 Launch 文件

**标准 ROS2 方式**（需要将项目配置为 ROS2 包）

```bash
cd /home/ihpc/code/ai-drone

ros2 launch ai_drone data_collection.launch.py \
  output_dir:=./dataset/flights \
  fps:=30 \
  image_size:=224 \
  capture_method:=obs
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `output_dir` | `./dataset/liftoff_data` | 输出目录 |
| `fps` | `30` | 采集帧率 |
| `image_size` | `224` | 图像尺寸 |
| `capture_method` | `obs` | 捕获方法 (obs/mss) |
| `obs_device` | `/dev/video10` | OBS 设备路径 |
| `enable_gamepad` | `true` | 启用遥控器控制 |
| `bindings_file` | `record/control_bindings.json` | 控制绑定文件 |

---

## 📊 对比

| 特性 | 完全自动 | 自动启动 | ROS2 Launch |
|------|---------|---------|-------------|
| **OBS 自动启动** | ✅ | ❌ | ❌ |
| **日志保存** | ✅ | ✅ | ✅ |
| **参数配置** | 简单 | 简单 | 完整 |
| **依赖 ROS2 包** | ❌ | ❌ | ✅ |
| **推荐用途** | 首次使用 | 日常使用 | 高级用户 |

---

## 🎮 使用流程

### 使用完全自动启动

```bash
# 1. 启动系统
./start_full_system.sh

# 2. 等待 OBS 启动（10 秒）
# 3. 在 OBS 中配置：
#    - 添加"窗口捕获"源
#    - 选择 Liftoff 窗口
#    - 点击"启动虚拟摄像头"
# 4. 按 Enter 继续

# 5. 系统准备完成，看到：
#    🚀 System Ready!
#    🎮 RadioMaster Controls:
#      - SH switch UP   → Start recording ▶️
#      - SA switch UP   → Stop recording ⏹️
#      - BTN_SOUTH      → Emergency stop 🛑

# 6. 启动 Liftoff（确保 UDP 输出已启用）

# 7. 使用 RadioMaster 控制录制：
#    - SH 向上 → 开始录制
#    - SA 向上 → 停止录制

# 8. 结束时按 Ctrl+C
```

---

## 🐛 故障排查

### 问题1: "Failed to load v4l2loopback"

**解决方案**：
```bash
# 检查模块是否可用
modinfo v4l2loopback

# 手动加载
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 验证
ls -l /dev/video10
```

### 问题2: "ROS2 not found"

**解决方案**：
```bash
# 手动 source ROS2 环境
source /opt/ros/jazzy/setup.bash

# 验证
echo $ROS_DISTRO
```

### 问题3: "OBS failed to start"

**解决方案**：
- 手动启动 OBS：`obs`
- 或者跳过 OBS 自动启动，使用方式 2

### 问题4: "Bridge failed to start"

**解决方案**：
```bash
# 查看日志
cat logs/bridge_*.log

# 检查端口占用
netstat -ulnp | grep 30001

# 手动测试 bridge
python3 liftoff_bridge_ros2.py
```

### 问题5: 数据采集无输出

**解决方案**：
```bash
# 查看日志
cat logs/capture_*.log

# 检查虚拟摄像头
v4l2-ctl --list-devices

# 检查 ROS2 话题
ros2 topic list | grep liftoff
ros2 topic echo /liftoff/rc
```

---

## 📁 日志文件

所有启动脚本都会保存日志到 `logs/` 目录：

```
logs/
├── bridge_20251020_153045.log   # ROS2 Bridge 日志
└── capture_20251020_153045.log  # 数据采集日志
```

查看实时日志：
```bash
# Bridge 日志
tail -f logs/bridge_*.log

# Capture 日志
tail -f logs/capture_*.log
```

---

## 🔧 高级配置

### 自定义 OBS 启动参数

编辑 `start_full_system.sh`，修改第 79 行：
```bash
obs --minimize-to-tray --startreplaybuffer > /dev/null 2>&1 &
```

### 自定义 Bridge 参数

编辑 `liftoff_bridge_ros2.py` 或在启动脚本中添加参数。

### 自定义采集参数

编辑启动脚本，修改 `python3 -m record.liftoff_capture` 命令的参数。

---

## 📚 相关文档

- [QUICK_START_RECORDING.md](QUICK_START_RECORDING.md) - 快速开始
- [DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md) - 详细指南
- [OBS_SETUP_GUIDE.md](OBS_SETUP_GUIDE.md) - OBS 配置

---

## 💡 推荐工作流

### 首次使用

```bash
# 1. 设置环境（一次性）
./setup_recording.sh

# 2. 使用完全自动启动
./start_full_system.sh
```

### 日常使用

```bash
# 1. 手动启动 OBS（保持配置）
obs &

# 2. 在 OBS 中启动虚拟摄像头

# 3. 使用自动启动脚本
./start_data_collection.sh ./dataset/today_flights
```

### 批量采集

```bash
# 为不同场景创建不同数据集
./start_data_collection.sh ./dataset/indoor
./start_data_collection.sh ./dataset/outdoor
./start_data_collection.sh ./dataset/acrobatic
```

---

**最后更新**：2025-10-20
**版本**：v2.0
