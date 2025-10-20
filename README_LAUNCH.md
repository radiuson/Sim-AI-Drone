# 🚀 快速启动参考

## 三种启动方式

### 1️⃣ 完全自动（最简单）

```bash
./start_full_system.sh
```

**自动完成**：
- ✅ 加载虚拟摄像头
- ✅ 启动 OBS
- ✅ 启动 ROS2 Bridge
- ✅ 启动数据采集

**你需要**：
1. 在 OBS 中添加窗口捕获
2. 启动虚拟摄像头
3. 启动 Liftoff

---

### 2️⃣ 半自动（推荐日常使用）

```bash
# 先启动 OBS
obs &

# 在 OBS 中配置好窗口捕获和虚拟摄像头

# 然后启动数据采集
./start_data_collection.sh ./dataset/my_flights
```

---

### 3️⃣ 手动启动（完全控制）

```bash
# 终端 1: 加载虚拟摄像头
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 终端 2: 启动 OBS
obs

# 终端 3: 启动 Bridge
./start_bridge.sh

# 终端 4: 启动数据采集
python -m record.liftoff_capture --output-dir ./dataset/flights
```

---

## 🎮 录制控制

| 操作 | RadioMaster | 键盘 |
|------|------------|------|
| 开始录制 | **SH ↑** | `r` |
| 停止录制 | **SA ↑** | `s` |
| 紧急停止 | **BTN_SOUTH** | - |
| 退出 | `Ctrl+C` | `q` |

---

## 📊 系统状态检查

```bash
# 检查虚拟摄像头
ls -l /dev/video10

# 检查 ROS2 话题
ros2 topic list | grep liftoff

# 检查 RadioMaster
ls -l /dev/input/js0

# 查看日志
tail -f logs/bridge_*.log
tail -f logs/capture_*.log
```

---

## 🐛 快速修复

| 问题 | 解决方案 |
|------|---------|
| 没有 `/dev/video10` | `sudo modprobe v4l2loopback ...` |
| ROS2 话题不存在 | 重启 `./start_bridge.sh` |
| 遥控器无反应 | `ls /dev/input/js0` 检查连接 |
| OBS 黑屏 | 检查窗口捕获源配置 |

---

## 📚 详细文档

- **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - 完整启动指南
- **[QUICK_START_RECORDING.md](QUICK_START_RECORDING.md)** - 录制指南
- **[DATA_COLLECTION_GUIDE.md](DATA_COLLECTION_GUIDE.md)** - 详细文档

---

## ⚡ 最快开始

```bash
# 只需两条命令！
./start_full_system.sh
# 然后在 OBS 中配置，按 Enter 继续
# 启动 Liftoff，开始飞行和录制！
```

🎉 就是这么简单！
