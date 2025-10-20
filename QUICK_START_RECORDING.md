# 快速开始 - 数据采集

使用 OBS + ROS2 + RadioMaster 进行全自动数据采集。

## 🚀 一键启动

### 第一步：环境准备（首次运行）

```bash
cd /home/ihpc/code/ai-drone

# 安装依赖
./setup_recording.sh

# 或手动安装
pip install inputs numpy pandas pillow opencv-python
```

### 第二步：启动系统组件

**终端 1 - 加载虚拟摄像头**：
```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"
```

**终端 2 - 启动 OBS**：
```bash
obs &
# 然后在 OBS 中：
# 1. 添加"窗口捕获"源 → 选择 Liftoff
# 2. 点击"启动虚拟摄像头"
```

**终端 3 - 启动 Liftoff**：
- 确保在 Settings → Extras → UDP Output 中启用
- Host: 127.0.0.1, Port: 30001

**终端 4 - 启动 ROS2 Bridge**：
```bash
cd /home/ihpc/code/ai-drone
./start_bridge.sh
```

**终端 5 - 启动数据采集**：
```bash
cd /home/ihpc/code/ai-drone

# 使用默认设置（推荐）
python -m record.liftoff_capture \
  --output-dir ./dataset/my_flights
```

---

## 🎮 使用 RadioMaster 控制录制

### 控制方式

| 操作 | RadioMaster 控制 | 说明 |
|------|-----------------|------|
| **开始录制** | **SH 开关向上** | 开始新的 episode |
| **停止录制** | **SA 开关向上** | 保存当前 episode |
| **紧急停止** | **BTN_SOUTH 按钮** | 放弃当前 episode |

### 录制流程

1. **准备飞行**：
   - 在 Liftoff 中选择地图
   - 确认所有系统正常运行

2. **开始录制**：
   - 将 **SH 开关向上拨**
   - 看到终端显示：`📹 Starting episode 0`

3. **飞行操作**：
   - 使用 RadioMaster 手动飞行
   - 建议每个 episode 10-30 秒

4. **停止录制**：
   - 将 **SA 开关向上拨**
   - 看到：`✓ Saved episode 0: XX frames`

5. **继续录制**：
   - 重复步骤 2-4，录制更多 episodes
   - 每次会自动创建新的 episode

6. **结束**：
   - 按 `Ctrl+C` 退出
   - 元数据自动保存

### 紧急情况

如果飞行失控或想放弃当前录制：
- **按下 BTN_SOUTH 按钮**（紧急停止）
- 当前 episode 会被丢弃
- 系统立即准备好录制下一个 episode

---

## 📊 数据集位置

```
dataset/my_flights/
├── videos/
│   ├── episode_000000_frame_000000.png
│   └── ...
├── data/
│   ├── episode_000000.parquet
│   └── ...
└── meta/
    └── info.json
```

---

## ⚙️ 自定义配置

### 更改输出目录

```bash
python -m record.liftoff_capture \
  --output-dir ./dataset/outdoor_flights
```

### 更改帧率

```bash
python -m record.liftoff_capture \
  --output-dir ./dataset/high_fps \
  --fps 60
```

### 禁用遥控器控制（使用键盘）

```bash
python -m record.liftoff_capture \
  --output-dir ./dataset/manual \
  --no-gamepad
```

然后使用键盘命令：
- `r` - 开始录制
- `s` - 停止录制
- `q` - 退出

---

## 🐛 故障排查

### 问题1: "Gamepad controller not available"

**解决方案**：
```bash
pip install inputs
```

### 问题2: 遥控器按键无反应

**检查**：
1. RadioMaster 是否在 Joystick 模式
2. 检查设备：`ls -la /dev/input/js0`
3. 测试遥控器输入：
   ```bash
   python3 -c "from inputs import get_gamepad; print(get_gamepad())"
   ```

### 问题3: "Failed to open OBS virtual camera"

**解决方案**：
```bash
# 1. 加载模块
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 2. 确认 OBS 虚拟摄像头已启动
```

### 问题4: "Failed to initialize ROS2 receiver"

**解决方案**：
```bash
# 1. 启动 bridge
./start_bridge.sh

# 2. 验证话题
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep liftoff
```

---

## 📈 最佳实践

### 录制建议

1. **热身飞行**：录制前先飞几分钟熟悉手感
2. **多样性**：不同地图、不同动作、不同速度
3. **质量优先**：只保存流畅的飞行，失控的用紧急停止丢弃
4. **适度长度**：每个 episode 10-30 秒最佳

### 数据管理

1. **分类存储**：
   ```bash
   --output-dir ./dataset/indoor_flights
   --output-dir ./dataset/outdoor_flights
   --output-dir ./dataset/acrobatic_flights
   ```

2. **定期备份**：
   ```bash
   tar -czf dataset_backup_$(date +%Y%m%d).tar.gz dataset/
   ```

3. **检查数据**：
   ```bash
   ls -lh dataset/my_flights/data/*.parquet
   du -sh dataset/my_flights/
   ```

---

## 🎓 下一步

数据采集完成后：

1. **训练模型**：参见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. **部署推理**：参见 [OBS_SETUP_GUIDE.md](OBS_SETUP_GUIDE.md)

---

**最后更新**：2025-10-20
**版本**：v2.0
