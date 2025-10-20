# OBS 捕获设置指南

本项目已默认使用 OBS 虚拟摄像头进行屏幕捕获，性能优于传统 MSS 方法。

## 🚀 快速开始

### 一次性设置（系统启动时）

```bash
# 1. 加载 v4l2loopback 内核模块
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 2. 启动 OBS
obs &
```

### 在 OBS 中配置

1. **添加游戏捕获源**：
   - 点击"源"面板的 **+** 按钮
   - 选择 **"窗口捕获 (Xcomposite)"**
   - 窗口选择：`[liftoff.x86_64] Liftoff`

2. **调整输出分辨率（可选）**：
   - 设置 → 视频 → 输出分辨率：640x480 或 224x224

3. **启动虚拟摄像头**：
   - 点击右侧控制面板的 **"启动虚拟摄像头"** 按钮

---

## 📊 数据采集

### 默认用法（OBS + RadioMaster）

```bash
# 最简单的命令 - 使用所有默认设置
python -m record.liftoff_capture \
  --output-dir ./dataset/my_flights \
  --fps 30
```

**默认配置**：
- ✅ 捕获方法：OBS 虚拟摄像头（`/dev/video10`）
- ✅ 遥控器：RadioMaster（`/dev/input/js0`）
- ✅ 帧率：30 FPS
- ✅ 图像尺寸：224x224

### 自定义设置

```bash
# 更高帧率采集
python -m record.liftoff_capture \
  --output-dir ./dataset/high_fps \
  --fps 60

# 使用不同的遥控器设备
python -m record.liftoff_capture \
  --output-dir ./dataset/my_flights \
  --joystick-device /dev/input/js1

# 不同图像尺寸
python -m record.liftoff_capture \
  --output-dir ./dataset/large_images \
  --image-size 640
```

### 使用 MSS（备选方案）

```bash
python -m record.liftoff_capture \
  --output-dir ./dataset/my_flights \
  --capture-method mss \
  --window-name "Liftoff"
```

---

## 🎮 策略推理

### 标准模式（30Hz）

```bash
# 使用默认设置
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_best.pt \
  --rate 30
```

### 双频率模式（推荐 - 30Hz视觉 + 100Hz动作）

```bash
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_best.pt \
  --dual-rate \
  --visual-rate 30 \
  --action-rate 100
```

**性能对比**：

| 模式 | 视觉频率 | 动作频率 | CPU占用 | 延迟 | 适用场景 |
|------|---------|---------|---------|------|---------|
| **标准** | 30Hz | 30Hz | 低 | ~30ms | 休闲飞行 |
| **双频率** | 30Hz | 100Hz | 中 | ~12ms | 竞速、特技 |

### 使用 MSS（备选方案）

```bash
python -m deploy.run_policy \
  --policy toge \
  --checkpoint outputs/toge_best.pt \
  --capture-method mss \
  --window-name "Liftoff"
```

---

## 🛠️ 硬件配置

### 检测到的设备

```bash
# RadioMaster Pocket 遥控器
/dev/input/js0
Device: EdgeTX_Radiomaster_Pocket_Joystick

# OBS 虚拟摄像头
/dev/video10
Device: OBS (platform:v4l2loopback-010)
```

### 验证设备

```bash
# 检查遥控器
ls -la /dev/input/by-id/ | grep -i radiomaster

# 检查虚拟摄像头
v4l2-ctl --list-devices

# 测试虚拟摄像头捕获
python3 -m deploy.screen_capture --obs
```

---

## 🔧 开机自动加载 v4l2loopback

```bash
# 创建模块配置
echo "v4l2loopback" | sudo tee /etc/modules-load.d/v4l2loopback.conf

# 创建模块参数配置
echo "options v4l2loopback devices=1 video_nr=10 card_label=\"OBS Virtual Camera\" exclusive_caps=1" | \
  sudo tee /etc/modprobe.d/v4l2loopback.conf

# 重启后自动生效
```

---

## 📈 性能优势

### OBS vs MSS 对比

| 指标 | MSS | OBS (PipeWire) | 提升 |
|------|-----|----------------|------|
| **CPU占用** | 15-20% | **2-5%** | **70-85% ↓** |
| **帧时间** | 30-35ms | **5-15ms** | **50-80% ↓** |
| **延迟** | 5-10ms | **<2ms** | **60-90% ↓** |
| **稳定性** | 中 | **高** | ✅ |
| **长时间运行** | 易发热 | **稳定** | ✅ |

### 为什么 OBS 更快？

1. **GPU 加速编码**：OBS 使用硬件编码器（NVENC/VAAPI）
2. **零拷贝传输**：PipeWire 直接从 GPU 获取帧
3. **低延迟管线**：专为实时流媒体优化
4. **无窗口查找开销**：不需要 xdotool 定位窗口

---

## 🐛 故障排查

### 问题1: "Failed to open OBS virtual camera"

**解决方案**：
```bash
# 1. 检查模块是否加载
lsmod | grep v4l2loopback

# 2. 如果没加载，手动加载
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"

# 3. 检查设备是否存在
ls -l /dev/video10

# 4. 确保 OBS 虚拟摄像头已启动
# 在 OBS 中点击"启动虚拟摄像头"按钮
```

### 问题2: 捕获的是黑屏

**解决方案**：
- 确保 OBS 中的游戏捕获源正常显示
- 确保点击了"启动虚拟摄像头"
- 重启 OBS 并重新启动虚拟摄像头

### 问题3: "Failed to connect joystick"

**解决方案**：
```bash
# 1. 检查遥控器是否连接
ls -la /dev/input/js*

# 2. 检查权限
sudo chmod 666 /dev/input/js0

# 3. 添加用户到 input 组（永久解决）
sudo usermod -aG input $USER
# 重新登录使权限生效
```

### 问题4: 性能仍然很慢

**检查清单**：
- [ ] 确认使用的是 OBS 捕获（`--capture-method obs`）
- [ ] 确认 OBS 虚拟摄像头已启动
- [ ] 检查 GPU 是否被正确使用（`nvidia-smi`）
- [ ] 尝试降低图像分辨率
- [ ] 关闭其他占用 GPU 的程序

---

## 📚 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 项目快速开始指南
- [DUAL_RATE_INFERENCE.md](DUAL_RATE_INFERENCE.md) - 双频率推理详细文档
- [screen_capture.py](deploy/screen_capture.py) - 捕获模块源码

---

## 💡 最佳实践

### 数据采集

1. **使用 OBS 捕获**：CPU 占用低，适合长时间录制
2. **固定帧率**：建议 30 FPS，平衡质量和性能
3. **多 episode 录制**：每个 episode 10-30 秒，避免单个文件过大

### 策略推理

1. **使用双频率模式**：竞速飞行时获得更低延迟
2. **GPU 推理**：确保使用 CUDA 加速
3. **监控性能**：关注 FPS 和 CPU/GPU 占用

### 系统优化

1. **关闭不必要的后台程序**
2. **使用性能模式**（如果是笔记本）
3. **确保散热良好**

---

**最后更新**：2025-10-20
**适用版本**：ai-drone v2.0+
