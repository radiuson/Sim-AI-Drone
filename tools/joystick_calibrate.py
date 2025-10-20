#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TOGE虚拟遥控器校准工具 - 用于测试和校准TOGE-AI-Controller

功能：
- 通过键盘手动控制TOGE虚拟遥控器
- 测试遥控器输出是否正确
- 在Liftoff中校准通道映射
- 支持增量控制模式（推荐）和瞬时控制模式

使用前准备：
1. 安装依赖:
   pip install evdev pynput

2. 添加用户到input组:
   sudo usermod -a -G input $USER
   # 然后重新登录

3. 加载uinput模块:
   sudo modprobe uinput

用法：
  # 启动校准工具
  python deploy/calibrate_toge_joystick.py

  # 指定模式和灵敏度
  python deploy/calibrate_toge_joystick.py --mode instant --sensitivity 0.08

按键映射：
  W/S     - 油门 Throttle (上/下)
  A/D     - 偏航 Yaw (左/右)
  I/K     - 俯仰 Pitch (前/后)
  J/L     - 横滚 Roll (左/右)

  Space   - 重置所有轴到中心位置 (0,0,0,0)
  T       - 解锁（ARM）- 油门设为最小值2秒
  R       - 显示当前原始值
  ESC/Q   - 退出

  [ / ]   - 降低/提高灵敏度

校准步骤：
1. 运行此脚本
2. 在Liftoff中进入: Settings → Controls → Add Controller
3. 选择 'TOGE-AI-Controller'
4. 按键盘键逐个测试和校准通道:
   - 按W/S测试油门通道 (Throttle)
   - 按A/D测试偏航通道 (Yaw)
   - 按I/K测试俯仰通道 (Pitch)
   - 按J/L测试横滚通道 (Roll)
5. 在Liftoff中为每个通道分配正确的功能
6. 测试解锁: 按T键，观察油门是否保持最小值2秒
"""

import sys
import time
import threading
from pathlib import Path

try:
    from pynput import keyboard
    from pynput.keyboard import Key
except ImportError:
    print("❌ pynput not installed. Please run: pip install pynput")
    sys.exit(1)

# 导入虚拟遥控器模块
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from deploy.virtual_joystick import VirtualJoystick
except ImportError:
    print("❌ Cannot import VirtualJoystick")
    print("   Make sure virtual_joystick.py is in deploy/ directory")
    sys.exit(1)


class TOGEJoystickCalibrator:
    """TOGE虚拟遥控器校准工具"""

    def __init__(self, mode='incremental', sensitivity=0.05):
        """
        初始化校准工具

        Args:
            mode: 'incremental' 或 'instant'
                  incremental: 按键增加值，松开保持（推荐用于校准）
                  instant: 按住移动，松开归零
            sensitivity: 灵敏度 (0.01 ~ 0.2)
        """
        self.mode = mode
        self.sensitivity = sensitivity

        # 当前轴的值 [-1.0, 1.0]
        self.throttle = 0.0
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # 当前按下的按键
        self.pressed_keys = set()

        # 创建TOGE虚拟遥控器
        print("Creating TOGE-AI-Controller virtual joystick...")
        self.js = VirtualJoystick(device_name="TOGE-AI-Controller")
        print("✓ Virtual joystick created")

        # 控制循环
        self.running = True
        self.lock = threading.Lock()

        # 解锁状态
        self.arming = False
        self.arm_start_time = 0.0
        self.arm_duration = 2.0  # 解锁持续时间

        # 统计
        self.send_count = 0
        self.start_time = time.time()

        # 按键映射
        self.key_map = {
            # 油门 (Throttle) - ABS_X
            'w': ('throttle', 1),
            's': ('throttle', -1),
            # 偏航 (Yaw) - ABS_Y
            'a': ('yaw', -1),
            'd': ('yaw', 1),
            # 俯仰 (Pitch) - ABS_Z
            'i': ('pitch', 1),
            'k': ('pitch', -1),
            # 横滚 (Roll) - ABS_RX
            'j': ('roll', -1),
            'l': ('roll', 1),
        }

        self._print_header()

    def _print_header(self):
        """打印欢迎信息"""
        print()
        print("=" * 80)
        print("TOGE Virtual Joystick Calibration Tool")
        print("=" * 80)
        print()
        print(f"Device Name: TOGE-AI-Controller")
        print(f"Mode: {self.mode}")
        print(f"Sensitivity: {self.sensitivity}")
        print()
        print("Channel Mapping (as configured in virtual_joystick.py):")
        print("  ABS_X  (Axis 0) → Throttle")
        print("  ABS_Y  (Axis 1) → Yaw")
        print("  ABS_Z  (Axis 2) → Pitch")
        print("  ABS_RX (Axis 3) → Roll")
        print()
        print("Keyboard Controls:")
        print("  W/S     - Throttle (Up/Down)         [ABS_X]")
        print("  A/D     - Yaw (Left/Right)           [ABS_Y]")
        print("  I/K     - Pitch (Forward/Back)       [ABS_Z]")
        print("  J/L     - Roll (Left/Right)          [ABS_RX]")
        print()
        print("  Space   - Reset all axes to center (0.0)")
        print("  T       - ARM (hold throttle at minimum for 2s)")
        print("  R       - Show raw values")
        print("  [ / ]   - Decrease/Increase sensitivity")
        print("  ESC/Q   - Quit")
        print()
        print("Calibration Steps:")
        print("  1. Start Liftoff")
        print("  2. Go to: Settings → Controls → Add Controller")
        print("  3. Select 'TOGE-AI-Controller' from the list")
        print("  4. Test each axis by pressing keys (W/S, A/D, I/K, J/L)")
        print("  5. Assign channels in Liftoff based on which keys move what")
        print("  6. Test ARM sequence with 'T' key")
        print()
        print("Press keys to control the virtual joystick...")
        print("=" * 80)
        print()

    def on_press(self, key):
        """按键按下事件"""
        try:
            # 获取按键字符
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()

                # 解锁键
                if key_char == 't':
                    self.start_arming()
                # 显示原始值
                elif key_char == 'r':
                    self.show_raw_values()
                else:
                    self.pressed_keys.add(key_char)

            elif key == Key.space:
                self.reset()
            elif key == Key.esc:
                self.stop()

        except Exception as e:
            pass

    def on_release(self, key):
        """按键释放事件"""
        try:
            # 处理特殊按键
            if key == Key.esc or (hasattr(key, 'char') and key.char == 'q'):
                self.stop()
                return False  # 停止监听

            # 获取按键字符
            if hasattr(key, 'char') and key.char:
                key_char = key.char.lower()

                # 灵敏度调节
                if key_char == '[':
                    self.sensitivity = max(0.01, self.sensitivity - 0.01)
                    print(f"\rSensitivity: {self.sensitivity:.2f}  ", end='', flush=True)
                    return
                elif key_char == ']':
                    self.sensitivity = min(0.2, self.sensitivity + 0.01)
                    print(f"\rSensitivity: {self.sensitivity:.2f}  ", end='', flush=True)
                    return

                # 移除按键
                if key_char in self.pressed_keys:
                    self.pressed_keys.discard(key_char)

                    # 瞬时模式：松开按键时对应轴归零
                    if self.mode == 'instant' and key_char in self.key_map:
                        axis, _ = self.key_map[key_char]
                        with self.lock:
                            setattr(self, axis, 0.0)

        except Exception as e:
            pass

    def update_axes(self):
        """更新轴的值（增量模式）"""
        if self.mode != 'incremental':
            return

        with self.lock:
            for key_char in self.pressed_keys:
                if key_char in self.key_map:
                    axis, direction = self.key_map[key_char]
                    current_value = getattr(self, axis)
                    new_value = current_value + direction * self.sensitivity
                    # 限制在 [-1, 1]
                    new_value = max(-1.0, min(1.0, new_value))
                    setattr(self, axis, new_value)

    def update_axes_instant(self):
        """更新轴的值（瞬时模式）"""
        if self.mode != 'instant':
            return

        # 先重置所有轴
        with self.lock:
            self.throttle = 0.0
            self.yaw = 0.0
            self.pitch = 0.0
            self.roll = 0.0

            # 根据当前按下的键设置值
            for key_char in self.pressed_keys:
                if key_char in self.key_map:
                    axis, direction = self.key_map[key_char]
                    # 瞬时模式直接设置为最大值
                    setattr(self, axis, float(direction))

    def start_arming(self):
        """开始解锁序列"""
        with self.lock:
            self.arming = True
            self.arm_start_time = time.time()
            # 注意：TOGE使用 [throttle, yaw, pitch, roll] 顺序
            # 解锁时油门设为最小值 -1.0
            self.throttle = -1.0
            self.yaw = 0.0
            self.pitch = 0.0
            self.roll = 0.0
        print("\r🔓 ARMING... (Throttle at minimum for 2s)                              ", end='', flush=True)

    def reset(self):
        """重置所有轴到中心"""
        with self.lock:
            self.arming = False
            self.throttle = 0.0
            self.yaw = 0.0
            self.pitch = 0.0
            self.roll = 0.0
        print("\r✓ Reset all axes to center (0.0)                                       ", end='', flush=True)

    def show_raw_values(self):
        """显示原始数值"""
        with self.lock:
            t, y, p, r = self.throttle, self.yaw, self.pitch, self.roll

        print(f"\r📊 Raw Values: Throttle={t:+.3f}, Yaw={y:+.3f}, Pitch={p:+.3f}, Roll={r:+.3f}    ", end='', flush=True)

    def send_current_state(self):
        """发送当前状态到虚拟joystick"""
        with self.lock:
            # TOGE动作顺序: [throttle, yaw, pitch, roll]
            action = [self.throttle, self.yaw, self.pitch, self.roll]

        self.js.send_action(action)
        self.send_count += 1

    def display_state(self):
        """显示当前状态"""
        with self.lock:
            t, y, p, r = self.throttle, self.yaw, self.pitch, self.roll

        # 创建可视化进度条
        def bar(value, width=15):
            """创建进度条 [-1, 1] -> 可视化"""
            center = width // 2
            pos = int((value + 1) / 2 * width)  # 映射 [-1,1] 到 [0,width]
            pos = max(0, min(width - 1, pos))  # 确保在范围内

            bar_str = ""
            for i in range(width):
                if i == center:
                    if i == pos:
                        bar_str += "●"  # 在中心位置
                    else:
                        bar_str += "|"  # 中心标记
                elif i == pos:
                    bar_str += "●"  # 当前位置
                elif (i < center < pos) or (pos < center < i):
                    bar_str += "─"  # 填充
                else:
                    bar_str += " "  # 空白
            return bar_str

        # 计算运行时间和频率
        elapsed = time.time() - self.start_time
        freq = self.send_count / elapsed if elapsed > 0 else 0

        # 格式化输出
        output = (
            f"T[{bar(t)}]{t:+.2f}  "
            f"Y[{bar(y)}]{y:+.2f}  "
            f"P[{bar(p)}]{p:+.2f}  "
            f"R[{bar(r)}]{r:+.2f}  "
            f"| {freq:.1f}Hz"
        )

        print(f"\r{output}", end='', flush=True)

    def control_loop(self):
        """主控制循环"""
        last_update = time.time()
        update_rate = 50  # Hz
        update_interval = 1.0 / update_rate

        while self.running:
            current_time = time.time()

            if current_time - last_update >= update_interval:
                # 检查解锁状态
                if self.arming:
                    elapsed = current_time - self.arm_start_time
                    if elapsed >= self.arm_duration:
                        # 解锁完成
                        with self.lock:
                            self.arming = False
                            self.throttle = 0.0  # 解锁完成后归零
                        print("\r✅ ARMED! Drone is ready to fly. Throttle reset to 0.0.              ", end='', flush=True)
                        time.sleep(1.0)  # 显示消息
                    # 解锁期间保持油门最小
                else:
                    # 正常控制：更新轴值
                    if self.mode == 'incremental':
                        self.update_axes()
                    else:
                        self.update_axes_instant()

                # 发送到虚拟joystick
                self.send_current_state()

                # 显示状态
                if not self.arming:  # 解锁期间不覆盖ARMING消息
                    self.display_state()

                last_update = current_time

            time.sleep(0.001)  # 避免CPU占用过高

    def start(self):
        """启动校准工具"""
        # 启动控制循环线程
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()

        # 启动键盘监听（阻塞）
        with keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        ) as listener:
            listener.join()

    def stop(self):
        """停止校准工具"""
        self.running = False
        print("\n")
        print("=" * 80)
        print("Stopping calibration tool...")
        print(f"Total commands sent: {self.send_count}")
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            print(f"Average frequency: {self.send_count / elapsed:.1f} Hz")
        print("=" * 80)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.js.close()
        print("✓ Virtual joystick closed")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TOGE Virtual Joystick Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start with default settings (incremental mode, sensitivity 0.05)
  python deploy/calibrate_toge_joystick.py

  # Use instant mode (press to move, release to center)
  python deploy/calibrate_toge_joystick.py --mode instant

  # Adjust sensitivity
  python deploy/calibrate_toge_joystick.py --sensitivity 0.08

Calibration Workflow:
  1. Run this script
  2. In Liftoff: Settings → Controls → Add Controller
  3. Select 'TOGE-AI-Controller'
  4. Press keyboard keys and observe which channels move in Liftoff
  5. Map channels: Throttle, Yaw, Pitch, Roll
  6. Test ARM with 'T' key (throttle should go to minimum for 2s)
  7. Save configuration in Liftoff

Channel Mapping Reference:
  ABS_X  → Throttle (W/S keys)
  ABS_Y  → Yaw      (A/D keys)
  ABS_Z  → Pitch    (I/K keys)
  ABS_RX → Roll     (J/L keys)
        """
    )

    parser.add_argument(
        '--mode',
        choices=['incremental', 'instant'],
        default='incremental',
        help='Control mode (default: incremental)'
    )

    parser.add_argument(
        '--sensitivity',
        type=float,
        default=0.05,
        help='Sensitivity (0.01 ~ 0.2, default: 0.05)'
    )

    args = parser.parse_args()

    try:
        with TOGEJoystickCalibrator(mode=args.mode, sensitivity=args.sensitivity) as calibrator:
            calibrator.start()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
