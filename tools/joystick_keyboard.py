#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
键盘控制虚拟游戏手柄 - 用于测试和手动控制Liftoff

功能：
- 通过键盘实时控制虚拟joystick
- 支持增量控制和瞬时控制两种模式
- 显示当前轴的状态

使用前准备：
1. 安装依赖:
   pip install evdev pynput

2. 添加用户到input组（避免需要root）:
   sudo usermod -a -G input $USER
   # 然后重新登录

3. 加载uinput模块：
   sudo modprobe uinput

用法：
  # 启动键盘控制
  python deploy/keyboard_joystick.py

按键映射：
  W/S     - 油门 Throttle (上/下)
  A/D     - 偏航 Yaw (左/右)
  I/K     - 俯仰 Pitch (前/后)
  J/L     - 横滚 Roll (左/右)

  Space   - 重置所有轴到中心位置
  T       - 解锁（ARM）- 油门设为最小值2秒
  ESC/Q   - 退出

  [ / ]   - 降低/提高灵敏度

模式：
  - 增量模式（默认）：按键增加/减少轴的值，松开保持
  - 瞬时模式：按住键轴移动，松开回中心
"""

import sys
import time
import threading
from collections import defaultdict

try:
    from pynput import keyboard
    from pynput.keyboard import Key
except ImportError:
    print("❌ pynput not installed. Please run: pip install pynput")
    sys.exit(1)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deploy.virtual_joystick import VirtualJoystick
except ImportError:
    print("❌ Cannot import VirtualJoystick")
    print("   Make sure virtual_joystick.py is in deploy/ directory")
    sys.exit(1)


class KeyboardJoystick:
    """键盘控制的虚拟joystick"""

    def __init__(self, mode='incremental', sensitivity=0.05):
        """
        初始化键盘控制器

        Args:
            mode: 'incremental' 或 'instant'
                  incremental: 按键增加值，松开保持
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

        # 虚拟joystick
        self.js = VirtualJoystick()

        # 控制循环
        self.running = True
        self.lock = threading.Lock()

        # 解锁状态
        self.arming = False
        self.arm_start_time = 0.0
        self.arm_duration = 2.0  # 解锁持续时间

        # 按键映射
        self.key_map = {
            # 油门 (Throttle)
            'w': ('throttle', 1),
            's': ('throttle', -1),
            # 偏航 (Yaw)
            'a': ('yaw', -1),
            'd': ('yaw', 1),
            # 俯仰 (Pitch)
            'i': ('pitch', 1),
            'k': ('pitch', -1),
            # 横滚 (Roll)
            'j': ('roll', -1),
            'l': ('roll', 1),
        }

        print()
        print("=" * 70)
        print("Keyboard Joystick Controller")
        print("=" * 70)
        print()
        print(f"Mode: {mode}")
        print(f"Sensitivity: {sensitivity}")
        print()
        print("Controls:")
        print("  W/S     - Throttle (Up/Down)")
        print("  A/D     - Yaw (Left/Right)")
        print("  I/K     - Pitch (Forward/Back)")
        print("  J/L     - Roll (Left/Right)")
        print()
        print("  Space   - Reset all axes to center")
        print("  T       - ARM (hold throttle at minimum for 2s)")
        print("  [ / ]   - Decrease/Increase sensitivity")
        print("  ESC/Q   - Quit")
        print()
        print("Press keys to control the virtual joystick...")
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
            self.throttle = -1.0  # 油门最小
            self.yaw = 0.0
            self.pitch = 0.0
            self.roll = 0.0
        print("\r🔓 ARMING... (hold throttle at minimum for 2s)", end='', flush=True)

    def reset(self):
        """重置所有轴到中心"""
        with self.lock:
            self.arming = False
            self.throttle = 0.0
            self.yaw = 0.0
            self.pitch = 0.0
            self.roll = 0.0
        print("\rReset all axes           ", end='', flush=True)

    def send_current_state(self):
        """发送当前状态到虚拟joystick"""
        with self.lock:
            action = [self.throttle, self.yaw, self.pitch, self.roll]

        self.js.send_action(action)

    def display_state(self):
        """显示当前状态"""
        with self.lock:
            t, y, p, r = self.throttle, self.yaw, self.pitch, self.roll

        # 创建可视化进度条
        def bar(value, width=20):
            center = width // 2
            pos = int((value + 1) / 2 * width)  # 映射 [-1,1] 到 [0,width]

            bar_str = ""
            for i in range(width):
                if i == center:
                    bar_str += "|"
                elif i == pos:
                    bar_str += "●"
                elif min(i, pos) < center < max(i, pos):
                    bar_str += "─"
                else:
                    bar_str += "─"
            return bar_str

        # 格式化输出
        output = (
            f"T:{bar(t)} {t:+.2f}  "
            f"Y:{bar(y)} {y:+.2f}  "
            f"P:{bar(p)} {p:+.2f}  "
            f"R:{bar(r)} {r:+.2f}"
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
                        print("\r✓ ARMED! Drone is ready to fly.                    ", end='', flush=True)
                        time.sleep(0.5)  # 显示消息
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
        """启动控制器"""
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
        """停止控制器"""
        self.running = False
        print("\n")
        print("Stopping...")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.js.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Keyboard-controlled Virtual Joystick for Liftoff",
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        with KeyboardJoystick(mode=args.mode, sensitivity=args.sensitivity) as kj:
            kj.start()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
