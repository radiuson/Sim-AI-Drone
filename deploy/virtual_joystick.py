#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟游戏手柄/遥控器 - 用于AI策略输出到Liftoff

【修改说明（by ChatGPT）】
- 问题：原始实现中「横滚（Roll）」方向与实际期望相反。
- 修改：在 send_action() 中将 roll 取反（roll = -roll），再写入 ABS_RX。
- 影响：对外部调用接口保持不变——依然使用 action=[throttle, yaw, pitch, roll]，其中
        roll 的约定含义仍是：-1=左滚, 0=水平, 1=右滚。内部自动取反以适配 Liftoff 实际识别方向。
- 文档同步：下方映射说明中已标注 “(Roll 已在代码中取反)”。

功能：
- 创建虚拟的Linux input设备
- 将AI预测的动作 [throttle, yaw, pitch, roll] 转换为joystick输入
- Liftoff可以识别为标准游戏手柄

使用前准备：
1. 安装evdev:
   pip install evdev

2. 添加用户到input组（避免需要root）:
   sudo usermod -a -G input $USER
   # 然后重新登录

3. 加载uinput模块：
   sudo modprobe uinput
   # 或永久添加：
   echo "uinput" | sudo tee /etc/modules-load.d/uinput.conf

用法：
  # 测试虚拟遥控器
  python deploy/virtual_joystick.py --test

  # 从命令行发送动作
  python deploy/virtual_joystick.py --action "0.5,-0.1,0.2,0.0"

  # 作为Python模块使用
  from deploy.virtual_joystick import VirtualJoystick
  js = VirtualJoystick()
  js.send_action([0.5, 0.0, 0.0, 0.0])
"""

import time
import argparse
from typing import List, Tuple
import numpy as np

try:
    from evdev import UInput, ecodes as e, AbsInfo
except ImportError:
    print("❌ evdev not installed. Please run: pip install evdev")
    raise


class VirtualJoystick:
    """虚拟游戏手柄，模拟4轴遥控器"""

    def __init__(self, device_name: str = "AI-Liftoff-Controller"):
        """
        初始化虚拟joystick

        Args:
            device_name: 设备名称，会显示在Liftoff的控制器列表中
        """
        self.device_name = device_name

        # 定义轴的范围 [-1, 1] 映射到 [-32767, 32767]
        # Linux joystick标准范围
        abs_info = AbsInfo(
            value=0,      # 初始值（中间）
            min=-32767,   # 最小值
            max=32767,    # 最大值
            fuzz=0,       # 噪声滤波
            flat=0,       # 死区
            resolution=0  # 分辨率
        )

        # 定义4个轴（油门、偏航、俯仰、横滚）
        capabilities = {
            e.EV_ABS: [
                (e.ABS_X, abs_info),      # 油门 (Throttle)
                (e.ABS_Y, abs_info),      # 偏航 (Yaw)
                (e.ABS_Z, abs_info),      # 俯仰 (Pitch)
                (e.ABS_RX, abs_info),     # 横滚 (Roll) —— 注意：代码内已取反
            ],
            e.EV_KEY: [
                e.BTN_GAMEPAD,  # 标识为游戏手柄
            ],
        }

        try:
            self.ui = UInput(capabilities, name=self.device_name)
            print(f"✓ Virtual joystick created: {self.device_name}")
            print(f"  Device path: {self.ui.device.path}")
            print(f"  Device node: /dev/input/event{self.ui.device.path.split('event')[-1]}")
            print()
            print("⚙️  Configure in Liftoff:")
            print("   Settings → Controls → Add Controller → Select 'AI-Liftoff-Controller'")
            print("   Then map: ABS_X=Throttle, ABS_Y=Yaw, ABS_Z=Pitch, ABS_RX=Roll (Roll 已在代码中取反)")
            print()
        except PermissionError:
            print("❌ Permission denied!")
            print("   Solution 1: Add user to input group:")
            print("     sudo usermod -a -G input $USER")
            print("     # Then logout and login again")
            print()
            print("   Solution 2: Run as root (not recommended):")
            print("     sudo python deploy/virtual_joystick.py")
            raise
        except OSError as os_err:
            if "No such file or directory" in str(os_err):
                print("❌ /dev/uinput not found!")
                print("   Solution: Load uinput module:")
                print("     sudo modprobe uinput")
                print("     # To load automatically on boot:")
                print("     echo 'uinput' | sudo tee /etc/modules-load.d/uinput.conf")
            raise

    def normalize_action(self, action: List[float]) -> List[int]:
        """
        将动作归一化到joystick范围

        Args:
            action: [throttle, yaw, pitch, roll] 范围通常在[-1, 1]

        Returns:
            [int, int, int, int] 范围在[-32767, 32767]
        """
        # 裁剪到[-1, 1]范围
        action = np.clip(action, -1.0, 1.0)

        # 映射到[-32767, 32767]
        normalized = (action * 32767).astype(np.int32)

        return normalized.tolist()

    def send_action(self, action: List[float]):
        """
        发送动作到虚拟joystick

        Args:
            action: [throttle, yaw, pitch, roll] 范围在[-1, 1]
                   throttle: -1=后退, 0=悬停, 1=前进
                   yaw:      -1=左转, 0=不转, 1=右转
                   pitch:    -1=后仰, 0=水平, 1=前倾
                   roll:     -1=左滚, 0=水平, 1=右滚  （对外语义保持不变，内部会取反以适配设备）
        """
        if len(action) != 4:
            raise ValueError(f"Action must have 4 values, got {len(action)}")

        # 归一化
        throttle, yaw, pitch, roll = self.normalize_action(action)

        # **** 修复点：将 Roll 取反后再写入 ****
        roll = -roll

        # 写入事件
        self.ui.write(e.EV_ABS, e.ABS_X, throttle)
        self.ui.write(e.EV_ABS, e.ABS_Y, yaw)
        self.ui.write(e.EV_ABS, e.ABS_Z, pitch)
        self.ui.write(e.EV_ABS, e.ABS_RX, roll)

        # 同步事件（重要！）
        self.ui.syn()

    def reset(self):
        """重置所有轴到中间位置（0）"""
        self.send_action([0.0, 0.0, 0.0, 0.0])

    def close(self):
        """关闭虚拟设备"""
        self.reset()  # 先重置
        time.sleep(0.1)
        self.ui.close()
        print("✓ Virtual joystick closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_joystick():
    """测试虚拟joystick"""
    print("="*70)
    print("Virtual Joystick Test")
    print("="*70)
    print()

    with VirtualJoystick() as js:
        print("Running test sequence...")
        print("Press Ctrl+C to stop")
        print()

        try:
            # 测试序列（语义不变；内部已对 Roll 取反输出）
            test_sequences = [
                ("Center (hover)", [0.0, 0.0, 0.0, 0.0]),
                ("Throttle up", [0.5, 0.0, 0.0, 0.0]),
                ("Yaw left", [0.0, -0.5, 0.0, 0.0]),
                ("Yaw right", [0.0, 0.5, 0.0, 0.0]),
                ("Pitch forward", [0.0, 0.0, 0.5, 0.0]),
                ("Pitch backward", [0.0, 0.0, -0.5, 0.0]),
                ("Roll left", [0.0, 0.0, 0.0, -0.5]),
                ("Roll right", [0.0, 0.0, 0.0, 0.5]),
                ("Combined", [0.3, 0.1, 0.2, -0.1]),
                ("Full throttle", [1.0, 0.0, 0.0, 0.0]),
                ("Reset", [0.0, 0.0, 0.0, 0.0]),
            ]

            for name, action in test_sequences:
                print(f"  {name:20s}: {action}")
                js.send_action(action)
                time.sleep(1.5)

            print()
            print("✓ Test complete!")
            print()
            print("To verify in another terminal:")
            print(f"  evtest {js.ui.device.path}")

        except KeyboardInterrupt:
            print("\n\n⚠ Test interrupted")


def send_single_action(action_str: str):
    """从命令行发送单个动作"""
    try:
        action = [float(x.strip()) for x in action_str.split(",")]
        if len(action) != 4:
            raise ValueError("Must provide 4 values")
    except Exception as e:
        print(f"❌ Invalid action format: {e}")
        print("   Expected: 'throttle,yaw,pitch,roll'")
        print("   Example: '0.5,-0.1,0.2,0.0'")
        return

    with VirtualJoystick() as js:
        print(f"Sending action: {action}")
        js.send_action(action)
        print("✓ Action sent")
        time.sleep(0.5)  # 保持一会儿


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Joystick for AI-controlled Liftoff",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test the virtual joystick
  python deploy/virtual_joystick.py --test

  # Send a single action (throttle, yaw, pitch, roll)
  python deploy/virtual_joystick.py --action "0.5,0.0,0.0,0.0"

  # Verify device in another terminal
  evtest /dev/input/eventX

Notes:
  - Values should be in range [-1, 1]
  - Throttle: -1=down, 0=hover, 1=up
  - Yaw:      -1=left, 0=center, 1=right
  - Pitch:    -1=back, 0=center, 1=forward
  - Roll:     -1=left, 0=center, 1=right  (内部已对 ABS_RX 取反)
        """
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test sequence"
    )
    parser.add_argument(
        "--action",
        type=str,
        help="Send single action as 'throttle,yaw,pitch,roll'"
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default="AI-Liftoff-Controller",
        help="Device name (default: AI-Liftoff-Controller)"
    )

    args = parser.parse_args()

    if args.test:
        test_joystick()
    elif args.action:
        send_single_action(args.action)
    else:
        parser.print_help()
        print()
        print("Quick start:")
        print("  python deploy/virtual_joystick.py --test")


if __name__ == "__main__":
    main()
