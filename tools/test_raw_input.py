#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始输入测试 - 显示 RadioMaster 的所有输入事件
帮助你找到正确的轴/按钮代码
Raw Input Test - Display all RadioMaster input events
Help you find the correct axis/button codes
"""

import sys
import time

try:
    from inputs import get_gamepad, devices
except ImportError:
    print("❌ 'inputs' library not installed")
    print("   Install with: pip install inputs")
    sys.exit(1)

def main():
    """主函数"""
    print("\n" + "="*70)
    print("RadioMaster 原始输入监视器")
    print("="*70)
    print()

    # 检查设备
    if not devices.gamepads:
        print("❌ 未检测到游戏手柄!")
        print("   请连接您的 RadioMaster 并确保其处于 USB Joystick 模式")
        sys.exit(1)

    print("✅ 检测到游戏手柄:")
    for gamepad in devices.gamepads:
        print(f"  - {gamepad}")
    print()

    print("="*70)
    print("监视所有输入事件...")
    print("="*70)
    print()
    print("操作说明:")
    print("  1. 移动 RadioMaster 上的每个开关")
    print("  2. 按下每个按钮")
    print("  3. 观察下面的输出以查看轴代码和值")
    print()
    print("注意查找:")
    print("  - SH 开关: 应显示一个轴 (例如 ABS_RUDDER) 值约为 2047 当处于向上位置时")
    print("  - SA 开关: 应显示一个轴 (例如 ABS_RY) 值约为 2047 当处于向上位置时")
    print("  - BTN_SOUTH: 按下时应显示 state=1")
    print()
    print("按 Ctrl+C 退出")
    print("="*70)
    print()
    print(f"{'时间':<12} | {'类型':<10} | {'代码':<15} | {'状态/值':<10}")
    print("-" * 70)

    try:
        while True:
            events = get_gamepad()
            for event in events:
                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp:<12} | {event.ev_type:<10} | {event.code:<15} | {event.state:<10}")

    except KeyboardInterrupt:
        print("\n\n🛑 监视已停止")
        print("\n现在您可以使用正确的轴代码更新 control_bindings.json!")

if __name__ == '__main__':
    main()