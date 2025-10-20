#!/usr/bin/env python3
"""
原始输入测试 - 显示 RadioMaster 的所有输入事件
帮助你找到正确的轴/按钮代码
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
    print("\n" + "="*70)
    print("RadioMaster Raw Input Monitor")
    print("="*70)
    print()

    # 检查设备
    if not devices.gamepads:
        print("❌ No gamepad detected!")
        print("   Please connect your RadioMaster and make sure it's in USB Joystick mode")
        sys.exit(1)

    print("✓ Gamepad detected:")
    for gamepad in devices.gamepads:
        print(f"  - {gamepad}")
    print()

    print("="*70)
    print("Monitoring ALL input events...")
    print("="*70)
    print()
    print("Instructions:")
    print("  1. Move each switch on your RadioMaster")
    print("  2. Press each button")
    print("  3. Watch the output below to see the axis codes and values")
    print()
    print("Look for:")
    print("  - SH switch: Should show an axis (e.g., ABS_RUDDER) with value ~2047 when UP")
    print("  - SA switch: Should show an axis (e.g., ABS_RY) with value ~2047 when UP")
    print("  - BTN_SOUTH: Should show state=1 when pressed")
    print()
    print("Press Ctrl+C to exit")
    print("="*70)
    print()
    print(f"{'Time':<12} | {'Type':<10} | {'Code':<15} | {'State/Value':<10}")
    print("-" * 70)

    try:
        while True:
            events = get_gamepad()
            for event in events:
                timestamp = time.strftime("%H:%M:%S")
                print(f"{timestamp:<12} | {event.ev_type:<10} | {event.code:<15} | {event.state:<10}")

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")
        print("\nNow you can update control_bindings.json with the correct axis codes!")

if __name__ == '__main__':
    main()
