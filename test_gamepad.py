#!/usr/bin/env python3
"""
测试 RadioMaster 遥控器输入
用于调试和验证遥控器开关是否正确触发
"""

import sys
import time
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
from record.gamepad_controller import GamepadController

def on_start():
    print("\n" + "="*60)
    print("✅ START RECORDING CALLBACK TRIGGERED!")
    print("="*60)
    print("   -> SH switch is UP")
    print()

def on_stop():
    print("\n" + "="*60)
    print("✅ STOP RECORDING CALLBACK TRIGGERED!")
    print("="*60)
    print("   -> SA switch is UP")
    print()

def on_emergency():
    print("\n" + "="*60)
    print("🛑 EMERGENCY STOP CALLBACK TRIGGERED!")
    print("="*60)
    print("   -> BTN_SOUTH button was pressed")
    print()

def main():
    print("\n" + "="*60)
    print("RadioMaster Gamepad Test Utility")
    print("="*60)
    print()

    try:
        # 创建控制器
        print("Initializing gamepad controller...")
        controller = GamepadController('record/control_bindings.json')
        print()

        # 注册回调
        print("Registering callbacks...")
        controller.register_callback('start_recording', on_start)
        controller.register_callback('stop_recording', on_stop)
        controller.register_callback('emergency_stop', on_emergency)
        print()

        # 启动监听
        print("Starting gamepad monitoring...")
        controller.start()
        print()

        print("="*60)
        print("🎮 Gamepad Test Active")
        print("="*60)
        print()
        print("Please test your RadioMaster controls:")
        print("  1. Move SH switch UP   → Should trigger START RECORDING")
        print("  2. Move SA switch UP   → Should trigger STOP RECORDING")
        print("  3. Press BTN_SOUTH     → Should trigger EMERGENCY STOP")
        print()
        print("You should see callback messages above when you operate the controls.")
        print()
        print("Press Ctrl+C to exit")
        print("="*60 + "\n")

        # 保持运行
        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'controller' in locals():
            controller.stop()
        print("\n✓ Test completed\n")

if __name__ == '__main__':
    main()
