#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式配置工具 - 飞行控制和录制按钮绑定 (Radiomaster 增强版)
Interactive Configuration Tool - Flight Control and Recording Button Bindings (Radiomaster Enhanced)
"""

import sys
import time
import json
import termios
import tty
import select
from pathlib import Path

try:
    from inputs import get_gamepad, devices
    INPUTS_AVAILABLE = True
except ImportError:
    INPUTS_AVAILABLE = False
    print("❌ Error: inputs library is required")
    print("   Run: pip install inputs")
    sys.exit(1)


# ---------------- 工具函数: 键盘检测 ----------------
def _kbhit():
    """检测是否有按键被按下（非阻塞）"""
    dr, dw, de = select.select([sys.stdin], [], [], 0)
    return dr != []

def _getch():
    """读取单个字符"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def _flush_input():
    """清空标准输入缓冲区"""
    termios.tcflush(sys.stdin, termios.TCIFLUSH)


# ---------------- 主配置类 ----------------
class BindingConfigurator:
    def __init__(self):
        self.bindings = {
            'flight_controls': {},     # 例如: "throttle": {"axis":"ABS_RZ","min":-32768,"max":32767}
            'recording_controls': {},  # 例如: "start_recording": {"axis":"ABS_RUDDER","value":2047}
            'liftoff_controls': {}
        }
        self.timeout = 30  # 每步最大等待时间（秒），给用户更多操作时间

    # ---------------------------------------------------
    def check_gamepad(self):
        """检查游戏手柄是否连接"""
        print("\n" + "="*70)
        print("检查游戏手柄连接")
        print("="*70)

        gamepads = devices.gamepads
        if not gamepads:
            print("未检测到游戏手柄!")
            print("\n请确保:")
            print("  1. 控制器已通过USB或蓝牙连接")
            print("  2. 系统已识别设备: ls /dev/input/js*")
            sys.exit(1)

        print(f"检测到 {len(gamepads)} 个游戏手柄:\n")
        for i, gamepad in enumerate(gamepads):
            print(f"  {i+1}. {gamepad.name}")
        return True

    # ---------------------------------------------------
    def wait_for_input(self, input_type, description, current_name=None, current_key=None):
        """交互式绑定: 记录最大/最小值或按钮值 (Radiomaster 增强版)"""
        # 开始前清空任何待处理的输入
        _flush_input()

        print(f"\n当前配置: {current_name} ({current_key})")
        print(f"  {description}")
        print("\n" + "="*70)
        print("操作说明:")
        if input_type == 'axis':
            print("  1. 第一步: 将摇杆推到最大位置")
            print("  2. 按回车键记录最大值")
            print("  3. 第二步: 将摇杆推到最小位置") 
            print("  4. 按回车键记录最小值")
            print("  5. 最后: 按回车键完成此轴的配置")
        else:
            print("  • 切换任何开关或旋钮产生显著变化")
            print("  • 按回车键确认绑定")
        print("="*70)
        print("\n实时输入监视 (按 Ctrl+C 退出):")
        print("-" * 70)

        axis_values, axis_min, axis_max = {}, {}, {}
        stage = "max" if input_type == "axis" else "button"
        last_print_time = 0
        start_time = time.time()
        last_event_display = ""
        selected_axis = None  # 存储选定的轴，确保整个过程中保持一致

        while True:
            # 检查是否超时 - 只在没有有效输入的情况下超时
            if time.time() - start_time > self.timeout and not axis_values:
                print(f"\n配置超时 ({self.timeout}秒)，请重新开始配置")
                return None
                
            try:
                events = get_gamepad()
                for event in events:
                    # 更新轴值
                    if input_type == 'axis' and event.ev_type == 'Absolute':
                        code, value = event.code, event.state
                        if code not in axis_values:
                            axis_values[code] = value
                            axis_min[code] = value
                            axis_max[code] = value
                        axis_values[code] = value
                        if value < axis_min[code]:
                            axis_min[code] = value
                        if value > axis_max[code]:
                            axis_max[code] = value

                    # 按钮事件处理
                    elif input_type == 'button':
                        if event.ev_type == 'Key' and event.state == 1:
                            print(f"\n检测到按钮按下: {event.code}")
                            time.sleep(0.5)
                            return {"axis": event.code, "value": 1}
                        elif event.ev_type == 'Absolute':
                            code, value = event.code, event.state
                            if code not in axis_values:
                                axis_values[code] = value
                                axis_min[code] = value
                                axis_max[code] = value
                            axis_values[code] = value
                            if value < axis_min[code]:
                                axis_min[code] = value
                            if value > axis_max[code]:
                                axis_max[code] = value

                    # 实时显示最新的输入事件（在一行内更新）
                    if time.time() - last_print_time > 0.1:  # 每100ms更新一次
                        timestamp = time.strftime("%H:%M:%S")
                        event_display = f"\r{timestamp} | {event.ev_type} | {event.code} | {event.state}"
                        # 只有当事件信息发生变化时才更新显示
                        if event_display != last_event_display:
                            # 清除当前行并显示新信息
                            print(event_display.ljust(70), end="", flush=True)
                            last_event_display = event_display
                        last_print_time = time.time()

                # ---------- 检查按键输入 ----------
                if _kbhit():
                    ch = _getch()
                    # 清除任何额外的待处理回车键
                    while _kbhit():
                        _getch()

                    # 检查是否是重新绑定键
                    if ch.lower() == 'r' and input_type == "axis" and stage in ["max", "min"]:
                        print(f"\n重新选择轴...")
                        # 重新选择变化最大的轴
                        if axis_values:
                            deltas = {k: abs(axis_max[k] - axis_min[k]) for k in axis_values.keys() if k in axis_max and k in axis_min}
                            if deltas:
                                selected_axis = max(deltas.items(), key=lambda x: x[1])[0]
                                print(f"检测到变化最大的轴: {selected_axis} (Δ={deltas[selected_axis]})")
                            else:
                                # 如果没有明显变化，选择值最大的轴
                                selected_axis = max(axis_max.items(), key=lambda x: x[1])[0]
                                print(f"选择值最大的轴: {selected_axis} (值={axis_max[selected_axis]})")
                            if stage == "max":
                                print(f"已记录 {current_name} 最大值轴: {selected_axis}")
                            else:
                                print(f"已记录 {current_name} 最小值轴: {selected_axis}")
                        if stage == "max":
                            print(f"\n请将 {current_name} 摇杆推到最小位置，然后按回车键继续...")
                            print("提示: 按 'r' 键可重新选择轴，按回车键继续...")
                        elif stage == "min":
                            print(f"\n请按回车键完成 {current_name} 的配置...")
                            print("提示: 按 'r' 键可重新选择轴，按回车键继续...")
                        time.sleep(0.5)
                        continue

                    if ch in ['\r', '\n']:
                        # 清除当前行的实时显示
                        print("\r" + " " * 70 + "\r", end="", flush=True)
                        
                        # ================== 飞行控制轴配置 ==================
                        if input_type == "axis":
                            if stage == "max":
                                if not axis_max:
                                    print(f"\n未检测到有效输入，请移动摇杆然后按回车")
                                    time.sleep(1)
                                    continue
                                # 选择变化最大的轴（最大值与最小值之差最大）
                                deltas = {k: abs(axis_max[k] - axis_min[k]) for k in axis_values.keys() if k in axis_max and k in axis_min}
                                if not deltas:
                                    print(f"\n未检测到有效轴变化，请移动摇杆然后按回车")
                                    time.sleep(1)
                                    continue
                                selected_axis = max(deltas.items(), key=lambda x: x[1])[0]
                                print(f"\n已选择变化最大的轴: {selected_axis} (变化量: {deltas[selected_axis]})")
                                print(f"当前最大值: {axis_max[selected_axis]}, 最小值: {axis_min[selected_axis]}")
                                stage = "min"
                                print(f"\n请将 {current_name} 摇杆推到最小位置，然后按回车键继续...")
                                print("提示: 按 'r' 键可重新选择轴，按回车键继续...")
                                time.sleep(0.5)

                            elif stage == "min":
                                # 在min阶段，我们应该使用之前选定的轴来获取最小值
                                if selected_axis and selected_axis in axis_min:
                                    print(f"已记录 {current_name} 轴数据: {selected_axis}")
                                    print(f"最大值: {axis_max[selected_axis]}, 最小值: {axis_min[selected_axis]}")
                                    stage = "done"
                                    print(f"\n请按回车键完成 {current_name} 的配置...")
                                    print("提示: 按 'r' 键可重新选择轴，按回车键继续...")
                                    time.sleep(0.5)
                                else:
                                    print(f"\n检测到有效轴失败，请重新操作")
                                    time.sleep(1)

                            elif stage == "done":
                                if selected_axis and selected_axis in axis_min and selected_axis in axis_max:
                                    # 计算结果并打印详细信息
                                    axis_min_val = axis_min[selected_axis]
                                    axis_max_val = axis_max[selected_axis]
                                    print(f"\n{current_name} ({current_key}) 配置完成!")
                                    print("="*50)
                                    print(f"  轴代码: {selected_axis}")
                                    print(f"  最小值: {axis_min_val}")
                                    print(f"  最大值: {axis_max_val}")
                                    print("="*50)
                                    # 等待用户按回车键确认
                                    print("按回车键继续下一个配置项，或按 'r' 键重新配置当前轴...")
                                    # 等待用户输入，最多等待30秒
                                    start_wait_time = time.time()
                                    reconfigure = False
                                    while time.time() - start_wait_time < 30:
                                        if _kbhit():
                                            ch = _getch()
                                            if ch in ['\r', '\n']:
                                                # 确认配置
                                                input("按回车键继续下一个配置项...")
                                                return {
                                                    "axis": selected_axis,
                                                    "min": axis_min_val,
                                                    "max": axis_max_val
                                                }
                                            elif ch.lower() == 'r':
                                                # 重新配置当前轴
                                                print(f"\n重新配置 {current_name} 轴...")
                                                # 重置状态并重新开始
                                                axis_values, axis_min, axis_max = {}, {}, {}
                                                stage = "max"
                                                selected_axis = None
                                                reconfigure = True
                                                print(f"请将 {current_name} 摇杆推到最大位置，然后按回车键记录...")
                                                break
                                        time.sleep(0.1)
                                    # 如果需要重新配置，继续循环
                                    if reconfigure:
                                        continue
                                    # 如果超时或用户没有按任何键，继续下一个配置项
                                    input("按回车键继续下一个配置项...")
                                    return {
                                        "axis": selected_axis,
                                        "min": axis_min_val,
                                        "max": axis_max_val
                                    }
                                else:
                                    print(f"配置过程中出现错误，无法获取轴数据")
                                    return None

                        # ================== 按钮配置 ==================
                        else:
                            if axis_values:
                                # 计算最大变化 Δ
                                deltas = {k: abs(axis_max[k] - axis_min[k]) for k in axis_values.keys() if k in axis_max and k in axis_min}
                                if deltas:
                                    selected_axis = max(deltas.items(), key=lambda x: x[1])[0]
                                    peak_value = axis_max[selected_axis] if abs(axis_max[selected_axis]) > abs(axis_min[selected_axis]) else axis_min[selected_axis]
                                    print(f"\n{current_name} ({current_key}) 配置完成!")
                                    print("="*50)
                                    print(f"  检测到变化最大的轴: {selected_axis}")
                                    print(f"  变化范围 (Δ): {deltas[selected_axis]}")
                                    print(f"  触发值: {peak_value}")
                                    print("="*50)
                                    # 等待用户按回车键确认
                                    input("按回车键继续下一个配置项...")
                                    return {"axis": selected_axis, "value": peak_value}
                                else:
                                    print(f"\n未检测到显著变化，请再次切换然后按回车")
                                    time.sleep(1)
                            else:
                                print(f"\n未检测到显著变化，请再次切换然后按回车")
                                time.sleep(1)

            except KeyboardInterrupt:
                print(f"\n用户中断")
                # 如果用户中断，询问是否要退出或重新开始
                response = input("是否要退出配置? (y/n): ").strip().lower()
                if response == 'y':
                    return None
                else:
                    # 重新开始当前配置项
                    print("重新开始当前配置项...")
                    _flush_input()
                    return self.wait_for_input(input_type, description, current_name, current_key)
            except Exception as e:
                if time.time() - start_time < 1:
                    print(f"输入错误: {e}")
                pass

    # ---------------------------------------------------
    def configure_flight_controls(self):
        """配置飞行控制轴"""
        print("\n" + "="*70)
        print("步骤 1: 配置飞行控制轴")
        print("="*70)
        print("\n您将为以下轴进行配置，请按照提示操作:")
        print("• 油门 (throttle): 左手柄 - 推到最高位置")
        print("• 偏航 (yaw): 左手柄 - 推到最右位置")
        print("• 俯仰 (pitch): 右手柄 - 推到最高位置")
        print("• 横滚 (roll): 右手柄 - 推到最右位置")
        print("\n" + "="*50)
        print("配置说明:")
        print("• 对每个轴，您将需要:")
        print("  ① 将摇杆推到最大位置，按回车记录")
        print("  ② 将摇杆推到最小位置，按回车记录") 
        print("  ③ 再次按回车完成此轴的配置")
        print("• 按 Ctrl+C 可随时退出配置")
        print("="*50)

        controls = [
            ('throttle', '油门', '左手柄 - 推到最高位置'),
            ('yaw', '偏航', '左手柄 - 推到最右位置'),
            ('pitch', '俯仰', '右手柄 - 推到最高位置'),
            ('roll', '横滚', '右手柄 - 推到最右位置'),
        ]
        for key, name_zh, instruction in controls:
            print(f"\n现在开始配置 {name_zh} ({key})...")
            info = self.wait_for_input(
                'axis',
                f"{instruction}",
                current_name=name_zh,
                current_key=key
            )
            if info:
                self.bindings['flight_controls'][key] = info
                print(f"{name_zh} 配置成功")
            else:
                print(f"跳过 {name_zh} 配置")

        print(f"\n飞行控制轴配置完成!")
        self._show_flight_summary()

    # ---------------------------------------------------
    def configure_recording_controls(self):
        """配置录制按钮"""
        print("\n" + "="*70)
        print("步骤 2: 配置录制控制按钮")
        print("="*70)
        print("\n您将为以下录制控制功能配置按钮:")
        print("• 开始录制: 按下您想要用于开始录制的按钮")
        print("• 停止录制: 按下您想要用于停止录制的按钮")
        print("• 紧急停止: 按下用于紧急停止的按钮")
        print("\n" + "="*50)
        print("配置说明:")
        print("• 对每个功能，您将需要:")
        print("  ① 按下相应的按钮")
        print("  ② 系统会自动检测并确认")
        print("• 按 Ctrl+C 可随时退出配置")
        print("="*50)

        controls = [
            ('start_recording', '开始录制', '按下您想要用于开始录制的按钮（推荐: A/X 或开关位置1）'),
            ('stop_recording', '停止录制', '按下您想要用于停止录制的按钮（推荐: B/○ 或开关位置2）'),
            ('emergency_stop', '紧急停止', '按下用于紧急停止的按钮（推荐: START 或三段开关中间位置）'),
        ]
        for key, name_zh, instruction in controls:
            print(f"\n现在开始配置 {name_zh} ({key})...")
            info = self.wait_for_input(
                'button',
                f"{instruction}",
                current_name=name_zh,
                current_key=key
            )
            if info:
                # 轴/开关触发（包含轴和值）
                if isinstance(info, dict) and "axis" in info and "value" in info:
                    self.bindings['recording_controls'][key] = {
                        "axis": info["axis"],
                        "value": info["value"]
                    }
                else:
                    # 兼容 Key 返回字符串的情况
                    self.bindings['recording_controls'][key] = {"axis": info, "value": 1}
                print(f"{name_zh} 配置成功")
            else:
                print(f"跳过 {name_zh} 配置")
                # 如果用户中断配置，可以选择退出或跳过
                try:
                    response = input("是否要退出配置? (y/n): ").strip().lower()
                    if response == 'y':
                        print("配置已取消")
                        return False
                except KeyboardInterrupt:
                    print("\n配置已取消")
                    return False

        print(f"\n录制控制按钮配置完成!")
        self._show_recording_summary()
        return True

    # ---------------------------------------------------
    def _show_flight_summary(self):
        print(f"\n{chr(0x2705)} 已配置的飞行控制轴:")
        if self.bindings['flight_controls']:
            for k, v in self.bindings['flight_controls'].items():
                axis_info = f"轴: {v.get('axis', 'N/A')}, 范围: [{v.get('min', 'N/A')}, {v.get('max', 'N/A')}]"
                print(f"  • {k}: {axis_info}")
        else:
            print("  (无配置)")

    def _show_recording_summary(self):
        print(f"\n{chr(0x2705)} 已配置的录制控制按钮:")
        if self.bindings['recording_controls']:
            for k, v in self.bindings['recording_controls'].items():
                button_info = f"轴: {v.get('axis', 'N/A')}, 触发值: {v.get('value', 'N/A')}"
                print(f"  • {k}: {button_info}")
        else:
            print("  (无配置)")

    # ---------------------------------------------------
    def save_bindings(self, filename='control_bindings.json'):
        """保存绑定配置到JSON文件"""
        try:
            filepath = Path(filename)
            cfg = {
                'version': '1.0',
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'bindings': self.bindings
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=2, ensure_ascii=False)
            print(f"\n{chr(0x1F4BE)} 配置已保存到: {filepath.absolute()}")
            print(f"  文件名: {filepath.name}")
            print(f"  保存路径: {filepath.parent}")
            return True
        except Exception as e:
            print(f"\n{chr(0x274C)} 保存配置文件时出错: {e}")
            return False

    # ---------------------------------------------------
    def run(self):
        """运行配置工具"""
        print("="*70)
        print(f"{chr(0x1F3AF)} Liftoff 录制系统 - 交互式配置工具")
        print("="*70)
        print("此工具将帮助您配置 RadioMaster 遥控器与 AI Drone 系统的连接")
        print("\n配置过程分为两个步骤:")
        print(f"  {chr(0x1F538)} 步骤 1: 配置飞行控制轴 (油门、偏航、俯仰、横滚)")
        print(f"  {chr(0x1F538)} 步骤 2: 配置录制控制按钮 (开始、停止、紧急停止)")
        print("\n请确保:")
        print("  • RadioMaster 遥控器已通过 USB 连接")
        print("  • 遥控器已开启并处于正常工作状态")
        print("="*70)
        
        if not self.check_gamepad():
            return

        print(f"\n{chr(0x1F914)} 是否要先测试游戏手柄输入? (y/n, 推荐): ", end="")
        resp = input().strip().lower()
        if resp == 'y' or resp == '':
            self.test_gamepad_input()

        print(f"\n{chr(0x1F680)} 准备开始配置...")
        try:
            input("按回车键开始配置过程...")
        except KeyboardInterrupt:
            print("\n配置已取消")
            return
        self.configure_flight_controls()
        if not self.configure_recording_controls():
            print("配置已取消")
            return
        self.show_full_summary()
        if not self.save_bindings():
            print("配置保存失败，但配置信息已在上方显示")
            return
        print(f"\n{chr(0x1F389)} 恭喜! 所有配置已完成!")
        print("现在您可以开始使用遥控器进行数据录制了!")

    # ---------------------------------------------------
    def show_full_summary(self):
        """显示完整配置摘要"""
        print("\n" + "="*70)
        print(f"{chr(0x1F4CB)} 最终配置摘要")
        print("="*70)
        for sec, data in self.bindings.items():
            section_name = {
                'flight_controls': '飞行控制轴',
                'recording_controls': '录制控制按钮',
                'liftoff_controls': 'Liftoff 特殊控制'
            }.get(sec, sec)
            
            print(f"\n{chr(0x1F4C1)} {section_name} [{sec}]:")
            if data:
                for k, v in data.items():
                    print(f"  • {k}: {v}")
            else:
                print("  (未配置)")

    def test_gamepad_input(self):
        """测试游戏手柄输入 - 显示所有输入事件"""
        print("\n" + "="*70)
        print("RadioMaster 原始输入监视器")
        print("="*70)
        print()
        
        # 检查设备
        if not devices.gamepads:
            print("❌ 未检测到游戏手柄!")
            print("   请连接您的 RadioMaster 并确保其处于 USB Joystick 模式")
            return

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
            event_count = 0
            max_events = 1000  # 限制显示的事件数量以避免输出过多
            while event_count < max_events:
                events = get_gamepad()
                for event in events:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"{timestamp:<12} | {event.ev_type:<10} | {event.code:<15} | {event.state:<10}")
                    event_count += 1
                    if event_count >= max_events:
                        print(f"\n已达到最大事件显示数量 ({max_events})，停止监视")
                        break

        except KeyboardInterrupt:
            print("\n\n🛑 监视已停止")
            print("\n现在您可以使用正确的轴代码更新 control_bindings.json!")
        except Exception as e:
            print(f"\n❌ 监视过程中出现错误: {e}")


# ---------------- 主入口点 ----------------
def main():
    """主函数"""
    print(f"{chr(0x1F44B)} 欢迎使用 AI Drone 遥控器配置工具!")
    try:
        BindingConfigurator().run()
    except KeyboardInterrupt:
        print(f"\n{chr(0x1F6AB)} 配置已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n{chr(0x274C)} 配置过程中出现错误: {e}")
        sys.exit(1)



if __name__ == '__main__':
    main()