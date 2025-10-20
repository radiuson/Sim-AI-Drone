#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
屏幕捕获模块 - 用于实时捕获Liftoff游戏画面

支持三种模式：
1. 窗口捕获：使用xdotool定位特定窗口 (mss)
2. 区域捕获：手动指定屏幕区域 (mss)
3. OBS虚拟摄像头：通过OBS捕获游戏画面 (推荐，低CPU占用)

使用mss库或OpenCV VideoCapture进行屏幕捕获
"""

import subprocess
import time
from typing import Optional, Tuple, Dict
import numpy as np
import mss
import cv2


class ScreenCapture:
    """屏幕捕获类 - 用于捕获Liftoff游戏画面"""

    def __init__(
        self,
        window_name: str = "Liftoff",
        monitor_region: Optional[Dict[str, int]] = None,
        target_size: Tuple[int, int] = (640, 480),
        auto_find_window: bool = True,
    ):
        """
        初始化屏幕捕获

        Args:
            window_name: 要捕获的窗口名称（用于xdotool搜索）
            monitor_region: 手动指定的捕获区域 {'left': x, 'top': y, 'width': w, 'height': h}
            target_size: 输出图像大小 (width, height)
            auto_find_window: 是否自动查找窗口
        """
        self.window_name = window_name
        self.target_size = target_size
        self.sct = mss.mss()
        self.monitor = None

        # 如果提供了手动区域，使用它
        if monitor_region is not None:
            self.monitor = monitor_region
            print(f"✓ Using manual monitor region: {monitor_region}")
        # 否则尝试自动查找窗口
        elif auto_find_window:
            self.monitor = self._find_window()
            if self.monitor:
                print(f"✓ Found window '{window_name}': {self.monitor}")
            else:
                print(f"⚠️  Could not find window '{window_name}'")
                print("   Available monitors:")
                for i, mon in enumerate(self.sct.monitors):
                    print(f"   Monitor {i}: {mon}")

        # 统计信息
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _find_window(self) -> Optional[Dict[str, int]]:
        """
        使用xdotool查找窗口位置和大小

        Returns:
            窗口区域字典，如果找不到则返回None
        """
        try:
            # 查找窗口ID
            result = subprocess.run(
                ['xdotool', 'search', '--name', self.window_name],
                capture_output=True,
                text=True,
                timeout=2.0
            )

            if result.returncode != 0 or not result.stdout.strip():
                return None

            # 获取第一个匹配的窗口
            window_id = result.stdout.strip().split('\n')[0]

            # 获取窗口几何信息
            result = subprocess.run(
                ['xdotool', 'getwindowgeometry', window_id],
                capture_output=True,
                text=True,
                timeout=2.0
            )

            if result.returncode != 0:
                return None

            # 解析几何信息
            lines = result.stdout.strip().split('\n')
            position_line = [l for l in lines if 'Position:' in l]
            geometry_line = [l for l in lines if 'Geometry:' in l]

            if not position_line or not geometry_line:
                return None

            # 解析位置 (x,y)
            pos_str = position_line[0].split('Position:')[1].split('(')[0].strip()
            x, y = map(int, pos_str.split(','))

            # 解析大小 (widthxheight)
            geom_str = geometry_line[0].split('Geometry:')[1].strip()
            width, height = map(int, geom_str.split('x'))

            return {
                'left': x,
                'top': y,
                'width': width,
                'height': height
            }

        except Exception as e:
            print(f"Error finding window: {e}")
            return None

    def update_window_position(self) -> bool:
        """
        更新窗口位置（当窗口移动时调用）

        Returns:
            是否成功更新
        """
        new_monitor = self._find_window()
        if new_monitor:
            self.monitor = new_monitor
            return True
        return False

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像

        Returns:
            BGR格式的图像数组 (H, W, 3)，如果失败返回None
        """
        if self.monitor is None:
            return None

        try:
            # 捕获屏幕
            screenshot = self.sct.grab(self.monitor)

            # 转换为numpy数组 (BGRA格式)
            frame_bgra = np.array(screenshot)

            # 转换为BGR格式
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

            # 调整大小到目标尺寸
            if frame_bgr.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame_bgr = cv2.resize(
                    frame_bgr,
                    self.target_size,
                    interpolation=cv2.INTER_LINEAR
                )

            # 更新统计
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            return frame_bgr

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def capture_frame_rgb(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像（RGB格式）

        Returns:
            RGB格式的图像数组 (H, W, 3)，如果失败返回None
        """
        frame_bgr = self.capture_frame()
        if frame_bgr is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def get_fps(self) -> float:
        """获取当前捕获帧率"""
        return self.fps

    def close(self):
        """释放资源"""
        if self.sct is not None:
            self.sct.close()
            self.sct = None


class OBSCapture:
    """OBS虚拟摄像头捕获类 - 通过v4l2loopback设备捕获"""

    def __init__(
        self,
        device_path: str = "/dev/video10",
        target_size: Tuple[int, int] = (640, 480),
        auto_detect: bool = True,
    ):
        """
        初始化OBS虚拟摄像头捕获

        Args:
            device_path: 虚拟摄像头设备路径 (默认 /dev/video10)
            target_size: 输出图像大小 (width, height)
            auto_detect: 是否自动检测OBS虚拟摄像头设备
        """
        self.target_size = target_size
        self.cap = None

        # 自动检测OBS虚拟摄像头
        if auto_detect:
            device_path = self._find_obs_device() or device_path

        # 打开设备
        self.device_path = device_path
        self._open_device()

        # 统计信息
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0

    def _find_obs_device(self) -> Optional[str]:
        """
        自动查找OBS虚拟摄像头设备

        Returns:
            设备路径，如果找不到返回None
        """
        try:
            # 查找所有video设备
            result = subprocess.run(
                ['v4l2-ctl', '--list-devices'],
                capture_output=True,
                text=True,
                timeout=2.0
            )

            if result.returncode != 0:
                return None

            # 解析输出，查找OBS设备
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                if 'OBS' in line or 'v4l2loopback' in line:
                    # 下一行通常是设备路径
                    if i + 1 < len(lines):
                        device_line = lines[i + 1].strip()
                        if device_line.startswith('/dev/video'):
                            return device_line

            return None

        except Exception as e:
            print(f"⚠️  Error detecting OBS device: {e}")
            return None

    def _open_device(self):
        """打开视频捕获设备"""
        try:
            # 尝试打开设备
            self.cap = cv2.VideoCapture(self.device_path, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                print(f"❌ Failed to open {self.device_path}")
                self.cap = None
                return

            # 设置捕获参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            # 设置缓冲区大小为1（降低延迟）
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 读取实际参数
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            print(f"✓ Opened OBS virtual camera: {self.device_path}")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")

        except Exception as e:
            print(f"❌ Error opening device: {e}")
            self.cap = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像

        Returns:
            BGR格式的图像数组 (H, W, 3)，如果失败返回None
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        try:
            # 读取帧
            ret, frame = self.cap.read()

            if not ret or frame is None:
                return None

            # 调整大小到目标尺寸（如果需要）
            if frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(
                    frame,
                    self.target_size,
                    interpolation=cv2.INTER_LINEAR
                )

            # 更新统计
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time

            return frame

        except Exception as e:
            print(f"⚠️  Error capturing frame: {e}")
            return None

    def capture_frame_rgb(self) -> Optional[np.ndarray]:
        """
        捕获一帧图像（RGB格式）

        Returns:
            RGB格式的图像数组 (H, W, 3)，如果失败返回None
        """
        frame_bgr = self.capture_frame()
        if frame_bgr is not None:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return None

    def get_fps(self) -> float:
        """获取当前捕获帧率"""
        return self.fps

    def close(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def test_screen_capture():
    """测试屏幕捕获功能"""
    print("=" * 60)
    print("Screen Capture Test")
    print("=" * 60)
    print()

    # 创建捕获器
    print("Initializing screen capture...")
    capture = ScreenCapture(
        window_name="Liftoff",
        target_size=(640, 480),
        auto_find_window=True
    )

    if capture.monitor is None:
        print()
        print("❌ Could not find Liftoff window!")
        print()
        print("Solutions:")
        print("1. Make sure Liftoff is running")
        print("2. Check the window name is correct")
        print("3. Or manually specify monitor region:")
        print()
        print("   # Example: Use the entire primary monitor")
        print("   capture = ScreenCapture(")
        print("       monitor_region={'left': 0, 'top': 0, 'width': 1920, 'height': 1080},")
        print("       auto_find_window=False")
        print("   )")
        return

    print()
    print("✓ Screen capture initialized!")
    print(f"  Monitor region: {capture.monitor}")
    print(f"  Target size: {capture.target_size}")
    print()
    print("Capturing frames... (Press 'q' to quit)")
    print()

    try:
        while True:
            # 捕获帧
            frame = capture.capture_frame()

            if frame is not None:
                # 显示帧率
                fps_text = f"FPS: {capture.get_fps():.1f}"
                cv2.putText(
                    frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                # 显示图像
                cv2.imshow('Screen Capture Test', frame)

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("⚠️  Failed to capture frame")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # 清理
        cv2.destroyAllWindows()
        capture.close()
        print()
        print("✓ Test completed")


def test_obs_capture():
    """测试OBS虚拟摄像头捕获"""
    print("=" * 60)
    print("OBS Virtual Camera Test")
    print("=" * 60)
    print()

    # 创建OBS捕获器
    print("Initializing OBS capture...")
    capture = OBSCapture(
        target_size=(640, 480),
        auto_detect=True
    )

    if capture.cap is None:
        print()
        print("❌ Could not open OBS virtual camera!")
        print()
        print("Solutions:")
        print("1. Make sure v4l2loopback is loaded:")
        print("   sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"OBS\"")
        print()
        print("2. Make sure OBS is running and virtual camera is started:")
        print("   - Open OBS Studio")
        print("   - Add game capture source (Liftoff window)")
        print("   - Click 'Start Virtual Camera' button")
        print()
        print("3. Check available devices:")
        print("   v4l2-ctl --list-devices")
        return

    print()
    print("✓ OBS capture initialized!")
    print(f"  Device: {capture.device_path}")
    print(f"  Target size: {capture.target_size}")
    print()
    print("Capturing frames... (Press 'q' to quit)")
    print()

    try:
        frame_count = 0
        while True:
            # 捕获帧
            frame = capture.capture_frame()

            if frame is not None:
                frame_count += 1

                # 显示帧率和帧计数
                fps_text = f"FPS: {capture.get_fps():.1f} | Frames: {frame_count}"
                cv2.putText(
                    frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                )

                # 添加提示文字
                cv2.putText(
                    frame, "OBS Virtual Camera", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )

                # 显示图像
                cv2.imshow('OBS Capture Test', frame)

                # 检查退出键
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("⚠️  Failed to capture frame (is OBS virtual camera started?)")
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # 清理
        cv2.destroyAllWindows()
        capture.close()
        print()
        print("✓ Test completed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--obs":
        test_obs_capture()
    else:
        print("Usage:")
        print("  python screen_capture.py           # Test mss screen capture")
        print("  python screen_capture.py --obs     # Test OBS virtual camera")
        print()

        if input("Test OBS capture? [y/N]: ").lower() == 'y':
            test_obs_capture()
        else:
            test_screen_capture()
