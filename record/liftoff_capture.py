"""
Liftoff数据采集工具
作用：
    从Liftoff模拟器采集图像、状态和遥控器输入，保存为LeRobot格式数据集
主要功能：
    - 实时屏幕捕获（支持mss和OBS虚拟摄像头）
    - 遥控器输入录制
    - 状态数据同步（需ROS2 bridge）
    - LeRobot格式数据集生成
依赖：
    mss, opencv-python, evdev, pandas, PIL
注意：
    需要先运行liftoff_bridge_ros2来提供状态数据
    需要物理遥控器或虚拟遥控器作为输入源
示例：
    # 使用默认设置（OBS + RadioMaster）
    python -m record.liftoff_capture \\
        --output-dir ./dataset/liftoff_data \\
        --fps 30

    # 自定义设置
    python -m record.liftoff_capture \\
        --output-dir ./dataset/liftoff_data \\
        --capture-method obs \\
        --joystick-device /dev/input/js0 \\
        --fps 30

    # 使用MSS屏幕捕获（备选方案）
    python -m record.liftoff_capture \\
        --output-dir ./dataset/liftoff_data \\
        --capture-method mss \\
        --window-name "Liftoff" \\
        --fps 30
"""

import time
import json
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import threading

# 导入屏幕捕获类
import sys
sys.path.append(str(Path(__file__).parent.parent))
from deploy.screen_capture import ScreenCapture, OBSCapture

# 尝试导入 ROS2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Joy, Imu
    from geometry_msgs.msg import TwistStamped
    HAS_ROS2 = True
except ImportError:
    HAS_ROS2 = False
    print("⚠️  ROS2 not available, will use mock data")

# 尝试导入遥控器控制
try:
    from record.gamepad_controller import GamepadController
    HAS_GAMEPAD = True
except ImportError:
    try:
        from gamepad_controller import GamepadController
        HAS_GAMEPAD = True
    except ImportError:
        HAS_GAMEPAD = False
        print("⚠️  Gamepad controller not available")


class ROS2DataReceiver:
    """ROS2 数据接收器 - 订阅 Liftoff bridge 数据"""

    def __init__(self):
        if not HAS_ROS2:
            print("⚠️  ROS2 not available")
            return

        # 初始化 ROS2
        if not rclpy.ok():
            rclpy.init()

        # 创建节点
        self.node = rclpy.create_node('liftoff_data_receiver')

        # 数据缓存
        self.latest_rc = None      # [throttle, yaw, pitch, roll]
        self.latest_state = None   # [vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
        self.data_lock = threading.Lock()

        # 线程控制
        self.running = True

        # 订阅话题
        self.sub_rc = self.node.create_subscription(
            Joy,
            '/liftoff/rc',
            self._rc_callback,
            10
        )

        self.sub_twist = self.node.create_subscription(
            TwistStamped,
            '/liftoff/twist',
            self._twist_callback,
            10
        )

        self.sub_imu = self.node.create_subscription(
            Imu,
            '/liftoff/imu',
            self._imu_callback,
            10
        )

        # 临时状态存储（用于组合数据）
        self._vel = [0.0, 0.0, 0.0]
        self._quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
        self._omega = [0.0, 0.0, 0.0]
        self._accel = [0.0, 0.0, 0.0]

        # 启动 ROS2 spin 线程
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()

        print("✓ ROS2 data receiver initialized")
        print("  Subscribing to:")
        print("    - /liftoff/rc (Joy)")
        print("    - /liftoff/twist (TwistStamped)")
        print("    - /liftoff/imu (Imu)")

    def _rc_callback(self, msg):
        """遥控器输入回调"""
        with self.data_lock:
            # Joy.axes = [throttle, yaw, pitch, roll]
            self.latest_rc = list(msg.axes[:4]) if len(msg.axes) >= 4 else [0.0, 0.0, 0.0, 0.0]

    def _twist_callback(self, msg):
        """速度回调"""
        self._vel = [
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z
        ]
        self._omega = [
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z
        ]
        self._update_state()

    def _imu_callback(self, msg):
        """IMU 回调"""
        self._quat = [
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ]
        self._accel = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ]
        self._update_state()

    def _update_state(self):
        """更新状态向量"""
        with self.data_lock:
            # 状态向量: [vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
            self.latest_state = (
                self._vel +
                self._quat +
                self._omega +
                self._accel
            )

    def _spin_loop(self):
        """ROS2 spin 循环"""
        while self.running and rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.1)
            except Exception as e:
                if self.running:  # 只在仍在运行时打印错误
                    print(f"⚠️  ROS2 spin error: {e}")
                break

    def get_rc_input(self) -> Optional[List[float]]:
        """获取最新的遥控器输入"""
        with self.data_lock:
            return self.latest_rc.copy() if self.latest_rc else None

    def get_state(self) -> Optional[np.ndarray]:
        """获取最新的状态数据"""
        with self.data_lock:
            if self.latest_state:
                return np.array(self.latest_state, dtype=np.float32)
            return None

    def shutdown(self):
        """关闭接收器"""
        print("  Shutting down ROS2 receiver...")
        self.running = False

        # 等待 spin 线程结束
        if self.spin_thread and self.spin_thread.is_alive():
            self.spin_thread.join(timeout=2.0)

        # 销毁节点
        if HAS_ROS2 and rclpy.ok():
            try:
                self.node.destroy_node()
            except Exception as e:
                print(f"  Warning: Error destroying node: {e}")

        print("  ✓ ROS2 receiver stopped")


class LiftoffCapture:
    """
    Liftoff数据采集器
    """

    def __init__(
        self,
        output_dir: str,
        window_name: str = "Liftoff",
        image_size: Tuple[int, int] = (224, 224),
        fps: int = 30,
        use_ros2: bool = True,
        capture_method: str = "mss",
        obs_device: str = "/dev/video10",
        enable_gamepad: bool = True,
        bindings_file: str = "record/control_bindings.json"
    ):
        """
        初始化采集器

        Args:
            output_dir: 输出数据集目录
            window_name: Liftoff窗口名称（仅用于MSS）
            image_size: 图像尺寸 (width, height)
            fps: 采集帧率
            use_ros2: 是否使用ROS2获取数据（推荐）
            capture_method: 捕获方法 ('mss' 或 'obs')
            obs_device: OBS虚拟摄像头设备路径（仅用于OBS）
            enable_gamepad: 是否启用遥控器按键控制录制（推荐）
            bindings_file: 遥控器控制绑定配置文件路径
        """
        self.output_dir = Path(output_dir)
        self.window_name = window_name
        self.image_size = image_size
        self.fps = fps
        self.interval = 1.0 / fps
        self.capture_method = capture_method
        self.use_ros2 = use_ros2 and HAS_ROS2
        self.enable_gamepad = enable_gamepad and HAS_GAMEPAD
        self.bindings_file = bindings_file

        # 创建输出目录结构
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.output_dir / 'videos'
        self.videos_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        self.meta_dir = self.output_dir / 'meta'
        self.meta_dir.mkdir(exist_ok=True)

        # 屏幕捕获（根据方法选择）
        self.capture = None
        if capture_method == "obs":
            print(f"Using OBS virtual camera capture (device: {obs_device})")
            self.capture = OBSCapture(
                device_path=obs_device,
                target_size=image_size,
                auto_detect=True
            )
            if self.capture.cap is None:
                raise RuntimeError("Failed to open OBS virtual camera!")
        else:
            print("Using MSS screen capture")
            self.capture = ScreenCapture(
                window_name=window_name,
                target_size=image_size,
                auto_find_window=True
            )
            if self.capture.monitor is None:
                raise RuntimeError(f"Failed to find window '{window_name}'!")

        # ROS2 数据接收器
        self.ros2_receiver = None
        if self.use_ros2:
            try:
                self.ros2_receiver = ROS2DataReceiver()
            except Exception as e:
                print(f"⚠️  Failed to initialize ROS2 receiver: {e}")
                print("    Will use mock data")
                self.ros2_receiver = None

        # Episode数据
        self.current_episode = 0
        self.frame_buffer = []
        self.is_recording = False

        # 遥控器控制
        self.gamepad_controller = None
        if self.enable_gamepad:
            try:
                self.gamepad_controller = GamepadController(bindings_file)
                self.gamepad_controller.register_callback('start_recording', self._gamepad_start_recording)
                self.gamepad_controller.register_callback('stop_recording', self._gamepad_stop_recording)
                self.gamepad_controller.register_callback('emergency_stop', self._gamepad_emergency_stop)
                self.gamepad_controller.start()
                print("✓ Gamepad controller enabled")
                
                # 显示从配置文件中读取的控制信息
                control_info = self.gamepad_controller.list_controls()
                print("  Gamepad Controls:")
                for control_name, binding in control_info.items():
                    axis = binding.get('axis', 'Unknown')
                    value = binding.get('value', 'Unknown')
                    control_labels = {
                        'start_recording': 'Start recording',
                        'stop_recording': 'Stop recording',
                        'emergency_stop': 'Emergency stop'
                    }
                    label = control_labels.get(control_name, control_name)
                    print(f"    - {label}: {axis} = {value}")
            except Exception as e:
                print(f"⚠️  Failed to initialize gamepad controller: {e}")
                self.gamepad_controller = None

        print(f"✓ LiftoffCapture initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Capture method: {capture_method}")
        print(f"  FPS: {self.fps}")
        print(f"  Image size: {self.image_size}")
        print()
        # 显示从配置文件读取的控制信息
        if self.gamepad_controller and self.gamepad_controller.recording_controls:
            print("🎮 Gamepad Controls Ready:")
            controls = self.gamepad_controller.recording_controls
            if 'start_recording' in controls:
                ctrl = controls['start_recording']
                print(f"  ▶️  {ctrl.get('axis', 'Unknown')}: Start recording")
            if 'stop_recording' in controls:
                ctrl = controls['stop_recording']
                print(f"  ⏹️  {ctrl.get('axis', 'Unknown')}: Stop recording")
            if 'emergency_stop' in controls:
                ctrl = controls['emergency_stop']
                print(f"  🚨 {ctrl.get('axis', 'Unknown')}: Emergency stop")
            print()

    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧图像（RGB格式）"""
        if self.capture is None:
            return None

        try:
            # 使用统一的捕获接口
            frame = self.capture.capture_frame_rgb()
            return frame

        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None

    def read_joystick(self) -> List[float]:
        """
        读取遥控器输入（从ROS2获取）

        Returns:
            [throttle, yaw, pitch, roll]
        """
        if self.ros2_receiver:
            rc_input = self.ros2_receiver.get_rc_input()
            if rc_input is not None:
                return rc_input

        # 默认值（悬停）
        return [0.0, 0.0, 0.0, 0.0]

    def get_state(self) -> np.ndarray:
        """
        获取状态数据（13维）从ROS2或返回模拟数据

        Returns:
            [vx, vy, vz, qw, qx, qy, qz, wx, wy, wz, ax, ay, az]
        """
        if self.ros2_receiver:
            state = self.ros2_receiver.get_state()
            if state is not None:
                return state

        # 默认值（悬停，零速度）
        return np.zeros(13, dtype=np.float32)

    def _gamepad_start_recording(self):
        """遥控器触发开始录制"""
        # 从游戏手柄控制器获取当前绑定信息
        binding_info = self.gamepad_controller.get_control_info('start_recording') if self.gamepad_controller else {}
        axis_name = binding_info.get('axis', 'Unknown')
        print("\n" + "="*60)
        print(f"🎮 RECORDING CONTROL: {axis_name} - START RECORDING")
        print("="*60)
        if not self.is_recording:
            print("✓ Starting new episode...")
            self.start_episode()
            print(f"📹 Episode {self.current_episode} started - Recording in progress")
        else:
            print(f"⚠️  Already recording episode {self.current_episode}! Ignoring command.")
        print("💡 Press SA switch UP to stop recording")
        print()

    def _gamepad_stop_recording(self):
        """遥控器触发停止录制"""
        # 从游戏手柄控制器获取当前绑定信息
        binding_info = self.gamepad_controller.get_control_info('stop_recording') if self.gamepad_controller else {}
        axis_name = binding_info.get('axis', 'Unknown')
        print("\n" + "="*60)
        print(f"🎮 RECORDING CONTROL: {axis_name} - STOP RECORDING")
        print("="*60)
        if self.is_recording:
            current_episode = self.current_episode
            print("✓ Stopping episode...")
            self.end_episode()
            print(f"✅ Episode {current_episode} saved successfully")
        else:
            print("⚠️  Not recording! Ignoring command.")
        print("💡 Press SH switch UP to start a new recording")
        print()

    def _gamepad_emergency_stop(self):
        """遥控器触发紧急停止"""
        # 从游戏手柄控制器获取当前绑定信息
        binding_info = self.gamepad_controller.get_control_info('emergency_stop') if self.gamepad_controller else {}
        axis_name = binding_info.get('axis', 'Unknown')
        print("\n" + "="*60)
        print(f"🚨 EMERGENCY STOP: {axis_name} - EMERGENCY STOP TRIGGERED")
        print("="*60)
        if self.is_recording:
            current_episode = self.current_episode
            print(f"⚠️  Discarding current episode {current_episode} ({len(self.frame_buffer)} frames)...")
            self.is_recording = False
            self.frame_buffer = []
            print("✅ Episode discarded - Ready for next recording")
        else:
            print("ℹ️  Not recording, emergency stop acknowledged")
            print("✅ System ready for next recording")
        print("💡 Press SH switch UP to start a new recording")
        print()

    def start_episode(self):
        """开始新的episode"""
        self.frame_buffer = []
        self.is_recording = True
        print(f"\n📹 Starting episode {self.current_episode}")
        print(f"📈 Recording started - Episode {self.current_episode}")

    def record_frame(self, timestamp: float):
        """
        记录一帧数据

        Args:
            timestamp: 当前时间戳（秒）
        """
        # 只在录制时记录
        if not self.is_recording:
            return

        # 捕获图像
        image = self.capture_frame()
        if image is None:
            return

        # 保存图像
        frame_idx = len(self.frame_buffer)
        image_filename = f"episode_{self.current_episode:06d}_frame_{frame_idx:06d}.png"
        image_path = self.videos_dir / image_filename
        Image.fromarray(image).save(image_path)

        # 读取遥控器
        action = self.read_joystick()

        # 获取状态
        state = self.get_state()

        # 添加到缓冲区
        self.frame_buffer.append({
            'episode_index': self.current_episode,
            'frame_index': frame_idx,
            'timestamp': timestamp,
            'observation.images.cam_front': str(image_filename),
            'observation.state': state.tolist(),
            'action': action
        })

    def end_episode(self):
        """结束当前episode并保存"""
        self.is_recording = False

        if not self.frame_buffer:
            print("⚠️  No frames recorded in this episode")
            return

        # 转换为DataFrame
        df = pd.DataFrame(self.frame_buffer)

        # 保存parquet
        episode_file = self.data_dir / f"episode_{self.current_episode:06d}.parquet"
        df.to_parquet(episode_file, index=False)

        print(f"✓ Saved episode {self.current_episode}: {len(self.frame_buffer)} frames")
        print(f"📁 Data saved to: {episode_file}")

        self.current_episode += 1
        self.frame_buffer = []

    def save_metadata(self):
        """保存数据集元数据"""
        info = {
            'codebase_version': '2.0',
            'robot_type': 'liftoff_fpv_drone',
            'fps': self.fps,
            'video_fps': self.fps,
            'features': {
                'observation.images.cam_front': {
                    'dtype': 'image',
                    'shape': [self.image_size[1], self.image_size[0], 3],
                    'info': 'Front camera view'
                },
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [13],
                    'names': ['vx', 'vy', 'vz', 'qw', 'qx', 'qy', 'qz', 'wx', 'wy', 'wz', 'ax', 'ay', 'az'],
                    'info': 'Velocity, quaternion, angular velocity, acceleration'
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [4],
                    'names': ['throttle', 'yaw', 'pitch', 'roll'],
                    'info': 'RC controller inputs'
                }
            }
        }

        info_file = self.meta_dir / 'info.json'
        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"✓ Saved metadata to {info_file}")

    def close(self):
        """清理资源"""
        print("\n🔄 Cleaning up resources...")

        if self.gamepad_controller:
            print("  Stopping gamepad controller...")
            try:
                self.gamepad_controller.stop()
                print("  ✓ Gamepad controller stopped")
            except Exception as e:
                print(f"  Warning: Error stopping gamepad: {e}")

        if self.ros2_receiver:
            try:
                self.ros2_receiver.shutdown()
            except Exception as e:
                print(f"  Warning: Error shutting down ROS2: {e}")

        if self.capture:
            print("  Closing video capture...")
            try:
                self.capture.close()
                print("  ✓ Video capture closed")
            except Exception as e:
                print(f"  Warning: Error closing capture: {e}")

        print("✓ All resources cleaned up")


def main():
    parser = argparse.ArgumentParser(
        description='Liftoff Data Capture Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output dataset directory'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='Liftoff',
        help='Liftoff window name'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size (square)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Capture FPS'
    )
    parser.add_argument(
        '--use-ros2',
        action='store_true',
        default=True,
        help='Use ROS2 to get RC input and state data (default: True)'
    )
    parser.add_argument(
        '--no-ros2',
        action='store_false',
        dest='use_ros2',
        help='Disable ROS2, use mock data instead'
    )
    parser.add_argument(
        '--capture-method',
        type=str,
        default='obs',
        choices=['mss', 'obs'],
        help='Screen capture method: obs (default, recommended) or mss'
    )
    parser.add_argument(
        '--obs-device',
        type=str,
        default='/dev/video10',
        help='OBS virtual camera device path (only for --capture-method obs)'
    )
    parser.add_argument(
        '--enable-gamepad',
        action='store_true',
        default=True,
        help='Enable gamepad/RC control for recording (default: True)'
    )
    parser.add_argument(
        '--no-gamepad',
        action='store_false',
        dest='enable_gamepad',
        help='Disable gamepad control'
    )
    parser.add_argument(
        '--bindings-file',
        type=str,
        default='record/control_bindings.json',
        help='Path to control bindings configuration file'
    )

    args = parser.parse_args()

    print("="*70)
    print("Liftoff Data Capture")
    print("="*70)
    print()

    # 创建采集器
    try:
        capture = LiftoffCapture(
            output_dir=args.output_dir,
            window_name=args.window_name,
            image_size=(args.image_size, args.image_size),
            fps=args.fps,
            use_ros2=args.use_ros2,
            capture_method=args.capture_method,
            obs_device=args.obs_device,
            enable_gamepad=args.enable_gamepad,
            bindings_file=args.bindings_file
        )
    except RuntimeError as e:
        print(f"\n❌ Failed to initialize capture: {e}")
        print("\nSolutions:")
        if args.capture_method == 'obs':
            print("  1. Make sure v4l2loopback is loaded:")
            print("     sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"OBS\"")
            print("  2. Make sure OBS is running with virtual camera started")
            print("  3. Or use --capture-method mss for traditional screen capture")
        else:
            print("  1. Make sure Liftoff is running")
            print("  2. Check the window name with: xdotool search --name Liftoff")
            print("  3. Or use --capture-method obs for OBS virtual camera")
        return

    if args.use_ros2:
        print()
        print("⚠️  Using ROS2 for data collection")
        print("   Make sure liftoff_bridge_ros2 is running:")
        print("   ros2 run <your_package> liftoff_bridge_ros2")
        print()

    print()
    if capture.gamepad_controller:
        # 显示从配置文件中读取的控制信息
        control_info = capture.gamepad_controller.list_controls()
        print("🎮 Gamepad Control Enabled:")
        for control_name, binding in control_info.items():
            axis = binding.get('axis', 'Unknown')
            value = binding.get('value', 'Unknown')
            control_labels = {
                'start_recording': 'Start recording',
                'stop_recording': 'Stop recording',
                'emergency_stop': 'Emergency stop'
            }
            label = control_labels.get(control_name, control_name)
            print(f"  - {label}: {axis} = {value}")
        print()
        print("   OR use keyboard:")
        print("  - Press 'r' to start recording")
        print("  - Press 's' to stop recording")
        print("  - Press 'q' to quit")
        print()
        print("▶️  Waiting for RC inputs... (or keyboard commands)")
    else:
        print("Keyboard Controls:")
        print("  - Press 'r' to start recording episode")
        print("  - Press 's' to stop and save current episode")
        print("  - Press 'q' to quit")
    print()

    # 自动录制模式（使用遥控器控制）
    if capture.gamepad_controller:
        try:
            print("▶️  System ready for recording")
            print("   Use RadioMaster switches to control recording:")
            
            # 显示从配置文件中读取的控制信息
            control_info = capture.gamepad_controller.list_controls()
            for control_name, binding in control_info.items():
                axis = binding.get('axis', 'Unknown')
                value = binding.get('value', 'Unknown')
                control_labels = {
                    'start_recording': '▶️  SH Switch UP: Start recording',
                    'stop_recording': '⏹️  SA Switch UP: Stop recording',
                    'emergency_stop': '🚨 BTN_SOUTH: Emergency stop'
                }
                label = control_labels.get(control_name, f"{control_name}: {axis} = {value}")
                print(f"     {label}")
            print()
            print("   Press Ctrl+C to quit\n")

            # 主循环 - 等待遥控器触发
            while True:
                # 定期记录数据（如果正在录制）
                if capture.is_recording:
                    timestamp = time.time()
                    capture.record_frame(timestamp)

                time.sleep(1.0 / capture.fps)

        except KeyboardInterrupt:
            print("\n")
            print("="*60)
            print("⚠️  Interrupted by user - shutting down...")
            print("="*60)

        finally:
            # 如果正在录制，保存当前 episode
            if capture.is_recording:
                print("   Saving current episode...")
                try:
                    capture.end_episode()
                except Exception as e:
                    print(f"   Warning: Error saving episode: {e}")

            # 保存元数据
            try:
                capture.save_metadata()
            except Exception as e:
                print(f"   Warning: Error saving metadata: {e}")

            # 清理资源
            try:
                capture.close()
            except Exception as e:
                print(f"   Warning: Error during cleanup: {e}")

            print("\n✓ Capture completed\n")

    else:
        # 手动键盘控制模式
        try:
            while True:
                # 简单的键盘控制
                key = input("Command (r/s/q): ").strip().lower()

                if key == 'r' and not capture.is_recording:
                    capture.start_episode()
                    print("Recording...")

                elif key == 's' and capture.is_recording:
                    capture.end_episode()
                    print("Stopped recording")

                elif key == 'q':
                    if capture.is_recording:
                        capture.end_episode()
                    break

        except KeyboardInterrupt:
            print("\n")
            print("="*60)
            print("⚠️  Interrupted by user - shutting down...")
            print("="*60)

        finally:
            # 保存元数据
            try:
                capture.save_metadata()
            except Exception as e:
                print(f"   Warning: Error saving metadata: {e}")

            # 清理资源
            try:
                capture.close()
            except Exception as e:
                print(f"   Warning: Error during cleanup: {e}")

            print("\n✓ Capture completed\n")


if __name__ == '__main__':
    main()
