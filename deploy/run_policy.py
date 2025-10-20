"""
统一策略部署入口
作用：
    加载训练好的模型，实时捕获Liftoff画面并控制虚拟遥控器
主要功能：
    - 支持多种策略（resnet_unet, toge等）
    - 屏幕捕获与图像预处理
    - 动作平滑（EMA）与限幅
    - 虚拟遥控器输出
依赖：
    torch, numpy, mss, opencv-python, evdev
注意：
    需要先在Liftoff中配置虚拟遥控器
    推荐在GPU上运行以获得更高帧率
示例：
    # 使用默认设置（OBS捕获 + 标准30Hz模式）
    python -m deploy.run_policy \\
        --policy toge \\
        --checkpoint outputs/toge_best.pt \\
        --rate 30

    # 双频率模式（推荐 - 30Hz视觉 + 100Hz动作）
    python -m deploy.run_policy \\
        --policy toge \\
        --checkpoint outputs/toge_best.pt \\
        --dual-rate \\
        --visual-rate 30 \\
        --action-rate 100

    # 使用MSS捕获（备选）
    python -m deploy.run_policy \\
        --policy toge \\
        --checkpoint outputs/toge_best.pt \\
        --capture-method mss \\
        --window-name "Liftoff"
"""

import time
import argparse
from pathlib import Path
from typing import Optional, Tuple
import threading
import numpy as np
import torch
import cv2
from torchvision import transforms

# 导入本地模块
from models import get_model, list_models
from deploy.screen_capture import ScreenCapture, OBSCapture
from deploy.virtual_joystick import VirtualJoystick


class PolicyRunner:
    """
    策略运行器
    """

    def __init__(
        self,
        policy_name: str,
        checkpoint_path: str,
        image_size: int = 224,
        device: str = 'cuda',
        ema_alpha: float = 0.2,
        max_action_change: float = 0.3,
        num_diffusion_steps: int = 10,
        dual_rate: bool = False,
        visual_rate: int = 30,
        action_rate: int = 100
    ):
        """
        初始化策略运行器

        Args:
            policy_name: 策略名称（如 'toge', 'resnet_unet'）
            checkpoint_path: 模型检查点路径
            image_size: 输入图像尺寸
            device: 运行设备 ('cuda' 或 'cpu')
            ema_alpha: EMA平滑系数（0=不平滑，1=完全跟随新值）
            max_action_change: 单步最大动作变化
            num_diffusion_steps: 扩散模型采样步数
            dual_rate: 是否启用双频率推理（视觉30Hz + 动作100Hz）
            visual_rate: 视觉编码频率（Hz）
            action_rate: 动作预测频率（Hz）
        """
        self.policy_name = policy_name
        self.image_size = image_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.ema_alpha = ema_alpha
        self.max_action_change = max_action_change
        self.num_diffusion_steps = num_diffusion_steps
        self.dual_rate = dual_rate
        self.visual_rate = visual_rate
        self.action_rate = action_rate

        print(f"Initializing PolicyRunner...")
        print(f"  Policy: {policy_name}")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Device: {self.device}")
        print(f"  Image size: {image_size}x{image_size}")
        print(f"  EMA alpha: {ema_alpha}")
        print(f"  Max action change: {max_action_change}")

        # 加载模型
        self.model = self._load_model(checkpoint_path)
        self.model.eval()

        # 检查模型是否支持双频率推理
        self.supports_dual_rate = hasattr(self.model, 'encode_visual') and \
                                   hasattr(self.model, 'predict_fast')

        if self.dual_rate:
            if not self.supports_dual_rate:
                print(f"⚠️  Dual-rate not supported by {policy_name}, falling back to standard mode")
                self.dual_rate = False
            else:
                print(f"  Dual-rate mode: Visual {visual_rate}Hz + Action {action_rate}Hz")

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 状态管理
        self.last_action = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.mock_state = np.zeros(13, dtype=np.float32)  # TODO: 从ROS2获取真实状态

        # 动作历史缓冲区（用于模型输入）
        # 获取模型需要的历史长度
        self.action_history_len = getattr(self.model, 'action_history_len', 4)
        self.use_action_history = getattr(self.model, 'use_action_history', False)

        if self.use_action_history:
            # 初始化为零动作的历史
            self.action_history_buffer = np.zeros(
                (self.action_history_len, 4),
                dtype=np.float32
            )
            print(f"  Action history enabled: {self.action_history_len} frames")
        else:
            self.action_history_buffer = None
            print(f"  Action history: disabled")

        # 双频率推理相关
        if self.dual_rate:
            self.visual_feat_cache = None  # 缓存的视觉特征
            self.visual_feat_lock = threading.Lock()  # 线程锁
            self.visual_encoding_thread = None
            self.stop_visual_thread = threading.Event()
            self.screen_capture = None  # 将在run时设置
            print(f"  Visual feature cache initialized")

        print("✓ PolicyRunner initialized successfully")

    def _load_model(self, checkpoint_path: str):
        """加载模型"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")

        # 创建模型实例
        # 注意：这里需要根据具体模型设置正确的参数
        if self.policy_name == 'toge':
            model = get_model(
                'toge',
                action_dim=4,
                state_dim=13,
                horizon=4,
                visual_backbone='efficientnet_b3',
                pretrained_backbone=False  # 使用训练后的权重
            )
        elif self.policy_name in ['resnet_unet', 'fpv_diffusion']:
            model = get_model(
                'resnet_unet',
                action_dim=4,
                horizon=4,
                pretrained_backbone=False
            )
        else:
            raise ValueError(f"Unknown policy: {self.policy_name}")

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        print("✓ Model loaded successfully")

        return model

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像

        Args:
            image: [H, W, 3] BGR图像

        Returns:
            [1, 3, H, W] 归一化的图像张量
        """
        # BGR → RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 变换
        image_tensor = self.transform(image_rgb)

        # 添加batch维度
        return image_tensor.unsqueeze(0)

    def _visual_encoding_loop(self):
        """
        视觉编码线程循环（30Hz）
        持续捕获图像并更新视觉特征缓存
        """
        loop_interval = 1.0 / self.visual_rate

        while not self.stop_visual_thread.is_set():
            loop_start = time.time()

            # 捕获图像
            if self.screen_capture is not None:
                image = self.screen_capture.capture_frame()
                if image is not None:
                    # 预处理图像
                    image_tensor = self.preprocess_image(image).to(self.device)

                    # 编码视觉特征
                    with torch.no_grad():
                        visual_feat = self.model.encode_visual(image_tensor)  # [1, base_dim]

                    # 更新缓存（线程安全）
                    with self.visual_feat_lock:
                        self.visual_feat_cache = visual_feat

            # 控制循环频率
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start_visual_encoding_thread(self, screen_capture):
        """
        启动视觉编码后台线程

        Args:
            screen_capture: ScreenCapture实例
        """
        if not self.dual_rate:
            return

        self.screen_capture = screen_capture
        self.stop_visual_thread.clear()
        self.visual_encoding_thread = threading.Thread(
            target=self._visual_encoding_loop,
            daemon=True
        )
        self.visual_encoding_thread.start()
        print(f"✓ Visual encoding thread started at {self.visual_rate}Hz")

    def stop_visual_encoding_thread(self):
        """停止视觉编码线程"""
        if self.dual_rate and self.visual_encoding_thread is not None:
            self.stop_visual_thread.set()
            self.visual_encoding_thread.join(timeout=2.0)
            print("✓ Visual encoding thread stopped")

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        预测动作（标准模式，用于非双频率推理）

        Args:
            image: [H, W, 3] BGR图像

        Returns:
            [4] 动作 (throttle, yaw, pitch, roll)
        """
        # 预处理
        image_tensor = self.preprocess_image(image).to(self.device)
        state_tensor = torch.from_numpy(self.mock_state).unsqueeze(0).to(self.device)

        # 准备动作历史张量（如果模型需要）
        action_history_tensor = None
        if self.use_action_history and self.action_history_buffer is not None:
            action_history_tensor = torch.from_numpy(
                self.action_history_buffer
            ).unsqueeze(0).to(self.device)  # [1, action_history_len, 4]

        # 推理
        action_seq = self.model.predict(
            image_tensor,
            state_tensor,
            action_history=action_history_tensor,
            num_diffusion_steps=self.num_diffusion_steps
        )  # [1, horizon, 4]

        # 取第一个时间步
        action = action_seq[0, 0].cpu().numpy()  # [4]

        # 限幅
        action = np.clip(action, -1.0, 1.0)

        # 限制单步变化
        action_change = action - self.last_action
        action_change = np.clip(action_change, -self.max_action_change, self.max_action_change)
        action = self.last_action + action_change

        # EMA平滑
        action = self.ema_alpha * action + (1 - self.ema_alpha) * self.last_action

        # 更新动作历史缓冲区（滚动窗口）
        if self.use_action_history and self.action_history_buffer is not None:
            # 移除最老的动作，添加新动作
            self.action_history_buffer = np.roll(self.action_history_buffer, -1, axis=0)
            self.action_history_buffer[-1] = action

        # 更新状态
        self.last_action = action

        return action

    @torch.no_grad()
    def predict_fast(self) -> np.ndarray:
        """
        快速预测动作（双频率模式，使用缓存的视觉特征）

        Returns:
            [4] 动作 (throttle, yaw, pitch, roll)
        """
        # 获取缓存的视觉特征（线程安全）
        with self.visual_feat_lock:
            if self.visual_feat_cache is None:
                # 如果还没有缓存，返回零动作
                return self.last_action
            visual_feat = self.visual_feat_cache

        # 准备状态张量
        state_tensor = torch.from_numpy(self.mock_state).unsqueeze(0).to(self.device)

        # 准备动作历史张量（如果模型需要）
        action_history_tensor = None
        if self.use_action_history and self.action_history_buffer is not None:
            action_history_tensor = torch.from_numpy(
                self.action_history_buffer
            ).unsqueeze(0).to(self.device)  # [1, action_history_len, 4]

        # 快速推理（跳过视觉编码）
        action_seq = self.model.predict_fast(
            visual_feat,
            state_tensor,
            action_history=action_history_tensor,
            num_diffusion_steps=self.num_diffusion_steps
        )  # [1, horizon, 4]

        # 取第一个时间步
        action = action_seq[0, 0].cpu().numpy()  # [4]

        # 限幅
        action = np.clip(action, -1.0, 1.0)

        # 限制单步变化
        action_change = action - self.last_action
        action_change = np.clip(action_change, -self.max_action_change, self.max_action_change)
        action = self.last_action + action_change

        # EMA平滑
        action = self.ema_alpha * action + (1 - self.ema_alpha) * self.last_action

        # 更新动作历史缓冲区（滚动窗口）
        if self.use_action_history and self.action_history_buffer is not None:
            # 移除最老的动作，添加新动作
            self.action_history_buffer = np.roll(self.action_history_buffer, -1, axis=0)
            self.action_history_buffer[-1] = action

        # 更新状态
        self.last_action = action

        return action


def main():
    parser = argparse.ArgumentParser(
        description='Unified Policy Runner for Liftoff Control',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run TOGE policy
  python -m deploy.run_policy \\
      --policy toge \\
      --checkpoint outputs/checkpoints/best.pt \\
      --window-name "Liftoff" \\
      --image-size 224 \\
      --rate 30

  # Run ResNet-UNet policy with custom settings
  python -m deploy.run_policy \\
      --policy resnet_unet \\
      --checkpoint outputs/fpv_diffusion_best.pt \\
      --ema 0.3 \\
      --max-action-change 0.2 \\
      --device cpu

Available policies: """ + ", ".join(list_models())
    )

    parser.add_argument(
        '--policy',
        type=str,
        required=True,
        choices=list_models(),
        help='Policy name'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='Liftoff',
        help='Liftoff window name for capture'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Input image size (square)'
    )
    parser.add_argument(
        '--rate',
        type=int,
        default=30,
        help='Control loop rate (Hz)'
    )
    parser.add_argument(
        '--ema',
        type=float,
        default=0.2,
        help='EMA smoothing alpha (0-1)'
    )
    parser.add_argument(
        '--max-action-change',
        type=float,
        default=0.3,
        help='Maximum action change per step'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run model on'
    )
    parser.add_argument(
        '--num-diffusion-steps',
        type=int,
        default=10,
        help='Number of diffusion sampling steps'
    )
    parser.add_argument(
        '--dual-rate',
        action='store_true',
        help='Enable dual-rate inference (visual 30Hz + action 100Hz)'
    )
    parser.add_argument(
        '--visual-rate',
        type=int,
        default=30,
        help='Visual encoding rate (Hz) for dual-rate mode'
    )
    parser.add_argument(
        '--action-rate',
        type=int,
        default=100,
        help='Action prediction rate (Hz) for dual-rate mode'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Show debug visualization'
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

    args = parser.parse_args()

    print("="*70)
    print("Policy Runner for Liftoff")
    print("="*70)
    print()

    # 创建策略运行器
    runner = PolicyRunner(
        policy_name=args.policy,
        checkpoint_path=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        ema_alpha=args.ema,
        max_action_change=args.max_action_change,
        num_diffusion_steps=args.num_diffusion_steps,
        dual_rate=args.dual_rate,
        visual_rate=args.visual_rate,
        action_rate=args.action_rate
    )

    print()
    print("Initializing screen capture and virtual joystick...")

    # 创建屏幕捕获（根据选择的方法）
    if args.capture_method == 'obs':
        print(f"Using OBS virtual camera capture (device: {args.obs_device})")
        capture = OBSCapture(
            device_path=args.obs_device,
            target_size=(args.image_size, args.image_size),
            auto_detect=True
        )

        if capture.cap is None:
            print()
            print("❌ Failed to open OBS virtual camera!")
            print("   Solutions:")
            print("   1. Make sure v4l2loopback is loaded:")
            print("      sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"OBS\"")
            print("   2. Make sure OBS is running with virtual camera started")
            print("   3. Or use --capture-method mss for traditional screen capture")
            return
    else:
        print("Using MSS screen capture")
        capture = ScreenCapture(
            window_name=args.window_name,
            target_size=(args.image_size, args.image_size),
            auto_find_window=True
        )

        if capture.monitor is None:
            print()
            print("❌ Failed to find Liftoff window!")
            print("   Please make sure Liftoff is running")
            print("   Or use --capture-method obs for OBS virtual camera")
            return

    # 创建虚拟遥控器
    joystick = VirtualJoystick(device_name="AI-Liftoff-Controller")

    # 启动双频率模式的视觉编码线程（如果启用）
    if runner.dual_rate:
        runner.start_visual_encoding_thread(capture)
        print(f"✓ Dual-rate mode active: Visual {args.visual_rate}Hz + Action {args.action_rate}Hz")

    print()
    print("="*70)
    print("Running policy control loop...")
    print("Press Ctrl+C to stop")
    print("="*70)
    print()

    # 控制循环频率：双频率模式使用action_rate，标准模式使用rate
    control_rate = args.action_rate if runner.dual_rate else args.rate
    loop_interval = 1.0 / control_rate
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time

    try:
        while True:
            loop_start = time.time()

            # 预测动作（根据模式选择方法）
            if runner.dual_rate:
                # 双频率模式：使用缓存的视觉特征
                action = runner.predict_fast()
            else:
                # 标准模式：完整推理
                image = capture.capture_frame()
                if image is None:
                    print("⚠️  Failed to capture frame")
                    time.sleep(0.1)
                    continue
                action = runner.predict(image)

            # 发送到虚拟遥控器
            joystick.send_action(action.tolist())

            # 统计
            frame_count += 1
            current_time = time.time()

            # 每秒打印一次状态
            if current_time - last_print_time >= 1.0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                mode_str = f"Dual-rate {args.visual_rate}/{args.action_rate}Hz" if runner.dual_rate else f"Standard {args.rate}Hz"
                print(
                    f"[{mode_str}] FPS: {fps:5.1f} | "
                    f"Action: [{action[0]:+.2f}, {action[1]:+.2f}, {action[2]:+.2f}, {action[3]:+.2f}] | "
                    f"Frames: {frame_count}"
                )
                last_print_time = current_time

            # Debug可视化（仅标准模式支持）
            if args.debug and not runner.dual_rate:
                debug_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # 绘制动作信息
                text = f"T:{action[0]:+.2f} Y:{action[1]:+.2f} P:{action[2]:+.2f} R:{action[3]:+.2f}"
                cv2.putText(debug_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Policy Debug', debug_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 控制循环频率
            loop_elapsed = time.time() - loop_start
            sleep_time = max(0, loop_interval - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")

    finally:
        # 停止视觉编码线程
        if runner.dual_rate:
            runner.stop_visual_encoding_thread()

        # 清理
        joystick.reset()
        joystick.close()
        capture.close()
        if args.debug:
            cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print()
        print("="*70)
        print(f"Session Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Average FPS: {avg_fps:.1f}")
        print("="*70)
        print("✓ Shutdown complete")


if __name__ == '__main__':
    main()
