#!/usr/bin/env python3
"""
捕获性能对比测试
比较 MSS vs OBS 的性能差异
"""

import time
import argparse
import numpy as np
import psutil
import os
from deploy.screen_capture import ScreenCapture, OBSCapture


def test_capture_performance(capture, method_name: str, num_frames: int = 300):
    """
    测试捕获性能

    Args:
        capture: 捕获对象（ScreenCapture 或 OBSCapture）
        method_name: 方法名称（用于显示）
        num_frames: 测试帧数
    """
    print(f"\n{'='*60}")
    print(f"Testing {method_name} Performance")
    print(f"{'='*60}")

    # 获取当前进程
    process = psutil.Process(os.getpid())

    # 预热（前10帧不计入统计）
    print("Warming up...")
    for _ in range(10):
        frame = capture.capture_frame()
        if frame is None:
            print("❌ Failed to capture frame during warmup!")
            return None

    # 记录初始状态
    cpu_percent_start = process.cpu_percent(interval=0.1)
    mem_info_start = process.memory_info()

    # 开始测试
    print(f"Capturing {num_frames} frames...")
    frame_times = []
    failed_frames = 0

    start_time = time.time()

    for i in range(num_frames):
        frame_start = time.time()

        frame = capture.capture_frame()

        if frame is None:
            failed_frames += 1
        else:
            frame_times.append(time.time() - frame_start)

    end_time = time.time()

    # 记录结束状态
    cpu_percent_end = process.cpu_percent(interval=0.1)
    mem_info_end = process.memory_info()

    # 计算统计
    total_time = end_time - start_time
    avg_fps = num_frames / total_time

    if frame_times:
        avg_frame_time = np.mean(frame_times) * 1000  # ms
        min_frame_time = np.min(frame_times) * 1000
        max_frame_time = np.max(frame_times) * 1000
        std_frame_time = np.std(frame_times) * 1000
    else:
        avg_frame_time = min_frame_time = max_frame_time = std_frame_time = 0

    cpu_usage = (cpu_percent_start + cpu_percent_end) / 2
    mem_usage_mb = (mem_info_end.rss - mem_info_start.rss) / 1024 / 1024

    # 打印结果
    print(f"\n{'='*60}")
    print(f"Results for {method_name}")
    print(f"{'='*60}")
    print(f"总帧数:         {num_frames}")
    print(f"失败帧数:       {failed_frames}")
    print(f"成功率:         {(num_frames - failed_frames) / num_frames * 100:.1f}%")
    print(f"\n--- 帧率 ---")
    print(f"平均FPS:        {avg_fps:.1f}")
    print(f"平均帧时间:     {avg_frame_time:.2f} ms")
    print(f"最小帧时间:     {min_frame_time:.2f} ms")
    print(f"最大帧时间:     {max_frame_time:.2f} ms")
    print(f"帧时间标准差:   {std_frame_time:.2f} ms")
    print(f"\n--- 资源占用 ---")
    print(f"CPU使用率:      {cpu_usage:.1f}%")
    print(f"内存增长:       {mem_usage_mb:.1f} MB")
    print(f"{'='*60}\n")

    return {
        'method': method_name,
        'fps': avg_fps,
        'frame_time_ms': avg_frame_time,
        'frame_time_std_ms': std_frame_time,
        'cpu_percent': cpu_usage,
        'mem_mb': mem_usage_mb,
        'failed_frames': failed_frames
    }


def main():
    parser = argparse.ArgumentParser(description='测试屏幕捕获性能')
    parser.add_argument(
        '--methods',
        nargs='+',
        choices=['mss', 'obs', 'both'],
        default=['both'],
        help='测试方法: mss, obs, 或 both'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=300,
        help='每个测试捕获的帧数'
    )
    parser.add_argument(
        '--window-name',
        type=str,
        default='Liftoff',
        help='Liftoff窗口名称（用于MSS）'
    )
    parser.add_argument(
        '--obs-device',
        type=str,
        default='/dev/video10',
        help='OBS虚拟摄像头设备路径'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='捕获图像尺寸'
    )

    args = parser.parse_args()

    print("="*60)
    print("Screen Capture Performance Test")
    print("="*60)
    print(f"测试帧数: {args.num_frames}")
    print(f"图像尺寸: {args.image_size}x{args.image_size}")
    print("="*60)

    results = []
    test_methods = args.methods if 'both' not in args.methods else ['mss', 'obs']

    # 测试 MSS
    if 'mss' in test_methods:
        print("\n准备测试 MSS 方法...")
        capture_mss = ScreenCapture(
            window_name=args.window_name,
            target_size=(args.image_size, args.image_size),
            auto_find_window=True
        )

        if capture_mss.monitor is None:
            print("❌ 无法找到Liftoff窗口，跳过MSS测试")
        else:
            result = test_capture_performance(capture_mss, "MSS", args.num_frames)
            if result:
                results.append(result)
            capture_mss.close()

            # 等待一下再测试下一个
            time.sleep(2)

    # 测试 OBS
    if 'obs' in test_methods:
        print("\n准备测试 OBS 方法...")
        capture_obs = OBSCapture(
            device_path=args.obs_device,
            target_size=(args.image_size, args.image_size),
            auto_detect=True
        )

        if capture_obs.cap is None:
            print("❌ 无法打开OBS虚拟摄像头，跳过OBS测试")
            print("   请确保:")
            print("   1. v4l2loopback已加载")
            print("   2. OBS正在运行且虚拟摄像头已启动")
        else:
            result = test_capture_performance(capture_obs, "OBS", args.num_frames)
            if result:
                results.append(result)
            capture_obs.close()

    # 打印对比
    if len(results) >= 2:
        print("\n" + "="*60)
        print("性能对比")
        print("="*60)

        mss_result = next((r for r in results if r['method'] == 'MSS'), None)
        obs_result = next((r for r in results if r['method'] == 'OBS'), None)

        if mss_result and obs_result:
            print(f"\n{'指标':<20} {'MSS':<15} {'OBS':<15} {'提升':<10}")
            print("-" * 60)

            fps_improvement = (obs_result['fps'] - mss_result['fps']) / mss_result['fps'] * 100
            print(f"{'FPS':<20} {mss_result['fps']:<15.1f} {obs_result['fps']:<15.1f} {fps_improvement:>+.1f}%")

            frame_time_improvement = (mss_result['frame_time_ms'] - obs_result['frame_time_ms']) / mss_result['frame_time_ms'] * 100
            print(f"{'帧时间 (ms)':<20} {mss_result['frame_time_ms']:<15.2f} {obs_result['frame_time_ms']:<15.2f} {frame_time_improvement:>+.1f}%")

            cpu_improvement = (mss_result['cpu_percent'] - obs_result['cpu_percent']) / mss_result['cpu_percent'] * 100
            print(f"{'CPU使用率 (%)':<20} {mss_result['cpu_percent']:<15.1f} {obs_result['cpu_percent']:<15.1f} {cpu_improvement:>+.1f}%")

            print("-" * 60)
            print(f"\n{'结论':<20} ", end='')
            if obs_result['fps'] > mss_result['fps']:
                print(f"OBS 比 MSS 快 {fps_improvement:.1f}%")
            else:
                print(f"MSS 比 OBS 快 {-fps_improvement:.1f}%")
            print("="*60)


if __name__ == '__main__':
    main()
