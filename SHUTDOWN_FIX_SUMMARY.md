# Shutdown 卡住问题修复总结

## 问题描述

运行 `start_data_collection.sh` 时，按 Ctrl+C 无法正常退出，程序卡在 ROS2 线程清理阶段。

### 错误现象

```
Exception in thread Thread-1 (_spin_loop):
========================================
Shutting down...
========================================
Stopping data capture (PID: 39788)...
^C   <-- 多次 Ctrl+C 无响应
```

## 根本原因

1. **ROS2 spin 线程无限循环**
   - `_spin_loop()` 使用 `while rclpy.ok()` 无限循环
   - 没有停止标志，daemon 线程无法正常退出
   - `join()` 会永远阻塞

2. **异常处理不完善**
   - 线程清理时没有捕获异常
   - ROS2 节点销毁时可能抛出异常导致卡住

3. **调试信息缺失**
   - 无法看到清理过程在哪个环节卡住

## 修复方案

### 1. 添加线程控制标志 ([liftoff_capture.py:96-97](record/liftoff_capture.py#L96-L97))

```python
class ROS2DataReceiver:
    def __init__(self):
        # ...
        # 线程控制
        self.running = True  # ✅ 新增停止标志
```

### 2. 修复 spin 循环 ([liftoff_capture.py:183-191](record/liftoff_capture.py#L183-L191))

```python
def _spin_loop(self):
    """ROS2 spin 循环"""
    while self.running and rclpy.ok():  # ✅ 检查 running 标志
        try:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        except Exception as e:
            if self.running:  # 只在仍在运行时打印错误
                print(f"⚠️  ROS2 spin error: {e}")
            break
```

**改进点：**
- 添加 `self.running` 检查，允许外部停止循环
- 增加异常捕获，防止意外错误导致线程卡死
- 增大 timeout 到 0.1s，减少 CPU 占用

### 3. 改进 shutdown 方法 ([liftoff_capture.py:205-221](record/liftoff_capture.py#L205-L221))

```python
def shutdown(self):
    """关闭接收器"""
    print("  Shutting down ROS2 receiver...")
    self.running = False  # ✅ 设置停止标志

    # 等待 spin 线程结束（最多 2 秒）
    if self.spin_thread and self.spin_thread.is_alive():
        self.spin_thread.join(timeout=2.0)  # ✅ 带超时的 join

    # 销毁节点
    if HAS_ROS2 and rclpy.ok():
        try:
            self.node.destroy_node()  # ✅ 捕获异常
        except Exception as e:
            print(f"  Warning: Error destroying node: {e}")

    print("  ✓ ROS2 receiver stopped")
```

**改进点：**
- 先设置 `running = False` 让循环退出
- 使用 `join(timeout=2.0)` 避免永久阻塞
- 捕获节点销毁异常，确保清理继续进行
- 添加详细的日志输出

### 4. 增强 LiftoffCapture.close() ([liftoff_capture.py:514-540](record/liftoff_capture.py#L514-L540))

```python
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
```

**改进点：**
- 添加详细的进度输出，可以看到清理进行到哪一步
- 每个清理步骤都有独立的异常捕获
- 确保一个组件失败不影响其他组件的清理

### 5. 改进异常处理 ([liftoff_capture.py:693-720](record/liftoff_capture.py#L693-L720))

```python
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
```

**改进点：**
- 每个清理步骤都有独立的 try-except
- 即使某步骤失败，也会继续执行后续清理
- 清晰的视觉分隔和状态提示

## 测试验证

### 测试步骤

1. **正常退出测试**
   ```bash
   ./start_data_collection.sh
   # 等待启动完成
   # 按 Ctrl+C
   # 应该看到清晰的关闭流程输出
   ```

2. **预期输出**
   ```
   ^C
   ============================================================
   ⚠️  Interrupted by user - shutting down...
   ============================================================

   🔄 Cleaning up resources...
     Stopping gamepad controller...
     ✓ Gamepad controller stopped
     Shutting down ROS2 receiver...
     ✓ ROS2 receiver stopped
     Closing video capture...
     ✓ Video capture closed
   ✓ All resources cleaned up

   ✓ Capture completed
   ```

3. **超时测试**
   - 如果 ROS2 线程真的卡死，2 秒后会强制继续
   - 不会永久阻塞

## 修改文件清单

- ✅ [record/liftoff_capture.py](record/liftoff_capture.py)
  - 第 96-97 行：添加 `running` 标志
  - 第 183-191 行：修复 `_spin_loop()`
  - 第 205-221 行：改进 `shutdown()`
  - 第 514-540 行：增强 `close()`
  - 第 693-720 行：改进 KeyboardInterrupt 处理（gamepad 模式）
  - 第 742-761 行：改进 KeyboardInterrupt 处理（keyboard 模式）

## 关键改进

| 问题 | 修复前 | 修复后 |
|------|--------|--------|
| ROS2 线程无法停止 | 无限循环 | `running` 标志控制 |
| join() 永久阻塞 | 无超时 | 2 秒超时 |
| 异常导致卡死 | 未捕获 | 全面异常处理 |
| 无法诊断卡点 | 无日志 | 详细进度输出 |
| 清理不完整 | 一个失败全失败 | 独立异常捕获 |

## 附加优化

- 将 `spin_once` timeout 从 0.01s 增加到 0.1s，降低 CPU 占用
- 在所有关键清理步骤添加日志，方便调试
- 确保 daemon 线程正确退出，不留僵尸进程

## 下次使用

现在可以放心使用 Ctrl+C 退出程序，应该能在 2-3 秒内干净地关闭所有组件。

如果仍然遇到问题，请检查日志输出，看看卡在哪个具体步骤。
