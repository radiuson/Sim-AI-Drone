"""
部署模块
作用：
    提供策略部署和实时控制功能
主要组件：
    - run_policy: 统一策略运行入口
    - screen_capture: 屏幕捕获
    - virtual_joystick: 虚拟遥控器
依赖：
    torch, mss, opencv-python, evdev
注意：
    需要加载训练好的模型检查点
示例：
    python -m deploy.run_policy --policy toge --checkpoint model.pt
"""

__all__ = []
