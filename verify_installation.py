#!/usr/bin/env python3
"""
安装验证脚本
检查所有模块是否可以正确导入和基本功能是否正常
"""

import sys
from pathlib import Path

print("="*70)
print("AI Drone Installation Verification")
print("="*70)
print()

# 检查Python版本
print(f"Python version: {sys.version}")
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required")
    sys.exit(1)
else:
    print("✓ Python version OK")
print()

# 检查必需的依赖
print("Checking dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'torchvision': 'TorchVision',
    'numpy': 'NumPy',
    'cv2': 'OpenCV',
    'PIL': 'Pillow',
    'mss': 'MSS',
    'evdev': 'evdev',
    'pynput': 'pynput',
    'pandas': 'Pandas',
    'yaml': 'PyYAML'
}

missing_deps = []
for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"  ✓ {name}")
    except ImportError:
        print(f"  ✗ {name} (missing)")
        missing_deps.append(name)

if missing_deps:
    print()
    print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
    print("   Install with: pip install -r requirements.txt")
    print()
else:
    print()
    print("✓ All dependencies installed")
    print()

# 检查目录结构
print("Checking directory structure...")
required_dirs = [
    'models',
    'record',
    'train',
    'deploy',
    'tools',
    'configs'
]

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"  ✓ {dir_name}/")
    else:
        print(f"  ✗ {dir_name}/ (missing)")

print()

# 检查关键文件
print("Checking key files...")
key_files = [
    'models/__init__.py',
    'models/resnet18_unet.py',
    'models/toge_policy.py',
    'record/datasets.py',
    'train/train.py',
    'deploy/run_policy.py',
    'tools/joystick_calibrate.py',
    'requirements.txt',
    'README.md'
]

for file_name in key_files:
    file_path = Path(file_name)
    if file_path.exists():
        print(f"  ✓ {file_name}")
    else:
        print(f"  ✗ {file_name} (missing)")

print()

# 尝试导入本地模块（如果依赖满足）
if not missing_deps:
    print("Testing module imports...")
    try:
        from models import list_models
        models = list_models()
        print(f"  ✓ models module: {', '.join(models)} available")
    except Exception as e:
        print(f"  ✗ models module: {e}")

    try:
        from record.datasets import FPVDataset
        print(f"  ✓ record.datasets module")
    except Exception as e:
        print(f"  ✗ record.datasets module: {e}")

    try:
        from train.engine import TrainingEngine
        print(f"  ✓ train.engine module")
    except Exception as e:
        print(f"  ✗ train.engine module: {e}")

    try:
        from deploy.virtual_joystick import VirtualJoystick
        print(f"  ✓ deploy.virtual_joystick module")
    except Exception as e:
        print(f"  ✗ deploy.virtual_joystick module: {e}")

    print()

# 检查系统依赖
print("Checking system dependencies...")
import subprocess

system_deps = {
    'xdotool': 'xdotool (for window detection)',
    'uinput': 'uinput module (for virtual joystick)'
}

for cmd, desc in system_deps.items():
    try:
        if cmd == 'uinput':
            result = subprocess.run(['lsmod'], capture_output=True, text=True)
            if 'uinput' in result.stdout:
                print(f"  ✓ {desc}")
            else:
                print(f"  ⚠ {desc} (not loaded, run: sudo modprobe uinput)")
        else:
            subprocess.run([cmd, '--version'], capture_output=True, check=True)
            print(f"  ✓ {desc}")
    except Exception:
        print(f"  ✗ {desc} (not found)")

print()
print("="*70)

if missing_deps:
    print("Status: Installation incomplete")
    print("Action: Install missing dependencies")
else:
    print("Status: Installation looks good!")
    print("Next steps:")
    print("  1. Record data: python -m record.liftoff_capture --help")
    print("  2. Train model: python -m train.train --help")
    print("  3. Deploy: python -m deploy.run_policy --help")

print("="*70)
