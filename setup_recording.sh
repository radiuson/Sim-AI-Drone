#!/bin/bash
# 数据采集环境设置脚本

echo "========================================"
echo "Setting up Data Collection Environment"
echo "========================================"
echo ""

# 安装 Python 依赖
echo "Installing Python dependencies..."
pip install inputs

echo ""
echo "✓ Setup completed!"
echo ""
echo "Next steps:"
echo "1. Connect RadioMaster in Joystick mode"
echo "2. Start Liftoff with UDP output enabled"
echo "3. Run ./start_bridge.sh"
echo "4. Run: python -m record.liftoff_capture --output-dir ./dataset/flights"
echo ""
