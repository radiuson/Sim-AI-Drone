#!/bin/bash
# Liftoff ROS2 Bridge 启动脚本

echo "========================================"
echo "Starting Liftoff ROS2 Bridge"
echo "========================================"
echo ""

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo "⚠️  ROS2 environment not sourced!"
    echo "   Trying to source ROS2 Jazzy..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        echo "✓ Sourced ROS2 Jazzy"
    else
        echo "❌ ROS2 not found! Please install ROS2 Jazzy"
        exit 1
    fi
else
    echo "✓ ROS2 $ROS_DISTRO detected"
fi

echo ""
echo "Bridge Configuration:"
echo "  Host: 127.0.0.1"
echo "  Port: 30001"
echo "  Print Rate: 2 Hz"
echo ""
echo "Published Topics:"
echo "  - /liftoff/pose  (PoseStamped)"
echo "  - /liftoff/twist (TwistStamped)"
echo "  - /liftoff/imu   (Imu)"
echo "  - /liftoff/rc    (Joy)"
echo ""
echo "Starting bridge... (Press Ctrl+C to stop)"
echo "========================================"
echo ""

# 启动 bridge
cd "$(dirname "$0")"
python3 liftoff_bridge_ros2.py
