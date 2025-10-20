#!/bin/bash
# 完整系统启动脚本
# 自动启动：v4l2loopback + OBS + ROS2 Bridge + 数据采集

# 默认参数
OUTPUT_DIR="${1:-./dataset/liftoff_data}"
FPS="${2:-30}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clear
echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║      Liftoff AI Data Collection System v2.0              ║
║                                                           ║
║      Full Automatic Startup                              ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

# PID 追踪
OBS_PID=""
BRIDGE_PID=""
CAPTURE_PID=""

# 清理函数
cleanup() {
    echo ""
    echo -e "${YELLOW}========================================"
    echo "Shutting down all components..."
    echo -e "========================================${NC}"

    # 停止数据采集
    if [ ! -z "$CAPTURE_PID" ] && ps -p $CAPTURE_PID > /dev/null; then
        echo "Stopping data capture..."
        kill $CAPTURE_PID 2>/dev/null
        wait $CAPTURE_PID 2>/dev/null
    fi

    # 停止 bridge
    if [ ! -z "$BRIDGE_PID" ] && ps -p $BRIDGE_PID > /dev/null; then
        echo "Stopping ROS2 bridge..."
        kill $BRIDGE_PID 2>/dev/null
        wait $BRIDGE_PID 2>/dev/null
    fi

    # 停止 OBS（如果我们启动的）
    if [ ! -z "$OBS_PID" ] && ps -p $OBS_PID > /dev/null; then
        echo "Stopping OBS..."
        kill $OBS_PID 2>/dev/null
    fi

    echo -e "${GREEN}✓ All components stopped${NC}"
    echo ""
    exit 0
}

trap cleanup SIGINT SIGTERM

# Step 1: 检查和加载 v4l2loopback
echo -e "${BLUE}[Step 1/5] Setting up virtual camera...${NC}"
if [ ! -e "/dev/video10" ]; then
    echo "  Loading v4l2loopback module..."
    sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS Virtual Camera" exclusive_caps=1

    if [ -e "/dev/video10" ]; then
        echo -e "${GREEN}  ✓ Virtual camera loaded at /dev/video10${NC}"
    else
        echo -e "${RED}  ❌ Failed to load v4l2loopback${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  ✓ Virtual camera already exists${NC}"
fi
echo ""

# Step 2: 检查 ROS2 环境
echo -e "${BLUE}[Step 2/5] Checking ROS2 environment...${NC}"
if [ -z "$ROS_DISTRO" ]; then
    echo "  Sourcing ROS2 Jazzy..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        echo -e "${GREEN}  ✓ ROS2 Jazzy loaded${NC}"
    else
        echo -e "${RED}  ❌ ROS2 not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}  ✓ ROS2 $ROS_DISTRO already loaded${NC}"
fi
echo ""

# Step 3: 启动 OBS
echo -e "${BLUE}[Step 3/5] Starting OBS Studio...${NC}"
if pgrep -x "obs" > /dev/null; then
    echo -e "${GREEN}  ✓ OBS already running${NC}"
else
    echo "  Launching OBS in background..."
    obs --minimize-to-tray --startreplaybuffer > /dev/null 2>&1 &
    OBS_PID=$!

    echo "  Waiting for OBS to initialize (10 seconds)..."
    for i in {10..1}; do
        echo -ne "  \r  Waiting... $i seconds  "
        sleep 1
    done
    echo ""

    if pgrep -x "obs" > /dev/null; then
        echo -e "${GREEN}  ✓ OBS started${NC}"
        echo -e "${YELLOW}  ⚠️  Please manually:${NC}"
        echo "     1. Add 'Window Capture' source for Liftoff"
        echo "     2. Click 'Start Virtual Camera' button"
        echo ""
        read -p "  Press Enter when OBS virtual camera is started..."
    else
        echo -e "${RED}  ❌ Failed to start OBS${NC}"
        exit 1
    fi
fi
echo ""

# Step 4: 启动 ROS2 Bridge
echo -e "${BLUE}[Step 4/5] Starting ROS2 Bridge...${NC}"

# 检查并清理已存在的 bridge 进程
EXISTING_BRIDGE=$(pgrep -f "python3 liftoff_bridge_ros2.py")
if [ ! -z "$EXISTING_BRIDGE" ]; then
    echo -e "${YELLOW}  ⚠️  Found existing bridge process (PID: $EXISTING_BRIDGE)${NC}"
    echo "  Killing existing bridge..."
    kill $EXISTING_BRIDGE 2>/dev/null
    sleep 1
    echo -e "${GREEN}  ✓ Existing bridge stopped${NC}"
fi

# 创建日志目录
mkdir -p "$SCRIPT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BRIDGE_LOG="$SCRIPT_DIR/logs/bridge_${TIMESTAMP}.log"

cd "$SCRIPT_DIR"
python3 liftoff_bridge_ros2.py > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

echo "  Waiting for bridge to start..."
sleep 3

if ps -p $BRIDGE_PID > /dev/null; then
    echo -e "${GREEN}  ✓ ROS2 Bridge running (PID: $BRIDGE_PID)${NC}"
    echo "    Log: $BRIDGE_LOG"

    # 验证 ROS2 话题
    if ros2 topic list 2>/dev/null | grep -q "/liftoff"; then
        echo -e "${GREEN}    ✓ ROS2 topics active:${NC}"
        ros2 topic list | grep liftoff | sed 's/^/      - /'
    fi
else
    echo -e "${RED}  ❌ Bridge failed to start${NC}"
    echo "    Check log: $BRIDGE_LOG"
    cleanup
    exit 1
fi
echo ""

# Step 5: 启动数据采集
echo -e "${BLUE}[Step 5/5] Starting Data Collection...${NC}"
echo "  Output directory: $OUTPUT_DIR"
echo "  FPS: $FPS"
echo ""

CAPTURE_LOG="$SCRIPT_DIR/logs/capture_${TIMESTAMP}.log"

echo -e "${GREEN}========================================"
echo "🚀 System Ready!"
echo -e "========================================${NC}"
echo ""
echo -e "${CYAN}📝 Configuration:${NC}"
echo "  - Virtual Camera: /dev/video10"
echo "  - ROS2 Bridge: UDP Port 30001"
echo "  - Output: $OUTPUT_DIR"
echo "  - FPS: $FPS"
echo ""
echo -e "${GREEN}🎮 RadioMaster Controls:${NC}"
echo "  - ${YELLOW}SH switch UP${NC}   → Start recording ▶️"
echo "  - ${YELLOW}SA switch UP${NC}   → Stop recording ⏹️"
echo "  - ${YELLOW}BTN_SOUTH${NC}      → Emergency stop 🛑"
echo ""
echo -e "${CYAN}📋 Checklist:${NC}"
echo "  ✓ Virtual camera loaded"
echo "  ✓ ROS2 environment ready"
echo "  ✓ OBS virtual camera active"
echo "  ✓ ROS2 Bridge running"
echo "  ✓ RadioMaster connected"
echo ""
echo -e "${YELLOW}⚠️  Make sure Liftoff is running with UDP output enabled!${NC}"
echo "   (Settings → Extras → UDP Output → 127.0.0.1:30001)"
echo ""
echo -e "${GREEN}Press Ctrl+C to stop all components${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 启动数据采集（前台）
python3 -m record.liftoff_capture \
    --output-dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    2>&1 | tee "$CAPTURE_LOG" &

CAPTURE_PID=$!

# 等待
wait $CAPTURE_PID

# 清理
cleanup
