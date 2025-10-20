#!/bin/bash
# 一键启动数据采集系统
# 包括：ROS2 Bridge + 数据采集器

# 默认参数
OUTPUT_DIR="${1:-./dataset/liftoff_data}"
FPS="${2:-30}"
IMAGE_SIZE="${3:-224}"
CAPTURE_METHOD="${4:-obs}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}========================================"
echo "Liftoff Data Collection System"
echo -e "========================================${NC}"
echo ""

# 检查 ROS2 环境
if [ -z "$ROS_DISTRO" ]; then
    echo -e "${YELLOW}⚠️  ROS2 environment not sourced!${NC}"
    echo "   Trying to source ROS2 Jazzy..."
    if [ -f "/opt/ros/jazzy/setup.bash" ]; then
        source /opt/ros/jazzy/setup.bash
        echo -e "${GREEN}✓ Sourced ROS2 Jazzy${NC}"
    else
        echo -e "${RED}❌ ROS2 not found! Please install ROS2 Jazzy${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ ROS2 $ROS_DISTRO detected${NC}"
fi

# 检查并清理占用 /dev/video10 的进程（排除用户正在使用的OBS）
cleanup_video_device() {
    echo -e "${BLUE}🔍 Checking for processes using /dev/video10...${NC}"
    
    # 查找使用 /dev/video10 的进程
    local pids=$(lsof /dev/video10 2>/dev/null | awk 'NR>1 {print $2}' | sort -u)
    
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}⚠️  Found processes using /dev/video10:${NC}"
        lsof /dev/video10 2>/dev/null | awk 'NR>1 {print "   - PID " $2 " (" $1 ")"}'
        
        local cleaned_pids=""
        
        # 优雅地终止进程（排除用户正在使用的OBS）
        for pid in $pids; do
            # 获取进程名
            local process_name=$(ps -p $pid -o comm= 2>/dev/null)
            
            # 如果是OBS且是用户手动启动的，跳过
            if [ "$process_name" = "obs" ]; then
                # 检查是否是我们刚刚启动的OBS
                if [ "$pid" = "$OBS_PID" ]; then
                    echo "   Stopping our OBS instance (PID $pid)..."
                    kill -TERM $pid 2>/dev/null
                    
                    # 等待进程结束，最多等待5秒
                    local count=0
                    while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                        sleep 0.5
                        count=$((count + 1))
                    done
                    
                    # 如果进程仍在运行，强制终止
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "   Force stopping our OBS instance (PID $pid)..."
                        kill -KILL $pid 2>/dev/null
                    fi
                    cleaned_pids="$cleaned_pids $pid"
                else
                    echo "   Skipping user's OBS instance (PID $pid)..."
                fi
            else
                # 非OBS进程，正常清理
                if ps -p $pid > /dev/null 2>&1; then
                    echo "   Stopping PID $pid ($process_name)..."
                    kill -TERM $pid 2>/dev/null
                    
                    # 等待进程结束，最多等待5秒
                    local count=0
                    while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                        sleep 0.5
                        count=$((count + 1))
                    done
                    
                    # 如果进程仍在运行，强制终止
                    if ps -p $pid > /dev/null 2>&1; then
                        echo "   Force stopping PID $pid ($process_name)..."
                        kill -KILL $pid 2>/dev/null
                    fi
                    cleaned_pids="$cleaned_pids $pid"
                fi
            fi
        done
        
        # 等待一小段时间确保设备释放
        if [ ! -z "$cleaned_pids" ]; then
            sleep 1
            echo -e "${GREEN}✓ Processes using /dev/video10 stopped${NC}"
        else
            echo -e "${GREEN}✓ No processes needed to be stopped${NC}"
        fi
    else
        echo -e "${GREEN}✓ No processes found using /dev/video10${NC}"
    fi
}

# 检查 v4l2loopback
check_and_load_v4l2loopback() {
    # 检查模块是否已加载
    if lsmod | grep -q v4l2loopback; then
        echo -e "${GREEN}✓ v4l2loopback module loaded${NC}"
    else
        echo -e "${YELLOW}⚠️  v4l2loopback module not loaded${NC}"
        echo "   Loading v4l2loopback module..."
        sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"
        
        if lsmod | grep -q v4l2loopback; then
            echo -e "${GREEN}✓ v4l2loopback module loaded${NC}"
        else
            echo -e "${RED}❌ Failed to load v4l2loopback module${NC}"
            echo "   Please run manually: sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=\"OBS\""
            return 1
        fi
    fi
    
    # 检查设备是否存在
    if [ -e "/dev/video10" ]; then
        echo -e "${GREEN}✓ Virtual camera /dev/video10 ready${NC}"
        # 检查设备权限
        if [ ! -r "/dev/video10" ] || [ ! -w "/dev/video10" ]; then
            echo -e "${YELLOW}⚠️  Fixing permissions for /dev/video10${NC}"
            sudo chmod 666 /dev/video10
            if [ -r "/dev/video10" ] && [ -w "/dev/video10" ]; then
                echo -e "${GREEN}✓ Permissions fixed for /dev/video10${NC}"
            else
                echo -e "${RED}❌ Failed to fix permissions for /dev/video10${NC}"
                return 1
            fi
        fi
        return 0
    else
        echo -e "${RED}❌ Virtual camera /dev/video10 not found${NC}"
        # 尝试重新加载模块
        echo "   Reloading v4l2loopback module..."
        sudo rmmod v4l2loopback 2>/dev/null
        sleep 1
        sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="OBS"
        
        if [ -e "/dev/video10" ]; then
            echo -e "${GREEN}✓ Virtual camera /dev/video10 loaded after reload${NC}"
            return 0
        else
            echo -e "${RED}❌ Failed to create virtual camera after reload${NC}"
            return 1
        fi
    fi
}

# 执行检查和加载
check_and_start_obs
if ! check_and_load_v4l2loopback; then
    exit 1
fi
cleanup_video_device

# 执行检查和加载
if ! check_and_load_v4l2loopback; then
    exit 1
fi

# 检查 RadioMaster
if [ -e "/dev/input/js0" ]; then
    echo -e "${GREEN}✓ RadioMaster detected at /dev/input/js0${NC}"
else
    echo -e "${YELLOW}⚠️  RadioMaster not detected${NC}"
    echo "   Gamepad control will be disabled"
fi

# 检查 OBS
check_and_start_obs() {
    if pgrep -x "obs" > /dev/null; then
        echo -e "${GREEN}✓ OBS is running${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  OBS is not running${NC}"
        echo "   Starting OBS automatically..."
        
        # 启动 OBS（后台运行）
        obs &>/dev/null &
        OBS_PID=$!
        
        # 等待 OBS 启动
        sleep 3
        
        if ps -p $OBS_PID > /dev/null 2>&1; then
            echo -e "${GREEN}✓ OBS started (PID: $OBS_PID)${NC}"
            echo -e "${YELLOW}⚠️  Please manually enable virtual camera in OBS${NC}"
            echo -e "${YELLOW}   Go to: Tools → Virtual Camera → Start${NC}"
            echo ""
            echo "Press Enter when ready to continue..."
            read -r
            return 0
        else
            echo -e "${RED}❌ Failed to start OBS${NC}"
            echo "   Please start OBS manually and enable virtual camera"
            echo "   Or use --capture-method mss"
            return 1
        fi
    fi
}

# 检查并启动 OBS
if ! check_and_start_obs; then
    echo -e "${YELLOW}Continuing without OBS...${NC}"
    echo "   Will use MSS screen capture method instead"
    # 修改捕获方法为 mss
    CAPTURE_METHOD="mss"
fi

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Output Directory: $OUTPUT_DIR"
echo "  FPS: $FPS"
echo "  Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Capture Method: $CAPTURE_METHOD"
echo ""

# 创建日志目录
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BRIDGE_LOG="$LOG_DIR/bridge_${TIMESTAMP}.log"
CAPTURE_LOG="$LOG_DIR/capture_${TIMESTAMP}.log"

echo -e "${GREEN}========================================"
echo "Starting Components..."
echo -e "========================================${NC}"
echo ""

# 检查并清理已存在的 bridge 进程
EXISTING_BRIDGE=$(pgrep -f "python3 liftoff_bridge_ros2.py")
if [ ! -z "$EXISTING_BRIDGE" ]; then
    echo -e "${YELLOW}⚠️  Found existing bridge process (PID: $EXISTING_BRIDGE)${NC}"
    echo "  Killing existing bridge..."
    kill $EXISTING_BRIDGE 2>/dev/null
    sleep 1
    echo -e "${GREEN}✓ Existing bridge stopped${NC}"
    echo ""
fi

# 启动 ROS2 Bridge（后台）
echo -e "${BLUE}[1/2] Starting ROS2 Bridge...${NC}"
cd "$SCRIPT_DIR"
python3 liftoff_bridge_ros2.py > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

# 等待 bridge 启动
sleep 2

# 检查 bridge 是否运行
if ps -p $BRIDGE_PID > /dev/null; then
    echo -e "${GREEN}✓ ROS2 Bridge started (PID: $BRIDGE_PID)${NC}"
    echo "  Log: $BRIDGE_LOG"
else
    echo -e "${RED}❌ Failed to start ROS2 Bridge${NC}"
    echo "  Check log: $BRIDGE_LOG"
    exit 1
fi

# 验证 ROS2 话题
echo "  Verifying ROS2 topics..."
sleep 1
if ros2 topic list 2>/dev/null | grep -q "/liftoff"; then
    echo -e "${GREEN}  ✓ ROS2 topics active${NC}"
    ros2 topic list | grep liftoff | sed 's/^/    - /'
else
    echo -e "${YELLOW}  ⚠️  ROS2 topics not found (bridge may need more time)${NC}"
fi

echo ""

# 启动数据采集器（前台）
echo -e "${BLUE}[2/2] Starting Data Capture...${NC}"
echo -e "${GREEN}✓ Data capture starting in foreground${NC}"
echo "  Log: $CAPTURE_LOG"
echo ""
echo -e "${YELLOW}========================================"
echo "System Ready!"
echo -e "========================================${NC}"
echo ""
echo -e "${GREEN}🎮 RadioMaster Controls:${NC}"
echo "  - SH switch UP: Start recording"
echo "  - SA switch UP: Stop recording"
echo "  - BTN_SOUTH: Emergency stop"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all components${NC}"
echo ""

# Trap Ctrl+C to cleanup
cleanup() {
    echo ""
    echo -e "${YELLOW}========================================"
    echo "Shutting down..."
    echo -e "========================================${NC}"
    
    # 取消信号陷阱，防止重复调用
    trap - SIGINT SIGTERM

    # 停止数据采集
    if [ ! -z "$CAPTURE_PID" ]; then
        echo "Stopping data capture (PID: $CAPTURE_PID)..."
        # 使用 -TERM 信号优雅停止，如果失败则使用 -KILL
        kill -TERM $CAPTURE_PID 2>/dev/null
        # 等待进程结束，最多等待10秒
        for i in {1..20}; do
            if ps -p $CAPTURE_PID > /dev/null; then
                sleep 0.5
            else
                break
            fi
        done
        # 如果进程仍在运行，强制终止
        if ps -p $CAPTURE_PID > /dev/null; then
            echo "Force stopping data capture (PID: $CAPTURE_PID)..."
            kill -KILL $CAPTURE_PID 2>/dev/null
        fi
    fi

    # 停止 bridge
    echo "Stopping ROS2 bridge (PID: $BRIDGE_PID)..."
    kill -TERM $BRIDGE_PID 2>/dev/null
    # 等待进程结束，最多等待5秒
    for i in {1..10}; do
        if ps -p $BRIDGE_PID > /dev/null; then
            sleep 0.5
        else
            break
        fi
    done
    # 如果进程仍在运行，强制终止
    if ps -p $BRIDGE_PID > /dev/null; then
        echo "Force stopping ROS2 bridge (PID: $BRIDGE_PID)..."
        kill -KILL $BRIDGE_PID 2>/dev/null
    fi

    echo -e "${GREEN}✓ All components stopped${NC}"
    echo ""
    echo "Logs saved to:"
    echo "  - Bridge: $BRIDGE_LOG"
    echo "  - Capture: $CAPTURE_LOG"
    echo ""
    exit 0
}

# 启动数据采集（前台，带实时输出）
python3 -m record.liftoff_capture \
    --output-dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    --image-size "$IMAGE_SIZE" \
    --capture-method "$CAPTURE_METHOD" \
    2>&1 | tee "$CAPTURE_LOG" &

CAPTURE_PID=$!

# 设置陷阱以捕获信号
trap cleanup SIGINT SIGTERM

# 实时显示数据采集状态
echo -e "${BLUE}📊 Monitoring data collection status...${NC}"
echo "  Press Ctrl+C to stop all components"
echo ""

# 创建状态监控函数
monitor_status() {
    local last_line=""
    local last_episode=""
    local recording_status="🔴 Not recording"
    
    while kill -0 $CAPTURE_PID 2>/dev/null; do
        # 读取最新的日志行
        if [ -f "$CAPTURE_LOG" ]; then
            local current_line=$(tail -n 1 "$CAPTURE_LOG" 2>/dev/null)
            
            # 检查是否是新的日志行
            if [ "$current_line" != "$last_line" ]; then
                # 检查是否开始录制
                if echo "$current_line" | grep -q "Starting episode"; then
                    last_episode=$(echo "$current_line" | grep -o "episode [0-9]*" | cut -d ' ' -f 2)
                    recording_status="🟢 Recording episode $last_episode"
                # 检查是否停止录制
                elif echo "$current_line" | grep -q "Saved episode"; then
                    local saved_episode=$(echo "$current_line" | grep -o "episode [0-9]*" | cut -d ' ' -f 2)
                    local frame_count=$(echo "$current_line" | grep -o "[0-9]* frames" | cut -d ' ' -f 1)
                    recording_status="✅ Saved episode $saved_episode ($frame_count frames)"
                # 检查紧急停止
                elif echo "$current_line" | grep -q "EMERGENCY STOP"; then
                    recording_status="🔴 Emergency stop - Not recording"
                # 检查遥控器控制消息
                elif echo "$current_line" | grep -q "RECORDING CONTROL:.*START RECORDING"; then
                    recording_status="🎮 Start recording command received"
                elif echo "$current_line" | grep -q "RECORDING CONTROL:.*STOP RECORDING"; then
                    recording_status="🎮 Stop recording command received"
                elif echo "$current_line" | grep -q "RECORDING CONTROL:.*EMERGENCY STOP"; then
                    recording_status="🚨 Emergency stop command received"
                fi
                
                last_line="$current_line"
            fi
        fi
        
        # 清除当前行并显示状态
        echo -ne "\r\033[K${BLUE}📊 Status:${NC} $recording_status | ${BLUE}Latest:${NC} ${last_line:0:60}..."
        sleep 0.5
    done
    
    # 清除状态行
    echo -ne "\r\033[K"
}

# 在后台启动状态监控
monitor_status &

# 等待数据采集进程
wait $CAPTURE_PID

# 等待监控进程结束
sleep 1

# 如果进程正常结束，执行清理
cleanup

