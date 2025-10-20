#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# AI-Drone 工程重构脚本
# ──────────────────────────────────────────────────────────────────────────────
# 功能：
#   1. 将当前工作区所有文件备份到 legacy/ai-drone-<时间戳>/
#   2. 创建全新的分层清晰的工程目录结构
#   3. 从 legacy 中提取并迁移关键模型文件
#   4. 生成所有必需的模块文件（models/record/train/deploy/tools）
# ──────────────────────────────────────────────────────────────────────────────

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
}

print_info() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ──────────────────────────────────────────────────────────────────────────────
# 步骤 1: 备份当前工作区到 legacy/
# ──────────────────────────────────────────────────────────────────────────────
print_header "步骤 1/3: 备份当前工作区"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LEGACY_DIR="legacy/ai-drone-${TIMESTAMP}"

echo "创建备份目录: ${LEGACY_DIR}"
mkdir -p "${LEGACY_DIR}"

# 获取要迁移的文件/目录列表（排除 legacy/ 本身和此脚本）
echo "收集要迁移的文件..."
ITEMS_TO_MOVE=$(ls -A | grep -v "^legacy$" | grep -v "^refactor_to_new_layout.sh$" || true)

if [ -z "$ITEMS_TO_MOVE" ]; then
    print_warning "没有找到需要迁移的文件"
else
    # 迁移文件
    for item in $ITEMS_TO_MOVE; do
        if [ -e "$item" ]; then
            echo "  移动: $item"
            mv "$item" "${LEGACY_DIR}/"
        fi
    done
    print_info "所有文件已迁移到 ${LEGACY_DIR}/"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 步骤 2: 创建新的目录结构
# ──────────────────────────────────────────────────────────────────────────────
print_header "步骤 2/3: 创建新工程目录结构"

# 创建目录
mkdir -p models
mkdir -p record
mkdir -p train
mkdir -p deploy
mkdir -p tools
mkdir -p configs
mkdir -p outputs/checkpoints
mkdir -p outputs/logs

print_info "目录结构创建完成"

# ──────────────────────────────────────────────────────────────────────────────
# 步骤 3: 标记迁移完成
# ──────────────────────────────────────────────────────────────────────────────
print_header "步骤 3/3: 完成迁移"

print_info "迁移脚本执行完成！"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  下一步操作${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "1. 现在将运行 Python 脚本来生成所有新的模块文件"
echo "2. 模型文件将从 ${LEGACY_DIR} 自动拷贝到 models/"
echo "3. 其他模块将自动生成"
echo ""
echo "请执行:"
echo "  python3 generate_new_modules.py"
echo ""
