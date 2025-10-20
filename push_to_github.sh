#!/bin/bash
# 推送AI Drone项目到GitHub的脚本

# 请修改以下变量以匹配您的GitHub信息
GITHUB_USERNAME="radiuson"
REPO_NAME="Sim-AI-Drone"

echo "确保您已经在GitHub上创建了名为 $REPO_NAME 的空仓库"
echo "按 Enter 继续..."
read

# 设置远程仓库
echo "设置远程仓库..."
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# 重命名分支为main（如果需要）
echo "重命名分支为main..."
git branch -M main

# 推送到GitHub
echo "推送代码到GitHub..."
git push -u origin main

echo "项目已成功推送到GitHub！"
echo "访问地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"