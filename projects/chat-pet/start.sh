#!/bin/bash

# ChatPet 启动脚本

echo "🐾 启动 ChatPet 服务器..."

cd "$(dirname "$0")/server"

# 检查依赖
if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
fi

# 启动服务器
echo "🚀 服务器启动在 ws://localhost:8080"
node server.js