#!/bin/bash
# 使用 Deepseek API 运行记忆管道

export OPENAI_API_KEY='sk-37ca0f11a7a74a7e804a9c511e231969'
export OPENAI_BASE_URL='https://api.deepseek.com/v1'

cd /root/.openclaw/workspace

echo "🧠 使用 Deepseek API 运行记忆管道..."
echo ""

# Stage 1: Extract
echo "📥 Stage 1: 提取结构化事实..."
python3 skills/memory-pipeline/scripts/memory-extract.py

# Stage 2: Link
echo ""
echo "🔗 Stage 2: 构建知识图谱..."
python3 skills/memory-pipeline/scripts/memory-link.py

# Stage 3: Briefing
echo ""
echo "📋 Stage 3: 生成每日简报..."
python3 skills/memory-pipeline/scripts/memory-briefing.py

echo ""
echo "✅ 记忆管道运行完成！"
