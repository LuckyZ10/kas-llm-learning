#!/bin/bash
# 使用 Deepseek API 运行记忆管道 (带较长超时)

export OPENAI_API_KEY='sk-37ca0f11a7a74a7e804a9c511e231969'
export OPENAI_BASE_URL='https://api.deepseek.com/v1'

cd /root/.openclaw/workspace

echo "🧠 使用 Deepseek API 运行记忆管道..."
echo "(Deepseek 处理较慢，请耐心等待 1-3 分钟)"
echo ""

# Stage 1: Extract
echo "📥 Stage 1/3: 提取结构化事实..."
timeout 180 python3 skills/memory-pipeline/scripts/memory-extract.py
if [ $? -eq 124 ]; then
    echo "⏱️ Stage 1 超时，继续下一步..."
fi

# Stage 2: Link
echo ""
echo "🔗 Stage 2/3: 构建知识图谱..."
timeout 180 python3 skills/memory-pipeline/scripts/memory-link.py
if [ $? -eq 124 ]; then
    echo "⏱️ Stage 2 超时，继续下一步..."
fi

# Stage 3: Briefing
echo ""
echo "📋 Stage 3/3: 生成每日简报..."
timeout 120 python3 skills/memory-pipeline/scripts/memory-briefing.py
if [ $? -eq 124 ]; then
    echo "⏱️ Stage 3 超时"
fi

echo ""
echo "✅ 记忆管道运行完成！"
echo ""
echo "生成的文件:"
ls -la memory/extracted.jsonl memory/knowledge-graph.json memory/knowledge-summary.md BRIEFING.md 2>/dev/null || echo "(部分文件可能因超时而未生成)"
