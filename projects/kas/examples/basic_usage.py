"""
示例：创建并使用 Code Reviewer Agent
"""
from pathlib import Path
from kas.core.ingestion import ingest_project
from kas.core.chat import ChatEngine

# 1. 吞食一个项目
print("=== 步骤1: 吞食项目 ===")
result = ingest_project(
    project_path="/path/to/your/project",
    agent_name="MyCodeReviewer"
)

print(f"Agent created: {result['agent'].name}")
print(f"Capabilities: {[c.name for c in result['agent'].capabilities]}")
print(f"Saved to: {result['output_path']}")

# 2. 与 Agent 对话
print("\n=== 步骤2: 使用 Agent ===")
engine = ChatEngine()

# 加载 Agent
agent = engine.load_agent("MyCodeReviewer")

# 对话
response = engine.chat(
    agent=agent,
    message="Review this Python function for potential issues"
)

print(f"Agent response: {response}")

# 3. 合体两个 Agent（示例）
print("\n=== 步骤3: 合体 Agent ===")
from kas.core.fusion import fuse_agents

result = fuse_agents(
    agent_paths=["path/to/agent1", "path/to/agent2"],
    strategy="synthesis",
    new_name="SuperAgent"
)

print(f"Fused agent: {result['agent'].name}")
print(f"Emergent capabilities: {result['emergent_capabilities']}")
