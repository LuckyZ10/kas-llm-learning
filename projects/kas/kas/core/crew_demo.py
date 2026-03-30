"""
KAS Crew - Agent 特种部队示例
合同审查流程演示
"""
import os
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

from kas.core.sandbox.supervisor import SandboxSupervisor, CrewConfig
from kas.core.crew_workflow import Workflow, WorkflowEngine, WorkflowTask
from kas.core.crew_memory import LayeredMemory

logger = logging.getLogger(__name__)


class ContractReviewCrew:
    """
    合同审查 Crew 示例
    
    角色:
    - Alice (协调员): 法律专家，协调沟通，汇总报告
    - Bob (OCR专家): 图片处理、文字识别
    - Carol (法律分析师): 条款分析、风险提示
    
    工作流程:
    1. Alice 确认审查需求
    2. Bob OCR识别 (如果含图片)
    3. Carol 分析条款 (依赖OCR结果)
    4. Carol 识别风险
    5. Alice 汇总报告
    
    使用示例:
        crew = ContractReviewCrew()
        crew.setup()  # 创建Crew和Agent
        result = crew.run(
            contract_text="合同内容...",
            has_image=True,
            image_path="contract.jpg"
        )
    """
    
    def __init__(self, supervisor: SandboxSupervisor = None, crew_name: str = None):
        """
        Args:
            supervisor: 沙盒监督器
            crew_name: Crew 名称 (默认 ContractReviewCrew)
        """
        self.supervisor = supervisor or SandboxSupervisor()
        self.crew_name = crew_name or "ContractReviewCrew"
        self.workflow_engine = WorkflowEngine(self.supervisor)
        
        # Agent 定义
        self.agents = {
            "Alice": {
                "name": "Alice",
                "role": "coordinator",
                "description": "法律专家，负责协调团队沟通，理解用户需求，分配任务，汇总最终报告",
                "system_prompt": """你是一位资深法律顾问，擅长合同审查和团队协调。

你的职责:
1. 理解用户的合同审查需求
2. 协调团队成员（Bob负责OCR识别，Carol负责法律分析）
3. 汇总各成员的分析结果
4. 生成清晰、专业的合同审查报告

沟通风格: 专业、清晰、有条理""",
                "equipment": ["web_search", "file_reader"]
            },
            "Bob": {
                "name": "Bob",
                "role": "ocr_expert",
                "description": "OCR专家，负责处理图片、识别文字，提取合同文本内容",
                "system_prompt": """你是一位OCR和文档处理专家。

你的职责:
1. 处理合同图片文件
2. 使用OCR技术提取文字内容
3. 整理和格式化识别结果
4. 标注识别不确定的部分

工作原则: 准确、高效、标注异常""",
                "equipment": ["ocr", "image_analysis", "file_reader"]
            },
            "Carol": {
                "name": "Carol",
                "role": "legal_analyst",
                "description": "法律分析师，负责分析合同条款，识别风险点，评估法律合规性",
                "system_prompt": """你是一位专业的合同法律分析师。

你的职责:
1. 分析合同条款的合法性和合理性
2. 识别潜在的法律风险
3. 评估双方权利义务是否对等
4. 检查是否有遗漏的重要条款

分析维度:
- 付款条款
- 违约责任
- 保密条款
- 知识产权
- 争议解决
- 合同终止

输出格式: 结构化分析 + 风险提示 + 修改建议""",
                "equipment": ["web_search", "file_reader", "pdf_parser"]
            }
        }
    
    def setup(self) -> bool:
        """
        设置 Crew
        
        1. 创建 Crew 配置
        2. 创建临时 Agent 配置
        3. 注入所有 Agent
        """
        logger.info("Setting up ContractReviewCrew...")
        
        # 检查是否已存在
        if self.crew_name in self.supervisor.list_crews():
            logger.info(f"Crew {self.crew_name} already exists")
            return True
        
        # 创建 Crew 配置
        crew_config = CrewConfig(
            name=self.crew_name,
            description="专业合同审查团队，包含协调员、OCR专家和法律分析师",
            members=[
                {"name": "Alice", "role": "coordinator", "description": self.agents["Alice"]["description"]},
                {"name": "Bob", "role": "ocr_expert", "description": self.agents["Bob"]["description"]},
                {"name": "Carol", "role": "legal_analyst", "description": self.agents["Carol"]["description"]}
            ],
            coordinator="Alice"  # 指定 Alice 为协调员
        )
        
        self.supervisor.create_crew(crew_config)
        
        # 注入所有 Agent
        for agent_name, agent_config in self.agents.items():
            # 保存临时 agent.yaml
            agent_path = self._create_temp_agent(agent_name, agent_config)
            
            # 注入到沙盒
            self.supervisor.inject_agent(self.crew_name, agent_config)
            logger.info(f"Agent {agent_name} injected")
        
        logger.info(f"ContractReviewCrew setup complete")
        return True
    
    def _create_temp_agent(self, name: str, config: Dict) -> Path:
        """创建临时 Agent 配置文件"""
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir()) / "kas_agents"
        temp_dir.mkdir(exist_ok=True)
        
        agent_dir = temp_dir / name
        agent_dir.mkdir(exist_ok=True)
        
        agent_file = agent_dir / "agent.yaml"
        
        yaml_content = f"""name: {name}
version: 0.1.0
description: {config['description']}

system_prompt: |
{config['system_prompt']}

capabilities:
  - name: review_contract
    description: 合同审查和分析
    type: domain
    confidence: 0.9

equipment:
"""
        for eq in config['equipment']:
            yaml_content += f"  - {eq}\n"
        
        agent_file.write_text(yaml_content, encoding='utf-8')
        return agent_dir
    
    def create_workflow(self, has_image: bool = False) -> Workflow:
        """
        创建工作流
        
        Args:
            has_image: 是否包含图片需要OCR处理
        """
        tasks = []
        
        # 步骤1: Alice 确认需求
        tasks.append(WorkflowTask(
            id="step_1",
            name="确认审查需求",
            agent="Alice",
            task="请确认用户的合同审查需求，包括：1)审查重点 2)特殊要求 3)分发任务给团队成员",
            timeout=30
        ))
        
        # 步骤2: Bob OCR识别 (条件执行)
        if has_image:
            tasks.append(WorkflowTask(
                id="step_2",
                name="OCR识别合同",
                agent="Bob",
                task="对合同图片进行OCR识别，提取完整文字内容。标注识别困难或不准确的部分。",
                depends_on=["step_1"],
                condition="context.get('has_image')",
                timeout=60
            ))
        
        # 步骤3: Carol 分析条款
        tasks.append(WorkflowTask(
            id="step_3",
            name="分析合同条款",
            agent="Carol",
            task="""分析合同条款，包括：
1. 付款条款是否合理
2. 违约责任是否对等
3. 保密条款是否完善
4. 知识产权归属
5. 争议解决方式
6. 合同终止条件

输出结构化分析报告。""",
            depends_on=["step_2"] if has_image else ["step_1"],
            timeout=120
        ))
        
        # 步骤4: Carol 识别风险
        tasks.append(WorkflowTask(
            id="step_4",
            name="识别法律风险",
            agent="Carol",
            task="基于条款分析，识别潜在法律风险，评估风险等级（高/中/低），提供修改建议。",
            depends_on=["step_3"],
            timeout=60
        ))
        
        # 步骤5: Alice 汇总报告
        tasks.append(WorkflowTask(
            id="step_5",
            name="汇总审查报告",
            agent="Alice",
            task="""汇总团队分析结果，生成最终合同审查报告：

报告结构:
1. 执行摘要
2. 条款分析总结
3. 风险提示清单
4. 修改建议
5. 总体评估

确保报告专业、清晰、可操作。""",
            depends_on=["step_4"],
            timeout=60
        ))
        
        return Workflow(
            name="contract_review",
            description="合同审查工作流",
            tasks=tasks,
            variables={"has_image": has_image}
        )
    
    def run(self, contract_text: str = "", has_image: bool = False,
            image_path: str = None, user_requirements: str = "") -> Dict[str, Any]:
        """
        运行合同审查流程
        
        Args:
            contract_text: 合同文本内容
            has_image: 是否包含图片
            image_path: 图片路径 (如果有)
            user_requirements: 用户特殊要求
        
        Returns:
            审查结果
        """
        logger.info(f"Starting contract review workflow (has_image={has_image})")
        
        # 确保 Crew 已启动
        status = self.supervisor.get_crew_status(self.crew_name)
        running_count = sum(
            1 for s in status.get('sandboxes', {}).values()
            if s.get('state') == 'running'
        )
        
        if running_count < len(self.agents):
            logger.info("Starting crew sandboxes...")
            self.supervisor.start_crew(self.crew_name, use_mock=True)
        
        # 创建工作流
        workflow = self.create_workflow(has_image=has_image)
        
        # 执行上下文
        context = {
            "contract_text": contract_text,
            "has_image": has_image,
            "image_path": image_path,
            "user_requirements": user_requirements,
            "review_type": "合同审查"
        }
        
        # 如果有图片，需要预处理
        if has_image and image_path:
            from kas.core.multimodal import MultimodalProcessor
            processor = MultimodalProcessor()
            processed = processor.process_file(image_path)
            context["image_ocr_text"] = processed.content if processed.file_type.value == "image" else ""
        
        # 执行工作流
        result = self.workflow_engine.execute(
            crew_name=self.crew_name,
            workflow=workflow,
            context=context
        )
        
        # 提取最终报告
        final_task = next(
            (t for t in workflow.tasks if t.id == "step_5"),
            None
        )
        
        return {
            "success": result["status"] == "completed",
            "execution_id": result["execution_id"],
            "final_report": final_task.result if final_task else None,
            "all_results": {t.id: t.to_dict() for t in workflow.tasks}
        }
    
    def interactive_review(self):
        """交互式合同审查"""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        console.print(Panel.fit(
            "[bold cyan]合同审查 Crew[/bold cyan]\n"
            "成员: Alice(协调员) | Bob(OCR专家) | Carol(法律分析师)",
            title="🚀 Agent 特种部队"
        ))
        
        # 获取用户输入
        console.print("\n[bold]请输入合同审查要求:[/bold]")
        requirements = console.input(" ")
        
        console.print("\n[bold]是否有图片附件?[/bold] (y/n): ", end="")
        has_image = console.input("").lower() == 'y'
        
        image_path = None
        if has_image:
            console.print("[bold]图片路径:[/bold] ", end="")
            image_path = console.input("")
        
        # 运行审查
        with console.status("[bold green]正在协调团队进行审查..."):
            result = self.run(
                user_requirements=requirements,
                has_image=has_image,
                image_path=image_path
            )
        
        # 显示结果
        if result["success"]:
            console.print("\n✅ [bold green]审查完成![/bold green]\n")
            if result["final_report"]:
                console.print(Panel(
                    str(result["final_report"]),
                    title="📋 审查报告",
                    border_style="cyan"
                ))
        else:
            console.print("\n❌ [bold red]审查失败[/bold red]")
            console.print_json(data=result["all_results"])
        
        return result


def demo():
    """演示函数"""
    print("=" * 60)
    print("KAS Agent 特种部队 - 合同审查演示")
    print("=" * 60)
    
    # 创建 Crew
    crew = ContractReviewCrew()
    crew.setup()
    
    # 运行交互式审查
    result = crew.interactive_review()
    
    print("\n" + "=" * 60)
    print("演示结束")
    print(f"执行ID: {result['execution_id']}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    demo()
