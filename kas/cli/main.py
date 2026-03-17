"""
KAS CLI - 命令行接口
简单优先的CLI实现
"""
import os
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from kas.core.ingestion import ingest_project
from kas.core.fusion import fuse_agents
from kas.core.chat import ChatEngine, SimpleLLMClient
from kas.core.llm_learning import LLMEnhancedLearningEngine

console = Console()


def get_llm_client(force_mock=False):
    """从环境变量获取LLM配置并创建客户端"""
    if force_mock:
        return None
    
    # 检测 key 类型并设置对应的 base_url
    if os.environ.get('OPENAI_API_KEY'):
        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('LLM_BASE_URL')  # OpenAI 可选自定义
        model = "gpt-3.5-turbo"
    elif os.environ.get('KIMI_API_KEY'):
        api_key = os.environ.get('KIMI_API_KEY')
        base_url = "https://api.moonshot.cn/v1"
        model = "moonshot-v1-8k"
    elif os.environ.get('DEEPSEEK_API_KEY'):
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        base_url = "https://api.deepseek.com/v1"
        model = "deepseek-chat"
    else:
        return None
    
    return SimpleLLMClient(api_key=api_key, base_url=base_url, model=model)


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Kimi Agent Studio - 专业开发者 Agent 孵化平台
    
    核心概念：代码吞食 → 能力提取 → Agent 进化
    """
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--name', '-n', help='Agent 名称（默认为项目目录名）')
@click.option('--output', '-o', help='输出目录')
@click.option('--model', '-m', default='gpt-3.5-turbo', help='LLM 模型')
def ingest(path, name, output, model):
    """吞食项目，提取能力，孵化 Agent"""
    console.print(f"\n🔍 [bold blue]Analyzing project:[/bold blue] {path}")
    
    try:
        # 创建LLM客户端（如果配置了API key）
        llm_client = None
        # TODO: 从配置文件读取API key
        
        # 吞食项目
        result = ingest_project(path, agent_name=name, llm_client=llm_client)
        agent = result['agent']
        
        # 显示结果
        console.print(f"\n✅ [bold green]Agent created:[/bold green] {agent.name}")
        console.print(f"📁 [dim]Location:[/dim] {result['output_path']}")
        
        # 显示能力表格
        table = Table(title="Extracted Capabilities")
        table.add_column("Capability", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Confidence", style="yellow")
        
        for cap in agent.capabilities:
            table.add_row(
                cap.name,
                cap.type.value,
                f"{cap.confidence:.0%}"
            )
        
        console.print(table)
        
        console.print(f"\n💡 [dim]Try:[/dim] [bold]kas chat {agent.name}[/bold]")
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.argument('agents', nargs=-1, required=True)
@click.option('--strategy', '-s', 
              type=click.Choice(['union', 'intersect', 'dominant', 'synthesis']),
              default='synthesis',
              help='合体策略')
@click.option('--name', '-n', required=True, help='新 Agent 名称')
def fuse(agents, strategy, name):
    """合体多个 Agent"""
    console.print(f"\n🔀 [bold blue]Fusing {len(agents)} agents...[/bold blue]")
    console.print(f"   Strategy: [cyan]{strategy}[/cyan]")
    
    try:
        # 转换Agent名称到路径
        agent_paths = []
        for agent_name in agents:
            agent_path = Path.home() / '.kas' / 'agents' / agent_name
            if not agent_path.exists():
                console.print(f"❌ Agent not found: {agent_name}")
                raise click.Abort()
            agent_paths.append(str(agent_path))
        
        # 合体
        result = fuse_agents(agent_paths, strategy=strategy, new_name=name)
        fused_agent = result['agent']
        
        console.print(f"\n✅ [bold green]Created:[/bold green] {fused_agent.name}")
        console.print(f"📁 [dim]Location:[/dim] {result['output_path']}")
        
        # 显示能力
        table = Table(title=f"Capabilities ({len(fused_agent.capabilities)} total)")
        table.add_column("Capability", style="cyan")
        table.add_column("Source", style="dim")
        
        for cap in fused_agent.capabilities:
            table.add_row(cap.name, cap.type.value)
        
        console.print(table)
        
        # 显示涌现能力
        if result['emergent_capabilities']:
            console.print("\n✨ [bold yellow]Emergent capabilities detected:[/bold yellow]")
            for cap in result['emergent_capabilities']:
                console.print(f"   + [green]{cap['name']}[/green]: {cap['description']}")
        
        console.print(f"\n💡 [dim]Try:[/dim] [bold]kas chat {fused_agent.name}[/bold]")
        
    except Exception as e:
        console.print(f"\n❌ [bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
@click.argument('agent')
@click.option('--interactive', '-i', is_flag=True, help='交互模式')
@click.option('--message', '-m', help='单次消息')
@click.option('--mock', is_flag=True, help='强制使用 Mock 模式')
def chat(agent, interactive, message, mock):
    """与 Agent 对话"""
    try:
        # 创建LLM客户端
        llm_client = get_llm_client(force_mock=mock)
        engine = ChatEngine(llm_client=llm_client)
        
        # 加载Agent
        loaded_agent = engine.load_agent(agent)
        
        if llm_client:
            console.print(f"\n[dim]🧠 Using LLM API for responses[/dim]")
        else:
            console.print(f"\n[dim]🤖 Mock mode - Set OPENAI_API_KEY for real responses[/dim]")
        
        if interactive:
            # 交互模式
            engine.interactive_chat(agent)
        elif message:
            # 单次对话
            response = engine.chat(loaded_agent, message)
            console.print(f"\n[bold cyan]{loaded_agent.name}:[/bold cyan] {response}\n")
        else:
            console.print("❌ Please provide --message or use --interactive mode")
            
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        raise click.Abort()


@cli.command()
def list():
    """列出所有 Agents"""
    agents_dir = Path.home() / '.kas' / 'agents'
    
    if not agents_dir.exists():
        console.print("📂 No agents directory found. Create your first agent with:")
        console.print("   [bold]kas ingest <project-path> --name <name>[/bold]")
        return
    
    table = Table(title="Installed Agents")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="dim")
    table.add_column("Capabilities", style="green")
    table.add_column("Source", style="dim")
    
    agent_count = 0
    for agent_dir in sorted(agents_dir.iterdir()):
        if agent_dir.is_dir():
            agent_file = agent_dir / 'agent.yaml'
            if agent_file.exists():
                try:
                    import yaml
                    with open(agent_file, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    caps = ', '.join([c['name'] for c in data.get('capabilities', [])[:3]])
                    if len(data.get('capabilities', [])) > 3:
                        caps += '...'
                    
                    table.add_row(
                        data['name'],
                        data.get('version', '0.1.0'),
                        caps or 'None',
                        Path(data.get('created_from', '')).name or 'Unknown'
                    )
                    agent_count += 1
                except Exception:
                    pass
    
    if agent_count > 0:
        console.print(table)
        console.print(f"\n[dim]Total: {agent_count} agents[/dim]")
    else:
        console.print("📂 No agents found. Create your first agent with:")
        console.print("   [bold]kas ingest <project-path> --name <name>[/bold]")


@cli.command()
@click.argument('agent')
def inspect(agent):
    """查看 Agent 详情"""
    try:
        engine = ChatEngine()
        loaded_agent = engine.load_agent(agent)
        
        # 显示详情面板
        console.print(Panel.fit(
            f"[bold cyan]{loaded_agent.name}[/bold cyan]\n"
            f"[dim]Version:[/dim] {loaded_agent.version}\n"
            f"[dim]Description:[/dim] {loaded_agent.description}\n"
            f"[dim]Created from:[/dim] {loaded_agent.created_from}",
            title="Agent Info"
        ))
        
        # 显示能力
        table = Table(title="Capabilities")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("Description", style="dim")
        
        for cap in loaded_agent.capabilities:
            table.add_row(
                cap.name,
                cap.type.value,
                f"{cap.confidence:.0%}",
                cap.description[:50] + '...' if len(cap.description) > 50 else cap.description
            )
        
        console.print(table)
        
        # 显示配置
        console.print(f"\n[bold]Model Config:[/bold]")
        for key, value in loaded_agent.model_config.items():
            console.print(f"  {key}: {value}")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.option('--force', is_flag=True, help='强制进化，忽略收敛状态')
@click.option('--output', '-o', help='输出报告到文件')
def evolve(agent, force, output):
    """🧬 触发 Agent 进化 (LLM 增强学习)"""
    try:
        console.print(f"\n🧠 [bold blue]Initializing LLM Learning Engine for {agent}...[/bold blue]")
        
        # 初始化学习引擎
        learning_engine = LLMEnhancedLearningEngine(agent)
        
        # 加载 agent 信息
        agent_dir = Path.home() / '.kas' / 'agents' / agent
        if not agent_dir.exists():
            console.print(f"❌ Agent not found: {agent}")
            return
        
        import yaml
        with open(agent_dir / 'agent.yaml', 'r') as f:
            agent_data = yaml.safe_load(f)
        
        current_prompt = agent_data.get('system_prompt', '')
        capabilities = [c['name'] for c in agent_data.get('capabilities', [])]
        
        if force:
            # 强制进化
            console.print(f"\n🧬 [bold yellow]Force evolving {agent}...[/bold yellow]")
            result = learning_engine.force_evolution(current_prompt, capabilities)
            
            console.print(Panel.fit(
                f"[bold cyan]🧬 Evolution Plan for {agent}[/bold cyan]\n\n"
                f"Generation: {result['generation']}\n"
                f"Top Recommendation: {result['analysis'].get('top_recommendation', 'N/A')}\n\n"
                f"[bold]Key Strengths:[/bold]\n"
                + "\n".join(f"  ✓ {s}" for s in result['analysis'].get('key_strengths', [])[:3]) + "\n\n"
                f"[bold]New Prompt Preview:[/bold]\n"
                f"{result['evolution_plan'].get('new_system_prompt', 'N/A')[:300]}...",
                title="🧬 Evolution Result"
            ))
            
            # 显示能力调整建议
            changes = result['evolution_plan'].get('capability_changes', [])
            if changes:
                table = Table(title="Suggested Capability Changes")
                table.add_column("Action", style="cyan")
                table.add_column("Capability", style="green")
                table.add_column("Reason", style="dim")
                for change in changes:
                    table.add_row(
                        change.get('action', 'unknown'),
                        change.get('name', 'unknown'),
                        change.get('reason', '')[:40]
                    )
                console.print(table)
        else:
            # 显示当前学习状态
            console.print(learning_engine.get_learning_report())
            
            # 检查是否应该进化
            should_evolve, reason = learning_engine.convergence_engine.should_trigger_evolution()
            if should_evolve:
                console.print(f"\n✨ [bold green]{reason}[/bold green]")
                console.print("Run with --force to trigger evolution.")
        
        if output:
            result = learning_engine.force_evolution(current_prompt, capabilities) if force else {
                'report': learning_engine.get_learning_report(),
                'metrics': {k: v.__dict__ for k, v in learning_engine.capability_metrics.items()}
            }
            with open(output, 'w') as f:
                import json
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            console.print(f"\n💾 Report saved to: {output}")
            
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())


def main():
    """入口点"""
    cli()


if __name__ == '__main__':
    main()
