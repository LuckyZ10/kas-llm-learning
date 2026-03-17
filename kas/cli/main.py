"""
KAS CLI - 命令行接口
简单优先的CLI实现
"""
import os
from datetime import datetime
import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from dataclasses import asdict

from kas.core.ingestion import ingest_project
from kas.core.fusion import fuse_agents
from kas.core.chat import ChatEngine, SimpleLLMClient
from kas.core.llm_learning import LLMEnhancedLearningEngine
from kas.core.config import get_config, init_config, setup_wizard
from kas.core.versioning import get_version_manager, auto_save_on_evolve
from kas.core.market import get_market, pack_agent, unpack_agent
from kas.core.validation import CapabilityValidator, validate_agent
from kas.core.stats import get_dashboard, record_conversation

console = Console()


def get_llm_client(force_mock=False):
    """从配置获取LLM客户端"""
    if force_mock:
        return None
    
    # 使用新的配置系统
    config = get_config()
    api_key = config.get_api_key()
    
    if not api_key:
        return None
    
    # 根据配置的 provider 创建客户端
    provider = config.llm.provider
    base_url = config.llm.base_url
    model = config.llm.model
    
    if provider == "kimi" and not base_url:
        base_url = "https://api.moonshot.cn/v1"
        model = model or "moonshot-v1-8k"
    elif provider == "deepseek" and not base_url:
        base_url = "https://api.deepseek.com/v1"
        model = model or "deepseek-chat"
    
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
@click.option('--apply', is_flag=True, help='真正应用进化（修改 agent.yaml）')
@click.option('--output', '-o', help='输出报告到文件')
def evolve(agent, force, apply, output):
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
            
            # 真正应用进化
            if apply:
                console.print(f"\n📝 [bold yellow]Applying evolution to {agent}...[/bold yellow]")
                
                # 更新 system_prompt
                new_prompt = result['evolution_plan'].get('new_system_prompt', current_prompt)
                if new_prompt != current_prompt:
                    agent_data['system_prompt'] = new_prompt
                    console.print(f"  ✓ Updated system_prompt")
                
                # 更新能力
                if changes:
                    current_caps = {c['type']: c for c in agent_data.get('capabilities', [])}
                    
                    for change in changes:
                        action = change.get('action')
                        cap_name = change.get('name')
                        cap_type = change.get('capability_type', 'CODE_REVIEW')
                        
                        if action == 'add':
                            if cap_type not in current_caps:
                                agent_data['capabilities'].append({
                                    'name': cap_name,
                                    'type': cap_type,
                                    'description': change.get('description', ''),
                                    'confidence': 0.7
                                })
                                console.print(f"  ✓ Added capability: {cap_name}")
                        
                        elif action == 'remove' and cap_type in current_caps:
                            agent_data['capabilities'] = [
                                c for c in agent_data['capabilities'] 
                                if c['type'] != cap_type
                            ]
                            console.print(f"  ✓ Removed capability: {cap_name}")
                        
                        elif action == 'enhance':
                            for c in agent_data['capabilities']:
                                if c['type'] == cap_type:
                                    c['confidence'] = min(c.get('confidence', 0.5) + 0.1, 0.95)
                                    console.print(f"  ✓ Enhanced capability: {cap_name}")
                
                # 增加代数
                agent_data['generation'] = result.get('generation', agent_data.get('generation', 0) + 1)
                
                # 创建备份
                backup_dir = agent_dir / 'backups'
                backup_dir.mkdir(exist_ok=True)
                backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_file = backup_dir / f"agent.yaml.backup.{backup_timestamp}"
                
                # 复制当前配置到备份
                import shutil
                shutil.copy2(agent_dir / 'agent.yaml', backup_file)
                console.print(f"  ✓ Created backup: {backup_file.name}")
                
                # 清理旧备份（只保留最近10个）
                backups = sorted(backup_dir.glob('agent.yaml.backup.*'))
                if len(backups) > 10:
                    for old_backup in backups[:-10]:
                        old_backup.unlink()
                
                # 保存到文件
                with open(agent_dir / 'agent.yaml', 'w') as f:
                    yaml.dump(agent_data, f, allow_unicode=True, default_flow_style=False)
                
                with open(agent_dir / 'system_prompt.txt', 'w') as f:
                    f.write(agent_data['system_prompt'])
                
                console.print(f"✅ Evolution applied successfully!")
            else:
                console.print(f"\n💡 Use --apply to actually modify {agent}")
            
            # 自动保存版本
            console.print(f"\n💾 [bold blue]Saving evolution to version history...[/bold blue]")
            
            # 重新加载当前 agent
            from kas.core.models import Agent
            current_agent = Agent.from_dict(agent_data)
            
            vm = get_version_manager(agent)
            version_id = vm.save_version(
                agent=current_agent,
                description=f"Evolution Gen{result['generation']}: {result['analysis'].get('top_recommendation', 'Auto evolution')[:50]}...",
                tags=["evolution", f"gen{result['generation']}"],
                changes=[c.get('action', 'unknown') + ': ' + c.get('name', '') for c in changes],
                quality_score=result['analysis'].get('confidence', 0) * 100
            )
            
            console.print(f"✅ Saved as version: [bold]{version_id}[/bold]")
            console.print(f"💡 Run 'kas versions {agent}' to see history")
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


@cli.command()
def config():
    """⚙️  配置 KAS 设置"""
    try:
        # 初始化配置
        init_config()
        config_manager = get_config()
        
        # 显示当前状态
        console.print(config_manager.get_status())
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command(name='config-setup')
def config_setup():
    """🔧 交互式配置向导"""
    try:
        setup_wizard()
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.option('--limit', '-n', default=10, help='显示最近 N 个版本')
def versions(agent, limit):
    """📚 查看 Agent 版本历史"""
    try:
        vm = get_version_manager(agent)
        versions_list = vm.list_versions(limit=limit)
        
        if not versions_list:
            console.print(f"📭 {agent} 暂无版本历史")
            console.print("提示: 运行 'kas evolve --force' 创建第一个版本")
            return
        
        console.print(vm.format_version_list(versions_list))
        
        # 显示统计
        stats = vm.get_statistics()
        console.print(f"\n📊 统计: 平均质量 {stats['average_quality']}, 最佳版本 {stats['best_version'] or 'N/A'}")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.argument('version_id')
def rollback(agent, version_id):
    """⏪ 回滚 Agent 到指定版本"""
    try:
        vm = get_version_manager(agent)
        
        # 确认版本存在
        version_info = vm.get_version(version_id)
        if not version_info:
            console.print(f"❌ 版本 {version_id} 不存在")
            return
        
        # 确认回滚
        from rich.prompt import Confirm
        if not Confirm.ask(f"确定要回滚 {agent} 到 {version_id} 吗?"):
            console.print("已取消")
            return
        
        # 执行回滚
        agent_data = vm.rollback(version_id)
        if agent_data:
            # 保存回滚后的版本到当前 agent
            agent_dir = Path.home() / '.kas' / 'agents' / agent
            agent_dir.mkdir(parents=True, exist_ok=True)
            
            import yaml
            with open(agent_dir / 'agent.yaml', 'w') as f:
                yaml.dump(agent_data.to_dict(), f, allow_unicode=True)
            
            with open(agent_dir / 'system_prompt.txt', 'w') as f:
                f.write(agent_data.system_prompt)
            
            console.print(f"✅ {agent} 已回滚到 {version_id}")
            console.print(f"💾 当前状态已保存为新的版本（以防后悔）")
        else:
            console.print(f"❌ 回滚失败")
            
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.argument('version_a')
@click.argument('version_b')
def diff(agent, version_a, version_b):
    """🔍 对比两个 Agent 版本"""
    try:
        vm = get_version_manager(agent)
        
        result = vm.compare_versions(version_a, version_b)
        
        if 'error' in result:
            console.print(f"❌ {result['error']}")
            return
        
        console.print(Panel.fit(
            f"[bold cyan]🔍 {agent} 版本对比[/bold cyan]\n\n"
            f"{version_a} vs {version_b}\n\n"
            f"[bold]能力变化:[/bold]\n"
            f"  + 新增: {', '.join(result['capabilities']['added']) or '无'}\n"
            f"  - 移除: {', '.join(result['capabilities']['removed']) or '无'}\n\n"
            f"[bold]配置变化:[/bold]\n"
            + "\n".join(f"  {k}: {v['from']} → {v['to']}" for k, v in result['config_changes'].items())
            + f"\n\n[bold]Prompt 变化:[/bold] {len(result['prompt_changes'])} 处",
            title="🔍 Version Diff"
        ))
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.option('--output', '-o', help='输出路径')
@click.option('--author', help='作者名')
@click.option('--email', help='作者邮箱')
@click.option('--tag', multiple=True, help='标签（可多次使用）')
def export(agent, output, author, email, tag):
    """📦 导出 Agent 为 .kas-agent 包"""
    try:
        # 加载 Agent
        from kas.core.models import Agent
        agent_dir = Path.home() / '.kas' / 'agents' / agent
        
        if not agent_dir.exists():
            console.print(f"❌ Agent not found: {agent}")
            return
        
        with open(agent_dir / 'agent.yaml', 'r') as f:
            import yaml
            agent_data = yaml.safe_load(f)
        
        agent_obj = Agent.from_dict(agent_data)
        
        # 打包
        tags = list(tag) if tag else []
        package_path = pack_agent(
            agent_obj,
            output_path=output,
            author=author or "Unknown",
            author_email=email or "",
            tags=tags
        )
        
        console.print(f"✅ 导出成功: {package_path}")
        console.print(f"📦 大小: {Path(package_path).stat().st_size / 1024:.1f} KB")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('package')
@click.option('--install', is_flag=True, help='直接安装')
def import_pkg(package, install):
    """📂 导入 .kas-agent 包"""
    try:
        agent = unpack_agent(package, install=install)
        
        if agent:
            console.print(f"✅ 导入成功: {agent.name} v{agent.version}")
            
            if install:
                console.print(f"💾 已安装到: ~/.kas/agents/{agent.name}")
                console.print(f"💡 使用: kas chat {agent.name}")
            else:
                console.print(f"📋 预览模式，使用 --install 安装")
        else:
            console.print(f"❌ 导入失败")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('package')
def market_publish(package):
    """🛒 发布包到本地市场"""
    try:
        market = get_market()
        
        if market.publish(package):
            console.print(f"✅ 发布成功")
            
            # 获取包信息
            pkg_path = Path(package)
            agent = unpack_agent(package)
            if agent:
                console.print(f"📦 {agent.name} v{agent.version}")
                console.print(f"💡 其他人可以用 'kas market install {agent.name}' 安装")
        else:
            console.print(f"❌ 发布失败")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('query', default="")
@click.option('--tag', multiple=True, help='按标签过滤')
def market_search(query, tag):
    """🔍 搜索市场"""
    try:
        market = get_market()
        tags = list(tag) if tag else None
        
        results = market.search(query, tags=tags)
        
        if results:
            console.print(market.format_search_results(results))
        else:
            console.print(f"📭 未找到匹配的 Agent")
            console.print(f"💡 尝试: kas market search python")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('name')
def market_install(name):
    """⬇️  从市场安装 Agent"""
    try:
        market = get_market()
        
        info = market.get_info(name)
        if not info:
            console.print(f"❌ 市场找不到: {name}")
            console.print(f"💡 搜索: kas market search {name}")
            return
        
        console.print(f"📦 安装 {name} v{info.version}...")
        
        if market.install(name):
            console.print(f"✅ 安装成功")
            console.print(f"💡 使用: kas chat {name}")
        else:
            console.print(f"❌ 安装失败")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
def market_list():
    """📋 列出已安装的 Agent"""
    try:
        market = get_market()
        installed = market.list_installed()
        
        if installed:
            console.print(f"\n📦 已安装 {len(installed)} 个 Agent:")
            for name in installed:
                console.print(f"  • {name}")
            console.print(f"\n💡 使用 'kas chat <name>' 开始对话")
        else:
            console.print(f"📭 未安装任何 Agent")
            console.print(f"💡 尝试: kas market search")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
def market_stats():
    """📊 市场统计"""
    try:
        market = get_market()
        stats = market.get_stats()
        
        console.print(Panel.fit(
            f"[bold cyan]📊 本地市场统计[/bold cyan]\n\n"
            f"📦 总包数: {stats.total_packages}\n"
            f"📥 总下载: {stats.total_downloads}\n"
            f"🕐 更新时间: {stats.last_updated[:19]}\n\n"
            f"[bold]热门标签:[/bold]\n"
            + "\n".join(f"  • {tag} ({count})" for tag, count in stats.top_tags[:5]),
            title="📊 Market Stats"
        ))
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('agent')
@click.option('--capability', '-c', help='只测试特定能力 (code_review/documentation/debugging)')
@click.option('--output', '-o', help='保存报告到文件')
def validate(agent, capability, output):
    """✅ 验证 Agent 能力"""
    try:
        from kas.core.models import Agent
        
        # 加载 Agent
        agent_dir = Path.home() / '.kas' / 'agents' / agent
        if not agent_dir.exists():
            console.print(f"❌ Agent not found: {agent}")
            return
        
        with open(agent_dir / 'agent.yaml', 'r') as f:
            import yaml
            agent_data = yaml.safe_load(f)
        
        agent_obj = Agent.from_dict(agent_data)
        
        console.print(f"\n🧪 [bold blue]开始验证 {agent}...[/bold blue]")
        console.print(f"   测试能力: {', '.join(c.name for c in agent_obj.capabilities)}\n")
        
        # 获取 LLM 客户端
        llm_client = get_llm_client()
        if not llm_client:
            console.print("⚠️  未配置 LLM API，无法验证")
            console.print("💡 运行 'kas config-setup' 配置 API Key")
            return
        
        # 创建验证器
        validator = CapabilityValidator(llm_client)
        
        # 运行测试
        if capability:
            from kas.core.models import CapabilityType
            cap_map = {
                'code_review': CapabilityType.CODE_REVIEW,
                'documentation': CapabilityType.DOCUMENTATION,
                'debugging': CapabilityType.DEBUGGING
            }
            cap_type = cap_map.get(capability.lower())
            if cap_type:
                report = validator.validate_capability(agent_obj, cap_type)
            else:
                console.print(f"❌ 未知能力: {capability}")
                return
        else:
            report = validator.validate(agent_obj)
        
        # 显示报告
        console.print(validator.format_report(report))
        
        # 保存报告
        if output:
            import json
            report_dict = {
                'agent_name': report.agent_name,
                'agent_version': report.agent_version,
                'timestamp': report.timestamp,
                'overall_score': report.overall_score,
                'total_tests': report.total_tests,
                'passed_tests': report.passed_tests,
                'failed_tests': report.failed_tests,
                'capability_scores': report.capability_scores,
                'summary': report.summary,
                'recommendations': report.recommendations,
                'results': [asdict(r) for r in report.results]
            }
            with open(output, 'w') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            console.print(f"\n💾 报告已保存: {output}")
        
        # 建议
        if report.overall_score < 60:
            console.print(f"\n💡 建议运行 'kas evolve {agent} --force' 来提升能力")
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        import traceback
        traceback.print_exc()


@cli.command()
@click.argument('agent', required=False)
@click.option('--days', '-d', default=30, help='统计天数')
def stats(agent, days):
    """📊 查看使用统计"""
    try:
        dashboard = get_dashboard()
        
        if agent:
            console.print(dashboard.show_agent_stats(agent, days))
        else:
            console.print(dashboard.show_overview())
        
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


def main():
    """入口点"""
    cli()


if __name__ == '__main__':
    main()
