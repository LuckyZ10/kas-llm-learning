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
from kas.core.cloud_market import get_cloud_client, CloudMarketError
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
    """Klaw Agent Studio - 专业开发者 Agent 孵化平台

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
@click.option('--attach', '-a', multiple=True, help='附加文件 (可多次使用)')
def chat(agent, interactive, message, mock, attach):
    """与 Agent 对话"""
    try:
        # 处理附件
        attachments = []
        if attach:
            from kas.core.multimodal import Attachment
            
            for file_path in attach:
                path = Path(file_path)
                if not path.exists():
                    console.print(f"❌ 文件不存在: {file_path}")
                    continue
                
                with open(path, 'rb') as f:
                    content = f.read()
                
                attachments.append(Attachment(
                    name=path.name,
                    content=content,
                    content_type="",
                    size=len(content)
                ))
            
            if attachments:
                console.print(f"📎 已加载 {len(attachments)} 个附件")
        
        # 创建LLM客户端
        llm_client = get_llm_client(force_mock=mock)
        
        # 如果有附件，使用多模态对话
        if attachments:
            from kas.core.multimodal import MultimodalChat
            engine = MultimodalChat(agent)
            
            if message:
                response = engine.run(message, attachments=attachments, use_mock=mock)
                console.print(f"\n[bold cyan]{agent}:[/bold cyan] {response}\n")
            else:
                console.print("❌ 使用附件时需要提供 --message")
        else:
            # 普通对话
            engine = ChatEngine(llm_client=llm_client)
            loaded_agent = engine.load_agent(agent)
            
            if llm_client:
                console.print(f"\n[dim]🧠 Using LLM API for responses[/dim]")
            else:
                console.print(f"\n[dim]🤖 Mock mode - Set OPENAI_API_KEY for real responses[/dim]")
            
            if interactive:
                engine.interactive_chat(agent)
            elif message:
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
@click.option('--cloud/--local', default=False, help='发布到云端市场 (默认本地)')
@click.option('--name', help='包名称 (默认从包中读取)')
@click.option('--version', help='版本号 (默认从包中读取)')
@click.option('--description', help='包描述')
@click.option('--tag', multiple=True, help='标签')
def market_publish(package, cloud, name, version, description, tag):
    """🛒 发布包到市场"""
    try:
        pkg_path = Path(package)
        if not pkg_path.exists():
            console.print(f"❌ 包文件不存在: {package}")
            return

        # 解包获取信息
        agent = unpack_agent(package)
        if not agent:
            console.print(f"❌ 无法解析包: {package}")
            return

        # 使用提供的参数或从包中读取
        pkg_name = name or agent.name
        pkg_version = version or agent.version
        pkg_description = description or agent.description or f"{agent.name} Agent"
        pkg_tags = list(tag) if tag else []

        # 发布到云端
        if cloud:
            try:
                client = get_cloud_client()
                if not client.is_available():
                    console.print("❌ 云端市场服务器不可用")
                    console.print("💡 请确保服务器在运行: cd kas-cloud && bash start.sh")
                    return

                console.print(f"☁️  正在发布到云端市场...")
                console.print(f"   名称: {pkg_name}")
                console.print(f"   版本: {pkg_version}")

                result = client.publish(
                    package_path=str(pkg_path),
                    name=pkg_name,
                    version=pkg_version,
                    description=pkg_description,
                    tags=pkg_tags
                )

                console.print(f"✅ 云端发布成功!")
                console.print(f"📦 {result.get('name')} v{result.get('version')}")
                console.print(f"💡 其他人可以用 'kas market install {pkg_name}' 安装")
                return

            except CloudMarketError as e:
                console.print(f"❌ 云端发布失败: {e}")
                console.print(f"💡 尝试发布到本地: kas market publish {package}")
                return

        # 发布到本地市场
        market = get_market()

        if market.publish(package):
            console.print(f"✅ 本地发布成功")
            console.print(f"📦 {agent.name} v{agent.version}")
            console.print(f"💡 其他人可以用 'kas market install {agent.name}' 安装")
        else:
            console.print(f"❌ 发布失败")

    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('query', default="")
@click.option('--tag', multiple=True, help='按标签过滤')
@click.option('--cloud/--local', default=True, help='使用云端或本地市场 (默认云端)')
def market_search(query, tag, cloud):
    """🔍 搜索市场"""
    try:
        tags = list(tag) if tag else None

        # 优先尝试云端市场
        if cloud:
            try:
                client = get_cloud_client()
                if client.is_available():
                    results = client.search(query, tags=tags)

                    if results:
                        table = Table(title=f"🔍 云端市场搜索结果: '{query or 'all'}'")
                        table.add_column("名称", style="cyan")
                        table.add_column("版本", style="green")
                        table.add_column("作者", style="yellow")
                        table.add_column("下载", justify="right", style="blue")
                        table.add_column("评分", justify="right", style="magenta")
                        table.add_column("标签", style="dim")

                        for pkg in results:
                            rating_str = f"⭐ {pkg.rating:.1f}" if pkg.rating > 0 else "-"
                            tags_str = ", ".join(pkg.tags[:3]) if pkg.tags else ""
                            table.add_row(
                                pkg.name,
                                pkg.version,
                                pkg.author_name,
                                str(pkg.downloads),
                                rating_str,
                                tags_str
                            )

                        console.print(table)
                        console.print(f"\n💡 使用 'kas market install <name>' 安装")
                    else:
                        console.print(f"📭 云端市场未找到: {query or '任何 Agent'}")

                    return
                else:
                    console.print("⚠️  云端市场不可用，回退到本地市场...")
            except CloudMarketError as e:
                console.print(f"⚠️  云端错误: {e}")
                console.print("🔄 回退到本地市场...")

        # 本地市场
        market = get_market()
        results = market.search(query, tags=tags)

        if results:
            console.print(market.format_search_results(results))
        else:
            console.print(f"📭 本地市场未找到: {query or '任何 Agent'}")
            console.print(f"💡 尝试: kas market search python")

    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.command()
@click.argument('name')
@click.option('--cloud/--local', default=True, help='从云端或本地市场安装 (默认云端)')
@click.option('--id', type=int, help='云端包 ID (精确安装)')
def market_install(name, cloud, id):
    """⬇️  从市场安装 Agent"""
    try:
        # 如果指定了云端 ID
        if id and cloud:
            try:
                client = get_cloud_client()
                if not client.is_available():
                    console.print("❌ 云端市场不可用")
                    return

                console.print(f"📦 从云端下载 (ID: {id})...")

                # 获取包信息
                info = client.get_package_info(id)
                console.print(f"📋 {info['name']} v{info['version']} by {info['author_name']}")

                # 下载
                download_path = client.download(id, output_dir='/tmp')
                console.print(f"📥 下载完成: {download_path}")

                # 安装
                from kas.core.market import unpack_agent
                agent = unpack_agent(download_path, install=True)
                if agent:
                    console.print(f"✅ 安装成功: {agent.name}")
                    console.print(f"💡 使用: kas chat {agent.name}")
                else:
                    console.print(f"❌ 安装失败")

                # 清理临时文件
                Path(download_path).unlink(missing_ok=True)
                return

            except CloudMarketError as e:
                console.print(f"❌ 云端安装失败: {e}")
                return

        # 优先尝试云端市场
        if cloud:
            try:
                client = get_cloud_client()
                if client.is_available():
                    results = client.search(name)

                    # 找精确匹配的
                    for pkg in results:
                        if pkg.name.lower() == name.lower():
                            console.print(f"📦 在云端市场找到: {pkg.name} v{pkg.version}")
                            console.print(f"   作者: {pkg.author_name} | 下载: {pkg.downloads} | 评分: ⭐ {pkg.rating:.1f}")

                            # 下载并安装
                            download_path = client.download(pkg.id, output_dir='/tmp')
                            agent = unpack_agent(download_path, install=True)

                            if agent:
                                console.print(f"✅ 安装成功!")
                                console.print(f"💡 使用: kas chat {agent.name}")

                            Path(download_path).unlink(missing_ok=True)
                            return

                    console.print(f"⚠️  云端市场未找到精确匹配: {name}")
                    console.print(f"💡 尝试搜索: kas market search {name}")
                    return

            except CloudMarketError as e:
                console.print(f"⚠️  云端错误: {e}")
                console.print("🔄 回退到本地市场...")

        # 本地市场
        market = get_market()
        info = market.get_info(name)
        if not info:
            console.print(f"❌ 本地市场找不到: {name}")
            console.print(f"💡 搜索云端: kas market search {name}")
            return

        console.print(f"📦 从本地市场安装 {name} v{info.version}...")

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


@cli.command()
@click.option('--host', default='127.0.0.1', help='绑定地址')
@click.option('--port', '-p', default=8080, help='端口号')
@click.option('--debug', is_flag=True, help='调试模式')
def dashboard(host, port, debug):
    """🌐 启动 Web 统计面板"""
    try:
        console.print(f"\n🚀 [bold blue]启动 KAS Dashboard...[/bold blue]")
        console.print(f"   地址: http://{host}:{port}")
        console.print(f"   按 Ctrl+C 停止\n")

        from kas.dashboard.app import run_dashboard
        run_dashboard(host=host, port=port, debug=debug)

    except ImportError as e:
        console.print(f"❌ [bold red]缺少依赖:[/bold red] {e}")
        console.print("💡 请安装 dashboard 依赖:")
        console.print("   pip install flask")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


@cli.group()
def knowledge():
    """📚 知识库管理"""
    pass


@knowledge.command('add')
@click.argument('agent')
@click.argument('document')
@click.option('--type', '-t', 'doc_type', default='file',
              type=click.Choice(['file', 'text', 'url']), help='文档类型')
@click.option('--tag', multiple=True, help='文档标签')
def knowledge_add(agent, document, doc_type, tag):
    """添加文档到知识库"""
    try:
        from kas.core.knowledge import get_knowledge_base

        kb = get_knowledge_base(agent)

        # 读取文档内容
        if doc_type == 'file':
            doc_path = Path(document)
            if not doc_path.exists():
                console.print(f"❌ 文件不存在: {document}")
                return

            content = doc_path.read_text(encoding='utf-8')
            metadata = {
                'source': str(doc_path),
                'filename': doc_path.name,
                'tags': list(tag) if tag else []
            }

        elif doc_type == 'text':
            content = document
            metadata = {'tags': list(tag) if tag else []}

        elif doc_type == 'url':
            import requests
            response = requests.get(document, timeout=30)
            content = response.text
            metadata = {
                'source': document,
                'tags': list(tag) if tag else []
            }

        # 添加到知识库
        doc_id = kb.add_document(content, metadata)

        console.print(f"✅ 文档已添加到知识库")
        console.print(f"   Agent: {agent}")
        console.print(f"   ID: {doc_id[:16]}...")
        console.print(f"   字符数: {len(content)}")

    except ImportError:
        console.print("❌ 缺少依赖: pip install chromadb")
    except Exception as e:
        console.print(f"❌ Error: {e}")


@knowledge.command('search')
@click.argument('agent')
@click.argument('query')
@click.option('--top', '-k', default=5, help='返回结果数量')
@click.option('--score', '-s', is_flag=True, help='显示相似度分数')
def knowledge_search(agent, query, top, score):
    """搜索知识库"""
    try:
        from kas.core.knowledge import get_knowledge_base
        from rich.table import Table

        kb = get_knowledge_base(agent)
        results = kb.search(query, top_k=top)

        if not results:
            console.print("📭 未找到相关文档")
            return

        console.print(f"\n🔍 搜索: '{query}'")
        console.print(f"   找到 {len(results)} 条结果\n")

        for i, result in enumerate(results, 1):
            doc = result.document
            preview = doc.content[:200].replace('\n', ' ') + "..."

            panel = Panel(
                f"[dim]{preview}[/dim]",
                title=f"#{i} [bold]{doc.metadata.get('filename', 'Document')}[/bold]",
                subtitle=f"Score: {result.score:.3f}" if score else None,
                border_style="blue"
            )
            console.print(panel)

    except ImportError:
        console.print("❌ 缺少依赖: pip install chromadb")
    except Exception as e:
        console.print(f"❌ Error: {e}")


@knowledge.command('list')
@click.argument('agent')
@click.option('--limit', '-l', default=20, help='显示数量')
def knowledge_list(agent, limit):
    """列出知识库文档"""
    try:
        from kas.core.knowledge import get_knowledge_base
        from rich.table import Table

        kb = get_knowledge_base(agent)
        docs = kb.list_documents(limit=limit)

        if not docs:
            console.print(f"📭 {agent} 的知识库为空")
            return

        table = Table(title=f"📚 {agent} 知识库 ({kb.count()} 文档)")
        table.add_column("ID", style="dim")
        table.add_column("来源")
        table.add_column("标签")
        table.add_column("创建时间", style="dim")

        for doc in docs:
            source = doc.metadata.get('filename', doc.metadata.get('source', 'Unknown'))
            tags = ", ".join(doc.metadata.get('tags', [])) or "-"
            created = doc.metadata.get('created_at', '-')[:10]
            table.add_row(doc.id[:8], source[:30], tags, created)

        console.print(table)

    except ImportError:
        console.print("❌ 缺少依赖: pip install chromadb")
    except Exception as e:
        console.print(f"❌ Error: {e}")


@knowledge.command('clear')
@click.argument('agent')
@click.confirmation_option(prompt='确定要清空知识库吗？')
def knowledge_clear(agent):
    """清空知识库"""
    try:
        from kas.core.knowledge import get_knowledge_base

        kb = get_knowledge_base(agent)
        kb.clear()

        console.print(f"🗑️ {agent} 的知识库已清空")

    except ImportError:
        console.print("❌ 缺少依赖: pip install chromadb")
    except Exception as e:
        console.print(f"❌ Error: {e}")


@cli.group()
def memory():
    """🧠 用户偏好记忆"""
    pass


@memory.command('show')
@click.argument('agent')
def memory_show(agent):
    """查看记忆"""
    try:
        from kas.core.knowledge import get_user_memory

        mem = get_user_memory(agent)
        data = mem.show()

        if not data:
            console.print(f"📝 {agent} 没有记忆")
            return

        console.print(f"\n🧠 {agent} 的记忆:")
        console.print()

        for key, value in data.items():
            if key != 'updated_at':
                console.print(f"  [bold]{key}:[/bold] {value}")

        if 'updated_at' in data:
            console.print(f"\n  [dim]更新时间: {data['updated_at']}[/dim]")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@memory.command('set')
@click.argument('agent')
@click.argument('key')
@click.argument('value')
def memory_set(agent, key, value):
    """设置记忆"""
    try:
        from kas.core.knowledge import get_user_memory

        mem = get_user_memory(agent)
        mem.set(key, value)

        console.print(f"✅ 已设置: {key} = {value}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@memory.command('clear')
@click.argument('agent')
@click.confirmation_option(prompt='确定要清空记忆吗？')
def memory_clear(agent):
    """清空记忆"""
    try:
        from kas.core.knowledge import get_user_memory

        mem = get_user_memory(agent)
        mem.clear()

        console.print(f"🗑️ {agent} 的记忆已清空")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@cli.group()
def workflow():
    """🔗 Agent 工作流编排"""
    pass


@workflow.command('create')
@click.argument('name')
@click.option('--desc', '-d', help='工作流描述')
def workflow_create(name, desc):
    """创建工作流"""
    try:
        from kas.core.workflow import get_workflow_engine

        engine = get_workflow_engine()
        wf = engine.create(name, description=desc)

        console.print(f"✅ 工作流已创建: {name}")
        if desc:
            console.print(f"   描述: {desc}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@workflow.command('add')
@click.argument('name')
@click.argument('agent')
@click.argument('task')
@click.option('--after', '-a', help='在指定步骤后执行')
def workflow_add(name, agent, task, after):
    """添加步骤到工作流"""
    try:
        from kas.core.workflow import get_workflow_engine

        engine = get_workflow_engine()
        wf = engine.load(name)

        if not wf:
            console.print(f"❌ 工作流不存在: {name}")
            return

        depends_on = [after] if after else []
        step_id = wf.add_step(agent, task, depends_on)
        engine.save(wf)

        console.print(f"✅ 步骤已添加: {step_id}")
        console.print(f"   Agent: {agent}")
        console.print(f"   任务: {task}")
        if after:
            console.print(f"   依赖: {after}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@workflow.command('list')
def workflow_list():
    """列出所有工作流"""
    try:
        from kas.core.workflow import get_workflow_engine

        engine = get_workflow_engine()
        workflows = engine.list_workflows()

        if not workflows:
            console.print("📭 暂无工作流")
            return

        table = Table(title="🔗 工作流列表")
        table.add_column("名称", style="bold")
        table.add_column("步骤数", justify="center")
        table.add_column("描述", style="dim")

        for wf_name in workflows:
            wf = engine.load(wf_name)
            table.add_row(
                wf_name,
                str(len(wf.steps)),
                wf.description or "-"
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error: {e}")


@workflow.command('show')
@click.argument('name')
def workflow_show(name):
    """查看工作流详情"""
    try:
        from kas.core.workflow import get_workflow_engine
        from rich.tree import Tree

        engine = get_workflow_engine()
        wf = engine.load(name)

        if not wf:
            console.print(f"❌ 工作流不存在: {name}")
            return

        console.print(f"\n🔗 [bold]{wf.name}[/bold]")
        if wf.description:
            console.print(f"   {wf.description}")
        console.print()

        if not wf.steps:
            console.print("   (暂无步骤)")
            return

        # 显示执行顺序
        try:
            order = wf.get_execution_order()
            console.print("[bold]执行顺序:[/bold]")

            for i, layer in enumerate(order, 1):
                console.print(f"\n  第{i}层:")
                for step_id in layer:
                    step = next(s for s in wf.steps if s.id == step_id)
                    dep_str = f" (依赖: {', '.join(step.depends_on)})" if step.depends_on else ""
                    console.print(f"    - [{step_id}] {step.agent_name}: {step.task}{dep_str}")

        except ValueError as e:
            console.print(f"   ⚠️ {e}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@workflow.command('run')
@click.argument('name')
@click.argument('context')
@click.option('--verbose', '-v', is_flag=True, help='显示详细输出')
@click.option('--timeout', '-t', default=300, help='单步骤超时时间（秒），默认300')
@click.option('--mock', is_flag=True, help='使用 mock 模式（不调用 LLM）')
def workflow_run(name, context, verbose, timeout, mock):
    """执行工作流"""
    try:
        from kas.core.workflow import get_workflow_engine, StepStatus

        engine = get_workflow_engine()
        wf = engine.load(name)

        if not wf:
            console.print(f"❌ 工作流不存在: {name}")
            return

        # 检查是否有循环依赖
        try:
            execution_order = wf.get_execution_order()
        except ValueError as e:
            console.print(f"❌ 工作流配置错误: {e}")
            console.print("💡 请检查步骤依赖关系是否有循环")
            return

        console.print(f"\n🚀 执行工作流: [bold]{name}[/bold]")
        console.print(f"   上下文: {context}")
        console.print(f"   步骤数: {len(wf.steps)}")
        console.print(f"   超时: {timeout}秒\n")

        # 进度回调
        def on_step_complete(step, output):
            status_icon = "✅" if step.status == StepStatus.COMPLETED else "❌"
            console.print(f"{status_icon} [{step.id}] {step.agent_name} 完成")
            if verbose and output:
                console.print(f"   输出: {output[:200]}...")
            if step.error:
                console.print(f"   错误: {step.error}")

        # 执行
        results = engine.execute(
            wf, context,
            callback=on_step_complete,
            timeout=timeout,
            use_mock=mock
        )

        console.print()
        if results['success']:
            console.print(f"✅ 工作流执行成功")
        else:
            console.print(f"❌ 工作流执行失败")
            if 'error' in results:
                console.print(f"   错误: {results['error']}")

        console.print(f"\n   开始: {results['started_at']}")
        console.print(f"   结束: {results.get('completed_at', 'N/A')}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@workflow.command('delete')
@click.argument('name')
@click.confirmation_option(prompt='确定要删除工作流吗？')
def workflow_delete(name):
    """删除工作流"""
    try:
        from kas.core.workflow import get_workflow_engine

        engine = get_workflow_engine()
        if engine.delete(name):
            console.print(f"🗑️ 工作流已删除: {name}")
        else:
            console.print(f"❌ 工作流不存在: {name}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@cli.group()
def abtest():
    """🧪 A/B 测试 - Agent 版本对比"""
    pass


@abtest.command('create')
@click.argument('name')
@click.argument('agent')
@click.argument('version_a')
@click.argument('version_b')
@click.option('--desc', '-d', help='测试描述')
@click.option('--size', '-s', default=100, help='样本量，默认100')
def abtest_create(name, agent, version_a, version_b, desc, size):
    """创建 A/B 测试"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        test = engine.create(
            name=name,
            agent_name=agent,
            version_a=version_a,
            version_b=version_b,
            description=desc,
            sample_size=size
        )

        console.print(f"✅ A/B 测试已创建")
        console.print(f"   ID: {test.id}")
        console.print(f"   名称: {name}")
        console.print(f"   Agent: {agent}")
        console.print(f"   版本 A: {version_a}")
        console.print(f"   版本 B: {version_b}")
        console.print(f"   样本量: {size}")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@abtest.command('list')
def abtest_list():
    """列出 A/B 测试"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        tests = engine.list_tests()

        if not tests:
            console.print("📭 暂无 A/B 测试")
            return

        table = Table(title="🧪 A/B 测试列表")
        table.add_column("ID", style="dim")
        table.add_column("名称", style="bold")
        table.add_column("Agent")
        table.add_column("版本 A")
        table.add_column("版本 B")
        table.add_column("状态")
        table.add_column("样本量")

        for test in tests:
            status_color = {
                'running': 'green',
                'paused': 'yellow',
                'completed': 'blue'
            }.get(test.status.value, 'white')

            table.add_row(
                test.id,
                test.name,
                test.agent_name,
                test.version_a,
                test.version_b,
                f"[{status_color}]{test.status.value}[/{status_color}]",
                str(test.sample_size)
            )

        console.print(table)

    except Exception as e:
        console.print(f"❌ Error: {e}")


@abtest.command('status')
@click.argument('test_id')
def abtest_status(test_id):
    """查看测试状态"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        stats = engine.get_stats(test_id)

        if not stats:
            console.print(f"❌ 测试不存在: {test_id}")
            return

        test = stats['test']

        console.print(f"\n🧪 [bold]{test['name']}[/bold]")
        console.print(f"   ID: {test_id}")
        console.print(f"   Agent: {test['agent_name']}")
        console.print(f"   版本 A: {test['version_a']}")
        console.print(f"   版本 B: {test['version_b']}")
        console.print(f"   状态: {test['status']}")
        console.print()

        # 统计
        completed = stats['completed_sessions']
        total = test['sample_size']

        console.print(f"[bold]进度:[/bold] {completed}/{total} ({completed/total*100:.1f}%)")
        console.print()

        if completed > 0:
            console.print(f"[bold]结果:[/bold]")
            console.print(f"   版本 A 胜出: {stats['a_wins']} 次 ({stats['a_wins']/completed*100:.1f}%)")
            console.print(f"   版本 B 胜出: {stats['b_wins']} 次 ({stats['b_wins']/completed*100:.1f}%)")
            console.print(f"   平局: {stats['ties']} 次")
            console.print()

            if stats['avg_rating_a']:
                console.print(f"   版本 A 平均评分: {stats['avg_rating_a']}")
            if stats['avg_rating_b']:
                console.print(f"   版本 B 平均评分: {stats['avg_rating_b']}")
            console.print()

            if stats['winner']:
                console.print(f"🎉 [bold green]当前胜者: 版本 {stats['winner']}[/bold green]")
                console.print(f"   置信度: {stats['confidence']*100:.1f}%")
            else:
                console.print("⚖️  暂无明确胜者")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@abtest.command('compare')
@click.argument('test_id')
@click.argument('prompt')
@click.option('--mock', is_flag=True, help='使用 mock 模式')
def abtest_compare(test_id, prompt, mock):
    """运行一次对比测试"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        test = engine.get_test(test_id)

        if not test:
            console.print(f"❌ 测试不存在: {test_id}")
            return

        # 创建会话
        session = engine.create_session(test_id, prompt)
        if not session:
            console.print("❌ 无法创建测试会话")
            return

        console.print(f"\n🧪 运行 A/B 对比测试")
        console.print(f"   会话 ID: {session.id}")
        console.print(f"   分配到: 版本 {session.assigned_version}")
        console.print()

        # 运行对比
        with console.status("[bold green]运行版本 A..."):
            from kas.core.chat import ChatEngine
            from kas.core.versioning import get_version_manager
            from kas.core.config import get_config
            from pathlib import Path
            from kas.core.models import Agent

            # 加载两个版本
            vm = get_version_manager(test.agent_name)
            agent_a = vm.load_version(test.version_a)

            if test.version_b == 'current':
                agent_path = Path(get_config().agents_dir) / test.agent_name
                agent_b = Agent.load(agent_path)
            else:
                agent_b = vm.load_version(test.version_b)

            # 运行
            chat_a = ChatEngine(test.agent_name)
            chat_a.agent = agent_a
            response_a = chat_a.run(prompt, use_mock=mock)

        with console.status("[bold green]运行版本 B..."):
            chat_b = ChatEngine(test.agent_name)
            chat_b.agent = agent_b
            response_b = chat_b.run(prompt, use_mock=mock)

        # 显示结果
        console.print()
        console.print(Panel(
            response_a[:500] + "..." if len(response_a) > 500 else response_a,
            title=f"[bold blue]版本 A ({test.version_a})[/bold blue]",
            border_style="blue"
        ))

        console.print(Panel(
            response_b[:500] + "..." if len(response_b) > 500 else response_b,
            title=f"[bold green]版本 B ({test.version_b})[/bold green]",
            border_style="green"
        ))

        console.print()
        console.print(f"💡 使用 [bold]kas abtest vote {test_id} {session.id} <A/B/tie>[/bold] 记录你的选择")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@abtest.command('vote')
@click.argument('test_id')
@click.argument('session_id')
@click.argument('choice', type=click.Choice(['A', 'B', 'tie']))
@click.option('--rating-a', type=int, help='版本 A 评分 1-5')
@click.option('--rating-b', type=int, help='版本 B 评分 1-5')
@click.option('--feedback', '-f', help='文字反馈')
def abtest_vote(test_id, session_id, choice, rating_a, rating_b, feedback):
    """记录测试投票"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        engine.record_result(
            session_id=session_id,
            test_id=test_id,
            user_choice=choice,
            rating_a=rating_a,
            rating_b=rating_b,
            feedback=feedback
        )

        console.print(f"✅ 投票已记录: {choice}")

        # 显示当前统计
        stats = engine.get_stats(test_id)
        console.print(f"\n   当前进度: {stats['completed_sessions']}/{stats['test']['sample_size']}")

        if stats['winner']:
            console.print(f"   当前胜者: 版本 {stats['winner']} ({stats['confidence']*100:.1f}%)")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@abtest.command('winner')
@click.argument('test_id')
@click.argument('version', type=click.Choice(['A', 'B']))
def abtest_winner(test_id, version):
    """手动宣布胜者"""
    try:
        from kas.core.abtest import get_abtest_engine

        engine = get_abtest_engine()
        if engine.declare_winner(test_id, version):
            console.print(f"🎉 已宣布版本 {version} 为胜者！")
        else:
            console.print(f"❌ 操作失败")

    except Exception as e:
        console.print(f"❌ Error: {e}")


@cli.group()
def equip():
    """🛠️ 装备管理 - 为 Agent 添加工具"""
    pass


@equip.command('list')
def equip_list():
    """列出所有可用装备"""
    try:
        from kas.core.equipment import get_equipment_pool
        
        pool = get_equipment_pool()
        equipments = pool.list_all()
        
        if not equipments:
            console.print("📭 暂无可用装备")
            return
        
        table = Table(title="🛠️ 可用装备列表")
        table.add_column("名称", style="bold cyan")
        table.add_column("类型", style="dim")
        table.add_column("描述")
        table.add_column("状态")
        
        for eq in equipments:
            status = "✅ 可用" if eq.get("status") != "error" else "❌ 错误"
            table.add_row(
                eq["name"],
                eq["type"],
                eq.get("description", "-"),
                status
            )
        
        console.print(table)
        console.print("\n💡 使用 [bold]kas equip add <agent> <equipment>[/bold] 为 Agent 添加装备")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")


@equip.command('add')
@click.argument('agent')
@click.argument('equipment')
@click.option('--config', '-c', help='JSON 配置字符串')
def equip_add(agent, equipment, config):
    """为 Agent 添加装备"""
    try:
        from kas.core.equipment import get_equipment_pool
        from kas.core.models import Agent
        from kas.core.config import get_config
        from pathlib import Path
        import json
        
        # 检查装备是否存在
        pool = get_equipment_pool()
        eq = pool.get(equipment)
        if not eq:
            console.print(f"❌ 装备不存在: {equipment}")
            console.print(f"💡 可用装备: {', '.join([e['name'] for e in pool.list_all()])}")
            return
        
        # 加载 Agent
        config_obj = get_config()
        agent_path = Path(config_obj.agents_dir) / agent
        
        if not agent_path.exists():
            console.print(f"❌ Agent 不存在: {agent}")
            return
        
        agent_obj = Agent.load(agent_path)
        
        # 添加装备到 Agent 配置
        if not hasattr(agent_obj, 'equipment'):
            agent_obj.equipment = []
        
        # 检查是否已存在
        if equipment in agent_obj.equipment:
            console.print(f"⚠️ Agent {agent} 已有装备: {equipment}")
            return
        
        agent_obj.equipment.append(equipment)
        
        # 保存
        agent_obj.save(agent_path)
        
        console.print(f"✅ 已为 Agent [bold]{agent}[/bold] 添加装备: [cyan]{equipment}[/cyan]")
        console.print(f"   当前装备: {', '.join(agent_obj.equipment)}")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")


@equip.command('remove')
@click.argument('agent')
@click.argument('equipment')
def equip_remove(agent, equipment):
    """从 Agent 移除装备"""
    try:
        from kas.core.models import Agent
        from kas.core.config import get_config
        from pathlib import Path
        
        config = get_config()
        agent_path = Path(config.agents_dir) / agent
        
        if not agent_path.exists():
            console.print(f"❌ Agent 不存在: {agent}")
            return
        
        agent_obj = Agent.load(agent_path)
        
        if not hasattr(agent_obj, 'equipment') or equipment not in agent_obj.equipment:
            console.print(f"⚠️ Agent {agent} 没有装备: {equipment}")
            return
        
        agent_obj.equipment.remove(equipment)
        agent_obj.save(agent_path)
        
        console.print(f"🗑️ 已从 Agent [bold]{agent}[/bold] 移除装备: [cyan]{equipment}[/cyan]")
        
    except Exception as e:
        console.print(f"❌ Error: {e}")


@equip.command('show')
@click.argument('agent')
def equip_show(agent):
    """查看 Agent 的装备"""
    try:
        from kas.core.models import Agent
        from kas.core.equipment import get_equipment_pool
        from kas.core.config import get_config
        from pathlib import Path
        
        config = get_config()
        agent_path = Path(config.agents_dir) / agent
        
        if not agent_path.exists():
            console.print(f"❌ Agent 不存在: {agent}")
            return
        
        agent_obj = Agent.load(agent_path)
        equipments = getattr(agent_obj, 'equipment', [])
        
        if not equipments:
            console.print(f"📭 Agent [bold]{agent}[/bold] 暂无装备")
            console.print(f"💡 使用 [bold]kas equip add {agent} <equipment>[/bold] 添加")
            return
        
        console.print(f"\n🛠️ Agent [bold]{agent}[/bold] 的装备:\n")
        
        pool = get_equipment_pool()
        for eq_name in equipments:
            eq = pool.get(eq_name)
            if eq:
                console.print(f"  ✅ [cyan]{eq_name}[/cyan] - {eq.config.description}")
            else:
                console.print(f"  ⚠️ [yellow]{eq_name}[/yellow] - 装备未找到")
        
        console.print()
        
    except Exception as e:
        console.print(f"❌ Error: {e}")


@equip.command('use')
@click.argument('equipment')
@click.option('--params', '-p', help='JSON 参数字符串')
def equip_use(equipment, params):
    """测试使用装备"""
    try:
        from kas.core.equipment import get_equipment_pool
        import json
        
        pool = get_equipment_pool()
        eq = pool.get(equipment)
        
        if not eq:
            console.print(f"❌ 装备不存在: {equipment}")
            return
        
        # 解析参数
        param_dict = {}
        if params:
            param_dict = json.loads(params)
        
        console.print(f"🔧 使用装备: [cyan]{equipment}[/cyan]")
        console.print(f"   参数: {param_dict}\n")
        
        with console.status("[bold green]执行中..."):
            result = pool.use(equipment, param_dict)
        
        console.print("✅ [bold]执行结果:[/bold]\n")
        console.print_json(data=result)
        
    except Exception as e:
        console.print(f"❌ Error: {e}")


@cli.command()
@click.option('--host', default='0.0.0.0', help='绑定地址')
@click.option('--port', '-p', default=3000, help='端口号')
@click.option('--reload', is_flag=True, help='开发模式（自动重载）')
def web(host, port, reload):
    """🌐 启动 Web 界面"""
    try:
        console.print(f"\n🌐 [bold blue]启动 KAS Web...[/bold blue]")
        console.print(f"   地址: http://{host}:{port}")
        console.print(f"   按 Ctrl+C 停止\n")
        
        from kas.web.app import run_web
        run_web(host=host, port=port, reload=reload)
        
    except ImportError as e:
        console.print(f"❌ [bold red]缺少依赖:[/bold red] {e}")
        console.print("💡 请安装 web 依赖:")
        console.print("   pip install fastapi uvicorn")
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")


def main():
    """入口点"""
    cli()


if __name__ == '__main__':
    main()
