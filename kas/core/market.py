"""
KAS Core - Agent Market
Agent 市场系统 - 像 App Store 一样分享 Agent

功能:
- .kas-agent 包格式 (ZIP)
- 打包/导入/导出
- 本地市场索引
- 搜索/安装/发布
- 元数据管理
"""

import json
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from .models import Agent


# 市场目录
MARKET_DIR = Path.home() / '.kas' / 'market'
MARKET_INDEX = MARKET_DIR / 'index.json'


@dataclass
class PackageInfo:
    """包信息"""
    name: str                    # Agent 名称
    version: str                 # 版本号
    description: str             # 描述
    author: str                  # 作者
    author_email: str            # 作者邮箱
    tags: List[str]              # 标签
    capabilities: List[str]      # 能力列表
    downloads: int = 0           # 下载次数
    rating: float = 0.0          # 评分
    created_at: str = ""         # 创建时间
    updated_at: str = ""         # 更新时间
    package_hash: str = ""       # 包哈希（用于校验）
    file_size: int = 0           # 文件大小
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


@dataclass
class MarketStats:
    """市场统计"""
    total_packages: int = 0
    total_downloads: int = 0
    top_tags: List[tuple] = None  # [(tag, count), ...]
    last_updated: str = ""


class PackagePacker:
    """
    包打包器 - 把 Agent 变成可分享的 .kas-agent 文件
    
    用法:
        packer = PackagePacker()
        packer.pack(agent, output_path="MyAgent.kas-agent")
        
        # 解包
        agent = packer.unpack("MyAgent.kas-agent", install=True)
    """
    
    def __init__(self):
        self.temp_dir = Path.home() / '.kas' / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def pack(self, agent: Agent, output_path: Optional[str] = None,
             author: str = "", author_email: str = "",
             tags: List[str] = None, description: str = "") -> str:
        """
        打包 Agent 为 .kas-agent 文件
        
        Args:
            agent: Agent 对象
            output_path: 输出路径（可选，默认使用 agent 名称）
            author: 作者名
            author_email: 作者邮箱
            tags: 标签列表
            description: 详细描述
        
        Returns:
            生成的包文件路径
        """
        # 确定输出路径
        if output_path is None:
            output_path = f"{agent.name}-v{agent.version}.kas-agent"
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建临时目录
        temp_build = self.temp_dir / f"pack_{agent.name}_{datetime.now().timestamp()}"
        temp_build.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 写入 agent.yaml
            agent_data = agent.to_dict()
            with open(temp_build / 'agent.yaml', 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(agent_data, f, allow_unicode=True, default_flow_style=False)
            
            # 2. 写入 system_prompt.txt（方便查看）
            with open(temp_build / 'system_prompt.txt', 'w', encoding='utf-8') as f:
                f.write(agent.system_prompt)
            
            # 3. 写入 capabilities.yaml
            caps_data = {
                'capabilities': [
                    {
                        'name': c.name,
                        'type': c.type.value,
                        'description': c.description,
                        'confidence': c.confidence
                    }
                    for c in agent.capabilities
                ]
            }
            with open(temp_build / 'capabilities.yaml', 'w', encoding='utf-8') as f:
                import yaml
                yaml.dump(caps_data, f, allow_unicode=True)
            
            # 4. 写入 manifest.json（元数据和校验）
            manifest = {
                'name': agent.name,
                'version': agent.version,
                'description': description or agent.description,
                'author': author,
                'author_email': author_email,
                'tags': tags or [],
                'created_at': datetime.now().isoformat(),
                'file_count': len(list(temp_build.iterdir())),
            }
            
            # 计算文件哈希
            manifest['files'] = {}
            for f in temp_build.iterdir():
                if f.is_file():
                    manifest['files'][f.name] = self._hash_file(f)
            
            with open(temp_build / 'manifest.json', 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # 5. 创建 ZIP 包（.kas-agent 就是 ZIP）
            zip_path = output.with_suffix('')
            shutil.make_archive(str(zip_path), 'zip', temp_build)
            
            # 重命名为 .kas-agent
            final_path = output.parent / f"{output.stem}.kas-agent"
            if final_path.exists():
                final_path.unlink()
            Path(f"{zip_path}.zip").rename(final_path)
            
            return str(final_path)
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_build, ignore_errors=True)
    
    def unpack(self, package_path: str, install: bool = False) -> Optional[Agent]:
        """
        解包 .kas-agent 文件
        
        Args:
            package_path: 包文件路径
            install: 是否安装到本地 agents 目录
        
        Returns:
            Agent 对象，失败返回 None
        """
        package = Path(package_path)
        if not package.exists():
            return None
        
        # 创建临时解压目录
        temp_extract = self.temp_dir / f"unpack_{package.stem}_{datetime.now().timestamp()}"
        temp_extract.mkdir(parents=True, exist_ok=True)
        
        try:
            # 解压 ZIP
            with zipfile.ZipFile(package, 'r') as zf:
                zf.extractall(temp_extract)
            
            # 验证 manifest
            manifest_file = temp_extract / 'manifest.json'
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                # 校验文件完整性
                for filename, expected_hash in manifest.get('files', {}).items():
                    file_path = temp_extract / filename
                    if file_path.exists():
                        actual_hash = self._hash_file(file_path)
                        if actual_hash != expected_hash:
                            print(f"⚠️  文件校验失败: {filename}")
            
            # 读取 agent.yaml
            agent_file = temp_extract / 'agent.yaml'
            if not agent_file.exists():
                return None
            
            with open(agent_file, 'r', encoding='utf-8') as f:
                import yaml
                agent_data = yaml.safe_load(f)
            
            agent = Agent.from_dict(agent_data)
            
            # 如果指定安装，复制到 agents 目录
            if install:
                self._install_agent(temp_extract, agent.name)
            
            return agent
            
        finally:
            # 清理临时目录
            shutil.rmtree(temp_extract, ignore_errors=True)
    
    def _hash_file(self, file_path: Path) -> str:
        """计算文件 MD5 哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _install_agent(self, source_dir: Path, agent_name: str):
        """安装 Agent 到本地目录"""
        target_dir = Path.home() / '.kas' / 'agents' / agent_name
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file in source_dir.iterdir():
            if file.is_file():
                shutil.copy2(file, target_dir / file.name)


class LocalMarket:
    """
    本地市场 - 管理本地的 Agent 包
    
    用法:
        market = LocalMarket()
        
        # 发布包
        market.publish(package_path)
        
        # 搜索
        results = market.search("python")
        
        # 安装
        market.install("MyAgent")
    """
    
    def __init__(self):
        MARKET_DIR.mkdir(parents=True, exist_ok=True)
        self.packages_dir = MARKET_DIR / 'packages'
        self.packages_dir.mkdir(exist_ok=True)
        self.index: Dict[str, PackageInfo] = {}
        self._load_index()
    
    def _load_index(self):
        """加载市场索引"""
        if MARKET_INDEX.exists():
            with open(MARKET_INDEX, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.index = {
                    k: PackageInfo(**v) for k, v in data.get('packages', {}).items()
                }
    
    def _save_index(self):
        """保存市场索引"""
        with open(MARKET_INDEX, 'w', encoding='utf-8') as f:
            json.dump({
                'packages': {k: asdict(v) for k, v in self.index.items()},
                'last_updated': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False, default=str)
    
    def publish(self, package_path: str, force: bool = False) -> bool:
        """
        发布包到市场
        
        Args:
            package_path: .kas-agent 文件路径
            force: 强制覆盖已存在的包
        
        Returns:
            是否成功
        """
        package = Path(package_path)
        if not package.exists():
            return False
        
        # 解包获取信息
        packer = PackagePacker()
        agent = packer.unpack(package_path)
        if not agent:
            return False
        
        # 检查是否已存在
        existing = self.index.get(agent.name)
        if existing and not force:
            print(f"⚠️  包 {agent.name} v{existing.version} 已存在")
            print(f"   使用 force=True 覆盖，或使用不同名称")
            return False
        
        # 读取 manifest
        temp_dir = packer.temp_dir / f"publish_{agent.name}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(package, 'r') as zf:
                zf.extractall(temp_dir)
            
            manifest = {}
            manifest_file = temp_dir / 'manifest.json'
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            
            # 复制到市场目录
            market_package = self.packages_dir / f"{agent.name}.kas-agent"
            
            # 如果已存在，保留 downloads 计数
            preserved_downloads = existing.downloads if existing else 0
            
            shutil.copy2(package, market_package)
            
            # 更新索引
            now = datetime.now().isoformat()
            package_info = PackageInfo(
                name=agent.name,
                version=agent.version,
                description=manifest.get('description', agent.description),
                author=manifest.get('author', 'Unknown'),
                author_email=manifest.get('author_email', ''),
                tags=manifest.get('tags', []),
                capabilities=[c.name for c in agent.capabilities],
                downloads=preserved_downloads,  # 保留下载计数
                file_size=package.stat().st_size,
                package_hash=packer._hash_file(package),
                created_at=existing.created_at if existing else now,
                updated_at=now  # 更新更新时间
            )
            
            self.index[agent.name] = package_info
            self._save_index()
            
            action = "更新" if existing else "发布"
            print(f"✅ {action}成功: {agent.name} v{agent.version}")
            if existing:
                print(f"   保留下载计数: {preserved_downloads}")
            
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def search(self, query: str = "", tags: List[str] = None) -> List[PackageInfo]:
        """
        搜索包
        
        Args:
            query: 关键词
            tags: 标签过滤
        
        Returns:
            匹配的包列表
        """
        results = []
        query_lower = query.lower()
        
        for name, info in self.index.items():
            # 关键词匹配
            if query and query_lower not in name.lower():
                if query_lower not in info.description.lower():
                    continue
            
            # 标签过滤
            if tags:
                if not any(tag in info.tags for tag in tags):
                    continue
            
            results.append(info)
        
        # 按下载量排序
        results.sort(key=lambda x: x.downloads, reverse=True)
        
        return results
    
    def install(self, package_name: str, version: str = None) -> bool:
        """
        安装包
        
        Args:
            package_name: 包名称
            version: 指定版本（可选）
        
        Returns:
            是否成功
        """
        if package_name not in self.index:
            return False
        
        package_info = self.index[package_name]
        package_path = self.packages_dir / f"{package_name}.kas-agent"
        
        if not package_path.exists():
            return False
        
        # 解包并安装
        packer = PackagePacker()
        agent = packer.unpack(str(package_path), install=True)
        
        if agent:
            # 更新下载计数
            package_info.downloads += 1
            self._save_index()
            return True
        
        return False
    
    def uninstall(self, package_name: str) -> bool:
        """卸载包"""
        agent_dir = Path.home() / '.kas' / 'agents' / package_name
        if agent_dir.exists():
            shutil.rmtree(agent_dir)
            return True
        return False
    
    def get_info(self, package_name: str) -> Optional[PackageInfo]:
        """获取包信息"""
        return self.index.get(package_name)
    
    def list_installed(self) -> List[str]:
        """列出已安装的包"""
        agents_dir = Path.home() / '.kas' / 'agents'
        if not agents_dir.exists():
            return []
        
        return [d.name for d in agents_dir.iterdir() if d.is_dir()]
    
    def get_stats(self) -> MarketStats:
        """获取市场统计"""
        stats = MarketStats(
            total_packages=len(self.index),
            total_downloads=sum(p.downloads for p in self.index.values()),
            last_updated=datetime.now().isoformat()
        )
        
        # 统计热门标签
        tag_counts = {}
        for info in self.index.values():
            for tag in info.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return stats
    
    def format_search_results(self, results: List[PackageInfo]) -> str:
        """格式化搜索结果（用于 CLI 显示）"""
        if not results:
            return "未找到匹配的 Agent"
        
        lines = [
            f"\n📦 找到 {len(results)} 个 Agent",
            "=" * 70
        ]
        
        for i, info in enumerate(results, 1):
            tags = f" [{', '.join(info.tags)}]" if info.tags else ""
            rating = f"⭐ {info.rating:.1f}" if info.rating > 0 else ""
            downloads = f"📥 {info.downloads}"
            
            lines.append(
                f"\n{i}. 📦 {info.name} v{info.version}{tags}"
                f"\n   作者: {info.author} <{info.author_email}>"
                f"\n   描述: {info.description[:60]}..."
                f"\n   能力: {', '.join(info.capabilities[:3])}"
                f"\n   {downloads} {rating}"
            )
        
        return "\n".join(lines)


# 便捷函数
def pack_agent(agent: Agent, output_path: str = None, **kwargs) -> str:
    """打包 Agent"""
    packer = PackagePacker()
    return packer.pack(agent, output_path, **kwargs)


def unpack_agent(package_path: str, install: bool = False) -> Optional[Agent]:
    """解包 Agent"""
    packer = PackagePacker()
    return packer.unpack(package_path, install)


def get_market() -> LocalMarket:
    """获取市场实例"""
    return LocalMarket()


if __name__ == "__main__":
    # 测试
    print("🧪 测试市场系统")
    
    from kas.core.models import Agent, Capability, CapabilityType
    
    # 创建测试 Agent
    agent = Agent(
        name="TestReviewer",
        version="1.0.0",
        description="一个代码审查 Agent",
        capabilities=[
            Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9),
            Capability("Bug Fix", CapabilityType.DEBUGGING, "修复 Bug", 0.8)
        ],
        system_prompt="你是一个代码审查专家...",
        created_from="/test/project"
    )
    
    # 测试打包
    print("\n📦 测试打包...")
    packer = PackagePacker()
    package_path = packer.pack(
        agent,
        author="Yilin.zhang",
        author_email="zhangyilin210@gmail.com",
        tags=["python", "code-review"],
        description="专业的 Python 代码审查 Agent"
    )
    print(f"✅ 打包成功: {package_path}")
    
    # 测试解包
    print("\n📂 测试解包...")
    unpacked = packer.unpack(package_path)
    print(f"✅ 解包成功: {unpacked.name} v{unpacked.version}")
    
    # 测试市场
    print("\n🛒 测试市场...")
    market = LocalMarket()
    
    # 发布
    if market.publish(package_path):
        print(f"✅ 发布到市场: {agent.name}")
    
    # 搜索
    results = market.search("code")
    print(f"✅ 搜索结果: {len(results)} 个")
    
    # 显示
    print(market.format_search_results(results))
    
    # 统计
    stats = market.get_stats()
    print(f"\n📊 市场统计: {stats.total_packages} 个包, {stats.total_downloads} 次下载")
