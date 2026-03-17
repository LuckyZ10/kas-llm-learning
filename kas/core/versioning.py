"""
KAS Core - Agent Version Management
Agent 版本管理系统 - 像游戏存档一样管理 Agent

功能：
- 自动保存每次进化的版本
- 版本对比（diff）
- 回滚到任意版本
- A/B 测试支持
- 版本标签和注释
"""

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import yaml

from .models import Agent


# 版本存储目录
VERSIONS_DIR = Path.home() / '.kas' / 'versions'


@dataclass
class VersionInfo:
    """版本信息"""
    version_id: str           # 版本 ID (如 v1.0, v2.0, 或 hash)
    agent_name: str
    timestamp: float
    description: str          # 版本描述
    parent_version: Optional[str] = None  # 父版本（用于追溯历史）
    generation: int = 0       # 代数
    quality_score: float = 0.0  # 质量评分
    tags: List[str] = None    # 标签（如 stable, experimental）
    changes: List[str] = None # 变更列表
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.changes is None:
            self.changes = []
    
    @property
    def formatted_time(self) -> str:
        """格式化时间"""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')


class VersionManager:
    """
    版本管理器 - 像游戏存档一样简单
    
    用法：
        vm = VersionManager("MyAgent")
        
        # 保存当前版本
        vm.save_version(agent, "初始版本")
        
        # 查看历史
        versions = vm.list_versions()
        
        # 对比两个版本
        diff = vm.compare_versions("v1.0", "v2.0")
        
        # 回滚
        agent = vm.rollback("v1.0")
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent_versions_dir = VERSIONS_DIR / agent_name
        self.agent_versions_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.agent_versions_dir / 'versions.json'
        self.versions: List[VersionInfo] = []
        self._load_metadata()
    
    def _load_metadata(self):
        """加载版本元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.versions = [VersionInfo(**v) for v in data.get('versions', [])]
    
    def _save_metadata(self):
        """保存版本元数据"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump({
                'agent_name': self.agent_name,
                'versions': [asdict(v) for v in self.versions]
            }, f, indent=2, default=str)
    
    def _generate_version_id(self) -> str:
        """生成版本 ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"v{len(self.versions) + 1}_{timestamp}"
    
    def _get_version_dir(self, version_id: str) -> Path:
        """获取版本目录"""
        return self.agent_versions_dir / version_id
    
    def save_version(self, agent: Agent, description: str = "",
                     tags: List[str] = None, changes: List[str] = None,
                     quality_score: float = 0.0) -> str:
        """
        保存 Agent 版本
        
        像游戏存档一样，保存当前 Agent 的完整状态
        
        Args:
            agent: Agent 对象
            description: 版本描述
            tags: 标签（如 ["stable", "production"]）
            changes: 变更列表
            quality_score: 质量评分
        
        Returns:
            版本 ID
        """
        version_id = self._generate_version_id()
        version_dir = self._get_version_dir(version_id)
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 Agent 数据
        agent_data = agent.to_dict()
        with open(version_dir / 'agent.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(agent_data, f, allow_unicode=True, default_flow_style=False)
        
        # 保存 system_prompt 单独文件（方便查看）
        with open(version_dir / 'system_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(agent.system_prompt)
        
        # 创建版本信息
        parent = self.versions[-1].version_id if self.versions else None
        
        version_info = VersionInfo(
            version_id=version_id,
            agent_name=self.agent_name,
            timestamp=datetime.now().timestamp(),
            description=description,
            parent_version=parent,
            generation=len(self.versions),
            quality_score=quality_score,
            tags=tags or [],
            changes=changes or []
        )
        
        self.versions.append(version_info)
        self._save_metadata()
        
        return version_id
    
    def load_version(self, version_id: str) -> Optional[Agent]:
        """
        加载指定版本的 Agent
        
        像游戏读档一样，恢复到之前的版本
        """
        version_dir = self._get_version_dir(version_id)
        
        if not version_dir.exists():
            return None
        
        agent_file = version_dir / 'agent.yaml'
        if not agent_file.exists():
            return None
        
        with open(agent_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return Agent.from_dict(data)
    
    def rollback(self, version_id: str) -> Optional[Agent]:
        """
        回滚到指定版本
        
        进化错了？一键回滚！
        """
        agent = self.load_version(version_id)
        if agent:
            # 同时保存当前状态（以防后悔）
            self.save_version(
                agent,
                description=f"从 {version_id} 回滚",
                tags=["rollback"]
            )
        return agent
    
    def list_versions(self, limit: int = None) -> List[VersionInfo]:
        """
        列出所有版本
        
        像游戏存档列表一样查看历史
        """
        versions = sorted(self.versions, key=lambda v: v.timestamp, reverse=True)
        if limit:
            versions = versions[:limit]
        return versions
    
    def get_version(self, version_id: str) -> Optional[VersionInfo]:
        """获取版本信息"""
        for v in self.versions:
            if v.version_id == version_id:
                return v
        return None
    
    def compare_versions(self, version_a: str, version_b: str) -> Dict:
        """
        对比两个版本
        
        看看进化带来了什么变化
        """
        agent_a = self.load_version(version_a)
        agent_b = self.load_version(version_b)
        
        if not agent_a or not agent_b:
            return {"error": "版本不存在"}
        
        # 对比 capabilities
        caps_a = {c.name: c for c in agent_a.capabilities}
        caps_b = {c.name: c for c in agent_b.capabilities}
        
        added_caps = set(caps_b.keys()) - set(caps_a.keys())
        removed_caps = set(caps_a.keys()) - set(caps_b.keys())
        
        # 对比 prompt
        prompt_diff = self._diff_text(agent_a.system_prompt, agent_b.system_prompt)
        
        # 对比配置
        config_changes = {}
        for key in set(agent_a.model_config.keys()) | set(agent_b.model_config.keys()):
            val_a = agent_a.model_config.get(key)
            val_b = agent_b.model_config.get(key)
            if val_a != val_b:
                config_changes[key] = {"from": val_a, "to": val_b}
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "capabilities": {
                "added": list(added_caps),
                "removed": list(removed_caps)
            },
            "prompt_changes": prompt_diff,
            "config_changes": config_changes
        }
    
    def _diff_text(self, text_a: str, text_b: str) -> List[Dict]:
        """简单的文本 diff"""
        lines_a = text_a.split('\n')
        lines_b = text_b.split('\n')
        
        changes = []
        max_lines = max(len(lines_a), len(lines_b))
        
        for i in range(max_lines):
            line_a = lines_a[i] if i < len(lines_a) else None
            line_b = lines_b[i] if i < len(lines_b) else None
            
            if line_a != line_b:
                changes.append({
                    "line": i + 1,
                    "old": line_a,
                    "new": line_b
                })
        
        return changes
    
    def tag_version(self, version_id: str, tag: str):
        """
        给版本打标签
        
        比如标记 "stable", "production", "experimental"
        """
        version = self.get_version(version_id)
        if version:
            if tag not in version.tags:
                version.tags.append(tag)
                self._save_metadata()
            return True
        return False
    
    def untag_version(self, version_id: str, tag: str):
        """移除标签"""
        version = self.get_version(version_id)
        if version and tag in version.tags:
            version.tags.remove(tag)
            self._save_metadata()
            return True
        return False
    
    def get_best_version(self) -> Optional[VersionInfo]:
        """
        获取最佳版本
        
        根据质量评分选择最好的版本
        """
        if not self.versions:
            return None
        
        # 按质量评分排序
        return max(self.versions, key=lambda v: v.quality_score)
    
    def get_version_tree(self) -> Dict:
        """
        获取版本树
        
        可视化版本演进历史
        """
        tree = {
            "agent_name": self.agent_name,
            "total_versions": len(self.versions),
            "tree": []
        }
        
        for v in self.versions:
            node = {
                "version_id": v.version_id,
                "generation": v.generation,
                "parent": v.parent_version,
                "description": v.description,
                "quality_score": v.quality_score,
                "tags": v.tags
            }
            tree["tree"].append(node)
        
        return tree
    
    def delete_version(self, version_id: str) -> bool:
        """
        删除版本
        
        清理不需要的旧版本
        """
        version = self.get_version(version_id)
        if not version:
            return False
        
        # 删除文件
        version_dir = self._get_version_dir(version_id)
        if version_dir.exists():
            shutil.rmtree(version_dir)
        
        # 从元数据移除
        self.versions = [v for v in self.versions if v.version_id != version_id]
        self._save_metadata()
        
        return True
    
    def export_version(self, version_id: str, output_path: str) -> bool:
        """
        导出版本为 .kas-agent 包
        """
        agent = self.load_version(version_id)
        if not agent:
            return False
        
        # TODO: 使用现有的打包功能
        # 这里简化处理，直接复制文件
        version_dir = self._get_version_dir(version_id)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建压缩包
        shutil.make_archive(
            base_name=str(output.with_suffix('')),
            format='zip',
            root_dir=version_dir
        )
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        获取版本统计
        """
        if not self.versions:
            return {
                "total_versions": 0,
                "average_quality": 0,
                "best_version": None
            }
        
        avg_quality = sum(v.quality_score for v in self.versions) / len(self.versions)
        best = self.get_best_version()
        
        return {
            "total_versions": len(self.versions),
            "average_quality": round(avg_quality, 2),
            "best_version": best.version_id if best else None,
            "best_quality": best.quality_score if best else 0,
            "first_created": self.versions[0].formatted_time,
            "last_updated": self.versions[-1].formatted_time
        }
    
    def format_version_list(self, versions: List[VersionInfo] = None) -> str:
        """
        格式化版本列表（用于 CLI 显示）
        """
        if versions is None:
            versions = self.list_versions()
        
        if not versions:
            return "暂无版本历史"
        
        lines = [
            f"\n📚 {self.agent_name} 版本历史",
            "=" * 60
        ]
        
        for v in versions:
            tags = f" [{', '.join(v.tags)}]" if v.tags else ""
            quality = f"⭐ {v.quality_score:.1f}" if v.quality_score > 0 else ""
            lines.append(
                f"\n🏷️  {v.version_id}{tags}"
                f"\n   时间: {v.formatted_time}"
                f"\n   代数: Gen{v.generation} {quality}"
                f"\n   描述: {v.description}"
            )
            if v.changes:
                lines.append(f"   变更: {', '.join(v.changes[:3])}")
        
        lines.append(f"\n总计: {len(versions)} 个版本")
        return "\n".join(lines)


# 便捷函数
def get_version_manager(agent_name: str) -> VersionManager:
    """获取 Agent 的版本管理器"""
    return VersionManager(agent_name)


def auto_save_on_evolve(agent: Agent, evolution_result: Dict):
    """
    进化完成后自动保存版本
    
    在 LLMEnhancedLearningEngine 中调用
    """
    vm = VersionManager(agent.name)
    
    plan = evolution_result.get('evolution_plan', {})
    analysis = evolution_result.get('analysis', {})
    
    version_id = vm.save_version(
        agent=agent,
        description=f"自动进化 - Gen{evolution_result.get('generation', 0)}",
        tags=["auto", "evolution"],
        changes=[c.get('action', 'unknown') for c in plan.get('capability_changes', [])],
        quality_score=evolution_result.get('quality_score', 0)
    )
    
    return version_id


if __name__ == "__main__":
    # 测试
    print("🧪 测试版本管理系统")
    
    vm = VersionManager("TestAgent")
    
    # 模拟保存版本
    from kas.core.models import Agent, Capability, CapabilityType
    
    agent = Agent(
        name="TestAgent",
        capabilities=[
            Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9),
            Capability("Docs", CapabilityType.DOCUMENTATION, "写文档", 0.8)
        ],
        system_prompt="你是一个测试 Agent"
    )
    
    v1 = vm.save_version(agent, "初始版本", tags=["stable"], quality_score=75.0)
    print(f"✅ 保存版本: {v1}")
    
    # 修改后保存新版本
    agent.system_prompt = "你是一个更聪明的测试 Agent"
    agent.capabilities.append(Capability("Test", CapabilityType.TEST_GENERATION, "测试", 0.7))
    v2 = vm.save_version(agent, "增加了测试能力", quality_score=85.0)
    print(f"✅ 保存版本: {v2}")
    
    # 列出版本
    print(vm.format_version_list())
    
    # 对比
    diff = vm.compare_versions(v1, v2)
    print(f"\n📊 版本对比:")
    print(f"   新增能力: {diff['capabilities']['added']}")
