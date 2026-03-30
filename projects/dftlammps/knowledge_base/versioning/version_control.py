"""
Version Control - 计算结果版本控制
===============================
提供类似Git的版本控制功能，用于管理计算结果的变更历史。
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from datetime import datetime
import hashlib
import json
import copy
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """版本状态"""
    DRAFT = "draft"
    COMMITTED = "committed"
    TAGGED = "tagged"
    ARCHIVED = "archived"


@dataclass
class VersionTag:
    """版本标签"""
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    created_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiffResult:
    """差异结果"""
    added: Dict[str, Any] = field(default_factory=dict)
    removed: Dict[str, Any] = field(default_factory=dict)
    modified: Dict[str, Any] = field(default_factory=dict)
    unchanged: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def has_changes(self) -> bool:
        """是否有变更"""
        return bool(self.added or self.removed or self.modified)
    
    def summary(self) -> str:
        """变更摘要"""
        return (
            f"Added: {len(self.added)}, "
            f"Removed: {len(self.removed)}, "
            f"Modified: {len(self.modified)}, "
            f"Unchanged: {len(self.unchanged)}"
        )


@dataclass
class CalculationVersion:
    """计算版本"""
    id: str
    parent_id: Optional[str] = None
    branch: str = "main"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    author: str = ""
    message: str = ""
    status: VersionStatus = VersionStatus.COMMITTED
    
    # 数据存储
    data_hash: str = ""
    data_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 计算信息
    calculation_type: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.data_hash and self.data_snapshot:
            self.data_hash = self._calculate_hash()
    
    def _generate_id(self) -> str:
        """生成版本ID"""
        timestamp = datetime.now().isoformat()
        content = f"{timestamp}:{self.author}:{self.message}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _calculate_hash(self) -> str:
        """计算数据哈希"""
        content = json.dumps(self.data_snapshot, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "branch": self.branch,
            "timestamp": self.timestamp,
            "author": self.author,
            "message": self.message,
            "status": self.status.value,
            "data_hash": self.data_hash,
            "tags": self.tags,
            "metadata": self.metadata,
            "calculation_type": self.calculation_type,
            "parameters": self.parameters,
            "results": self.results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalculationVersion":
        """从字典创建版本"""
        return cls(
            id=data.get("id", ""),
            parent_id=data.get("parent_id"),
            branch=data.get("branch", "main"),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            author=data.get("author", ""),
            message=data.get("message", ""),
            status=VersionStatus(data.get("status", "committed")),
            data_hash=data.get("data_hash", ""),
            data_snapshot=data.get("data_snapshot", {}),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            calculation_type=data.get("calculation_type", ""),
            parameters=data.get("parameters", {}),
            results=data.get("results", {})
        )


class VersionComparator:
    """版本比较器"""
    
    @staticmethod
    def compare(v1: CalculationVersion, v2: CalculationVersion) -> DiffResult:
        """比较两个版本"""
        return VersionComparator.diff_dicts(
            v1.data_snapshot,
            v2.data_snapshot
        )
    
    @staticmethod
    def diff_dicts(
        old: Dict[str, Any],
        new: Dict[str, Any],
        path: str = ""
    ) -> DiffResult:
        """递归比较两个字典"""
        result = DiffResult()
        
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in old:
                result.added[current_path] = new[key]
            elif key not in new:
                result.removed[current_path] = old[key]
            elif type(old[key]) != type(new[key]):
                result.modified[current_path] = {
                    "old": old[key],
                    "new": new[key]
                }
            elif isinstance(old[key], dict):
                nested = VersionComparator.diff_dicts(old[key], new[key], current_path)
                result.added.update(nested.added)
                result.removed.update(nested.removed)
                result.modified.update(nested.modified)
                result.unchanged.extend(nested.unchanged)
            elif isinstance(old[key], (list, tuple)):
                if list(old[key]) != list(new[key]):
                    result.modified[current_path] = {
                        "old": old[key],
                        "new": new[key]
                    }
                else:
                    result.unchanged.append(current_path)
            elif old[key] != new[key]:
                result.modified[current_path] = {
                    "old": old[key],
                    "new": new[key]
                }
            else:
                result.unchanged.append(current_path)
        
        return result
    
    @staticmethod
    def compute_statistics(versions: List[CalculationVersion]) -> Dict[str, Any]:
        """计算版本统计信息"""
        if not versions:
            return {}
        
        # 计算每个参数的变更频率
        param_changes = {}
        for i in range(1, len(versions)):
            diff = VersionComparator.compare(versions[i-1], versions[i])
            for path in diff.modified.keys():
                param_changes[path] = param_changes.get(path, 0) + 1
        
        return {
            "total_versions": len(versions),
            "total_changes": sum(param_changes.values()),
            "most_changed_params": sorted(
                param_changes.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            "branches": list(set(v.branch for v in versions)),
            "authors": list(set(v.author for v in versions)),
            "time_range": {
                "first": min(v.timestamp for v in versions),
                "last": max(v.timestamp for v in versions)
            }
        }


class BranchManager:
    """分支管理器"""
    
    def __init__(self):
        self._branches: Dict[str, Dict[str, Any]] = {
            "main": {
                "head": None,
                "created_at": datetime.now().isoformat(),
                "description": "Main branch"
            }
        }
        self._current_branch: str = "main"
    
    def create_branch(
        self,
        name: str,
        from_branch: Optional[str] = None,
        from_version: Optional[str] = None,
        description: str = ""
    ) -> bool:
        """创建新分支"""
        if name in self._branches:
            logger.warning(f"Branch {name} already exists")
            return False
        
        base_branch = from_branch or self._current_branch
        base_version = from_version or self._branches[base_branch].get("head")
        
        self._branches[name] = {
            "head": base_version,
            "parent_branch": base_branch,
            "parent_version": base_version,
            "created_at": datetime.now().isoformat(),
            "description": description
        }
        
        logger.info(f"Created branch {name} from {base_branch}")
        return True
    
    def delete_branch(self, name: str, force: bool = False) -> bool:
        """删除分支"""
        if name == "main" and not force:
            logger.warning("Cannot delete main branch without force=True")
            return False
        
        if name in self._branches:
            del self._branches[name]
            if self._current_branch == name:
                self._current_branch = "main"
            return True
        return False
    
    def switch_branch(self, name: str) -> bool:
        """切换分支"""
        if name not in self._branches:
            logger.error(f"Branch {name} does not exist")
            return False
        
        self._current_branch = name
        logger.info(f"Switched to branch {name}")
        return True
    
    def merge_branch(
        self,
        source: str,
        target: Optional[str] = None,
        strategy: str = "auto"
    ) -> Optional[str]:
        """
        合并分支
        
        Args:
            source: 源分支
            target: 目标分支 (默认当前分支)
            strategy: 合并策略 (auto, ours, theirs)
            
        Returns:
            合并后的版本ID，失败返回None
        """
        target = target or self._current_branch
        
        if source not in self._branches or target not in self._branches:
            logger.error("Source or target branch does not exist")
            return None
        
        # 这里应该实现实际的合并逻辑
        logger.info(f"Merged branch {source} into {target}")
        return None
    
    def list_branches(self) -> List[str]:
        """列出所有分支"""
        return list(self._branches.keys())
    
    def get_current_branch(self) -> str:
        """获取当前分支"""
        return self._current_branch
    
    def get_branch_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取分支信息"""
        return self._branches.get(name)


class VersionControl:
    """
    版本控制系统
    
    提供类似Git的版本控制功能，用于管理计算结果的历史。
    """
    
    def __init__(self, storage_backend: Optional[Any] = None):
        self.storage = storage_backend
        self._versions: Dict[str, CalculationVersion] = {}
        self._branches = BranchManager()
        self._tags: Dict[str, VersionTag] = {}
        self._comparator = VersionComparator()
    
    def commit(
        self,
        data: Dict[str, Any],
        message: str,
        author: str = "",
        calculation_type: str = "",
        parameters: Optional[Dict[str, Any]] = None,
        results: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CalculationVersion:
        """
        提交新版本
        
        Args:
            data: 完整数据快照
            message: 提交信息
            author: 作者
            calculation_type: 计算类型
            parameters: 计算参数
            results: 计算结果
            metadata: 元数据
            
        Returns:
            新版本对象
        """
        current_branch = self._branches.get_current_branch()
        branch_info = self._branches.get_branch_info(current_branch)
        parent_id = branch_info.get("head") if branch_info else None
        
        version = CalculationVersion(
            parent_id=parent_id,
            branch=current_branch,
            author=author,
            message=message,
            status=VersionStatus.COMMITTED,
            data_snapshot=copy.deepcopy(data),
            calculation_type=calculation_type,
            parameters=parameters or {},
            results=results or {},
            metadata=metadata or {}
        )
        
        # 存储版本
        self._versions[version.id] = version
        
        # 更新分支头
        self._branches._branches[current_branch]["head"] = version.id
        
        # 持久化
        if self.storage:
            self._persist_version(version)
        
        logger.info(f"Committed version {version.id[:8]} on branch {current_branch}")
        return version
    
    def checkout(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        检出指定版本
        
        Args:
            version_id: 版本ID
            
        Returns:
            版本数据快照
        """
        version = self._versions.get(version_id)
        if version:
            return copy.deepcopy(version.data_snapshot)
        return None
    
    def revert(
        self,
        version_id: str,
        message: str = "Revert",
        author: str = ""
    ) -> Optional[CalculationVersion]:
        """
        回退到指定版本
        
        Args:
            version_id: 目标版本ID
            message: 回退信息
            author: 作者
            
        Returns:
            新版本对象
        """
        target_version = self._versions.get(version_id)
        if not target_version:
            logger.error(f"Version {version_id} not found")
            return None
        
        # 创建回退版本
        return self.commit(
            data=target_version.data_snapshot,
            message=f"{message}: {target_version.message}",
            author=author
        )
    
    def tag(
        self,
        version_id: str,
        tag_name: str,
        description: str = "",
        created_by: str = ""
    ) -> bool:
        """
        为版本打标签
        
        Args:
            version_id: 版本ID
            tag_name: 标签名
            description: 标签描述
            created_by: 创建者
            
        Returns:
            是否成功
        """
        if version_id not in self._versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        if tag_name in self._tags:
            logger.error(f"Tag {tag_name} already exists")
            return False
        
        tag = VersionTag(
            name=tag_name,
            description=description,
            created_by=created_by
        )
        
        self._tags[tag_name] = tag
        self._versions[version_id].tags.append(tag_name)
        
        logger.info(f"Tagged version {version_id[:8]} with {tag_name}")
        return True
    
    def get_tag(self, tag_name: str) -> Optional[VersionTag]:
        """获取标签"""
        return self._tags.get(tag_name)
    
    def list_tags(self) -> List[str]:
        """列出所有标签"""
        return list(self._tags.keys())
    
    def diff(
        self,
        version_id1: str,
        version_id2: str
    ) -> Optional[DiffResult]:
        """
        比较两个版本
        
        Args:
            version_id1: 第一个版本ID
            version_id2: 第二个版本ID
            
        Returns:
            差异结果
        """
        v1 = self._versions.get(version_id1)
        v2 = self._versions.get(version_id2)
        
        if not v1 or not v2:
            logger.error("One or both versions not found")
            return None
        
        return self._comparator.compare(v1, v2)
    
    def log(
        self,
        branch: Optional[str] = None,
        limit: int = 10
    ) -> List[CalculationVersion]:
        """
        获取提交历史
        
        Args:
            branch: 分支名 (None则使用当前分支)
            limit: 返回数量限制
            
        Returns:
            版本列表
        """
        branch = branch or self._branches.get_current_branch()
        branch_info = self._branches.get_branch_info(branch)
        
        if not branch_info:
            return []
        
        versions = []
        current_id = branch_info.get("head")
        
        while current_id and len(versions) < limit:
            version = self._versions.get(current_id)
            if version:
                versions.append(version)
                current_id = version.parent_id
            else:
                break
        
        return versions
    
    def get_version(self, version_id: str) -> Optional[CalculationVersion]:
        """获取版本"""
        return self._versions.get(version_id)
    
    def get_version_by_tag(self, tag_name: str) -> Optional[CalculationVersion]:
        """通过标签获取版本"""
        for version in self._versions.values():
            if tag_name in version.tags:
                return version
        return None
    
    def get_latest(self, branch: Optional[str] = None) -> Optional[CalculationVersion]:
        """获取最新版本"""
        branch = branch or self._branches.get_current_branch()
        branch_info = self._branches.get_branch_info(branch)
        
        if branch_info and branch_info.get("head"):
            return self._versions.get(branch_info["head"])
        return None
    
    def get_lineage(self, version_id: str) -> List[str]:
        """
        获取版本血统
        
        Args:
            version_id: 版本ID
            
        Returns:
            祖先版本ID列表
        """
        lineage = []
        current_id = version_id
        
        while current_id:
            version = self._versions.get(current_id)
            if version:
                lineage.append(current_id)
                current_id = version.parent_id
            else:
                break
        
        return lineage
    
    def find_common_ancestor(
        self,
        version_id1: str,
        version_id2: str
    ) -> Optional[str]:
        """
        查找共同祖先
        
        Args:
            version_id1: 第一个版本ID
            version_id2: 第二个版本ID
            
        Returns:
            共同祖先版本ID
        """
        lineage1 = set(self.get_lineage(version_id1))
        lineage2 = self.get_lineage(version_id2)
        
        for v in lineage2:
            if v in lineage1:
                return v
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取版本统计"""
        all_versions = list(self._versions.values())
        return self._comparator.compute_statistics(all_versions)
    
    def export_history(self, output_path: str):
        """导出版本历史到文件"""
        history = {
            "versions": [v.to_dict() for v in self._versions.values()],
            "tags": {k: v.to_dict() for k, v in self._tags.items()},
            "branches": self._branches._branches
        }
        
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Exported version history to {output_path}")
    
    def import_history(self, input_path: str):
        """从文件导入版本历史"""
        with open(input_path, 'r') as f:
            history = json.load(f)
        
        # 导入版本
        for v_data in history.get("versions", []):
            version = CalculationVersion.from_dict(v_data)
            self._versions[version.id] = version
        
        # 导入标签
        for name, t_data in history.get("tags", {}).items():
            self._tags[name] = VersionTag(**t_data)
        
        # 导入分支
        self._branches._branches.update(history.get("branches", {}))
        
        logger.info(f"Imported version history from {input_path}")
    
    def _persist_version(self, version: CalculationVersion):
        """持久化版本到存储"""
        if self.storage:
            # 这里应该调用存储后端的API
            pass
    
    # 分支操作代理
    @property
    def branches(self) -> BranchManager:
        """分支管理器"""
        return self._branches


def create_version_control(storage_backend: Optional[Any] = None) -> VersionControl:
    """
    工厂函数：创建版本控制系统
    
    Args:
        storage_backend: 存储后端
        
    Returns:
        版本控制系统实例
    """
    return VersionControl(storage_backend)
