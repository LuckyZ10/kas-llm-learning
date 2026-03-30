"""
DFT-LAMMPS 全局模块注册中心
============================
自动发现、版本管理、依赖解析

提供统一的模块注册、发现和依赖管理机制，支持插件式架构。

Author: DFT-LAMMPS Team
Phase: 56 - Orchestration System
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generic, List, Optional, Set, Type, TypeVar,
    Union, get_type_hints, get_origin, get_args, Protocol
)
from functools import wraps
import pkgutil


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("module_registry")


class ModuleState(Enum):
    """模块生命周期状态"""
    DISCOVERED = auto()      # 已发现
    REGISTERED = auto()      # 已注册
    INITIALIZING = auto()    # 初始化中
    ACTIVE = auto()          # 活跃可用
    ERROR = auto()           # 错误状态
    DISABLED = auto()        # 已禁用
    DEPRECATED = auto()      # 已弃用


class CapabilityType(Enum):
    """能力类型"""
    CALCULATION = "calculation"      # 计算能力
    SIMULATION = "simulation"        # 模拟能力
    ANALYSIS = "analysis"            # 分析能力
    ML = "ml"                        # 机器学习
    VISUALIZATION = "visualization"  # 可视化
    IO = "io"                        # 输入输出
    ORCHESTRATION = "orchestration"  # 编排能力
    UTILITY = "utility"              # 工具类


@dataclass
class ModuleVersion:
    """语义化版本"""
    major: int = 0
    minor: int = 0
    patch: int = 0
    prerelease: Optional[str] = None
    build: Optional[str] = None
    
    @classmethod
    def from_string(cls, version_str: str) -> ModuleVersion:
        """从字符串解析版本"""
        # 处理 prerelease 和 build metadata
        build_part = None
        if '+' in version_str:
            version_str, build_part = version_str.split('+', 1)
        
        prerelease_part = None
        if '-' in version_str:
            version_str, prerelease_part = version_str.split('-', 1)
        
        parts = version_str.split('.')
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
        
        return cls(major, minor, patch, prerelease_part, build_part)
    
    def __str__(self) -> str:
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: ModuleVersion) -> bool:
        """版本比较"""
        if (self.major, self.minor, self.patch) != (other.major, other.minor, other.patch):
            return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        # 处理 prerelease (没有 prerelease 的版本更大)
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        return False
    
    def __le__(self, other: ModuleVersion) -> bool:
        return self == other or self < other
    
    def __gt__(self, other: ModuleVersion) -> bool:
        return not self <= other
    
    def __ge__(self, other: ModuleVersion) -> bool:
        return not self < other
    
    def is_compatible_with(self, other: ModuleVersion) -> bool:
        """检查兼容性（主版本相同）"""
        return self.major == other.major


@dataclass
class Capability:
    """能力定义"""
    name: str                           # 能力名称
    capability_type: CapabilityType     # 能力类型
    description: str = ""               # 描述
    input_schema: Dict[str, Any] = field(default_factory=dict)   # 输入schema
    output_schema: Dict[str, Any] = field(default_factory=dict)  # 输出schema
    parameters: Dict[str, Any] = field(default_factory=dict)     # 参数定义
    tags: List[str] = field(default_factory=list)                # 标签
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "capability_type": self.capability_type.value,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "parameters": self.parameters,
            "tags": self.tags
        }


@dataclass
class Dependency:
    """依赖定义"""
    module_name: str                    # 依赖模块名
    version_constraint: str = "*"       # 版本约束
    optional: bool = False              # 是否可选
    features: List[str] = field(default_factory=list)  # 需要的特性
    
    def matches_version(self, version: ModuleVersion) -> bool:
        """检查版本是否满足约束"""
        # 简化的版本约束解析
        if self.version_constraint == "*":
            return True
        if self.version_constraint.startswith("^"):
            # ^1.2.3 兼容 1.x.x
            min_ver = ModuleVersion.from_string(self.version_constraint[1:])
            return version.major == min_ver.major and version >= min_ver
        if self.version_constraint.startswith("~"):
            # ~1.2.3 兼容 1.2.x
            min_ver = ModuleVersion.from_string(self.version_constraint[1:])
            return (version.major == min_ver.major and 
                    version.minor == min_ver.minor and 
                    version >= min_ver)
        if self.version_constraint.startswith(">="):
            min_ver = ModuleVersion.from_string(self.version_constraint[2:])
            return version >= min_ver
        if self.version_constraint.startswith(">"):
            min_ver = ModuleVersion.from_string(self.version_constraint[1:])
            return version > min_ver
        try:
            exact_ver = ModuleVersion.from_string(self.version_constraint)
            return version == exact_ver
        except:
            return False


@dataclass
class ModuleMetadata:
    """模块元数据"""
    name: str                           # 模块名称
    version: ModuleVersion              # 版本
    description: str = ""               # 描述
    author: str = ""                    # 作者
    license: str = ""                   # 许可证
    homepage: str = ""                  # 主页
    repository: str = ""                # 代码仓库
    documentation: str = ""             # 文档链接
    keywords: List[str] = field(default_factory=list)     # 关键词
    categories: List[str] = field(default_factory=list)   # 分类
    
    # 运行时信息
    entry_point: Optional[str] = None   # 入口点
    class_name: Optional[str] = None    # 主类名
    module_path: Optional[str] = None   # 模块路径
    file_hash: Optional[str] = None     # 文件哈希
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": str(self.version),
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "homepage": self.homepage,
            "repository": self.repository,
            "documentation": self.documentation,
            "keywords": self.keywords,
            "categories": self.categories,
            "entry_point": self.entry_point,
            "class_name": self.class_name,
            "module_path": self.module_path,
            "file_hash": self.file_hash
        }


@dataclass
class RegisteredModule:
    """已注册模块"""
    metadata: ModuleMetadata            # 元数据
    capabilities: List[Capability] = field(default_factory=list)  # 能力列表
    dependencies: List[Dependency] = field(default_factory=list)  # 依赖列表
    state: ModuleState = ModuleState.DISCOVERED  # 状态
    instance: Any = None                # 模块实例
    error_message: Optional[str] = None # 错误信息
    registration_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "capabilities": [c.to_dict() for c in self.capabilities],
            "dependencies": [asdict(d) for d in self.dependencies],
            "state": self.state.name,
            "registration_time": self.registration_time,
            "last_update": self.last_update,
            "error_message": self.error_message
        }


# 类型变量
T = TypeVar('T')


class ModuleInterface(ABC):
    """模块接口基类"""
    
    @property
    @abstractmethod
    def metadata(self) -> ModuleMetadata:
        """返回模块元数据"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[Capability]:
        """返回模块能力列表"""
        pass
    
    @property
    def dependencies(self) -> List[Dependency]:
        """返回依赖列表（默认空）"""
        return []
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """初始化模块"""
        pass
    
    def shutdown(self) -> None:
        """关闭模块（可选实现）"""
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {"status": "ok", "message": "Module is healthy"}


# 装饰器：注册能力
_CAPABILITY_REGISTRY: Dict[str, List[Capability]] = defaultdict(list)


def capability(
    name: str,
    cap_type: CapabilityType,
    description: str = "",
    input_schema: Optional[Dict[str, Any]] = None,
    output_schema: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
):
    """
    装饰器：将函数标记为模块能力
    
    Example:
        @capability("vasp_energy", CapabilityType.CALCULATION, "计算DFT能量")
        def calculate_energy(structure):
            ...
    """
    def decorator(func: Callable) -> Callable:
        func._is_capability = True
        func._capability_name = name
        func._capability_type = cap_type
        func._capability_description = description
        func._capability_input_schema = input_schema or {}
        func._capability_output_schema = output_schema or {}
        func._capability_tags = tags or []
        
        # 自动推导schema
        try:
            hints = get_type_hints(func)
            if 'return' in hints:
                func._capability_output_schema['type'] = str(hints['return'])
        except:
            pass
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def module(
    name: str,
    version: str,
    description: str = "",
    author: str = "",
    dependencies: Optional[List[Dependency]] = None
):
    """
    装饰器：标记模块类
    
    Example:
        @module("vasp_calculator", "2.0.0", "VASP计算器")
        class VASPCalculator(ModuleInterface):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        cls._is_module = True
        cls._module_name = name
        cls._module_version = ModuleVersion.from_string(version)
        cls._module_description = description
        cls._module_author = author
        cls._module_dependencies = dependencies or []
        return cls
    return decorator


class DependencyResolver:
    """依赖解析器"""
    
    def __init__(self, registry: ModuleRegistry):
        self.registry = registry
        self._cache: Dict[str, List[str]] = {}
    
    def resolve(
        self, 
        module_name: str, 
        version_constraint: str = "*"
    ) -> Optional[RegisteredModule]:
        """解析依赖，返回匹配的模块"""
        candidates = self.registry.find_modules(name=module_name)
        
        # 按版本降序排序
        candidates.sort(key=lambda m: m.metadata.version, reverse=True)
        
        dep = Dependency(module_name, version_constraint)
        for candidate in candidates:
            if dep.matches_version(candidate.metadata.version):
                return candidate
        
        return None
    
    def resolve_all(
        self, 
        dependencies: List[Dependency]
    ) -> Dict[str, RegisteredModule]:
        """解析所有依赖"""
        resolved = {}
        for dep in dependencies:
            module = self.resolve(dep.module_name, dep.version_constraint)
            if module:
                resolved[dep.module_name] = module
            elif not dep.optional:
                raise DependencyError(f"Required dependency not found: {dep.module_name}")
        return resolved
    
    def check_conflicts(self, modules: List[RegisteredModule]) -> List[str]:
        """检查依赖冲突"""
        conflicts = []
        
        # 收集所有依赖要求
        all_deps: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for mod in modules:
            for dep in mod.dependencies:
                all_deps[dep.module_name].append((mod.metadata.name, dep.version_constraint))
        
        # 检查冲突
        for dep_name, requirements in all_deps.items():
            if len(requirements) > 1:
                # 检查是否所有约束都能被同一个版本满足
                versions = []
                for mod_name, constraint in requirements:
                    candidates = self.registry.find_modules(name=dep_name)
                    valid_versions = [
                        c.metadata.version for c in candidates
                        if Dependency(dep_name, constraint).matches_version(c.metadata.version)
                    ]
                    versions.append(set(valid_versions))
                
                # 找交集
                if versions:
                    common = versions[0]
                    for v in versions[1:]:
                        common &= v
                    if not common:
                        conflicts.append(
                            f"Version conflict for {dep_name}: {requirements}"
                        )
        
        return conflicts
    
    def get_dependency_graph(
        self, 
        module_name: str
    ) -> Dict[str, Any]:
        """获取依赖图"""
        visited = set()
        graph = {"name": module_name, "dependencies": []}
        
        def build_graph(name: str, parent: Dict[str, Any]):
            if name in visited:
                return
            visited.add(name)
            
            module = self.registry.get_module(name)
            if not module:
                return
            
            for dep in module.dependencies:
                dep_node = {
                    "name": dep.module_name,
                    "constraint": dep.version_constraint,
                    "optional": dep.optional,
                    "dependencies": []
                }
                parent["dependencies"].append(dep_node)
                build_graph(dep.module_name, dep_node)
        
        build_graph(module_name, graph)
        return graph


class DependencyError(Exception):
    """依赖错误"""
    pass


class ModuleRegistry:
    """
    全局模块注册中心
    
    单例模式实现，提供统一的模块注册、发现和管理。
    
    Example:
        registry = ModuleRegistry.get_instance()
        
        # 注册模块
        registry.register(my_module)
        
        # 发现模块
        modules = registry.find_modules(capability_type=CapabilityType.CALCULATION)
        
        # 获取模块
        vasp = registry.get_module("vasp_calculator")
    """
    
    _instance: Optional[ModuleRegistry] = None
    _lock = threading.RLock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    @classmethod
    def get_instance(cls) -> ModuleRegistry:
        """获取注册中心实例"""
        return cls()
    
    def __init__(self):
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            self._modules: Dict[str, RegisteredModule] = {}
            self._capability_index: Dict[str, List[str]] = defaultdict(list)
            self._type_index: Dict[CapabilityType, List[str]] = defaultdict(list)
            self._tag_index: Dict[str, List[str]] = defaultdict(list)
            self._category_index: Dict[str, List[str]] = defaultdict(list)
            self._resolver = DependencyResolver(self)
            self._observers: List[Callable[[str, ModuleState, Optional[RegisteredModule]], None]] = []
            self._auto_discovery_paths: List[str] = []
            self._initialized = True
    
    def register(
        self, 
        module: Union[ModuleInterface, Type[ModuleInterface], RegisteredModule],
        force: bool = False
    ) -> bool:
        """
        注册模块
        
        Args:
            module: 要注册的模块
            force: 是否强制覆盖已存在模块
        
        Returns:
            注册是否成功
        """
        with self._lock:
            if isinstance(module, RegisteredModule):
                registered = module
            elif isinstance(module, type) and issubclass(module, ModuleInterface):
                # 从类创建
                instance = module()
                registered = self._create_registration(instance)
            elif isinstance(module, ModuleInterface):
                registered = self._create_registration(module)
            else:
                raise TypeError(f"Cannot register module of type {type(module)}")
            
            name = registered.metadata.name
            
            # 检查是否已存在
            if name in self._modules and not force:
                existing = self._modules[name]
                if existing.metadata.version >= registered.metadata.version:
                    logger.warning(f"Module {name} already registered with version {existing.metadata.version}")
                    return False
            
            # 注册
            self._modules[name] = registered
            registered.state = ModuleState.REGISTERED
            
            # 更新索引
            self._update_indices(registered)
            
            logger.info(f"Registered module: {name} v{registered.metadata.version}")
            self._notify_observers(name, ModuleState.REGISTERED, registered)
            
            return True
    
    def unregister(self, module_name: str) -> bool:
        """注销模块"""
        with self._lock:
            if module_name not in self._modules:
                return False
            
            module = self._modules.pop(module_name)
            self._remove_indices(module)
            
            logger.info(f"Unregistered module: {module_name}")
            self._notify_observers(module_name, ModuleState.DISABLED, None)
            return True
    
    def get_module(self, name: str) -> Optional[RegisteredModule]:
        """获取模块"""
        return self._modules.get(name)
    
    def find_modules(
        self,
        name: Optional[str] = None,
        capability_type: Optional[CapabilityType] = None,
        capability_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        state: Optional[ModuleState] = None,
        keyword: Optional[str] = None
    ) -> List[RegisteredModule]:
        """
        搜索模块
        
        支持多条件组合查询
        """
        candidates = list(self._modules.values())
        
        if name:
            candidates = [m for m in candidates if name.lower() in m.metadata.name.lower()]
        
        if capability_type:
            names = set(self._type_index.get(capability_type, []))
            candidates = [m for m in candidates if m.metadata.name in names]
        
        if capability_name:
            names = set(self._capability_index.get(capability_name, []))
            candidates = [m for m in candidates if m.metadata.name in names]
        
        if tags:
            tag_names = set()
            for tag in tags:
                tag_names.update(self._tag_index.get(tag, []))
            candidates = [m for m in candidates if m.metadata.name in tag_names]
        
        if categories:
            cat_names = set()
            for cat in categories:
                cat_names.update(self._category_index.get(cat, []))
            candidates = [m for m in candidates if m.metadata.name in cat_names]
        
        if state:
            candidates = [m for m in candidates if m.state == state]
        
        if keyword:
            keyword_lower = keyword.lower()
            candidates = [
                m for m in candidates
                if (keyword_lower in m.metadata.name.lower() or
                    keyword_lower in m.metadata.description.lower() or
                    any(keyword_lower in k.lower() for k in m.metadata.keywords))
            ]
        
        return candidates
    
    def get_all_modules(self) -> List[RegisteredModule]:
        """获取所有模块"""
        return list(self._modules.values())
    
    def get_capabilities(self, module_name: str) -> List[Capability]:
        """获取模块的能力列表"""
        module = self._modules.get(module_name)
        return module.capabilities if module else []
    
    def initialize_module(
        self, 
        name: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        初始化模块
        
        自动解析并初始化依赖
        """
        module = self._modules.get(name)
        if not module:
            logger.error(f"Module not found: {name}")
            return False
        
        if module.state == ModuleState.ACTIVE:
            return True
        
        try:
            module.state = ModuleState.INITIALIZING
            
            # 解析依赖
            deps = self._resolver.resolve_all(module.dependencies)
            
            # 初始化依赖
            for dep_name, dep_module in deps.items():
                if dep_module.state != ModuleState.ACTIVE:
                    if not self.initialize_module(dep_name, context):
                        raise DependencyError(f"Failed to initialize dependency: {dep_name}")
            
            # 初始化模块
            if module.instance:
                ctx = context or {}
                ctx['dependencies'] = deps
                success = module.instance.initialize(ctx)
                
                if success:
                    module.state = ModuleState.ACTIVE
                    logger.info(f"Module initialized: {name}")
                    self._notify_observers(name, ModuleState.ACTIVE, module)
                else:
                    module.state = ModuleState.ERROR
                    module.error_message = "Initialization returned False"
                    
            return module.state == ModuleState.ACTIVE
            
        except Exception as e:
            module.state = ModuleState.ERROR
            module.error_message = str(e)
            logger.error(f"Failed to initialize module {name}: {e}")
            return False
    
    def initialize_all(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """初始化所有模块"""
        results = {}
        for name in self._modules:
            results[name] = self.initialize_module(name, context)
        return results
    
    def shutdown_module(self, name: str) -> bool:
        """关闭模块"""
        module = self._modules.get(name)
        if not module or not module.instance:
            return False
        
        try:
            if hasattr(module.instance, 'shutdown'):
                module.instance.shutdown()
            module.state = ModuleState.DISABLED
            logger.info(f"Module shutdown: {name}")
            return True
        except Exception as e:
            logger.error(f"Error shutting down module {name}: {e}")
            return False
    
    def shutdown_all(self) -> None:
        """关闭所有模块"""
        for name in list(self._modules.keys()):
            self.shutdown_module(name)
    
    def auto_discover(
        self, 
        paths: Optional[List[str]] = None,
        package_prefix: str = "dftlammps"
    ) -> List[RegisteredModule]:
        """
        自动发现模块
        
        扫描指定路径，自动注册带有 @module 装饰器的类
        """
        discovered = []
        
        search_paths = paths or self._auto_discovery_paths
        if not search_paths:
            # 默认扫描 dftlammps 包
            try:
                import dftlammps
                search_paths = [os.path.dirname(dftlammps.__file__)]
            except ImportError:
                search_paths = ["."]
        
        for path in search_paths:
            discovered.extend(self._scan_path(path, package_prefix))
        
        return discovered
    
    def add_discovery_path(self, path: str) -> None:
        """添加自动发现路径"""
        self._auto_discovery_paths.append(path)
    
    def add_observer(
        self, 
        callback: Callable[[str, ModuleState, Optional[RegisteredModule]], None]
    ) -> None:
        """添加状态观察者"""
        self._observers.append(callback)
    
    def get_resolver(self) -> DependencyResolver:
        """获取依赖解析器"""
        return self._resolver
    
    def export_registry(self, format: str = "json") -> str:
        """导出注册表"""
        data = {
            "modules": {name: mod.to_dict() for name, mod in self._modules.items()},
            "export_time": time.time()
        }
        
        if format == "json":
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_registry(self, data: str, format: str = "json") -> None:
        """导入注册表（仅元数据）"""
        if format == "json":
            parsed = json.loads(data)
            # 这里只导入元数据，不创建实例
            logger.info(f"Imported registry with {len(parsed.get('modules', {}))} modules")
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_modules": len(self._modules),
            "active_modules": len([m for m in self._modules.values() if m.state == ModuleState.ACTIVE]),
            "by_state": {
                state.name: len([m for m in self._modules.values() if m.state == state])
                for state in ModuleState
            },
            "by_type": {
                cap_type.name: len(names)
                for cap_type, names in self._type_index.items()
            },
            "total_capabilities": len(self._capability_index),
            "total_tags": len(self._tag_index)
        }
    
    def _create_registration(
        self, 
        instance: ModuleInterface
    ) -> RegisteredModule:
        """从实例创建注册信息"""
        metadata = instance.metadata
        capabilities = instance.capabilities
        dependencies = instance.dependencies
        
        return RegisteredModule(
            metadata=metadata,
            capabilities=capabilities,
            dependencies=dependencies,
            instance=instance
        )
    
    def _update_indices(self, module: RegisteredModule) -> None:
        """更新索引"""
        name = module.metadata.name
        
        # 能力索引
        for cap in module.capabilities:
            self._capability_index[cap.name].append(name)
            self._type_index[cap.capability_type].append(name)
            for tag in cap.tags:
                self._tag_index[tag].append(name)
        
        # 关键词索引
        for keyword in module.metadata.keywords:
            self._tag_index[keyword].append(name)
        
        # 分类索引
        for category in module.metadata.categories:
            self._category_index[category].append(name)
    
    def _remove_indices(self, module: RegisteredModule) -> None:
        """移除索引"""
        name = module.metadata.name
        
        for cap in module.capabilities:
            if name in self._capability_index.get(cap.name, []):
                self._capability_index[cap.name].remove(name)
            if name in self._type_index.get(cap.capability_type, []):
                self._type_index[cap.capability_type].remove(name)
        
        for keyword in module.metadata.keywords:
            if name in self._tag_index.get(keyword, []):
                self._tag_index[keyword].remove(name)
        
        for category in module.metadata.categories:
            if name in self._category_index.get(category, []):
                self._category_index[category].remove(name)
    
    def _scan_path(
        self, 
        path: str, 
        package_prefix: str
    ) -> List[RegisteredModule]:
        """扫描路径发现模块"""
        discovered = []
        
        try:
            for importer, modname, ispkg in pkgutil.walk_packages([path], package_prefix + "."):
                try:
                    module = importlib.import_module(modname)
                    
                    # 查找带有 @module 装饰器的类
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, '_is_module') and obj._is_module:
                            try:
                                instance = obj()
                                if self.register(instance):
                                    discovered.append(self._modules[instance.metadata.name])
                            except Exception as e:
                                logger.warning(f"Failed to instantiate module {name}: {e}")
                    
                except Exception as e:
                    logger.debug(f"Failed to import {modname}: {e}")
                    
        except Exception as e:
            logger.error(f"Error scanning path {path}: {e}")
        
        return discovered
    
    def _notify_observers(
        self, 
        name: str, 
        state: ModuleState,
        module: Optional[RegisteredModule]
    ) -> None:
        """通知观察者"""
        for observer in self._observers:
            try:
                observer(name, state, module)
            except Exception as e:
                logger.error(f"Observer error: {e}")


# 便捷的模块注册装饰器（使用全局注册中心）
def register_module(
    name: Optional[str] = None,
    version: Optional[str] = None,
    description: str = "",
    auto_init: bool = False
):
    """
    便捷装饰器：自动注册模块到全局注册中心
    
    Example:
        @register_module("my_calculator", "1.0.0")
        class MyCalculator(ModuleInterface):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        registry = ModuleRegistry.get_instance()
        
        # 设置模块元数据
        if name:
            cls._module_name = name
        if version:
            cls._module_version = ModuleVersion.from_string(version)
        cls._module_description = description
        
        # 创建实例并注册
        try:
            instance = cls()
            registry.register(instance)
            
            if auto_init:
                registry.initialize_module(instance.metadata.name)
                
        except Exception as e:
            logger.error(f"Failed to auto-register module {cls.__name__}: {e}")
        
        return cls
    return decorator


# 全局访问函数
def get_registry() -> ModuleRegistry:
    """获取全局注册中心实例"""
    return ModuleRegistry.get_instance()


def find_modules(**kwargs) -> List[RegisteredModule]:
    """便捷函数：搜索模块"""
    return get_registry().find_modules(**kwargs)


def get_module(name: str) -> Optional[RegisteredModule]:
    """便捷函数：获取模块"""
    return get_registry().get_module(name)


def initialize_module(name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """便捷函数：初始化模块"""
    return get_registry().initialize_module(name, context)