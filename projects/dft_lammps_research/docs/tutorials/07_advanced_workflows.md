# 07 - 高级工作流定制 | Advanced Workflow Customization

> **学习目标**: 掌握工作流定制、插件开发和自动化部署  
> **Learning Goal**: Master workflow customization, plugin development, and automation

---

## 📋 目录 | Table of Contents

1. [工作流架构 | Workflow Architecture](#1-工作流架构--workflow-architecture)
2. [自定义阶段 | Custom Stages](#2-自定义阶段--custom-stages)
3. [插件开发 | Plugin Development](#3-插件开发--plugin-development)
4. [事件驱动架构 | Event-Driven Architecture](#4-事件驱动架构--event-driven-architecture)
5. [监控与日志 | Monitoring & Logging](#5-监控与日志--monitoring--logging)
6. [最佳实践 | Best Practices](#6-最佳实践--best-practices)
7. [案例研究 | Case Studies](#7-案例研究--case-studies)

---

## 1. 工作流架构 | Workflow Architecture

### 1.1 插件化架构 | Plugin Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    工作流核心引擎                                │
│                  Workflow Core Engine                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              阶段管理器 (Stage Manager)                   │  │
│  │              • 依赖解析                                  │  │
│  │              • 执行调度                                  │  │
│  │              • 状态管理                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐             │
│         ▼                    ▼                    ▼             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │  内置阶段   │      │  插件阶段   │      │  自定义阶段  │     │
│  │  Built-in  │      │  Plugins    │      │  Custom     │     │
│  ├─────────────┤      ├─────────────┤      ├─────────────┤     │
│  │ • Fetch     │      │ • NEB       │      │ • User      │     │
│  │ • DFT       │      │ • Phonon    │      │   Defined   │     │
│  │ • ML Train  │      │ • Defect    │      │ • External  │     │
│  │ • MD        │      │ • Interface │      │   Tools     │     │
│  └─────────────┘      └─────────────┘      └─────────────┘     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              事件总线 (Event Bus)                         │  │
│  │              • 阶段完成事件                              │  │
│  │              • 错误事件                                  │  │
│  │              • 自定义事件                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心接口 | Core Interfaces

```python
"""
工作流核心接口定义
Workflow Core Interface Definitions
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class StageStatus(Enum):
    """阶段状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageContext:
    """阶段执行上下文"""
    workflow_id: str
    stage_name: str
    work_dir: str
    config: Dict[str, Any]
    input_data: Dict[str, Any]
    shared_state: Dict[str, Any]  # 跨阶段共享状态


class WorkflowStage(ABC):
    """
    工作流阶段基类
    Base class for workflow stages
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.status = StageStatus.PENDING
        self.output_data = {}
    
    @abstractmethod
    def validate_inputs(self, context: StageContext) -> bool:
        """
        验证输入数据
        Validate input data
        """
        pass
    
    @abstractmethod
    def execute(self, context: StageContext) -> Dict[str, Any]:
        """
        执行阶段
        Execute the stage
        
        Returns:
            输出数据字典
        """
        pass
    
    @abstractmethod
    def rollback(self, context: StageContext):
        """
        回滚操作
        Rollback operations
        """
        pass
    
    def pre_execute(self, context: StageContext):
        """执行前钩子"""
        self.status = StageStatus.RUNNING
    
    def post_execute(self, context: StageContext, result: Dict[str, Any]):
        """执行后钩子"""
        self.output_data = result
        self.status = StageStatus.COMPLETED
    
    def on_error(self, context: StageContext, error: Exception):
        """错误处理"""
        self.status = StageStatus.FAILED
        # 可在这里发送通知、记录日志等


class WorkflowPlugin(ABC):
    """
    工作流插件基类
    Base class for workflow plugins
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """插件版本"""
        pass
    
    @abstractmethod
    def register_stages(self) -> List[type]:
        """
        注册插件提供的阶段类
        Register stage classes provided by this plugin
        """
        pass
    
    @abstractmethod
    def register_hooks(self, workflow_engine):
        """
        注册钩子函数
        Register hook functions
        """
        pass
```

---

## 2. 自定义阶段 | Custom Stages

### 2.1 NEB计算阶段 | NEB Calculation Stage

```python
"""
NEB (Nudged Elastic Band) 计算阶段
用于计算扩散能垒
"""
import os
import numpy as np
from pathlib import Path
from ase import Atoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.io import read, write

from .workflow_core import WorkflowStage, StageContext, StageStatus


class NEBStage(WorkflowStage):
    """
    NEB计算阶段
    
    计算离子在材料中的迁移路径和能垒
    """
    
    def __init__(self, config: Dict = None):
        default_config = {
            'n_images': 7,           # NEB图像数
            'k_spring': 0.1,         # 弹簧常数 (eV/Å²)
            'fmax': 0.05,            # 力收敛标准 (eV/Å)
            'climb': True,           # 是否使用climbing image
            'method': 'improvedtangent',  # NEB方法
        }
        if config:
            default_config.update(config)
        super().__init__("neb_calculation", default_config)
    
    def validate_inputs(self, context: StageContext) -> bool:
        """验证输入"""
        required = ['initial_structure', 'final_structure', 'calculator']
        for key in required:
            if key not in context.input_data:
                raise ValueError(f"Missing required input: {key}")
        return True
    
    def execute(self, context: StageContext) -> Dict:
        """执行NEB计算"""
        
        # 获取输入
        initial = context.input_data['initial_structure']
        final = context.input_data['final_structure']
        calculator = context.input_data['calculator']
        
        n_images = self.config['n_images']
        work_dir = Path(context.work_dir) / "neb"
        work_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建NEB图像 (线性插值)
        images = [initial]
        for i in range(n_images):
            image = initial.copy()
            # 线性插值位置
            t = (i + 1) / (n_images + 1)
            pos = initial.get_positions() * (1 - t) + final.get_positions() * t
            image.set_positions(pos)
            image.calc = calculator
            images.append(image)
        images.append(final)
        
        # 创建NEB对象
        neb = NEB(
            images,
            k=self.config['k_spring'],
            climb=self.config['climb'],
            method=self.config['method']
        )
        
        # 优化
        optimizer = BFGS(neb, logfile=str(work_dir / 'neb.log'))
        optimizer.run(fmax=self.config['fmax'])
        
        # 提取结果
        energies = [image.get_potential_energy() for image in images]
        barrier = max(energies) - energies[0]
        
        # 保存结果
        results = {
            'barrier': float(barrier),
            'energies': [float(e) for e in energies],
            'images': images,
            'work_dir': str(work_dir)
        }
        
        # 保存图像
        for i, image in enumerate(images):
            write(work_dir / f"image_{i}.vasp", image)
        
        return results
    
    def rollback(self, context: StageContext):
        """回滚"""
        work_dir = Path(context.work_dir) / "neb"
        if work_dir.exists():
            import shutil
            shutil.rmtree(work_dir)


# 使用示例
"""
# 在工作流中使用NEB阶段
config = {
    'n_images': 9,
    'climb': True,
    'fmax': 0.05
}

neb_stage = NEBStage(config)

context = StageContext(
    workflow_id="test",
    stage_name="neb",
    work_dir="./output",
    config=config,
    input_data={
        'initial_structure': initial_atoms,
        'final_structure': final_atoms,
        'calculator': ml_calculator
    },
    shared_state={}
)

result = neb_stage.execute(context)
print(f"Migration barrier: {result['barrier']:.3f} eV")
"""
```

### 2.2 声子计算阶段 | Phonon Calculation Stage

```python
"""
声子计算阶段
Phonon Calculation Stage
"""
from ase.phonons import Phonons
from ase.io import read
import numpy as np


class PhononStage(WorkflowStage):
    """声子计算阶段"""
    
    def __init__(self, config: Dict = None):
        default_config = {
            'supercell': (2, 2, 2),
            'delta': 0.01,           # 位移大小 (Å)
            'npoints': 100,          # DOS点数
            'q_path': 'GXMGRX',      # 高对称路径
        }
        if config:
            default_config.update(config)
        super().__init__("phonon_calculation", default_config)
    
    def execute(self, context: StageContext) -> Dict:
        """执行声子计算"""
        
        atoms = context.input_data['structure']
        calculator = context.input_data['calculator']
        
        # 创建Phonons对象
        ph = Phonons(
            atoms,
            calculator,
            supercell=self.config['supercell'],
            delta=self.config['delta']
        )
        
        # 计算力常数
        ph.run()
        
        # 读取力常数
        ph.read(acoustic=True)
        
        # 计算能带结构
        path = atoms.cell.bandpath(
            path=self.config['q_path'],
            npoints=self.config['npoints']
        )
        
        bs = ph.get_band_structure(path)
        
        # 计算DOS
        dos = ph.get_dos(kpts=(20, 20, 20))
        
        # 检查虚频
        frequencies = bs.get_frequencies()
        imaginary_modes = np.sum(frequencies < 0)
        
        # 热力学性质
        thermo = ph.get_thermochemistry(temperatures=[300, 500, 800])
        
        return {
            'band_structure': bs,
            'dos': dos,
            'imaginary_modes': int(imaginary_modes),
            'is_stable': imaginary_modes == 0,
            'thermochemistry': thermo,
        }
```

---

## 3. 插件开发 | Plugin Development

### 3.1 插件示例 | Plugin Example

```python
"""
电池材料分析插件
Battery Materials Analysis Plugin
"""
from workflow_core import WorkflowPlugin, WorkflowStage
from typing import List


class BatteryAnalysisPlugin(WorkflowPlugin):
    """电池材料分析插件"""
    
    @property
    def name(self) -> str:
        return "battery_analysis"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def register_stages(self) -> List[type]:
        """注册阶段"""
        return [
            VoltageProfileStage,
            IonicConductivityStage,
            VolumeExpansionStage,
        ]
    
    def register_hooks(self, workflow_engine):
        """注册钩子"""
        # 注册事件监听器
        workflow_engine.on_stage_complete(
            "dft_calculation",
            self._analyze_battery_properties
        )
    
    def _analyze_battery_properties(self, context, result):
        """分析电池性质的后处理"""
        # 计算电压、容量等
        pass


class VoltageProfileStage(WorkflowStage):
    """电压曲线计算阶段"""
    
    def __init__(self, config=None):
        super().__init__("voltage_profile", config)
    
    def execute(self, context) -> dict:
        """计算电压曲线"""
        # 实现电压曲线计算
        pass


class IonicConductivityStage(WorkflowStage):
    """离子电导率计算阶段"""
    
    def __init__(self, config=None):
        super().__init__("ionic_conductivity", config)
    
    def execute(self, context) -> dict:
        """计算离子电导率"""
        # 实现电导率计算
        pass


# 插件入口点
entry_points = {
    'workflow.plugins': [
        'battery = battery_plugin:BatteryAnalysisPlugin',
    ]
}
```

### 3.2 插件加载 | Plugin Loading

```python
"""
插件加载器
Plugin Loader
"""
import importlib
import pkgutil
from typing import List, Type


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins = {}
        self.stages = {}
    
    def load_plugin(self, plugin_class: Type[WorkflowPlugin]):
        """加载插件"""
        plugin = plugin_class()
        
        self.plugins[plugin.name] = plugin
        
        # 注册阶段
        for stage_class in plugin.register_stages():
            stage_name = stage_class.__name__
            self.stages[stage_name] = stage_class
        
        print(f"Loaded plugin: {plugin.name} v{plugin.version}")
        print(f"  Registered stages: {list(self.stages.keys())}")
    
    def discover_plugins(self, namespace: str = "workflow.plugins"):
        """自动发现插件"""
        try:
            import importlib.metadata as metadata
        except ImportError:
            import importlib_metadata as metadata
        
        eps = metadata.entry_points()
        if namespace in eps:
            for ep in eps[namespace]:
                plugin_class = ep.load()
                self.load_plugin(plugin_class)
    
    def get_stage(self, name: str) -> Type[WorkflowStage]:
        """获取阶段类"""
        return self.stages.get(name)
```

---

## 4. 事件驱动架构 | Event-Driven Architecture

### 4.1 事件系统 | Event System

```python
"""
事件驱动工作流系统
Event-Driven Workflow System
"""
from typing import Callable, List, Dict
from dataclasses import dataclass
from datetime import datetime
import threading
import queue


@dataclass
class WorkflowEvent:
    """工作流事件"""
    event_type: str
    workflow_id: str
    stage_name: str
    timestamp: datetime
    data: Dict


class EventBus:
    """事件总线"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.event_queue = queue.Queue()
        self._running = False
        self._thread = None
    
    def subscribe(self, event_type: str, handler: Callable):
        """订阅事件"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    def publish(self, event: WorkflowEvent):
        """发布事件"""
        self.event_queue.put(event)
    
    def start(self):
        """启动事件处理线程"""
        self._running = True
        self._thread = threading.Thread(target=self._process_events)
        self._thread.start()
    
    def stop(self):
        """停止事件处理"""
        self._running = False
        self._thread.join()
    
    def _process_events(self):
        """处理事件队列"""
        while self._running:
            try:
                event = self.event_queue.get(timeout=1)
                handlers = self.subscribers.get(event.event_type, [])
                
                for handler in handlers:
                    try:
                        handler(event)
                    except Exception as e:
                        print(f"Handler error: {e}")
                        
            except queue.Empty:
                continue


class EventDrivenWorkflowEngine:
    """事件驱动工作流引擎"""
    
    def __init__(self):
        self.event_bus = EventBus()
        self.stages: Dict[str, WorkflowStage] = {}
        self.stage_order: List[str] = []
        self.dependencies: Dict[str, List[str]] = {}
    
    def add_stage(self, stage: WorkflowStage, depends_on: List[str] = None):
        """添加阶段"""
        self.stages[stage.name] = stage
        self.stage_order.append(stage.name)
        self.dependencies[stage.name] = depends_on or []
    
    def on_stage_complete(self, stage_name: str, handler: Callable):
        """注册阶段完成回调"""
        self.event_bus.subscribe(f"stage_completed_{stage_name}", handler)
    
    def on_stage_fail(self, stage_name: str, handler: Callable):
        """注册阶段失败回调"""
        self.event_bus.subscribe(f"stage_failed_{stage_name}", handler)
    
    def execute(self, workflow_id: str, initial_context: Dict):
        """执行工作流"""
        self.event_bus.start()
        
        shared_state = {}
        completed_stages = set()
        
        while len(completed_stages) < len(self.stages):
            # 找出可执行的stage
            ready_stages = [
                name for name in self.stage_order
                if name not in completed_stages
                and all(dep in completed_stages for dep in self.dependencies[name])
            ]
            
            for stage_name in ready_stages:
                stage = self.stages[stage_name]
                
                context = StageContext(
                    workflow_id=workflow_id,
                    stage_name=stage_name,
                    work_dir=f"./{workflow_id}/{stage_name}",
                    config=stage.config,
                    input_data=initial_context,
                    shared_state=shared_state
                )
                
                try:
                    result = stage.execute(context)
                    shared_state[stage_name] = result
                    completed_stages.add(stage_name)
                    
                    # 发布完成事件
                    self.event_bus.publish(WorkflowEvent(
                        event_type=f"stage_completed_{stage_name}",
                        workflow_id=workflow_id,
                        stage_name=stage_name,
                        timestamp=datetime.now(),
                        data=result
                    ))
                    
                except Exception as e:
                    # 发布失败事件
                    self.event_bus.publish(WorkflowEvent(
                        event_type=f"stage_failed_{stage_name}",
                        workflow_id=workflow_id,
                        stage_name=stage_name,
                        timestamp=datetime.now(),
                        data={'error': str(e)}
                    ))
                    
                    # 处理错误
                    stage.on_error(context, e)
                    
                    if not self.config.get('continue_on_error', False):
                        raise
        
        self.event_bus.stop()
        return shared_state
```

---

## 5. 监控与日志 | Monitoring & Logging

### 5.1 工作流监控器 | Workflow Monitor

```python
"""
工作流监控器
Workflow Monitor
"""
import json
import time
from datetime import datetime
from pathlib import Path
import psutil


class WorkflowMonitor:
    """工作流监控器"""
    
    def __init__(self, workflow_id: str, log_dir: str = "./logs"):
        self.workflow_id = workflow_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.events = []
        self.metrics = {
            'cpu_percent': [],
            'memory_percent': [],
            'disk_io': [],
        }
    
    def log_event(self, stage_name: str, event_type: str, data: dict = None):
        """记录事件"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'workflow_id': self.workflow_id,
            'stage': stage_name,
            'event_type': event_type,
            'data': data or {}
        }
        self.events.append(event)
        
        # 实时写入
        log_file = self.log_dir / f"{self.workflow_id}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def record_metrics(self):
        """记录系统指标"""
        self.metrics['cpu_percent'].append(psutil.cpu_percent())
        self.metrics['memory_percent'].append(psutil.virtual_memory().percent)
        
        # 磁盘IO
        io = psutil.disk_io_counters()
        self.metrics['disk_io'].append({
            'read_bytes': io.read_bytes,
            'write_bytes': io.write_bytes
        })
    
    def generate_report(self) -> dict:
        """生成监控报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'workflow_id': self.workflow_id,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'total_events': len(self.events),
            'stages': self._analyze_stages(),
            'resource_usage': {
                'avg_cpu': sum(self.metrics['cpu_percent']) / len(self.metrics['cpu_percent']),
                'max_memory': max(self.metrics['memory_percent']),
            }
        }
        
        # 保存报告
        report_file = self.log_dir / f"{self.workflow_id}_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _analyze_stages(self) -> dict:
        """分析各阶段性能"""
        stage_stats = {}
        
        for event in self.events:
            stage = event['stage']
            if stage not in stage_stats:
                stage_stats[stage] = {
                    'events': [],
                    'start_time': None,
                    'end_time': None
                }
            
            stage_stats[stage]['events'].append(event)
            
            if event['event_type'] == 'started':
                stage_stats[stage]['start_time'] = event['timestamp']
            elif event['event_type'] == 'completed':
                stage_stats[stage]['end_time'] = event['timestamp']
        
        return stage_stats
```

### 5.2 Web仪表板 | Web Dashboard

```python
"""
工作流Web仪表板
Workflow Web Dashboard
"""
from flask import Flask, jsonify, render_template
import json
from pathlib import Path


class WorkflowDashboard:
    """工作流仪表板"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.app = Flask(__name__)
        self.log_dir = Path(log_dir)
        self._setup_routes()
    
    def _setup_routes(self):
        """设置路由"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html')
        
        @self.app.route('/api/workflows')
        def list_workflows():
            """列出所有工作流"""
            workflows = []
            for log_file in self.log_dir.glob('*.jsonl'):
                workflow_id = log_file.stem
                workflows.append({
                    'id': workflow_id,
                    'log_file': str(log_file)
                })
            return jsonify(workflows)
        
        @self.app.route('/api/workflow/<workflow_id>')
        def get_workflow(workflow_id: str):
            """获取工作流详情"""
            report_file = self.log_dir / f"{workflow_id}_report.json"
            
            if report_file.exists():
                with open(report_file) as f:
                    report = json.load(f)
                return jsonify(report)
            
            return jsonify({'error': 'Workflow not found'}), 404
        
        @self.app.route('/api/workflow/<workflow_id>/events')
        def get_workflow_events(workflow_id: str):
            """获取工作流事件流"""
            log_file = self.log_dir / f"{workflow_id}.jsonl"
            
            events = []
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        events.append(json.loads(line))
            
            return jsonify(events)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """运行仪表板"""
        self.app.run(host=host, port=port, debug=debug)


# HTML模板示例 (templates/dashboard.html)
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Workflow Dashboard</h1>
    
    <div id="workflow-list"></div>
    <div id="workflow-details"></div>
    
    <script>
        // JavaScript代码加载和显示工作流数据
        async function loadWorkflows() {
            const response = await fetch('/api/workflows');
            const workflows = await response.json();
            
            const list = document.getElementById('workflow-list');
            list.innerHTML = '<ul>' + 
                workflows.map(w => `
                    <li><a href="#" onclick="loadWorkflow('${w.id}')"${w.id}</a></li>
                `).join('') + 
                '</ul>';
        }
        
        async function loadWorkflow(id) {
            const response = await fetch(`/api/workflow/${id}`);
            const data = await response.json();
            
            document.getElementById('workflow-details').innerHTML = `
                <h2>${data.workflow_id}</h2>
                <p>Duration: ${data.duration_seconds}s</p>
                <p>Stages: ${Object.keys(data.stages).join(', ')}</p>
            `;
        }
        
        loadWorkflows();
    </script>
</body>
</html>
"""
```

---

## 6. 最佳实践 | Best Practices

### 6.1 设计原则 | Design Principles

1. **单一职责原则**: 每个阶段只做一件事
2. **可配置性**: 所有参数通过配置文件控制
3. **可测试性**: 每个阶段可独立测试
4. **可观察性**: 完善的日志和监控
5. **容错性**: 优雅处理错误和回滚

### 6.2 配置管理 | Configuration Management

```python
"""
配置管理器
Configuration Manager
"""
from pathlib import Path
import yaml
import json
from dataclasses import dataclass


@dataclass
class WorkflowConfig:
    """工作流配置"""
    name: str
    version: str
    stages: List[Dict]
    global_settings: Dict
    resources: Dict


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> WorkflowConfig:
        """加载配置文件"""
        with open(self.config_path) as f:
            if self.config_path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return WorkflowConfig(**data)
    
    def get_stage_config(self, stage_name: str) -> Dict:
        """获取阶段配置"""
        for stage in self.config.stages:
            if stage['name'] == stage_name:
                return stage.get('config', {})
        return {}
    
    def validate(self) -> bool:
        """验证配置有效性"""
        # 检查必需字段
        required = ['name', 'stages']
        for field in required:
            if not getattr(self.config, field):
                raise ValueError(f"Missing required field: {field}")
        
        # 检查阶段依赖
        stage_names = {s['name'] for s in self.config.stages}
        for stage in self.config.stages:
            deps = stage.get('depends_on', [])
            for dep in deps:
                if dep not in stage_names:
                    raise ValueError(f"Unknown dependency: {dep}")
        
        return True
```

---

## 7. 案例研究 | Case Studies

### 7.1 电池材料全工作流 | Battery Materials Workflow

```yaml
# battery_workflow.yaml
name: battery_materials_screening
version: "2.0.0"

global_settings:
  working_dir: "./battery_workflow"
  log_level: INFO
  cleanup_on_success: false

resources:
  default_cores: 32
  default_memory: "128G"
  max_walltime: "48:00:00"

stages:
  - name: structure_fetch
    type: built_in
    config:
      source: materials_project
      query:
        elements: [Li, S]
        max_entries: 100

  - name: dft_relaxation
    type: built_in
    depends_on: [structure_fetch]
    config:
      code: vasp
      functional: PBE
      encut: 520
      kpoints_density: 0.25

  - name: phonon_calculation
    type: plugin
    plugin: battery_analysis
    depends_on: [dft_relaxation]
    config:
      supercell: [2, 2, 2]
      npoints: 100

  - name: voltage_profile
    type: plugin
    plugin: battery_analysis
    depends_on: [dft_relaxation]
    config:
      voltage_range: [0, 5]

  - name: report_generation
    type: custom
    depends_on: [phonon_calculation, voltage_profile]
    config:
      template: battery_report.html
      output_format: pdf
```

---

**结束语**: 恭喜完成所有教程！你现在可以构建自己的材料计算工作流了。

**参考资源**:
- [ASE Documentation](https://wiki.fysik.dtu.dk/ase/)
- [Pymatgen Documentation](https://pymatgen.org/)
- [DeePMD-kit Documentation](https://deepmd.readthedocs.io/)
- [FireWorks Documentation](https://materialsproject.github.io/fireworks/)
