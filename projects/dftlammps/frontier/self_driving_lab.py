"""
self_driving_lab.py
自动驾驶实验室接口

连接自动化实验平台, 实现材料合成-表征-优化的闭环。
支持粉末合成、溶液合成、气相沉积等多种合成方法。

References:
- Roch et al. (2020) "ChemOS: An orchestration software to democratize autonomous discovery"
- MacLeod et al. (2020) "Self-driving laboratory for accelerated discovery of thin-film materials"
- 2024进展: 自动驾驶实验室用于电池材料发现
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import numpy as np
from collections import defaultdict
import time


@dataclass
class SynthesisParameters:
    """合成参数"""
    temperature: float = 25.0           # °C
    time: float = 1.0                   # hours
    precursors: Dict[str, float] = field(default_factory=dict)  # 前驱体和摩尔比
    atmosphere: str = "air"             # 气氛
    pressure: float = 1.0               # atm
    heating_rate: float = 5.0           # °C/min
    cooling_rate: float = 5.0           # °C/min
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CharacterizationData:
    """表征数据"""
    xrd_pattern: Optional[np.ndarray] = None
    sem_image: Optional[np.ndarray] = None
    composition: Optional[Dict[str, float]] = None
    particle_size: Optional[float] = None
    surface_area: Optional[float] = None
    band_gap: Optional[float] = None
    conductivity: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    synthesis_params: SynthesisParameters
    characterization: CharacterizationData
    target_properties: Dict[str, float]
    success: bool = True
    notes: str = ""


class SynthesisPlanner:
    """
    合成规划器
    
    基于目标材料推荐合成路线
    """
    
    def __init__(self):
        # 合成方法数据库
        self.synthesis_methods = {
            'solid_state': {
                'name': 'Solid State Reaction',
                'temp_range': (800, 1200),
                'time_range': (12, 48),
                'atmosphere': ['air', 'Ar', 'N2', 'O2'],
                'suitable_for': ['oxides', 'ceramics']
            },
            'sol_gel': {
                'name': 'Sol-Gel Synthesis',
                'temp_range': (300, 800),
                'time_range': (2, 24),
                'atmosphere': ['air'],
                'suitable_for': ['oxides', 'nanoparticles']
            },
            'hydrothermal': {
                'name': 'Hydrothermal Synthesis',
                'temp_range': (100, 250),
                'time_range': (6, 72),
                'atmosphere': ['autogenous'],
                'suitable_for': ['nanocrystals', 'zeolites']
            },
            'co_precipitation': {
                'name': 'Co-precipitation',
                'temp_range': (25, 100),
                'time_range': (1, 6),
                'atmosphere': ['air', 'N2'],
                'suitable_for': ['mixed oxides', 'catalysts']
            },
            'mechanochemical': {
                'name': 'Mechanochemical Synthesis',
                'temp_range': (25, 100),
                'time_range': (0.5, 4),
                'atmosphere': ['air', 'Ar'],
                'suitable_for': ['MOFs', 'polymorphs']
            }
        }
        
        # 前驱体数据库
        self.precursor_database = {
            'Li': ['Li2CO3', 'LiOH', 'LiNO3', 'CH3COOLi'],
            'Na': ['Na2CO3', 'NaOH', 'NaNO3'],
            'Fe': ['Fe2O3', 'Fe3O4', 'Fe(NO3)3', 'FeC2O4'],
            'Co': ['CoO', 'Co3O4', 'Co(NO3)2', 'Co(CH3COO)2'],
            'Ni': ['NiO', 'Ni(NO3)2', 'Ni(CH3COO)2'],
            'Mn': ['MnO2', 'Mn2O3', 'Mn(NO3)2', 'Mn(CH3COO)2'],
            'O': ['O2'],  # 通常来自气氛
        }
    
    def plan_synthesis(
        self,
        target_formula: str,
        target_properties: Optional[Dict[str, float]] = None,
        constraints: Optional[Dict] = None
    ) -> List[SynthesisParameters]:
        """
        规划合成路线
        
        Returns:
            候选合成参数列表
        """
        candidates = []
        
        # 解析化学式
        composition = self._parse_formula(target_formula)
        
        # 选择合成方法
        suitable_methods = self._select_synthesis_methods(composition)
        
        for method_name in suitable_methods:
            method = self.synthesis_methods[method_name]
            
            # 选择前驱体
            precursors = self._select_precursors(composition)
            
            # 生成参数组合
            for temp in np.linspace(method['temp_range'][0], method['temp_range'][1], 3):
                for time in np.linspace(method['time_range'][0], method['time_range'][1], 3):
                    params = SynthesisParameters(
                        temperature=float(temp),
                        time=float(time),
                        precursors=precursors,
                        atmosphere=method['atmosphere'][0],
                        heating_rate=5.0,
                        cooling_rate=5.0
                    )
                    candidates.append(params)
        
        return candidates[:10]  # 限制候选数量
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """解析化学式"""
        import re
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        composition = {}
        for elem, count in matches:
            composition[elem] = float(count) if count else 1.0
        
        return composition
    
    def _select_synthesis_methods(self, composition: Dict[str, float]) -> List[str]:
        """选择适合的合成方法"""
        # 启发式规则
        elements = set(composition.keys())
        
        methods = []
        
        # 氧化物优先选择固相或溶胶-凝胶
        if 'O' in elements or any(e in ['Fe', 'Co', 'Ni', 'Mn', 'Ti'] for e in elements):
            methods.extend(['solid_state', 'sol_gel', 'co_precipitation'])
        
        # 钠/锂离子材料
        if 'Li' in elements or 'Na' in elements:
            methods.append('solid_state')
        
        # 纳米材料
        if len(composition) <= 3:
            methods.extend(['hydrothermal', 'sol_gel'])
        
        return list(set(methods)) if methods else ['solid_state']
    
    def _select_precursors(self, composition: Dict[str, float]) -> Dict[str, float]:
        """选择前驱体"""
        precursors = {}
        
        for elem, stoich in composition.items():
            if elem in self.precursor_database:
                # 选择第一个可用前驱体
                precursors[self.precursor_database[elem][0]] = stoich
        
        return precursors


class SelfDrivingLab:
    """
    自动驾驶实验室主控制器
    
    整合合成、表征和优化
    """
    
    def __init__(
        self,
        synthesis_planner: Optional[SynthesisPlanner] = None,
        optimizer: Optional[Any] = None
    ):
        self.planner = synthesis_planner or SynthesisPlanner()
        self.optimizer = optimizer
        
        # 实验历史
        self.experiment_history: List[ExperimentResult] = []
        
        # 目标
        self.target_properties: Dict[str, Tuple[float, float]] = {}
        
        # 状态
        self.is_running = False
        
    def set_target(
        self,
        target_formula: str,
        target_properties: Dict[str, Tuple[float, float]],
        constraints: Optional[Dict] = None
    ):
        """
        设置发现目标
        
        Args:
            target_formula: 目标化学式
            target_properties: {property_name: (target_value, tolerance)}
            constraints: 合成约束
        """
        self.target_formula = target_formula
        self.target_properties = target_properties
        self.constraints = constraints or {}
        
        print(f"Target set: {target_formula}")
        print(f"Target properties: {target_properties}")
    
    def run_discovery(
        self,
        max_experiments: int = 100,
        batch_size: int = 1
    ) -> List[ExperimentResult]:
        """
        运行发现循环
        
        Args:
            max_experiments: 最大实验次数
            batch_size: 每批实验数
        """
        self.is_running = True
        
        # 生成初始候选
        candidates = self.planner.plan_synthesis(
            self.target_formula,
            constraints=self.constraints
        )
        
        for exp_num in range(max_experiments):
            if not self.is_running:
                break
            
            print(f"\n=== Experiment {exp_num + 1}/{max_experiments} ===")
            
            # 选择下一个实验
            if self.optimizer and len(self.experiment_history) > 5:
                params = self._suggest_next_experiment()
            else:
                params = candidates[exp_num % len(candidates)]
            
            # 执行实验 (模拟)
            result = self._execute_experiment(params)
            
            # 记录
            self.experiment_history.append(result)
            
            # 检查是否达到目标
            if self._check_target_achieved(result):
                print(f"\n✓ Target achieved after {exp_num + 1} experiments!")
                break
            
            # 更新候选
            if exp_num % 10 == 9:
                candidates = self._generate_new_candidates()
        
        self.is_running = False
        return self.experiment_history
    
    def _execute_experiment(
        self,
        params: SynthesisParameters
    ) -> ExperimentResult:
        """
        执行单个实验 (模拟)
        
        实际系统中, 这里会控制真实设备
        """
        print(f"Synthesizing with:")
        print(f"  Temperature: {params.temperature}°C")
        print(f"  Time: {params.time} hours")
        print(f"  Precursors: {params.precursors}")
        
        # 模拟合成结果
        # 基于参数和目标属性计算"成功程度"
        char_data = self._simulate_characterization(params)
        
        # 生成实验ID
        exp_id = f"EXP_{len(self.experiment_history) + 1:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        result = ExperimentResult(
            experiment_id=exp_id,
            synthesis_params=params,
            characterization=char_data,
            target_properties=self._evaluate_properties(char_data),
            success=True
        )
        
        print(f"Results:")
        for prop, value in result.target_properties.items():
            print(f"  {prop}: {value:.4f}")
        
        return result
    
    def _simulate_characterization(
        self,
        params: SynthesisParameters
    ) -> CharacterizationData:
        """模拟表征结果"""
        # 基于合成参数计算性质 (简化模型)
        
        # 温度影响结晶度
        temp_factor = params.temperature / 1000.0
        
        # 时间影响反应程度
        time_factor = min(params.time / 24.0, 1.0)
        
        # 计算性质
        particle_size = 100 + (1 - temp_factor) * 900  # nm
        surface_area = 50 * temp_factor * time_factor  # m2/g
        
        # XRD模式 (模拟)
        xrd = np.random.rand(1000) * temp_factor * time_factor
        
        return CharacterizationData(
            xrd_pattern=xrd,
            particle_size=particle_size,
            surface_area=surface_area,
            band_gap=2.0 + np.random.rand() * 1.0,
            conductivity=np.random.rand() * temp_factor * 100
        )
    
    def _evaluate_properties(
        self,
        char_data: CharacterizationData
    ) -> Dict[str, float]:
        """从表征数据提取目标性质"""
        properties = {}
        
        if char_data.band_gap is not None:
            properties['band_gap'] = char_data.band_gap
        
        if char_data.conductivity is not None:
            properties['conductivity'] = char_data.conductivity
        
        if char_data.particle_size is not None:
            properties['particle_size'] = char_data.particle_size
        
        return properties
    
    def _check_target_achieved(self, result: ExperimentResult) -> bool:
        """检查是否达到目标"""
        for prop_name, (target, tolerance) in self.target_properties.items():
            if prop_name in result.target_properties:
                actual = result.target_properties[prop_name]
                if abs(actual - target) > tolerance:
                    return False
            else:
                return False
        return True
    
    def _suggest_next_experiment(self) -> SynthesisParameters:
        """使用优化器建议下一个实验"""
        # 简化: 随机变异最好的实验
        best = max(self.experiment_history, 
                  key=lambda r: self._score_result(r))
        
        params = best.synthesis_params
        
        # 变异
        new_temp = params.temperature + np.random.randn() * 50
        new_temp = np.clip(new_temp, 100, 1200)
        
        new_time = params.time + np.random.randn() * 2
        new_time = max(0.5, new_time)
        
        return SynthesisParameters(
            temperature=float(new_temp),
            time=float(new_time),
            precursors=params.precursors,
            atmosphere=params.atmosphere
        )
    
    def _score_result(self, result: ExperimentResult) -> float:
        """评分实验结果"""
        score = 0.0
        
        for prop_name, (target, tolerance) in self.target_properties.items():
            if prop_name in result.target_properties:
                actual = result.target_properties[prop_name]
                error = abs(actual - target) / tolerance
                score += max(0, 1 - error)
        
        return score
    
    def _generate_new_candidates(self) -> List[SynthesisParameters]:
        """基于历史生成新候选"""
        return self.planner.plan_synthesis(self.target_formula)
    
    def get_summary(self) -> Dict:
        """获取实验摘要"""
        return {
            'total_experiments': len(self.experiment_history),
            'successful_experiments': sum(1 for r in self.experiment_history if r.success),
            'best_result': self._get_best_result(),
            'property_evolution': self._get_property_evolution()
        }
    
    def _get_best_result(self) -> Optional[ExperimentResult]:
        """获取最佳结果"""
        if not self.experiment_history:
            return None
        return max(self.experiment_history, key=lambda r: self._score_result(r))
    
    def _get_property_evolution(self) -> Dict[str, List[float]]:
        """获取性质演化"""
        evolution = defaultdict(list)
        
        for result in self.experiment_history:
            for prop, value in result.target_properties.items():
                evolution[prop].append(value)
        
        return dict(evolution)


class LabAutomationInterface:
    """
    实验室自动化接口
    
    连接实际硬件设备的抽象接口
    """
    
    def __init__(self):
        self.connected_devices = {}
        
    def connect_device(self, device_name: str, device_config: Dict):
        """连接设备"""
        self.connected_devices[device_name] = device_config
        print(f"Connected: {device_name}")
    
    def send_command(self, device_name: str, command: str, params: Dict) -> Any:
        """发送命令到设备"""
        if device_name not in self.connected_devices:
            raise RuntimeError(f"Device {device_name} not connected")
        
        # 实际系统中, 这里会发送硬件命令
        print(f"Command to {device_name}: {command} {params}")
        return {"status": "success"}
    
    def read_sensor(self, device_name: str, sensor_name: str) -> float:
        """读取传感器"""
        # 模拟传感器读数
        return np.random.rand() * 100


if __name__ == "__main__":
    print("=" * 60)
    print("Self-Driving Lab Demo")
    print("=" * 60)
    
    # 创建实验室
    lab = SelfDrivingLab()
    
    # 设置发现目标
    print("\n1. Setting Discovery Target")
    print("-" * 40)
    
    lab.set_target(
        target_formula="LiFePO4",
        target_properties={
            'band_gap': (2.0, 0.2),
            'conductivity': (50.0, 10.0)
        },
        constraints={'max_temperature': 800}
    )
    
    # 运行发现
    print("\n2. Running Discovery Loop")
    print("-" * 40)
    
    results = lab.run_discovery(max_experiments=10, batch_size=1)
    
    # 输出摘要
    print("\n3. Discovery Summary")
    print("-" * 40)
    
    summary = lab.get_summary()
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Successful: {summary['successful_experiments']}")
    
    if summary['best_result']:
        print(f"\nBest result:")
        print(f"  ID: {summary['best_result'].experiment_id}")
        print(f"  Properties: {summary['best_result'].target_properties}")
    
    print("\n4. Synthesis Planning Demo")
    print("-" * 40)
    
    planner = SynthesisPlanner()
    candidates = planner.plan_synthesis("LiCoO2")
    
    print(f"Generated {len(candidates)} synthesis candidates for LiCoO2")
    for i, cand in enumerate(candidates[:3]):
        print(f"\nCandidate {i+1}:")
        print(f"  Temperature: {cand.temperature}°C")
        print(f"  Time: {cand.time} hours")
        print(f"  Precursors: {cand.precursors}")
    
    print("\n" + "=" * 60)
    print("Self-Driving Lab Demo completed!")
    print("Key features:")
    print("- Automated synthesis planning")
    print("- Closed-loop discovery")
    print("- Multi-objective optimization")
    print("- Hardware abstraction interface")
    print("=" * 60)
