"""
干预模拟器 - Intervention Simulator
预测"如果...会怎样"的因果效应
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import warnings


@dataclass
class Intervention:
    """干预定义"""
    target: str
    value: Any
    intervention_type: str = "do"  # "do", "shift", "conditional"
    conditions: Optional[Dict[str, Any]] = None
    
    def __repr__(self):
        if self.intervention_type == "do":
            return f"do({self.target}={self.value})"
        elif self.intervention_type == "shift":
            return f"shift({self.target}, {self.value})"
        else:
            return f"{self.target}={self.value} | {self.conditions}"


@dataclass
class InterventionResult:
    """干预结果"""
    intervention: Intervention
    outcomes: Dict[str, Any]
    outcome_distribution: Optional[Dict[str, np.ndarray]] = None
    expected_outcomes: Dict[str, float] = field(default_factory=dict)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def __repr__(self):
        return f"Intervention({self.intervention}): {self.expected_outcomes}"


class CausalSimulator:
    """
    因果模拟器
    基于因果模型进行干预模拟
    """
    
    def __init__(
        self,
        structural_equations: Optional[Dict[str, Callable]] = None,
        noise_distributions: Optional[Dict[str, Tuple]] = None
    ):
        self.equations = structural_equations or {}
        self.noise_dist = noise_distributions or {}
        self.variables: Set[str] = set()
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        
        # 分析依赖关系
        self._analyze_dependencies()
    
    def add_equation(
        self,
        variable: str,
        equation: Callable,
        noise_mean: float = 0.0,
        noise_std: float = 1.0
    ):
        """添加结构方程"""
        self.equations[variable] = equation
        self.noise_dist[variable] = (noise_mean, noise_std)
        self.variables.add(variable)
        self._analyze_dependencies()
    
    def _analyze_dependencies(self):
        """分析变量依赖关系"""
        self.parents = defaultdict(set)
        self.children = defaultdict(set)
        
        # 简化的依赖分析
        # 实际应该通过代码分析或显式指定
        for var in self.equations:
            self.variables.add(var)
    
    def simulate(
        self,
        intervention: Intervention,
        n_samples: int = 1000,
        return_distribution: bool = True
    ) -> InterventionResult:
        """
        模拟干预
        
        Args:
            intervention: 干预
            n_samples: 模拟样本数
            return_distribution: 是否返回分布
        
        Returns:
            干预结果
        """
        outcomes = defaultdict(list)
        
        for _ in range(n_samples):
            # 生成外生噪声
            noise = {
                var: np.random.normal(*self.noise_dist.get(var, (0, 1)))
                for var in self.variables
            }
            
            # 应用干预
            if intervention.intervention_type == "do":
                noise[intervention.target] = intervention.value
            
            # 计算变量值
            values = self._compute_values(noise, intervention)
            
            for var, val in values.items():
                outcomes[var].append(val)
        
        # 整理结果
        outcome_arrays = {var: np.array(vals) for var, vals in outcomes.items()}
        
        expected = {
            var: np.mean(vals)
            for var, vals in outcome_arrays.items()
        }
        
        ci = {
            var: (np.percentile(vals, 2.5), np.percentile(vals, 97.5))
            for var, vals in outcome_arrays.items()
        }
        
        result = InterventionResult(
            intervention=intervention,
            outcomes={var: vals[0] for var, vals in outcome_arrays.items()},
            expected_outcomes=expected,
            confidence_intervals=ci
        )
        
        if return_distribution:
            result.outcome_distribution = outcome_arrays
        
        return result
    
    def _compute_values(
        self,
        noise: Dict[str, float],
        intervention: Intervention
    ) -> Dict[str, float]:
        """计算变量值"""
        values = {}
        
        # 拓扑排序（简化：按变量名顺序）
        for var in sorted(self.variables):
            if intervention.intervention_type == "do" and var == intervention.target:
                values[var] = intervention.value
            elif var in self.equations:
                try:
                    values[var] = self.equations[var](values, noise[var])
                except:
                    values[var] = noise[var]
            else:
                values[var] = noise[var]
        
        return values
    
    def compare_interventions(
        self,
        interventions: List[Intervention],
        target_variable: str,
        n_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        比较多个干预
        
        Args:
            interventions: 干预列表
            target_variable: 目标变量
            n_samples: 样本数
        
        Returns:
            比较结果
        """
        results = []
        
        for intervention in interventions:
            result = self.simulate(intervention, n_samples)
            results.append(result)
        
        # 整理比较结果
        comparison = {
            "interventions": [str(i) for i in interventions],
            "expected_outcomes": [
                r.expected_outcomes.get(target_variable, 0)
                for r in results
            ],
            "confidence_intervals": [
                r.confidence_intervals.get(target_variable, (0, 0))
                for r in results
            ]
        }
        
        if all(r.outcome_distribution for r in results):
            comparison["distributions"] = [
                r.outcome_distribution[target_variable]
                for r in results
            ]
        
        return comparison
    
    def find_optimal_intervention(
        self,
        target_variable: str,
        intervention_variables: List[str],
        intervention_values: List[Any],
        maximize: bool = True,
        n_samples: int = 500
    ) -> Tuple[Intervention, float]:
        """
        寻找最优干预
        
        Args:
            target_variable: 目标变量
            intervention_variables: 可干预变量
            intervention_values: 干预值列表
            maximize: 是否最大化
            n_samples: 样本数
        
        Returns:
            最优干预和期望结果
        """
        best_intervention = None
        best_outcome = -np.inf if maximize else np.inf
        
        for var in intervention_variables:
            for val in intervention_values:
                intervention = Intervention(var, val)
                result = self.simulate(intervention, n_samples, return_distribution=False)
                
                outcome = result.expected_outcomes.get(target_variable, 0)
                
                if maximize and outcome > best_outcome:
                    best_outcome = outcome
                    best_intervention = intervention
                elif not maximize and outcome < best_outcome:
                    best_outcome = outcome
                    best_intervention = intervention
        
        return best_intervention, best_outcome


class SensitivityAnalyzer:
    """
    敏感性分析器
    分析结果对干预的敏感性
    """
    
    def __init__(self, simulator: CausalSimulator):
        self.simulator = simulator
    
    def analyze(
        self,
        target_variable: str,
        intervention_variable: str,
        intervention_range: Tuple[float, float],
        n_points: int = 20,
        n_samples: int = 500
    ) -> Dict[str, Any]:
        """
        分析敏感性
        
        Args:
            target_variable: 目标变量
            intervention_variable: 干预变量
            intervention_range: 干预值范围
            n_points: 采样点数
            n_samples: 每点样本数
        
        Returns:
            敏感性分析结果
        """
        values = np.linspace(intervention_range[0], intervention_range[1], n_points)
        
        expected_outcomes = []
        lower_bounds = []
        upper_bounds = []
        
        for val in values:
            intervention = Intervention(intervention_variable, val)
            result = self.simulator.simulate(intervention, n_samples)
            
            expected = result.expected_outcomes.get(target_variable, 0)
            ci = result.confidence_intervals.get(target_variable, (0, 0))
            
            expected_outcomes.append(expected)
            lower_bounds.append(ci[0])
            upper_bounds.append(ci[1])
        
        # 计算敏感性指标
        sensitivity = np.gradient(expected_outcomes, values)
        
        return {
            "intervention_values": values.tolist(),
            "expected_outcomes": expected_outcomes,
            "confidence_intervals": list(zip(lower_bounds, upper_bounds)),
            "sensitivity": sensitivity.tolist(),
            "max_sensitivity": max(abs(s) for s in sensitivity),
            "average_sensitivity": np.mean(np.abs(sensitivity))
        }
    
    def tornado_analysis(
        self,
        target_variable: str,
        intervention_variables: List[str],
        baseline_values: Dict[str, float],
        perturbation: float = 0.1,
        n_samples: int = 500
    ) -> Dict[str, float]:
        """
        龙卷风分析
        
        Args:
            target_variable: 目标变量
            intervention_variables: 干预变量
            baseline_values: 基线值
            perturbation: 扰动幅度
            n_samples: 样本数
        
        Returns:
            各变量的影响程度
        """
        # 基线结果
        baseline_result = self.simulate_baseline(baseline_values, n_samples)
        baseline_outcome = baseline_result.expected_outcomes.get(target_variable, 0)
        
        effects = {}
        
        for var in intervention_variables:
            # 正向扰动
            intervention_plus = Intervention(var, baseline_values[var] * (1 + perturbation))
            result_plus = self.simulator.simulate(intervention_plus, n_samples)
            outcome_plus = result_plus.expected_outcomes.get(target_variable, 0)
            
            # 负向扰动
            intervention_minus = Intervention(var, baseline_values[var] * (1 - perturbation))
            result_minus = self.simulator.simulate(intervention_minus, n_samples)
            outcome_minus = result_minus.expected_outcomes.get(target_variable, 0)
            
            # 影响程度
            effect = abs(outcome_plus - outcome_minus) / 2
            effects[var] = effect
        
        # 按影响排序
        effects = dict(sorted(effects.items(), key=lambda x: x[1], reverse=True))
        
        return effects
    
    def simulate_baseline(
        self,
        baseline_values: Dict[str, float],
        n_samples: int
    ) -> InterventionResult:
        """模拟基线情况（无干预）"""
        # 使用一个虚拟干预
        dummy_intervention = Intervention("__dummy__", 0)
        return self.simulator.simulate(dummy_intervention, n_samples)


class ScenarioAnalyzer:
    """
    场景分析器
    分析不同场景下的结果
    """
    
    def __init__(self, simulator: CausalSimulator):
        self.simulator = simulator
    
    def analyze_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        target_variables: List[str],
        n_samples: int = 500
    ) -> Dict[str, Any]:
        """
        分析多个场景
        
        Args:
            scenarios: 场景列表（每个场景是变量值字典）
            target_variables: 目标变量
            n_samples: 样本数
        
        Returns:
            场景分析结果
        """
        results = []
        
        for i, scenario in enumerate(scenarios):
            # 将场景转换为干预
            interventions = [
                Intervention(var, val)
                for var, val in scenario.items()
            ]
            
            # 模拟每个干预（简化：只模拟第一个）
            if interventions:
                result = self.simulator.simulate(interventions[0], n_samples)
            else:
                result = self.simulator.simulate(
                    Intervention("__dummy__", 0), n_samples
                )
            
            scenario_result = {
                "scenario_id": i,
                "scenario": scenario,
                "expected_outcomes": {
                    var: result.expected_outcomes.get(var, 0)
                    for var in target_variables
                },
                "confidence_intervals": {
                    var: result.confidence_intervals.get(var, (0, 0))
                    for var in target_variables
                }
            }
            
            results.append(scenario_result)
        
        return {
            "scenarios": results,
            "target_variables": target_variables
        }
    
    def what_if_analysis(
        self,
        base_scenario: Dict[str, Any],
        changes: List[Dict[str, Any]],
        target_variable: str,
        n_samples: int = 500
    ) -> Dict[str, Any]:
        """
        "如果...会怎样"分析
        
        Args:
            base_scenario: 基准场景
            changes: 改变列表
            target_variable: 目标变量
            n_samples: 样本数
        
        Returns:
            分析结果
        """
        # 基准结果
        base_result = self.simulator.simulate(
            Intervention("__dummy__", 0), n_samples
        )
        baseline = base_result.expected_outcomes.get(target_variable, 0)
        
        # 应用每个改变
        results = []
        for change in changes:
            new_scenario = {**base_scenario, **change}
            
            # 转换为干预
            for var, val in change.items():
                intervention = Intervention(var, val)
                result = self.simulator.simulate(intervention, n_samples)
                
                outcome = result.expected_outcomes.get(target_variable, 0)
                
                results.append({
                    "change": change,
                    "outcome": outcome,
                    "difference": outcome - baseline,
                    "relative_change": (outcome - baseline) / baseline if baseline != 0 else 0
                })
        
        return {
            "baseline": baseline,
            "results": results,
            "target_variable": target_variable
        }


class PolicySimulator:
    """
    策略模拟器
    模拟不同策略的长期效果
    """
    
    def __init__(self, simulator: CausalSimulator):
        self.simulator = simulator
    
    def simulate_policy(
        self,
        policy: Callable[[Dict[str, float]], Dict[str, Any]],
        initial_state: Dict[str, float],
        n_periods: int = 10,
        n_trajectories: int = 100
    ) -> Dict[str, Any]:
        """
        模拟策略
        
        Args:
            policy: 策略函数（状态 -> 行动）
            initial_state: 初始状态
            n_periods: 模拟周期数
            n_trajectories: 轨迹数
        
        Returns:
            模拟结果
        """
        trajectories = []
        
        for _ in range(n_trajectories):
            trajectory = [initial_state.copy()]
            state = initial_state.copy()
            
            for _ in range(n_periods):
                # 应用策略
                action = policy(state)
                
                # 模拟一步
                next_state = self._simulate_step(state, action)
                trajectory.append(next_state)
                state = next_state
            
            trajectories.append(trajectory)
        
        # 分析轨迹
        return self._analyze_trajectories(trajectories, n_periods)
    
    def _simulate_step(
        self,
        state: Dict[str, float],
        action: Dict[str, Any]
    ) -> Dict[str, float]:
        """模拟单步"""
        # 应用行动作为干预
        intervention = Intervention(
            list(action.keys())[0] if action else "__dummy__",
            list(action.values())[0] if action else 0
        )
        
        result = self.simulator.simulate(intervention, n_samples=1)
        
        return result.expected_outcomes
    
    def _analyze_trajectories(
        self,
        trajectories: List[List[Dict[str, float]]],
        n_periods: int
    ) -> Dict[str, Any]:
        """分析轨迹"""
        # 计算每个时期的统计量
        period_stats = []
        
        for t in range(n_periods + 1):
            states_at_t = [traj[t] for traj in trajectories]
            
            stats = {}
            for var in states_at_t[0].keys():
                values = [s[var] for s in states_at_t]
                stats[var] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values)
                }
            
            period_stats.append(stats)
        
        return {
            "trajectories": trajectories,
            "period_statistics": period_stats,
            "final_state_distribution": trajectories[-1]
        }


def demo():
    """演示干预模拟器"""
    print("=" * 60)
    print("干预模拟器演示")
    print("=" * 60)
    
    # 创建因果模拟器
    simulator = CausalSimulator()
    
    # 定义电池制造过程的因果模型
    # 变量：温度 -> 反应速率 -> 材料质量 -> 电池容量
    
    simulator.add_equation(
        "temperature",
        lambda vals, noise: 150 + 20 * noise,  # 温度 ~ N(150, 400)
        noise_mean=0,
        noise_std=1
    )
    
    simulator.add_equation(
        "reaction_rate",
        lambda vals, noise: 0.5 + 0.01 * (vals.get("temperature", 150) - 150) + 0.1 * noise,
        noise_mean=0,
        noise_std=1
    )
    
    simulator.add_equation(
        "material_quality",
        lambda vals, noise: 80 + 20 * vals.get("reaction_rate", 0.5) - 0.05 * (vals.get("temperature", 150) - 160)**2 + 5 * noise,
        noise_mean=0,
        noise_std=1
    )
    
    simulator.add_equation(
        "capacity",
        lambda vals, noise: 200 + 2 * vals.get("material_quality", 80) + 10 * noise,
        noise_mean=0,
        noise_std=1
    )
    
    print("\n1. 因果模型已建立:")
    print("   温度 -> 反应速率 -> 材料质量 -> 电池容量")
    
    # 干预模拟
    print("\n2. 干预模拟:")
    
    # 干预1：提高温度
    intervention1 = Intervention("temperature", 180)
    result1 = simulator.simulate(intervention1, n_samples=1000)
    
    print(f"   干预: {intervention1}")
    print(f"   预期容量: {result1.expected_outcomes.get('capacity', 0):.2f}")
    print(f"   95%置信区间: {result1.confidence_intervals.get('capacity', (0, 0))}")
    
    # 干预2：降低温度
    intervention2 = Intervention("temperature", 140)
    result2 = simulator.simulate(intervention2, n_samples=1000)
    
    print(f"\n   干预: {intervention2}")
    print(f"   预期容量: {result2.expected_outcomes.get('capacity', 0):.2f}")
    print(f"   95%置信区间: {result2.confidence_intervals.get('capacity', (0, 0))}")
    
    # 比较干预
    print("\n3. 干预比较:")
    comparison = simulator.compare_interventions(
        [intervention1, intervention2],
        target_variable="capacity",
        n_samples=1000
    )
    
    for i, (interv, outcome) in enumerate(zip(comparison["interventions"], 
                                               comparison["expected_outcomes"])):
        print(f"   {interv}: 预期容量 = {outcome:.2f}")
    
    # 敏感性分析
    print("\n4. 敏感性分析:")
    analyzer = SensitivityAnalyzer(simulator)
    
    sensitivity = analyzer.analyze(
        target_variable="capacity",
        intervention_variable="temperature",
        intervention_range=(120, 200),
        n_points=10,
        n_samples=500
    )
    
    print(f"   温度范围: 120-200°C")
    print(f"   最大敏感性: {sensitivity['max_sensitivity']:.4f}")
    print(f"   平均敏感性: {sensitivity['average_sensitivity']:.4f}")
    
    # 龙卷风分析
    print("\n5. 龙卷风分析:")
    tornado = analyzer.tornado_analysis(
        target_variable="capacity",
        intervention_variables=["temperature", "reaction_rate"],
        baseline_values={"temperature": 150, "reaction_rate": 0.5},
        perturbation=0.2
    )
    
    print("   变量影响程度（排序）:")
    for var, effect in tornado.items():
        print(f"     {var}: {effect:.2f}")
    
    # 场景分析
    print("\n6. 场景分析:")
    scenario_analyzer = ScenarioAnalyzer(simulator)
    
    scenarios = [
        {"temperature": 160},  # 高温场景
        {"temperature": 140},  # 低温场景
        {"reaction_rate": 0.8},  # 高反应速率
    ]
    
    scenario_results = scenario_analyzer.analyze_scenarios(
        scenarios,
        target_variables=["capacity", "material_quality"],
        n_samples=500
    )
    
    for result in scenario_results["scenarios"]:
        print(f"   场景 {result['scenario_id']}: {result['scenario']}")
        print(f"     预期容量: {result['expected_outcomes'].get('capacity', 0):.2f}")
        print(f"     材料质量: {result['expected_outcomes'].get('material_quality', 0):.2f}")
    
    # What-if分析
    print("\n7. What-if分析:")
    what_if = scenario_analyzer.what_if_analysis(
        base_scenario={"temperature": 150},
        changes=[
            {"temperature": 160},
            {"temperature": 170},
            {"temperature": 180},
        ],
        target_variable="capacity",
        n_samples=500
    )
    
    print(f"   基准容量: {what_if['baseline']:.2f}")
    print("   改变温度的影响:")
    for result in what_if["results"]:
        temp = result["change"]["temperature"]
        print(f"     温度={temp}°C: 容量={result['outcome']:.2f} ({result['difference']:+.2f})")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
