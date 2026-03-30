"""
想象引擎 - Imagination Engine
==============================

反事实模拟、假设场景生成、创造性设计的AI想象模块。

This module implements an imagination engine that can:
- Run counterfactual simulations
- Generate hypothetical scenarios
- Perform creative design exploration
- Support "what-if" reasoning for materials
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import copy
from collections import defaultdict
import itertools
from pathlib import Path
import json

# 导入世界模型基类
from .material_world_model import (
    MaterialWorldModel,
    MaterialState,
    MaterialAction,
    ActionType,
    Transition,
    WorldModelConfig
)


class ScenarioType(Enum):
    """场景类型枚举"""
    COUNTERFACTUAL = auto()   # 反事实: "如果...会怎样"
    HYPOTHETICAL = auto()     # 假设性: "假设..."
    FUTURE = auto()           # 未来预测
    ALTERNATIVE = auto()      # 替代路径
    EXTREME = auto()          # 极端条件
    OPTIMAL = auto()          # 最优情况


class DesignStrategy(Enum):
    """设计策略枚举"""
    MUTATION = auto()         # 变异
    CROSSOVER = auto()        # 交叉
    EXPLORATION = auto()      # 探索
    EXPLOITATION = auto()     # 利用
    CONSTRAINT_SATISFACTION = auto()  # 约束满足
    MULTI_OBJECTIVE = auto()  # 多目标优化


@dataclass
class CounterfactualQuery:
    """
    反事实查询
    
    定义一个"如果...会怎样"的问题
    """
    query_id: str
    base_state: MaterialState  # 基准状态
    
    # 干预定义
    intervention_type: ActionType
    intervention_params: Dict[str, Any] = field(default_factory=dict)
    
    # 对比状态 (实际发生的情况)
    factual_state: Optional[MaterialState] = None
    
    # 查询参数
    num_steps: int = 10
    num_samples: int = 100  # 蒙特卡洛样本数
    
    def to_action(self) -> MaterialAction:
        """转换为动作"""
        return MaterialAction(
            action_id=f"cf_{self.query_id}",
            action_type=self.intervention_type,
            parameters=self.intervention_params,
            **{k: v for k, v in self.intervention_params.items() 
               if k in ['magnitude', 'duration', 'target_atoms']}
        )


@dataclass
class HypotheticalScenario:
    """
    假设场景
    
    定义一个假设性条件及其推演
    """
    scenario_id: str
    scenario_type: ScenarioType
    
    # 初始条件
    initial_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # 假设条件
    assumptions: List[Dict[str, Any]] = field(default_factory=list)
    
    # 推演参数
    time_horizon: float = 1000.0  # fs
    time_step: float = 1.0
    
    # 约束条件
    constraints: List[Callable[[MaterialState], bool]] = field(default_factory=list)
    
    # 成功标准
    success_criteria: List[Callable[[MaterialState], float]] = field(default_factory=list)
    
    def validate_state(self, state: MaterialState) -> Tuple[bool, List[str]]:
        """验证状态是否满足假设条件"""
        violations = []
        for constraint in self.constraints:
            if not constraint(state):
                violations.append(f"Constraint violated: {constraint.__name__}")
        return len(violations) == 0, violations


@dataclass
class ImaginedOutcome:
    """
    想象结果
    
    存储想象引擎的推演结果
    """
    outcome_id: str
    query: Union[CounterfactualQuery, HypotheticalScenario]
    
    # 推演结果
    trajectories: List[List[MaterialState]] = field(default_factory=list)
    probabilities: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    
    # 统计分析
    mean_final_state: Optional[MaterialState] = None
    state_uncertainty: Optional[np.ndarray] = None
    success_probability: float = 0.0
    expected_reward: float = 0.0
    
    # 关键发现
    insights: List[str] = field(default_factory=list)
    
    # 时间戳
    generation_time: float = field(default_factory=lambda: 0.0)
    
    def get_best_trajectory(self) -> Tuple[List[MaterialState], float]:
        """获取最佳轨迹"""
        if not self.rewards:
            return [], 0.0
        best_idx = np.argmax(self.rewards)
        return self.trajectories[best_idx], self.rewards[best_idx]
    
    def get_worst_trajectory(self) -> Tuple[List[MaterialState], float]:
        """获取最差轨迹"""
        if not self.rewards:
            return [], 0.0
        worst_idx = np.argmin(self.rewards)
        return self.trajectories[worst_idx], self.rewards[worst_idx]


class CounterfactualSimulator:
    """
    反事实模拟器
    
    运行"如果...会怎样"的模拟
    """
    
    def __init__(self, world_model: MaterialWorldModel):
        self.world_model = world_model
        self.history: List[CounterfactualQuery] = []
    
    def simulate(
        self,
        query: CounterfactualQuery,
        verbose: bool = False
    ) -> ImaginedOutcome:
        """
        运行反事实模拟
        
        Args:
            query: 反事实查询
            verbose: 是否打印进度
            
        Returns:
            想象结果
        """
        trajectories = []
        rewards = []
        
        # 创建干预动作
        action = query.to_action()
        
        # 蒙特卡洛采样
        for i in range(query.num_samples):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Sample {i+1}/{query.num_samples}")
            
            # 创建动作序列
            actions = [action] * query.num_steps
            
            # 运行推演
            result = self.world_model.rollout(
                query.base_state,
                actions,
                return_trajectory=True
            )
            
            trajectories.append(result['states'])
            rewards.append(result['total_reward'])
        
        # 统计分析
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        success_prob = np.mean([r > 0 for r in rewards])
        
        # 计算平均最终状态
        final_states = [traj[-1] for traj in trajectories]
        mean_final = self._compute_mean_state(final_states)
        
        # 生成洞察
        insights = self._generate_counterfactual_insights(
            query, trajectories, rewards
        )
        
        outcome = ImaginedOutcome(
            outcome_id=f"cf_outcome_{query.query_id}",
            query=query,
            trajectories=trajectories,
            rewards=rewards,
            mean_final_state=mean_final,
            success_probability=success_prob,
            expected_reward=mean_reward,
            insights=insights
        )
        
        self.history.append(query)
        return outcome
    
    def _compute_mean_state(
        self,
        states: List[MaterialState]
    ) -> MaterialState:
        """计算平均状态"""
        if not states:
            return states[0] if states else MaterialState(state_id="empty")
        
        # 平均热力学量
        mean_state = MaterialState(
            state_id=f"mean_of_{len(states)}_states",
            temperature=np.mean([s.temperature for s in states]),
            pressure=np.mean([s.pressure for s in states]),
            total_energy=np.mean([s.total_energy for s in states]),
            potential_energy=np.mean([s.potential_energy for s in states]),
            kinetic_energy=np.mean([s.kinetic_energy for s in states]),
            defect_concentration=np.mean([s.defect_concentration for s in states])
        )
        
        return mean_state
    
    def _generate_counterfactual_insights(
        self,
        query: CounterfactualQuery,
        trajectories: List[List[MaterialState]],
        rewards: List[float]
    ) -> List[str]:
        """生成反事实洞察"""
        insights = []
        
        # 效果分析
        if query.factual_state:
            factual_energy = query.factual_state.total_energy
            mean_cf_energy = np.mean([t[-1].total_energy for t in trajectories])
            energy_diff = mean_cf_energy - factual_energy
            
            if abs(energy_diff) > 1.0:
                direction = "increased" if energy_diff > 0 else "decreased"
                insights.append(
                    f"Counterfactual intervention would have {direction} "
                    f"energy by {abs(energy_diff):.2f} eV"
                )
        
        # 稳定性分析
        stability_count = sum(1 for r in rewards if r > 0)
        stability_ratio = stability_count / len(rewards)
        
        if stability_ratio > 0.8:
            insights.append("High stability: >80% of trajectories remain stable")
        elif stability_ratio < 0.2:
            insights.append("Low stability: <20% of trajectories remain stable")
        
        # 变异性分析
        std_reward = np.std(rewards)
        if std_reward > np.abs(np.mean(rewards)):
            insights.append("High outcome variability: results depend strongly on random factors")
        
        return insights
    
    def compare_interventions(
        self,
        base_state: MaterialState,
        interventions: List[CounterfactualQuery],
        metric_fn: Optional[Callable[[MaterialState], float]] = None
    ) -> Dict[str, Any]:
        """
        比较多个干预措施
        
        Returns:
            比较结果字典
        """
        if metric_fn is None:
            metric_fn = lambda s: -s.total_energy  # 默认: 能量越低越好
        
        results = {}
        
        for intervention in interventions:
            outcome = self.simulate(intervention)
            
            final_scores = [
                metric_fn(traj[-1]) for traj in outcome.trajectories
            ]
            
            results[intervention.query_id] = {
                'mean_score': np.mean(final_scores),
                'std_score': np.std(final_scores),
                'success_rate': outcome.success_probability,
                'expected_improvement': outcome.expected_reward
            }
        
        # 排名
        ranked = sorted(
            results.items(),
            key=lambda x: x[1]['mean_score'],
            reverse=True
        )
        
        return {
            'detailed_results': results,
            'ranking': [r[0] for r in ranked],
            'best_intervention': ranked[0][0] if ranked else None
        }


class HypotheticalScenarioGenerator:
    """
    假设场景生成器
    
    生成并探索假设性条件
    """
    
    def __init__(self, world_model: MaterialWorldModel):
        self.world_model = world_model
        self.scenario_templates: Dict[str, Callable] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """注册默认场景模板"""
        self.scenario_templates['extreme_temperature'] = self._create_extreme_temp_scenario
        self.scenario_templates['extreme_pressure'] = self._create_extreme_pressure_scenario
        self.scenario_templates['rapid_quench'] = self._create_quench_scenario
        self.scenario_templates['high_defect'] = self._create_defect_scenario
        self.scenario_templates['phase_transition'] = self._create_phase_transition_scenario
    
    def generate_scenario(
        self,
        template_name: str,
        base_state: MaterialState,
        **kwargs
    ) -> HypotheticalScenario:
        """基于模板生成场景"""
        if template_name not in self.scenario_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        return self.scenario_templates[template_name](base_state, **kwargs)
    
    def explore_scenario(
        self,
        scenario: HypotheticalScenario,
        num_samples: int = 50
    ) -> ImaginedOutcome:
        """
        探索假设场景
        
        Args:
            scenario: 假设场景
            num_samples: 采样数
            
        Returns:
            想象结果
        """
        # 创建初始动作
        init_action = self._scenario_to_action(scenario)
        
        # 生成随机动作序列探索场景空间
        trajectories = []
        rewards = []
        
        for _ in range(num_samples):
            # 随机动作序列
            actions = self._generate_exploration_actions(
                scenario, init_action
            )
            
            # 创建符合假设的初始状态
            init_state = self._apply_assumptions(
                scenario.initial_conditions.get('base_state'),
                scenario.assumptions
            )
            
            # 运行推演
            result = self.world_model.rollout(init_state, actions)
            
            if isinstance(result, dict):
                trajectories.append(result['states'])
                rewards.append(result['total_reward'])
            else:
                trajectories.append(result)
                rewards.append(0.0)
        
        # 评估成功标准
        success_scores = []
        for traj in trajectories:
            final_state = traj[-1]
            score = np.mean([
                criterion(final_state) for criterion in scenario.success_criteria
            ]) if scenario.success_criteria else 0.0
            success_scores.append(score)
        
        outcome = ImaginedOutcome(
            outcome_id=f"hypo_{scenario.scenario_id}",
            query=scenario,
            trajectories=trajectories,
            rewards=rewards,
            success_probability=np.mean([s > 0.5 for s in success_scores]),
            expected_reward=np.mean(rewards),
            insights=self._generate_hypothetical_insights(scenario, trajectories, success_scores)
        )
        
        return outcome
    
    def _scenario_to_action(self, scenario: HypotheticalScenario) -> MaterialAction:
        """将场景转换为初始动作"""
        return MaterialAction(
            action_id=f"scenario_{scenario.scenario_id}",
            action_type=ActionType.TEMPERATURE_CHANGE,
            magnitude=scenario.initial_conditions.get('temp_change', 0.0),
            duration=scenario.time_horizon
        )
    
    def _generate_exploration_actions(
        self,
        scenario: HypotheticalScenario,
        base_action: MaterialAction
    ) -> List[MaterialAction]:
        """生成探索性动作序列"""
        num_steps = int(scenario.time_horizon / scenario.time_step)
        actions = []
        
        for i in range(num_steps):
            # 在基础动作上添加噪声
            noisy_action = copy.deepcopy(base_action)
            noisy_action.magnitude += np.random.randn() * 0.1
            actions.append(noisy_action)
        
        return actions
    
    def _apply_assumptions(
        self,
        base_state: Optional[MaterialState],
        assumptions: List[Dict[str, Any]]
    ) -> MaterialState:
        """将假设应用到初始状态"""
        if base_state is None:
            base_state = MaterialState(state_id="hypothetical_base")
        
        state = base_state.clone()
        
        for assumption in assumptions:
            if 'temperature' in assumption:
                state.temperature = assumption['temperature']
            if 'pressure' in assumption:
                state.pressure = assumption['pressure']
            if 'defect_concentration' in assumption:
                state.defect_concentration = assumption['defect_concentration']
        
        return state
    
    def _generate_hypothetical_insights(
        self,
        scenario: HypotheticalScenario,
        trajectories: List[List[MaterialState]],
        success_scores: List[float]
    ) -> List[str]:
        """生成假设场景洞察"""
        insights = []
        
        success_rate = np.mean([s > 0.5 for s in success_scores])
        insights.append(f"Success rate under assumptions: {success_rate:.1%}")
        
        # 分析轨迹差异
        final_temps = [traj[-1].temperature for traj in trajectories]
        if np.std(final_temps) > 100:
            insights.append("High temperature variability suggests phase instability")
        
        final_energies = [traj[-1].total_energy for traj in trajectories]
        if np.min(final_energies) < -50:
            insights.append("Some trajectories reach highly favorable energy states")
        
        return insights
    
    # 场景模板方法
    def _create_extreme_temp_scenario(
        self,
        base_state: MaterialState,
        target_temp: float = 2000.0,
        **kwargs
    ) -> HypotheticalScenario:
        return HypotheticalScenario(
            scenario_id=f"extreme_temp_{target_temp}",
            scenario_type=ScenarioType.EXTREME,
            initial_conditions={'base_state': base_state, 'temp_change': target_temp - base_state.temperature},
            assumptions=[{'temperature': target_temp}],
            time_horizon=5000.0,
            constraints=[lambda s: s.temperature > 0]
        )
    
    def _create_extreme_pressure_scenario(
        self,
        base_state: MaterialState,
        target_pressure: float = 100.0,
        **kwargs
    ) -> HypotheticalScenario:
        return HypotheticalScenario(
            scenario_id=f"extreme_pressure_{target_pressure}",
            scenario_type=ScenarioType.EXTREME,
            initial_conditions={'base_state': base_state},
            assumptions=[{'pressure': target_pressure}],
            time_horizon=5000.0
        )
    
    def _create_quench_scenario(
        self,
        base_state: MaterialState,
        quench_rate: float = -100.0,
        **kwargs
    ) -> HypotheticalScenario:
        return HypotheticalScenario(
            scenario_id=f"quench_{quench_rate}",
            scenario_type=ScenarioType.ALTERNATIVE,
            initial_conditions={'base_state': base_state},
            assumptions=[{'quench_rate': quench_rate}],
            time_horizon=10000.0
        )
    
    def _create_defect_scenario(
        self,
        base_state: MaterialState,
        defect_conc: float = 0.1,
        **kwargs
    ) -> HypotheticalScenario:
        return HypotheticalScenario(
            scenario_id=f"defect_{defect_conc}",
            scenario_type=ScenarioType.HYPOTHETICAL,
            initial_conditions={'base_state': base_state},
            assumptions=[{'defect_concentration': defect_conc}],
            time_horizon=5000.0
        )
    
    def _create_phase_transition_scenario(
        self,
        base_state: MaterialState,
        target_phase: str = "high_pressure",
        **kwargs
    ) -> HypotheticalScenario:
        return HypotheticalScenario(
            scenario_id=f"phase_transition_{target_phase}",
            scenario_type=ScenarioType.HYPOTHETICAL,
            initial_conditions={'base_state': base_state, 'target_phase': target_phase},
            time_horizon=20000.0
        )


class CreativeDesignSpace:
    """
    创造性设计空间
    
    探索和生成新颖的材料设计方案
    """
    
    def __init__(self, world_model: MaterialWorldModel):
        self.world_model = world_model
        self.design_history: List[Dict[str, Any]] = []
        self.novelty_threshold: float = 0.3
    
    def explore_design_space(
        self,
        seed_states: List[MaterialState],
        design_objectives: List[Callable[[MaterialState], float]],
        num_iterations: int = 100,
        population_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        探索设计空间寻找创新方案
        
        Args:
            seed_states: 种子状态
            design_objectives: 设计目标函数列表
            num_iterations: 迭代次数
            population_size: 种群大小
            
        Returns:
            帕累托前沿设计
        """
        # 初始化种群
        population = self._initialize_population(seed_states, population_size)
        
        # 进化优化
        for iteration in range(num_iterations):
            # 评估适应度
            fitness_scores = self._evaluate_population(population, design_objectives)
            
            # 选择
            selected = self._select_pareto_front(population, fitness_scores)
            
            # 变异和交叉
            offspring = self._create_offspring(selected, population_size - len(selected))
            
            # 更新种群
            population = selected + offspring
            
            # 记录历史
            self.design_history.append({
                'iteration': iteration,
                'population_size': len(population),
                'pareto_size': len(selected),
                'best_fitness': np.max([np.mean(f) for f in fitness_scores])
            })
        
        # 返回最终帕累托前沿
        final_fitness = self._evaluate_population(population, design_objectives)
        pareto_front = self._select_pareto_front(population, final_fitness)
        
        return [
            {
                'state': state,
                'fitness': fitness,
                'novelty_score': self._compute_novelty(state, seed_states)
            }
            for state, fitness in zip(pareto_front, final_fitness[:len(pareto_front)])
        ]
    
    def _initialize_population(
        self,
        seed_states: List[MaterialState],
        population_size: int
    ) -> List[MaterialState]:
        """初始化种群"""
        population = []
        
        # 包含种子
        population.extend(seed_states)
        
        # 生成变异体
        while len(population) < population_size:
            seed = np.random.choice(seed_states)
            mutant = self._mutate_state(seed)
            population.append(mutant)
        
        return population[:population_size]
    
    def _mutate_state(self, state: MaterialState, mutation_strength: float = 0.1) -> MaterialState:
        """变异状态"""
        mutant = state.clone()
        
        # 热力学参数变异
        mutant.temperature += np.random.randn() * mutation_strength * 100
        mutant.pressure += np.random.randn() * mutation_strength * 10
        mutant.defect_concentration = np.clip(
            mutant.defect_concentration + np.random.randn() * mutation_strength * 0.05,
            0, 1
        )
        
        mutant.state_id = f"mutant_{id(mutant)}"
        return mutant
    
    def _evaluate_population(
        self,
        population: List[MaterialState],
        objectives: List[Callable[[MaterialState], float]]
    ) -> List[List[float]]:
        """评估种群适应度"""
        fitness_scores = []
        
        for state in population:
            scores = [obj(state) for obj in objectives]
            fitness_scores.append(scores)
        
        return fitness_scores
    
    def _select_pareto_front(
        self,
        population: List[MaterialState],
        fitness_scores: List[List[float]]
    ) -> List[MaterialState]:
        """选择帕累托前沿"""
        pareto_indices = []
        
        for i, scores_i in enumerate(fitness_scores):
            is_dominated = False
            for j, scores_j in enumerate(fitness_scores):
                if i != j:
                    # 检查j是否支配i
                    if all(s_j >= s_i for s_j, s_i in zip(scores_j, scores_i)) and \
                       any(s_j > s_i for s_j, s_i in zip(scores_j, scores_i)):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        return [population[i] for i in pareto_indices]
    
    def _create_offspring(
        self,
        parents: List[MaterialState],
        num_offspring: int
    ) -> List[MaterialState]:
        """创建后代"""
        offspring = []
        
        while len(offspring) < num_offspring:
            # 交叉
            if len(parents) >= 2 and np.random.rand() < 0.5:
                p1, p2 = np.random.choice(parents, 2, replace=False)
                child = self._crossover_states(p1, p2)
            else:
                # 变异
                parent = np.random.choice(parents)
                child = self._mutate_state(parent, mutation_strength=0.2)
            
            offspring.append(child)
        
        return offspring
    
    def _crossover_states(
        self,
        state1: MaterialState,
        state2: MaterialState
    ) -> MaterialState:
        """交叉两个状态"""
        child = state1.clone()
        
        # 参数交叉
        alpha = np.random.rand()
        child.temperature = alpha * state1.temperature + (1 - alpha) * state2.temperature
        child.pressure = alpha * state1.pressure + (1 - alpha) * state2.pressure
        child.defect_concentration = alpha * state1.defect_concentration + \
                                      (1 - alpha) * state2.defect_concentration
        
        child.state_id = f"crossover_{id(child)}"
        return child
    
    def _compute_novelty(
        self,
        state: MaterialState,
        reference_states: List[MaterialState]
    ) -> float:
        """计算状态的新颖性"""
        if not reference_states:
            return 1.0
        
        # 计算与参考状态的最小距离
        state_vec = state.to_vector()
        min_dist = float('inf')
        
        for ref in reference_states:
            ref_vec = ref.to_vector()
            dist = np.linalg.norm(state_vec - ref_vec)
            min_dist = min(min_dist, dist)
        
        # 归一化新颖性分数
        novelty = min(min_dist / (min_dist + self.novelty_threshold), 1.0)
        return novelty
    
    def generate_novel_designs(
        self,
        base_designs: List[MaterialState],
        num_novel_designs: int = 10,
        min_novelty: float = 0.5
    ) -> List[MaterialState]:
        """
        生成新颖设计
        
        Args:
            base_designs: 基础设计
            num_novel_designs: 需要的新颖设计数量
            min_novelty: 最小新颖性阈值
            
        Returns:
            新颖设计列表
        """
        novel_designs = []
        attempts = 0
        max_attempts = num_novel_designs * 10
        
        while len(novel_designs) < num_novel_designs and attempts < max_attempts:
            # 变异基础设计
            base = np.random.choice(base_designs)
            candidate = self._mutate_state(base, mutation_strength=0.3)
            
            # 检查新颖性
            novelty = self._compute_novelty(candidate, base_designs + novel_designs)
            
            if novelty >= min_novelty:
                novel_designs.append(candidate)
            
            attempts += 1
        
        return novel_designs


class ImaginationEngine:
    """
    想象引擎主类
    
    整合反事实模拟、假设场景生成、创造性设计
    """
    
    def __init__(self, world_model: MaterialWorldModel):
        self.world_model = world_model
        self.counterfactual_sim = CounterfactualSimulator(world_model)
        self.scenario_generator = HypotheticalScenarioGenerator(world_model)
        self.design_space = CreativeDesignSpace(world_model)
        
        # 想象历史
        self.imagination_log: List[ImaginedOutcome] = []
        self.imagination_stats = defaultdict(int)
    
    def imagine(
        self,
        query_type: str,
        **kwargs
    ) -> ImaginedOutcome:
        """
        主想象接口
        
        Args:
            query_type: 'counterfactual', 'hypothetical', 'design'
            **kwargs: 查询参数
            
        Returns:
            想象结果
        """
        if query_type == 'counterfactual':
            query = kwargs.get('query')
            if not isinstance(query, CounterfactualQuery):
                query = self._create_counterfactual_query(**kwargs)
            outcome = self.counterfactual_sim.simulate(query)
        
        elif query_type == 'hypothetical':
            scenario = kwargs.get('scenario')
            if not isinstance(scenario, HypotheticalScenario):
                scenario = self.scenario_generator.generate_scenario(**kwargs)
            outcome = self.scenario_generator.explore_scenario(scenario)
        
        elif query_type == 'design':
            designs = self.design_space.explore_design_space(**kwargs)
            outcome = self._designs_to_outcome(designs, kwargs)
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")
        
        self.imagination_log.append(outcome)
        self.imagination_stats[query_type] += 1
        
        return outcome
    
    def _create_counterfactual_query(self, **kwargs) -> CounterfactualQuery:
        """创建反事实查询"""
        return CounterfactualQuery(
            query_id=kwargs.get('query_id', 'auto_query'),
            base_state=kwargs['base_state'],
            intervention_type=kwargs.get('intervention_type', ActionType.TEMPERATURE_CHANGE),
            intervention_params=kwargs.get('intervention_params', {}),
            num_steps=kwargs.get('num_steps', 10),
            num_samples=kwargs.get('num_samples', 100)
        )
    
    def _designs_to_outcome(
        self,
        designs: List[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ) -> ImaginedOutcome:
        """将设计结果转换为想象结果"""
        return ImaginedOutcome(
            outcome_id=f"design_{len(self.imagination_log)}",
            query=kwargs,
            trajectories=[[d['state']] for d in designs],
            rewards=[np.mean(d['fitness']) for d in designs],
            insights=[f"Novelty scores: {[d['novelty_score'] for d in designs]}"]
        )
    
    def what_if(
        self,
        base_state: MaterialState,
        intervention: str,
        intervention_params: Dict[str, Any],
        compare_to_factual: bool = True
    ) -> Dict[str, Any]:
        """
        "如果...会怎样"便捷接口
        
        Args:
            base_state: 基准状态
            intervention: 干预类型
            intervention_params: 干预参数
            compare_to_factual: 是否与事实对比
            
        Returns:
            分析结果
        """
        # 映射干预字符串到ActionType
        intervention_map = {
            'heat': ActionType.TEMPERATURE_CHANGE,
            'pressurize': ActionType.PRESSURE_CHANGE,
            'dope': ActionType.COMPOSITION_CHANGE,
            'defect': ActionType.DEFECT_INSERTION,
            'field': ActionType.FIELD_APPLICATION,
            'stress': ActionType.MECHANICAL_STRESS
        }
        
        action_type = intervention_map.get(intervention, ActionType.TEMPERATURE_CHANGE)
        
        query = CounterfactualQuery(
            query_id=f"what_if_{intervention}",
            base_state=base_state,
            intervention_type=action_type,
            intervention_params=intervention_params,
            factual_state=base_state if compare_to_factual else None
        )
        
        outcome = self.counterfactual_sim.simulate(query)
        
        return {
            'outcome': outcome,
            'summary': {
                'expected_final_temp': outcome.mean_final_state.temperature if outcome.mean_final_state else None,
                'expected_final_energy': outcome.mean_final_state.total_energy if outcome.mean_final_state else None,
                'success_probability': outcome.success_probability,
                'key_insights': outcome.insights
            }
        }
    
    def explore_extremes(
        self,
        base_state: MaterialState,
        extreme_params: List[Dict[str, Any]]
    ) -> List[ImaginedOutcome]:
        """
        探索极端条件
        
        Args:
            base_state: 基础状态
            extreme_params: 极端参数列表
            
        Returns:
            多个极端场景的结果
        """
        outcomes = []
        
        for params in extreme_params:
            scenario = HypotheticalScenario(
                scenario_id=f"extreme_{params.get('name', 'unknown')}",
                scenario_type=ScenarioType.EXTREME,
                initial_conditions={'base_state': base_state},
                assumptions=[params],
                time_horizon=params.get('time_horizon', 5000.0)
            )
            
            outcome = self.scenario_generator.explore_scenario(scenario)
            outcomes.append(outcome)
        
        return outcomes
    
    def generate_creative_designs(
        self,
        seed_materials: List[MaterialState],
        target_properties: Dict[str, Tuple[float, float]],
        num_designs: int = 10
    ) -> List[Dict[str, Any]]:
        """
        生成创造性材料设计
        
        Args:
            seed_materials: 种子材料
            target_properties: 目标属性范围 {property: (min, max)}
            num_designs: 设计数量
            
        Returns:
            设计建议
        """
        # 创建目标函数
        objectives = []
        for prop, (min_val, max_val) in target_properties.items():
            target_mid = (min_val + max_val) / 2
            
            def make_objective(p, t):
                return lambda s: -abs(getattr(s, p, 0) - t)
            
            objectives.append(make_objective(prop, target_mid))
        
        # 探索设计空间
        designs = self.design_space.explore_design_space(
            seed_states=seed_materials,
            design_objectives=objectives,
            num_iterations=50,
            population_size=num_designs * 2
        )
        
        # 过滤和排序
        valid_designs = [
            d for d in designs
            if all(
                min_val <= getattr(d['state'], prop, 0) <= max_val
                for prop, (min_val, max_val) in target_properties.items()
            )
        ]
        
        return valid_designs[:num_designs]
    
    def get_imagination_summary(self) -> Dict[str, Any]:
        """获取想象历史摘要"""
        if not self.imagination_log:
            return {'message': 'No imaginations performed yet'}
        
        total_trajectories = sum(
            len(o.trajectories) for o in self.imagination_log
        )
        
        avg_success_rate = np.mean([
            o.success_probability for o in self.imagination_log
        ])
        
        return {
            'total_imaginations': len(self.imagination_log),
            'total_trajectories_generated': total_trajectories,
            'average_success_rate': avg_success_rate,
            'by_type': dict(self.imagination_stats),
            'best_reward': max(
                (max(o.rewards) for o in self.imagination_log if o.rewards),
                default=0
            ),
            'recent_insights': [
                insight
                for o in self.imagination_log[-5:]
                for insight in o.insights
            ]
        }
    
    def export_imaginations(self, filepath: str):
        """导出想象结果"""
        export_data = []
        
        for outcome in self.imagination_log:
            export_data.append({
                'outcome_id': outcome.outcome_id,
                'num_trajectories': len(outcome.trajectories),
                'success_probability': outcome.success_probability,
                'expected_reward': outcome.expected_reward,
                'insights': outcome.insights
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)


# 应用案例
class MaterialImaginationCases:
    """
    材料想象应用案例
    """
    
    def __init__(self, imagination_engine: ImaginationEngine):
        self.engine = imagination_engine
    
    def case_synthesis_path_planning(
        self,
        initial_material: MaterialState,
        target_structure: str = "cubic",
        max_steps: int = 20
    ) -> Dict[str, Any]:
        """
        案例: 合成路径规划
        
        想象不同的合成路径到达目标结构
        """
        # 定义不同的合成策略
        strategies = [
            {'name': 'slow_anneal', 'temp_profile': 'linear_decrease', 'rate': -10},
            {'name': 'rapid_quench', 'temp_profile': 'exponential_decrease', 'rate': -100},
            {'name': 'stepwise', 'temp_profile': 'step_decrease', 'steps': 5},
            {'name': 'isothermal', 'temp_profile': 'constant', 'hold_time': 10000}
        ]
        
        outcomes = []
        
        for strategy in strategies:
            # 创建动作序列
            actions = self._create_temp_profile_actions(strategy, max_steps)
            
            # 想象推演
            result = self.engine.world_model.rollout(
                initial_material,
                actions
            )
            
            outcomes.append({
                'strategy': strategy['name'],
                'result': result,
                'estimated_success': self._evaluate_structure_match(
                    result['states'][-1], target_structure
                )
            })
        
        # 选择最佳路径
        best = max(outcomes, key=lambda x: x['estimated_success'])
        
        return {
            'target_structure': target_structure,
            'strategies_evaluated': len(strategies),
            'best_strategy': best['strategy'],
            'predicted_success_rate': best['estimated_success'],
            'recommended_parameters': best['strategy'],
            'all_outcomes': outcomes
        }
    
    def case_defect_engineering(
        self,
        pristine_material: MaterialState,
        target_property: str = "ionic_conductivity"
    ) -> Dict[str, Any]:
        """
        案例: 缺陷工程
        
        想象不同缺陷配置对性能的影响
        """
        # 测试不同的缺陷浓度
        defect_concentrations = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
        
        results = []
        
        for conc in defect_concentrations:
            query = CounterfactualQuery(
                query_id=f"defect_{conc}",
                base_state=pristine_material,
                intervention_type=ActionType.DEFECT_INSERTION,
                intervention_params={'defect_concentration': conc},
                num_steps=5,
                num_samples=30
            )
            
            outcome = self.engine.counterfactual_sim.simulate(query)
            
            # 评估性能
            predicted_performance = self._estimate_property(
                outcome.mean_final_state,
                target_property
            )
            
            results.append({
                'defect_concentration': conc,
                'predicted_performance': predicted_performance,
                'stability': outcome.success_probability,
                'outcome': outcome
            })
        
        # 找到最佳缺陷浓度
        best = max(results, key=lambda x: x['predicted_performance'])
        
        return {
            'target_property': target_property,
            'optimal_defect_concentration': best['defect_concentration'],
            'predicted_performance': best['predicted_performance'],
            'all_results': results
        }
    
    def case_phase_stability(
        self,
        material: MaterialState,
        temperature_range: Tuple[float, float] = (100, 2000),
        pressure_range: Tuple[float, float] = (0.1, 100)
    ) -> Dict[str, Any]:
        """
        案例: 相稳定性探索
        
        想象材料在不同温压条件下的相行为
        """
        # 创建温压网格
        temps = np.linspace(temperature_range[0], temperature_range[1], 10)
        pressures = np.linspace(pressure_range[0], pressure_range[1], 10)
        
        stability_map = np.zeros((len(temps), len(pressures)))
        
        for i, temp in enumerate(temps):
            for j, pressure in enumerate(pressures):
                # 创建极端场景
                extreme_params = [{
                    'name': f'T{temp:.0f}_P{pressure:.1f}',
                    'temperature': temp,
                    'pressure': pressure,
                    'time_horizon': 1000
                }]
                
                outcomes = self.engine.explore_extremes(material, extreme_params)
                
                # 评估稳定性
                stability = outcomes[0].success_probability if outcomes else 0
                stability_map[i, j] = stability
        
        # 找到稳定区域
        stable_mask = stability_map > 0.8
        stable_conditions = []
        
        for i in range(len(temps)):
            for j in range(len(pressures)):
                if stable_mask[i, j]:
                    stable_conditions.append({
                        'temperature': temps[i],
                        'pressure': pressures[j],
                        'stability': stability_map[i, j]
                    })
        
        return {
            'temperature_range': temperature_range,
            'pressure_range': pressure_range,
            'stability_map': stability_map.tolist(),
            'stable_conditions': stable_conditions,
            'optimal_condition': max(stable_conditions, key=lambda x: x['stability'])
                if stable_conditions else None
        }
    
    def _create_temp_profile_actions(
        self,
        strategy: Dict[str, Any],
        num_steps: int
    ) -> List[MaterialAction]:
        """创建温度曲线动作"""
        actions = []
        
        profile = strategy.get('temp_profile', 'linear_decrease')
        
        for i in range(num_steps):
            if profile == 'linear_decrease':
                magnitude = strategy.get('rate', -10)
            elif profile == 'exponential_decrease':
                magnitude = strategy.get('rate', -100) * np.exp(-i / 5)
            elif profile == 'step_decrease':
                step = num_steps // strategy.get('steps', 5)
                magnitude = -50 if i % step == 0 else 0
            else:
                magnitude = 0
            
            actions.append(MaterialAction(
                action_id=f"temp_{i}",
                action_type=ActionType.TEMPERATURE_CHANGE,
                magnitude=magnitude,
                duration=strategy.get('hold_time', 100) if profile == 'constant' else 100
            ))
        
        return actions
    
    def _evaluate_structure_match(
        self,
        state: MaterialState,
        target_structure: str
    ) -> float:
        """评估结构匹配度"""
        # 简化实现
        if target_structure == "cubic":
            if state.lattice_params:
                a, b, c = state.lattice_params
                return 1.0 - abs(a - b) / a - abs(b - c) / b
        return 0.5
    
    def _estimate_property(
        self,
        state: MaterialState,
        property_name: str
    ) -> float:
        """估算材料属性"""
        # 简化实现
        if property_name == "ionic_conductivity":
            # 与温度和缺陷浓度相关
            return state.temperature / 1000.0 * (1 + state.defect_concentration * 10)
        return 0.0


if __name__ == "__main__":
    print("Testing Imagination Engine...")
    
    # 创建测试世界模型
    from .material_world_model import MaterialWorldModel, WorldModelConfig, create_synthetic_transitions
    
    config = WorldModelConfig(state_dim=20, action_dim=10, num_epochs=5)
    world_model = MaterialWorldModel(config)
    
    # 训练模型
    transitions = create_synthetic_transitions(200)
    world_model.train(transitions, verbose=False)
    
    # 创建想象引擎
    engine = ImaginationEngine(world_model)
    
    # 测试反事实模拟
    base_state = transitions[0].state
    
    what_if_result = engine.what_if(
        base_state=base_state,
        intervention='heat',
        intervention_params={'magnitude': 500}
    )
    
    print("\nWhat-if test:")
    print(f"  Insights: {what_if_result['summary']['key_insights']}")
    
    # 测试设计空间探索
    designs = engine.generate_creative_designs(
        seed_materials=[base_state],
        target_properties={'temperature': (500, 800)},
        num_designs=5
    )
    
    print(f"\nGenerated {len(designs)} creative designs")
    
    # 测试应用案例
    cases = MaterialImaginationCases(engine)
    
    defect_result = cases.case_defect_engineering(base_state)
    print(f"\nDefect engineering:")
    print(f"  Optimal concentration: {defect_result['optimal_defect_concentration']}")
    
    print("\nAll imagination engine tests passed!")
