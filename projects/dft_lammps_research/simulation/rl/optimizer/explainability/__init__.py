#!/usr/bin/env python3
"""
可解释性模块 - 强化学习材料优化的可解释性工具

包含:
- 注意力可视化
- 优化轨迹分析
- 化学直觉提取
- 反事实解释
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionWeights:
    """注意力权重"""
    layer_name: str
    weights: np.ndarray  # (n_heads, seq_len, seq_len) 或 (n_heads, n_atoms, n_atoms)
    atoms: Optional[List[str]] = None  # 对应的原子类型


class AttentionVisualizer:
    """
    注意力可视化器
    
    可视化图神经网络中的注意力权重，
    帮助理解模型关注哪些原子或结构特征。
    """
    
    def __init__(self):
        self.attention_history = []
    
    def extract_attention(self, model: Any, state: np.ndarray) -> List[AttentionWeights]:
        """
        从模型中提取注意力权重
        
        Args:
            model: 神经网络模型 (需要有注意力层)
            state: 输入状态
            
        Returns:
            注意力权重列表
        """
        attentions = []
        
        # 这里需要模型支持注意力提取
        # 简化实现
        
        return attentions
    
    def visualize_atom_attention(
        self,
        attention: AttentionWeights,
        structure: Any,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        可视化原子级别的注意力
        
        Args:
            attention: 注意力权重
            structure: 晶体结构
            save_path: 保存路径
            
        Returns:
            可视化数据
        """
        # 计算每个原子的重要性
        atom_importance = attention.weights.mean(axis=(0, 1))  # 平均所有头和查询
        
        # 归一化
        atom_importance = (atom_importance - atom_importance.min()) / \
                         (atom_importance.max() - atom_importance.min() + 1e-10)
        
        result = {
            'atom_importance': atom_importance.tolist(),
            'atoms': attention.atoms,
            'top_atoms': np.argsort(atom_importance)[-5:].tolist()
        }
        
        return result
    
    def get_structure_attention_map(
        self,
        attention: AttentionWeights
    ) -> np.ndarray:
        """
        获取结构注意力热图
        
        Returns:
            注意力矩阵
        """
        # 平均所有注意力头
        attention_map = attention.weights.mean(axis=0)
        
        return attention_map
    
    def identify_important_sites(
        self,
        attention: AttentionWeights,
        structure: Any,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        识别重要位点
        
        Args:
            attention: 注意力权重
            structure: 晶体结构
            threshold: 重要性阈值
            
        Returns:
            重要位点列表
        """
        atom_importance = attention.weights.mean(axis=(0, 1))
        
        important_sites = []
        for i, importance in enumerate(atom_importance):
            if importance > threshold * atom_importance.max():
                important_sites.append({
                    'index': i,
                    'atom_type': attention.atoms[i] if attention.atoms else 'Unknown',
                    'importance': float(importance),
                    'position': structure.positions[i] if hasattr(structure, 'positions') else None
                })
        
        return important_sites


class TrajectoryAnalyzer:
    """
    优化轨迹分析器
    
    分析RL优化过程的轨迹，提取有价值的化学洞察。
    """
    
    def __init__(self):
        self.trajectories = []
    
    def add_trajectory(self, trajectory: List[Dict]):
        """添加轨迹"""
        self.trajectories.append(trajectory)
    
    def analyze_action_distribution(self) -> Dict[str, Any]:
        """分析动作分布"""
        action_counts = {}
        
        for traj in self.trajectories:
            for step in traj:
                action_type = step.get('action_type', 'unknown')
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        total = sum(action_counts.values())
        action_freq = {k: v / total for k, v in action_counts.items()}
        
        return {
            'counts': action_counts,
            'frequencies': action_freq,
            'most_common': sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def analyze_reward_progression(self) -> Dict[str, Any]:
        """分析奖励进展"""
        all_rewards = []
        
        for traj in self.trajectories:
            rewards = [step.get('reward', 0) for step in traj]
            all_rewards.append(rewards)
        
        # 计算统计
        max_rewards = [max(r) for r in all_rewards if r]
        final_rewards = [r[-1] for r in all_rewards if r]
        
        return {
            'mean_max_reward': np.mean(max_rewards) if max_rewards else 0,
            'mean_final_reward': np.mean(final_rewards) if final_rewards else 0,
            'improvement_rate': np.mean([
                (r[-1] - r[0]) / max(len(r), 1) for r in all_rewards if r
            ])
        }
    
    def identify_common_patterns(self, min_support: int = 3) -> List[Dict]:
        """
        识别常见优化模式
        
        Args:
            min_support: 最小支持度
            
        Returns:
            发现的模式列表
        """
        patterns = []
        
        # 分析动作序列
        action_sequences = []
        for traj in self.trajectories:
            actions = [step.get('action_type', '') for step in traj]
            action_sequences.append(actions)
        
        # 寻找频繁子序列 (简化实现)
        # 实际应使用序列模式挖掘算法
        
        # 统计常见动作对
        action_pairs = {}
        for seq in action_sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                action_pairs[pair] = action_pairs.get(pair, 0) + 1
        
        # 筛选高频对
        for pair, count in action_pairs.items():
            if count >= min_support:
                patterns.append({
                    'pattern': f"{pair[0]} -> {pair[1]}",
                    'support': count,
                    'type': 'action_sequence'
                })
        
        return sorted(patterns, key=lambda x: x['support'], reverse=True)
    
    def find_convergence_points(self) -> List[Dict]:
        """找到轨迹收敛点"""
        convergence_points = []
        
        for traj in self.trajectories:
            if len(traj) < 10:
                continue
            
            # 寻找奖励变化很小的连续步骤
            rewards = [step.get('reward', 0) for step in traj]
            
            for i in range(5, len(rewards)):
                recent_changes = [abs(rewards[j] - rewards[j-1]) for j in range(i-4, i+1)]
                if max(recent_changes) < 0.01:  # 收敛阈值
                    convergence_points.append({
                        'step': i,
                        'reward': rewards[i],
                        'structure': traj[i].get('structure')
                    })
                    break
        
        return convergence_points


class ChemicalIntuitionExtractor:
    """
    化学直觉提取器
    
    从优化过程中提取可解释的化学规则和直觉。
    """
    
    def __init__(self):
        self.element_correlations = {}
        self.successful_combinations = []
        self.structure_patterns = []
    
    def extract_element_correlations(self, structures: List[Any], rewards: List[float]) -> Dict[str, Any]:
        """
        提取元素相关性
        
        分析哪些元素组合与高奖励相关。
        """
        correlations = {}
        
        # 收集所有元素对
        element_pairs = {}
        
        for struct, reward in zip(structures, rewards):
            composition = struct.get_composition()
            elements = list(composition.keys())
            
            for i, elem1 in enumerate(elements):
                for elem2 in elements[i+1:]:
                    pair = tuple(sorted([elem1, elem2]))
                    if pair not in element_pairs:
                        element_pairs[pair] = []
                    element_pairs[pair].append(reward)
        
        # 计算平均奖励
        for pair, rewards_list in element_pairs.items():
            avg_reward = np.mean(rewards_list)
            correlations[f"{pair[0]}-{pair[1]}"] = {
                'mean_reward': avg_reward,
                'count': len(rewards_list),
                'significance': avg_reward * np.log(len(rewards_list) + 1)  # 加权
            }
        
        # 排序
        sorted_correlations = sorted(
            correlations.items(),
            key=lambda x: x[1]['significance'],
            reverse=True
        )
        
        return {
            'top_pairs': sorted_correlations[:10],
            'all_correlations': correlations
        }
    
    def extract_composition_rules(self, structures: List[Any], rewards: List[float]) -> List[Dict]:
        """
        提取组成规则
        
        例如: "Li含量在20-30%时性能最佳"
        """
        rules = []
        
        # 分析各元素的最佳含量范围
        element_rewards = {}
        
        for struct, reward in zip(structures, rewards):
            composition = struct.get_composition()
            total = sum(composition.values())
            
            for elem, count in composition.items():
                fraction = count / total if total > 0 else 0
                
                if elem not in element_rewards:
                    element_rewards[elem] = []
                element_rewards[elem].append((fraction, reward))
        
        # 找出每个元素的最佳范围
        for elem, data in element_rewards.items():
            fractions, rew = zip(*data)
            fractions = np.array(fractions)
            rew = np.array(rew)
            
            # 找出高奖励对应的范围
            high_reward_mask = rew > np.percentile(rew, 75)
            if np.sum(high_reward_mask) > 0:
                best_fractions = fractions[high_reward_mask]
                
                rules.append({
                    'element': elem,
                    'optimal_range': (float(best_fractions.min()), float(best_fractions.max())),
                    'mean_fraction': float(best_fractions.mean()),
                    'confidence': float(np.mean(rew[high_reward_mask]))
                })
        
        return sorted(rules, key=lambda x: x['confidence'], reverse=True)
    
    def identify_key_structural_features(self, structures: List[Any], rewards: List[float]) -> List[Dict]:
        """识别关键结构特征"""
        features = []
        
        # 分析结构参数与高奖励的关系
        for struct, reward in zip(structures, rewards):
            if reward < np.percentile(rewards, 75):
                continue  # 只分析高奖励结构
            
            # 提取结构特征
            feature_vector = self._extract_structure_features(struct)
            
            features.append({
                'features': feature_vector,
                'reward': reward
            })
        
        # 找出共同特征 (简化)
        if features:
            feature_matrix = np.array([f['features'] for f in features])
            mean_features = feature_matrix.mean(axis=0)
            
            feature_names = ['volume', 'density', 'coordination', 'bond_variance']
            
            return [
                {'feature': name, 'mean_value': float(val)}
                for name, val in zip(feature_names, mean_features)
            ]
        
        return []
    
    def _extract_structure_features(self, structure: Any) -> np.ndarray:
        """提取结构特征向量"""
        features = []
        
        # 体积
        features.append(structure.get_volume() if hasattr(structure, 'get_volume') else 0)
        
        # 密度
        features.append(structure.get_density() if hasattr(structure, 'get_density') else 0)
        
        # 平均配位数 (简化)
        features.append(6.0)  # 占位符
        
        # 键长方差 (简化)
        features.append(0.1)  # 占位符
        
        return np.array(features)
    
    def generate_design_guidelines(self) -> Dict[str, List[str]]:
        """生成设计指导原则"""
        guidelines = {
            'composition': [],
            'structure': [],
            'synthesis': []
        }
        
        # 基于提取的规则生成指导
        # 简化实现
        
        guidelines['composition'].append("保持主族元素与过渡金属比例在1:1到2:1之间")
        guidelines['composition'].append("避免使用含量超过50%的单一元素")
        
        guidelines['structure'].append("确保足够的离子传输通道")
        guidelines['structure'].append("维持适当的晶格常数以平衡稳定性和离子迁移率")
        
        guidelines['synthesis'].append("优先选择固相合成路线")
        
        return guidelines


class CounterfactualExplainer:
    """
    反事实解释器
    
    生成"如果...会怎样"类型的解释，
    帮助理解模型的决策边界。
    """
    
    def __init__(self, model: Any):
        self.model = model
    
    def generate_counterfactual(
        self,
        structure: Any,
        target_reward: float,
        max_changes: int = 3
    ) -> Optional[Dict]:
        """
        生成反事实解释
        
        Args:
            structure: 当前结构
            target_reward: 目标奖励值
            max_changes: 最大允许修改数
            
        Returns:
            反事实解释
        """
        # 当前奖励
        current_state = self._structure_to_state(structure)
        current_reward = self._predict_reward(current_state)
        
        if current_reward >= target_reward:
            return None  # 已经满足目标
        
        # 尝试小的修改
        best_counterfactual = None
        best_reward = current_reward
        
        # 尝试替换元素
        composition = structure.get_composition()
        
        for elem_to_replace in composition:
            for replacement in ['Li', 'Na', 'Mg', 'Al', 'Si']:
                if replacement == elem_to_replace:
                    continue
                
                # 创建修改后的结构
                modified = self._replace_element(structure, elem_to_replace, replacement)
                modified_state = self._structure_to_state(modified)
                modified_reward = self._predict_reward(modified_state)
                
                if modified_reward > best_reward and modified_reward >= target_reward:
                    best_reward = modified_reward
                    best_counterfactual = {
                        'original_structure': structure,
                        'modified_structure': modified,
                        'change': f"Replace {elem_to_replace} with {replacement}",
                        'original_reward': current_reward,
                        'modified_reward': modified_reward,
                        'improvement': modified_reward - current_reward
                    }
        
        return best_counterfactual
    
    def explain_prediction(self, structure: Any) -> Dict[str, Any]:
        """
        解释预测结果
        
        Returns:
            解释详情
        """
        state = self._structure_to_state(structure)
        reward = self._predict_reward(state)
        
        explanation = {
            'predicted_reward': reward,
            'key_factors': []
        }
        
        # 识别关键影响因素 (简化)
        composition = structure.get_composition()
        
        for elem, count in composition.items():
            if count > 0.3:  # 主要组分
                explanation['key_factors'].append(f"高{elem}含量")
        
        return explanation
    
    def _structure_to_state(self, structure: Any) -> np.ndarray:
        """结构转状态向量"""
        # 简化实现
        return np.zeros(128)
    
    def _predict_reward(self, state: np.ndarray) -> float:
        """预测奖励"""
        # 简化实现
        return 0.0
    
    def _replace_element(self, structure: Any, old_elem: str, new_elem: str):
        """替换元素"""
        # 简化实现
        return structure


class ExplainabilityReport:
    """可解释性报告生成器"""
    
    def __init__(
        self,
        attention_viz: AttentionVisualizer,
        trajectory_analyzer: TrajectoryAnalyzer,
        intuition_extractor: ChemicalIntuitionExtractor,
        counterfactual_explainer: CounterfactualExplainer
    ):
        self.attention_viz = attention_viz
        self.trajectory_analyzer = trajectory_analyzer
        self.intuition_extractor = intuition_extractor
        self.counterfactual_explainer = counterfactual_explainer
    
    def generate_report(self) -> Dict[str, Any]:
        """生成完整可解释性报告"""
        report = {
            'summary': {
                'n_trajectories_analyzed': len(self.trajectory_analyzer.trajectories),
            },
            'optimization_analysis': self.trajectory_analyzer.analyze_reward_progression(),
            'action_patterns': self.trajectory_analyzer.identify_common_patterns(),
            'design_guidelines': self.intuition_extractor.generate_design_guidelines(),
        }
        
        return report
    
    def export_html(self, filename: str):
        """导出HTML报告"""
        report = self.generate_report()
        
        html = f"""
        <html>
        <head><title>RL材料优化可解释性报告</title></head>
        <body>
            <h1>强化学习材料优化可解释性报告</h1>
            <h2>优化分析</h2>
            <p>平均最大奖励: {report['optimization_analysis'].get('mean_max_reward', 0):.4f}</p>
            <p>改进率: {report['optimization_analysis'].get('improvement_rate', 0):.4f}</p>
            
            <h2>常见模式</h2>
            <ul>
        """
        
        for pattern in report['action_patterns'][:5]:
            html += f"<li>{pattern['pattern']} (支持度: {pattern['support']})</li>"
        
        html += """
            </ul>
        </body>
        </html>
        """
        
        with open(filename, 'w') as f:
            f.write(html)
        
        logger.info(f"Report exported to {filename}")
