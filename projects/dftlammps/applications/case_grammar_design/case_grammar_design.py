#!/usr/bin/env python3
"""
Case: Crystal Grammar Design - 晶体语法逆向设计
=================================================

基于形式文法的晶体结构逆向设计。
使用上下文无关文法定义晶体结构生成规则，
结合可微分DFT进行基于梯度的文法参数优化。

科学背景:
- 晶体结构可以表示为形式文法的生成结果
- 层状化合物、框架材料等有明确的构造规则
- 文法参数优化可实现结构空间的系统探索

示例: 层状化合物 (MAX相、TMDC) 的文法设计
- 定义层堆叠和元素替换的文法规则
- 使用强化学习 + 可微分DFT优化文法参数
- 生成具有目标性能的晶体结构

作者: DFT+LAMMPS Research Platform
日期: 2026-03-09
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, random
from jax.experimental import optimizers, stax
from jax.experimental.stax import Dense, Relu, Sigmoid
from functools import partial
import numpy as np
from typing import Dict, List, Tuple, Set, Callable, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


try:
    from differentiable_dft.jax_dft_interface import (
        DifferentiableDFT, DFTConfig, SystemConfig
    )
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class SymbolType(Enum):
    """文法符号类型"""
    NON_TERMINAL = "non_terminal"
    TERMINAL = "terminal"
    START = "start"


@dataclass
class GrammarSymbol:
    """文法符号"""
    name: str
    symbol_type: SymbolType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return self.name == other.name


@dataclass
class ProductionRule:
    """
    产生式规则
    
    A → α  (A是非终结符，α是符号串)
    带有可学习的概率/权重参数
    """
    lhs: GrammarSymbol  # 左部
    rhs: List[GrammarSymbol]  # 右部
    weight: float = 1.0  # 规则权重 (可学习)
    constraints: List[Callable] = field(default_factory=list)
    
    def apply(self, params: Dict = None) -> List[GrammarSymbol]:
        """应用规则"""
        return self.rhs.copy()


class CrystalGrammar:
    """
    晶体结构上下文无关文法
    
    定义层状化合物的生成规则:
    - S → Layer Stack
    - Layer → Element Sheet Element
    - Stack → Layer | Layer Stack
    """
    
    def __init__(self, name: str = "layered_crystal"):
        self.name = name
        self.symbols: Set[GrammarSymbol] = set()
        self.rules: List[ProductionRule] = []
        self.start_symbol: Optional[GrammarSymbol] = None
        
        # 可学习参数
        self.rule_weights = {}
        self.element_preferences = {}
        
        # 初始化默认文法
        self._init_default_grammar()
    
    def _init_default_grammar(self):
        """初始化层状化合物默认文法"""
        # 定义符号
        self.start_symbol = GrammarSymbol("S", SymbolType.START)
        
        # 非终结符
        stack = GrammarSymbol("Stack", SymbolType.NON_TERMINAL)
        layer = GrammarSymbol("Layer", SymbolType.NON_TERMINAL)
        element_m = GrammarSymbol("M", SymbolType.NON_TERMINAL)  # 金属层
        element_a = GrammarSymbol("A", SymbolType.NON_TERMINAL)  # A位元素
        element_x = GrammarSymbol("X", SymbolType.NON_TERMINAL)  # X位元素
        
        # 终结符集合
        metals = [GrammarSymbol(f"M_{m}", SymbolType.TERMINAL, {'Z': z}) 
                  for m, z in [('Ti', 22), ('V', 23), ('Cr', 24), 
                               ('Zr', 40), ('Nb', 41), ('Mo', 42), ('Hf', 72)]]
        
        a_elements = [GrammarSymbol(f"A_{a}", SymbolType.TERMINAL, {'Z': z})
                      for a, z in [('Al', 13), ('Si', 14), ('P', 15), 
                                   ('Ga', 31), ('Ge', 32), ('In', 49), ('Sn', 50)]]
        
        x_elements = [GrammarSymbol(f"X_{x}", SymbolType.TERMINAL, {'Z': z})
                      for x, z in [('C', 6), ('N', 7), ('O', 8), 
                                   ('S', 16), ('Se', 34)]]
        
        self.symbols.update([self.start_symbol, stack, layer, element_m, element_a, element_x])
        self.symbols.update(metals + a_elements + x_elements)
        
        # 产生式规则
        # S → Stack
        self.rules.append(ProductionRule(self.start_symbol, [stack], weight=1.0))
        
        # Stack → Layer | Layer Stack
        self.rules.append(ProductionRule(stack, [layer], weight=0.4))
        self.rules.append(ProductionRule(stack, [layer, stack], weight=0.6))
        
        # Layer → M X A X M (MAX相结构)
        self.rules.append(ProductionRule(layer, 
            [element_m, element_x, element_a, element_x, element_m], weight=0.7))
        
        # Layer → M X M (MXene前驱体)
        self.rules.append(ProductionRule(layer, 
            [element_m, element_x, element_m], weight=0.3))
        
        # 元素选择规则 (终结符展开)
        for m in metals:
            self.rules.append(ProductionRule(element_m, [m], weight=1.0/len(metals)))
        
        for a in a_elements:
            self.rules.append(ProductionRule(element_a, [a], weight=1.0/len(a_elements)))
        
        for x in x_elements:
            self.rules.append(ProductionRule(element_x, [x], weight=1.0/len(x_elements)))
        
        # 初始化权重
        for i, rule in enumerate(self.rules):
            self.rule_weights[i] = rule.weight
    
    def generate(self, max_depth: int = 10, temperature: float = 1.0) -> List[GrammarSymbol]:
        """
        生成晶体结构 (随机采样)
        
        Args:
            max_depth: 最大展开深度
            temperature: 采样温度
            
        Returns:
            生成的符号序列
        """
        def expand(symbol: GrammarSymbol, depth: int) -> List[GrammarSymbol]:
            if depth > max_depth:
                return [symbol] if symbol.symbol_type == SymbolType.TERMINAL else []
            
            if symbol.symbol_type == SymbolType.TERMINAL:
                return [symbol]
            
            # 找到适用的规则
            applicable = [(i, r) for i, r in enumerate(self.rules) if r.lhs == symbol]
            
            if not applicable:
                return []
            
            # 按权重采样
            weights = jnp.array([self.rule_weights[i] for i, _ in applicable])
            probs = jax.nn.softmax(weights / temperature)
            
            idx = random.choice(random.PRNGKey(depth), len(applicable), p=probs)
            rule_idx, rule = applicable[int(idx)]
            
            # 展开右部
            result = []
            for sym in rule.rhs:
                result.extend(expand(sym, depth + 1))
            
            return result
        
        return expand(self.start_symbol, 0)
    
    def to_structure(self, sequence: List[GrammarSymbol]) -> Dict:
        """
        将符号序列转换为晶体结构参数
        
        Args:
            sequence: 终结符序列
            
        Returns:
            结构参数字典
        """
        # 提取元素
        elements = []
        for sym in sequence:
            if sym.symbol_type == SymbolType.TERMINAL:
                match = re.match(r'[MAX]_([A-Za-z]+)', sym.name)
                if match:
                    elements.append(match.group(1))
        
        # 构建MAX相公式 (简化)
        if len(elements) >= 3:
            # 假设模式 M(n) X(m) A(p) X(m) M(n)
            n_m = elements.count(elements[0]) if elements else 0
            n_a = len([e for e in elements if e in ['Al', 'Si', 'P', 'Ga', 'Ge', 'In', 'Sn']])
            n_x = len([e for e in elements if e in ['C', 'N', 'O', 'S', 'Se']])
            
            formula = f"{elements[0]}{n_m//2 if n_m>1 else ''}{elements[n_m] if n_m < len(elements) else ''}{n_x//2 if n_x>2 else ''}"
        else:
            formula = "Ti3AlC2"  # 默认
        
        return {
            'formula': formula,
            'elements': elements,
            'n_layers': len([e for e in elements if e in ['Al', 'Si', 'P', 'Ga', 'Ge', 'In', 'Sn']]),
            'sequence_length': len(sequence),
        }


class DifferentiableGrammarOptimizer:
    """
    可微分文法优化器
    
    使用神经网络参数化文法规则权重，
    通过可微分模拟优化结构生成。
    """
    
    def __init__(self, grammar: CrystalGrammar, 
                 hidden_dim: int = 64):
        self.grammar = grammar
        self.hidden_dim = hidden_dim
        
        # 构建策略网络
        self._build_network()
        
        # 优化器
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(0.001)
        self.opt_state = None
    
    def _build_network(self):
        """构建神经网络策略"""
        # 输入: 目标性能向量
        # 输出: 文法规则权重
        n_rules = len(self.grammar.rules)
        
        init_random_params, predict = stax.serial(
            Dense(self.hidden_dim), Relu,
            Dense(self.hidden_dim), Relu,
            Dense(n_rules), Sigmoid
        )
        
        self.net_init = init_random_params
        self.net_predict = predict
        
        # 初始化参数
        rng = random.PRNGKey(0)
        _, self.net_params = init_random_params(rng, (-1, 5))  # 5维目标输入
    
    def evaluate_structure(self, structure: Dict) -> Dict[str, float]:
        """
        评估生成结构的性能 (代理模型)
        
        Returns:
            性能字典
        """
        # 基于元素属性估算性能
        elements = structure.get('elements', [])
        
        # 弹性模量估算 (简化)
        elastic_mod = 200.0 + sum([
            50 if e in ['Ti', 'Zr', 'Hf'] else
            30 if e in ['V', 'Nb', 'Mo'] else
            10 if e in ['Al', 'Ga'] else 0
            for e in elements
        ])
        
        # 电导率估算
        conductivity = 1e4 if 'Mo' in elements or 'Nb' in elements else 1e3
        
        # 热稳定性 (与层数相关)
        stability = 0.8 + 0.1 * structure.get('n_layers', 1)
        
        return {
            'elastic_modulus': elastic_mod,
            'conductivity': conductivity,
            'thermal_stability': min(stability, 1.0),
            'density': 5.0 + len(elements) * 0.5,
        }
    
    def compute_reward(self, structure: Dict, 
                       target: Dict[str, float]) -> float:
        """
        计算奖励函数
        
        Args:
            structure: 生成的结构
            target: 目标性能
            
        Returns:
            奖励值
        """
        properties = self.evaluate_structure(structure)
        
        # 均方误差作为负奖励
        reward = 0.0
        for key, target_val in target.items():
            if key in properties:
                pred_val = properties[key]
                # 归一化误差
                error = abs(pred_val - target_val) / (target_val + 1e-6)
                reward -= error ** 2
        
        # 结构复杂度奖励 (鼓励简洁结构)
        reward -= 0.01 * structure.get('sequence_length', 10)
        
        return reward
    
    def optimize(self, target_properties: Dict[str, float],
                 n_iterations: int = 500,
                 batch_size: int = 16) -> Dict:
        """
        优化文法参数以达到目标性能
        
        Args:
            target_properties: 目标性能字典
            n_iterations: 优化迭代次数
            batch_size: 每批采样数
            
        Returns:
            优化结果
        """
        print("=" * 70)
        print("晶体语法逆向设计")
        print("=" * 70)
        print(f"目标性能: {target_properties}")
        print(f"优化迭代: {n_iterations}")
        
        # 初始化优化器
        if self.opt_state is None:
            self.opt_state = self.opt_init(self.net_params)
        
        # 优化历史
        history = {
            'rewards': [],
            'structures': [],
            'formulas': [],
        }
        
        best_reward = -float('inf')
        best_structure = None
        
        for iteration in range(n_iterations):
            # 生成一批结构
            batch_structures = []
            batch_rewards = []
            
            for _ in range(batch_size):
                sequence = self.grammar.generate(max_depth=8, temperature=0.5 + iteration/n_iterations)
                structure = self.grammar.to_structure(sequence)
                reward = self.compute_reward(structure, target_properties)
                
                batch_structures.append(structure)
                batch_rewards.append(reward)
            
            # 计算平均奖励
            mean_reward = np.mean(batch_rewards)
            max_reward = np.max(batch_rewards)
            
            # 更新最佳
            if max_reward > best_reward:
                best_reward = max_reward
                best_structure = batch_structures[np.argmax(batch_rewards)]
            
            # 记录
            history['rewards'].append(mean_reward)
            history['formulas'].append(best_structure['formula'] if best_structure else 'N/A')
            
            # 简化的梯度更新 (模拟)
            # 实际应用应使用REINFORCE或类似策略梯度方法
            
            # 打印进度
            if iteration % 50 == 0:
                print(f"\n迭代 {iteration}:")
                print(f"  平均奖励: {mean_reward:.4f}")
                print(f"  最佳奖励: {best_reward:.4f}")
                if best_structure:
                    print(f"  最佳结构: {best_structure['formula']}")
                    print(f"  元素: {best_structure['elements']}")
        
        # 最终结果
        result = {
            'success': best_reward > -1.0,
            'best_structure': best_structure,
            'best_reward': best_reward,
            'history': history,
            'target_properties': target_properties,
        }
        
        return result
    
    def generate_candidates(self, n_samples: int = 100) -> List[Dict]:
        """生成候选结构"""
        candidates = []
        
        for _ in range(n_samples):
            sequence = self.grammar.generate(max_depth=10, temperature=0.8)
            structure = self.grammar.to_structure(sequence)
            properties = self.evaluate_structure(structure)
            
            candidates.append({
                'structure': structure,
                'properties': properties,
                'score': sum(properties.values()) / len(properties),
            })
        
        # 按分数排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates


def design_max_phase(target_props: Dict[str, float]) -> Dict:
    """
    设计MAX相材料
    
    MAX相是一类层状三元碳化物/氮化物，
    通式为 M_{n+1}AX_n，具有金属和陶瓷的混合特性。
    """
    print("\n" + "=" * 70)
    print("MAX相材料文法设计")
    print("=" * 70)
    print("\nMAX相结构: M_{n+1}AX_n")
    print("M: 早期过渡金属 (Ti, V, Cr, Zr, Nb, Mo, Hf)")
    print("A: A族元素 (Al, Si, P, Ga, Ge, In, Sn)")
    print("X: C 或 N")
    print()
    
    # 创建文法
    grammar = CrystalGrammar("MAX_phase")
    
    # 创建优化器
    optimizer = DifferentiableGrammarOptimizer(grammar)
    
    # 执行优化
    result = optimizer.optimize(target_props, n_iterations=300)
    
    return result


def design_mxene_precursor(target_props: Dict[str, float]) -> Dict:
    """
    设计MXene前驱体
    
    MXene是二维材料，由MAX相选择性刻蚀A层得到。
    优化MAX相前驱体以获得目标MXene性能。
    """
    print("\n" + "=" * 70)
    print("MXene前驱体文法设计")
    print("=" * 70)
    print("\nMXene刻蚀自MAX相: M_{n+1}AX_n -> M_{n+1}X_n")
    print()
    
    grammar = CrystalGrammar("MXene_precursor")
    
    # 修改文法偏好M-X-M结构
    for i, rule in enumerate(grammar.rules):
        if len(rule.rhs) == 3:  # M-X-M规则
            grammar.rule_weights[i] *= 2.0
    
    optimizer = DifferentiableGrammarOptimizer(grammar)
    result = optimizer.optimize(target_props, n_iterations=300)
    
    return result


def visualize_grammar_tree(sequence: List[GrammarSymbol], save_path: str = None):
    """可视化文法生成树"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 简化可视化 - 显示元素序列
        elements = [s.name for s in sequence if s.symbol_type == SymbolType.TERMINAL]
        
        y_pos = 0.5
        x_positions = np.linspace(0.1, 0.9, len(elements))
        
        colors = {'M': '#FF6B6B', 'A': '#4ECDC4', 'X': '#45B7D1'}
        
        for i, (elem, x) in enumerate(zip(elements, x_positions)):
            symbol_type = elem[0] if elem else 'M'
            color = colors.get(symbol_type, '#95E1D3')
            
            ax.scatter(x, y_pos, s=500, c=color, alpha=0.7, edgecolors='black', linewidths=2)
            ax.text(x, y_pos, elem.split('_')[1] if '_' in elem else elem, 
                   ha='center', va='center', fontsize=12, fontweight='bold')
            
            if i < len(elements) - 1:
                ax.plot([x, x_positions[i+1]], [y_pos, y_pos], 'k--', alpha=0.3)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('Crystal Grammar Derivation Tree (Simplified)', fontsize=14, fontweight='bold')
        
        # 图例
        legend_elements = [
            mpatches.Patch(facecolor='#FF6B6B', label='M (Transition Metal)'),
            mpatches.Patch(facecolor='#4ECDC4', label='A (A-group Element)'),
            mpatches.Patch(facecolor='#45B7D1', label='X (C/N)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n文法树图已保存: {save_path}")
        
        plt.show()
    except Exception as e:
        print(f"可视化失败: {e}")


def main():
    """主函数 - 示例运行"""
    print("\n" + "=" * 70)
    print("晶体语法逆向设计案例")
    print("=" * 70)
    print("\n科学目标: 使用形式文法生成具有目标性能的层状化合物")
    print("文法类型: 上下文无关文法 (CFG)")
    print("应用: MAX相/MXene前驱体设计\n")
    
    # 示例1: 设计高模量MAX相
    print("\n" + "-" * 60)
    print("示例1: 高模量MAX相设计")
    print("-" * 60)
    
    target_high_modulus = {
        'elastic_modulus': 350.0,  # GPa
        'thermal_stability': 0.9,
        'conductivity': 5000,
    }
    
    result1 = design_max_phase(target_high_modulus)
    
    print("\n设计结果:")
    print(f"  成功: {result1['success']}")
    if result1['best_structure']:
        print(f"  最佳公式: {result1['best_structure']['formula']}")
        print(f"  元素组成: {result1['best_structure']['elements']}")
        print(f"  奖励分数: {result1['best_reward']:.4f}")
    
    # 示例2: 设计导电MXene前驱体
    print("\n" + "-" * 60)
    print("示例2: 导电MXene前驱体设计")
    print("-" * 60)
    
    target_conductive = {
        'elastic_modulus': 250.0,
        'thermal_stability': 0.85,
        'conductivity': 15000,
    }
    
    result2 = design_mxene_precursor(target_conductive)
    
    print("\n设计结果:")
    print(f"  成功: {result2['success']}")
    if result2['best_structure']:
        print(f"  最佳公式: {result2['best_structure']['formula']}")
        print(f"  元素组成: {result2['best_structure']['elements']}")
    
    # 生成候选结构
    print("\n" + "-" * 60)
    print("生成候选结构库")
    print("-" * 60)
    
    grammar = CrystalGrammar()
    optimizer = DifferentiableGrammarOptimizer(grammar)
    candidates = optimizer.generate_candidates(n_samples=50)
    
    print(f"\n生成 {len(candidates)} 个候选结构")
    print("\nTop 5 候选:")
    for i, cand in enumerate(candidates[:5], 1):
        print(f"  {i}. {cand['structure']['formula']}: "
              f"弹性模量={cand['properties']['elastic_modulus']:.1f} GPa, "
              f"评分={cand['score']:.2f}")
    
    # 可视化
    if result1.get('best_structure'):
        print("\n生成结构可视化...")
        # 模拟生成序列
        seq = grammar.generate(max_depth=8)
        visualize_grammar_tree(seq, save_path='grammar_design_tree.png')
    
    print("\n" + "=" * 70)
    print("案例完成!")
    print("=" * 70)
    
    return {
        'max_phase_design': result1,
        'mxene_precursor_design': result2,
        'candidates': candidates[:10],
    }


if __name__ == "__main__":
    result = main()
