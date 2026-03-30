"""
电池性能因果图自动发现
演示如何使用因果发现算法分析电池性能影响因素
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import warnings


def generate_battery_data(
    n_samples: int = 1000,
    noise_level: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    生成电池性能数据
    
    真实的因果结构：
    - 材料类型 -> 容量
    - 电解液浓度 -> 离子电导率
    - 温度 -> 内阻, 循环寿命
    - 内阻 -> 能量密度
    - 循环寿命, 能量密度 -> 综合性能
    """
    np.random.seed(random_state)
    
    # 外生变量
    material_type = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])
    electrolyte_conc = np.random.uniform(0.5, 2.0, n_samples)
    temperature = np.random.normal(25, 10, n_samples)
    pressure = np.random.normal(1.0, 0.1, n_samples)
    
    # 内生变量
    # 容量受材料类型影响
    capacity = 200 + 50 * material_type - 0.5 * (temperature - 25)**2 + noise_level * np.random.randn(n_samples)
    capacity = np.clip(capacity, 100, 400)
    
    # 离子电导率受电解液浓度和温度影响
    ionic_conductivity = 0.1 * electrolyte_conc + 0.005 * temperature - 0.0001 * (temperature - 30)**2
    ionic_conductivity += noise_level * np.random.randn(n_samples)
    ionic_conductivity = np.clip(ionic_conductivity, 0.01, 0.5)
    
    # 内阻受温度影响
    internal_resistance = 0.1 + 0.001 * (temperature - 25)**2 - 0.02 * material_type
    internal_resistance += noise_level * np.random.randn(n_samples) * 0.01
    internal_resistance = np.clip(internal_resistance, 0.02, 0.3)
    
    # 循环寿命受温度和材料类型影响
    cycle_life = 1000 - 5 * np.abs(temperature - 25) + 200 * material_type
    cycle_life += noise_level * np.random.randn(n_samples) * 50
    cycle_life = np.clip(cycle_life, 100, 2000)
    
    # 能量密度受容量和内阻影响
    energy_density = capacity * 0.8 - internal_resistance * 100
    energy_density += noise_level * np.random.randn(n_samples) * 10
    energy_density = np.clip(energy_density, 50, 300)
    
    # 综合性能
    overall_performance = (
        0.3 * capacity / 400 +
        0.2 * cycle_life / 2000 +
        0.3 * energy_density / 300 +
        0.2 * ionic_conductivity / 0.5
    ) * 100
    overall_performance += noise_level * np.random.randn(n_samples) * 5
    overall_performance = np.clip(overall_performance, 0, 100)
    
    # 创建DataFrame
    data = pd.DataFrame({
        'material_type': material_type,
        'electrolyte_conc': electrolyte_conc,
        'temperature': temperature,
        'pressure': pressure,
        'capacity': capacity,
        'ionic_conductivity': ionic_conductivity,
        'internal_resistance': internal_resistance,
        'cycle_life': cycle_life,
        'energy_density': energy_density,
        'overall_performance': overall_performance
    })
    
    return data


def discover_battery_causal_graph(
    data: pd.DataFrame,
    algorithm: str = "pc",
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    发现电池性能的因果图
    
    Args:
        data: 电池性能数据
        algorithm: 因果发现算法
        alpha: 显著性水平
    
    Returns:
        因果发现结果
    """
    import sys
    sys.path.insert(0, '/root/.openclaw/workspace/dftlammps')
    
    from neuro_symbolic.causal_discovery import CausalDiscovery, IndependenceTest
    
    # 准备数据
    X = data.values
    var_names = list(data.columns)
    
    print(f"数据形状: {X.shape}")
    print(f"变量: {var_names}")
    
    # 创建因果发现器
    cd = CausalDiscovery()
    
    # 发现因果图
    print(f"\n使用 {algorithm.upper()} 算法发现因果结构...")
    
    if algorithm == "pc":
        test = IndependenceTest(method="fisher_z", alpha=alpha)
        graph = cd.discover(
            X, algorithm="pc", node_names=var_names,
            independence_test=test, verbose=True
        )
    elif algorithm == "ges":
        graph = cd.discover(
            X, algorithm="ges", node_names=var_names,
            score_type="bic", verbose=True
        )
    elif algorithm == "notears":
        graph = cd.discover(
            X, algorithm="notears", node_names=var_names,
            lambda1=0.1, w_threshold=0.3, verbose=False
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 提取边
    edges = []
    for edge in graph.edges:
        edges.append({
            'from': var_names[edge.source],
            'to': var_names[edge.target],
            'type': edge.edge_type.value,
            'weight': edge.weight
        })
    
    # 分析因果效应
    causal_effects = analyze_causal_effects(graph, var_names, data)
    
    return {
        'graph': graph,
        'edges': edges,
        'var_names': var_names,
        'is_dag': graph.is_dag(),
        'n_edges': len(edges),
        'causal_effects': causal_effects
    }


def analyze_causal_effects(
    graph,
    var_names: List[str],
    data: pd.DataFrame
) -> Dict[str, List[Dict]]:
    """
    分析因果效应
    
    Args:
        graph: 因果图
        var_names: 变量名称
        data: 数据
    
    Returns:
        因果效应分析
    """
    effects = {}
    
    # 对每个变量，找出其因果效应
    for i, var in enumerate(var_names):
        var_effects = []
        
        # 找出该变量的子节点（直接效应）
        children = graph.get_children(i)
        for child_idx in children:
            effect = {
                'target': var_names[child_idx],
                'type': 'direct',
                'path': f"{var} -> {var_names[child_idx]}"
            }
            var_effects.append(effect)
        
        # 找出后代节点（总效应）
        descendants = graph.get_descendants(i)
        for desc_idx in descendants:
            if desc_idx not in children:  # 只添加间接效应
                # 找到路径
                paths = find_causal_paths(graph, i, desc_idx, var_names)
                effect = {
                    'target': var_names[desc_idx],
                    'type': 'indirect',
                    'paths': paths
                }
                var_effects.append(effect)
        
        if var_effects:
            effects[var] = var_effects
    
    return effects


def find_causal_paths(
    graph,
    source: int,
    target: int,
    var_names: List[str],
    max_length: int = 5
) -> List[str]:
    """找出因果路径"""
    paths = []
    
    def dfs(current: int, path: List[int]):
        if len(path) > max_length:
            return
        if current == target:
            path_str = " -> ".join(var_names[i] for i in path)
            paths.append(path_str)
            return
        
        for child in graph.get_children(current):
            if child not in path:  # 避免环
                dfs(child, path + [child])
    
    dfs(source, [source])
    return paths


def generate_insights(results: Dict[str, Any]) -> List[str]:
    """
    生成洞察
    
    Args:
        results: 因果发现结果
    
    Returns:
        洞察列表
    """
    insights = []
    
    edges = results['edges']
    
    # 识别关键驱动因素
    outgoing_counts = {}
    incoming_counts = {}
    
    for edge in edges:
        from_var = edge['from']
        to_var = edge['to']
        outgoing_counts[from_var] = outgoing_counts.get(from_var, 0) + 1
        incoming_counts[to_var] = incoming_counts.get(to_var, 0) + 1
    
    # 找出最重要的驱动因素
    if outgoing_counts:
        top_drivers = sorted(outgoing_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"关键驱动因素: {', '.join([d[0] for d in top_drivers])}")
    
    # 找出最被影响的变量
    if incoming_counts:
        top_effects = sorted(incoming_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"最受影响的性能指标: {', '.join([e[0] for e in top_effects])}")
    
    # 检查特定模式
    # 温度影响
    temp_edges = [e for e in edges if e['from'] == 'temperature']
    if temp_edges:
        affected = [e['to'] for e in temp_edges]
        insights.append(f"温度直接影响: {', '.join(affected)}")
    
    # 材料类型影响
    material_edges = [e for e in edges if e['from'] == 'material_type']
    if material_edges:
        affected = [e['to'] for e in material_edges]
        insights.append(f"材料类型直接影响: {', '.join(affected)}")
    
    # 综合性能的因果路径
    performance_parents = [e['from'] for e in edges if e['to'] == 'overall_performance']
    if performance_parents:
        insights.append(f"影响综合性能的直接因素: {', '.join(performance_parents)}")
    
    return insights


def recommend_interventions(
    results: Dict[str, Any],
    target: str = "overall_performance"
) -> List[Dict]:
    """
    推荐干预措施
    
    Args:
        results: 因果发现结果
        target: 目标变量
    
    Returns:
        干预建议
    """
    recommendations = []
    
    # 找到影响目标的所有祖先
    target_idx = results['var_names'].index(target)
    ancestors = results['graph'].get_ancestors(target_idx)
    
    # 如果没有发现祖先，返回一般性建议
    if not ancestors:
        return [{"variable": "数据不足", "recommendation": "需要更多数据来建立因果联系"}]
    
    # 对每个祖先变量生成建议
    for ancestor_idx in ancestors:
        var_name = results['var_names'][ancestor_idx]
        
        # 根据变量类型生成建议
        if var_name == "temperature":
            recommendations.append({
                "variable": var_name,
                "recommendation": "优化工作温度至25°C附近可获得最佳性能",
                "expected_impact": "高",
                "rationale": "温度通过影响内阻和循环寿命间接影响综合性能"
            })
        elif var_name == "material_type":
            recommendations.append({
                "variable": var_name,
                "recommendation": "选择高等级材料类型（type=2）",
                "expected_impact": "高",
                "rationale": "材料类型对容量和循环寿命有显著影响"
            })
        elif var_name == "electrolyte_conc":
            recommendations.append({
                "variable": var_name,
                "recommendation": "优化电解液浓度至1.2-1.5 M范围",
                "expected_impact": "中",
                "rationale": "电解液浓度影响离子电导率"
            })
    
    return recommendations


def visualize_causal_graph(
    results: Dict[str, Any],
    output_path: Optional[str] = None
):
    """
    可视化因果图
    
    Args:
        results: 因果发现结果
        output_path: 输出路径
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # 创建图
        G = nx.DiGraph()
        
        for edge in results['edges']:
            G.add_edge(edge['from'], edge['to'], weight=edge.get('weight', 1.0))
        
        # 绘图
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # 节点颜色
        node_colors = []
        for node in G.nodes():
            if 'performance' in node:
                node_colors.append('lightgreen')
            elif node in ['temperature', 'pressure']:
                node_colors.append('lightblue')
            elif node in ['material_type', 'electrolyte_conc']:
                node_colors.append('lightyellow')
            else:
                node_colors.append('lightcoral')
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               arrowsize=20, node_size=2000)
        
        plt.title("Battery Performance Causal Graph", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"因果图已保存至: {output_path}")
        
        plt.show()
        
    except ImportError:
        print("可视化需要 matplotlib 和 networkx")


def run_battery_causal_discovery():
    """运行电池因果发现完整流程"""
    print("=" * 70)
    print("电池性能因果图自动发现")
    print("=" * 70)
    
    # 1. 生成数据
    print("\n1. 生成合成电池数据...")
    data = generate_battery_data(n_samples=500, noise_level=0.05)
    print(f"   生成了 {len(data)} 个样本")
    print(f"   变量: {list(data.columns)}")
    print("\n   数据统计:")
    print(data.describe().round(2))
    
    # 2. 发现因果图
    print("\n2. 因果发现...")
    results = discover_battery_causal_graph(data, algorithm="notears")
    
    print(f"\n   发现 {results['n_edges']} 条因果边")
    print(f"   是否为DAG: {results['is_dag']}")
    
    print("\n   发现的因果边:")
    for edge in results['edges']:
        print(f"     {edge['from']} -> {edge['to']} (weight: {edge['weight']:.3f})")
    
    # 3. 因果效应分析
    print("\n3. 因果效应分析:")
    for var, effects in results['causal_effects'].items():
        print(f"\n   {var}:")
        for effect in effects[:3]:  # 只显示前3个
            if effect['type'] == 'direct':
                print(f"     直接影响 -> {effect['target']}")
            else:
                print(f"     间接影响 -> {effect['target']}")
                for path in effect['paths'][:2]:
                    print(f"       路径: {path}")
    
    # 4. 生成洞察
    print("\n4. 关键洞察:")
    insights = generate_insights(results)
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # 5. 干预建议
    print("\n5. 优化建议:")
    recommendations = recommend_interventions(results)
    for rec in recommendations:
        print(f"\n   变量: {rec['variable']}")
        print(f"   建议: {rec['recommendation']}")
        print(f"   预期影响: {rec['expected_impact']}")
        print(f"   依据: {rec['rationale']}")
    
    # 6. 可视化
    print("\n6. 可视化因果图...")
    try:
        visualize_causal_graph(results)
    except Exception as e:
        print(f"   可视化跳过: {e}")
    
    print("\n" + "=" * 70)
    print("分析完成!")
    print("=" * 70)
    
    return results


def demo():
    """演示函数"""
    return run_battery_causal_discovery()


if __name__ == "__main__":
    demo()
