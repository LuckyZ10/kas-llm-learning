#!/usr/bin/env python3
"""
神经符号融合与因果发现引擎 - 主运行脚本
Neuro-Symbolic Fusion and Causal Discovery Engine - Main Runner

运行所有模块的演示和测试
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_neural_perception_demo():
    """运行神经感知演示"""
    print("\n" + "=" * 70)
    print("【模块1】神经感知层 (Neural Perception)")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic.neural_perception import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_symbolic_reasoning_demo():
    """运行符号推理演示"""
    print("\n" + "=" * 70)
    print("【模块2】符号推理引擎 (Symbolic Reasoning)")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic.symbolic_reasoning import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_causal_discovery_demo():
    """运行因果发现演示"""
    print("\n" + "=" * 70)
    print("【模块3】因果发现算法 (Causal Discovery)")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic.causal_discovery import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_neural_symbolic_bridge_demo():
    """运行神经符号桥接演示"""
    print("\n" + "=" * 70)
    print("【模块4】神经-符号桥接 (Neural-Symbolic Bridge)")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic.neural_symbolic_bridge import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_explainable_ai_demo():
    """运行可解释AI演示"""
    print("\n" + "=" * 70)
    print("【模块5】可解释AI模块 (Explainable AI)")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic.explainable_ai import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_structural_equation_demo():
    """运行结构方程模型演示"""
    print("\n" + "=" * 70)
    print("【模块6】结构方程模型 (Structural Equation Model)")
    print("=" * 70)
    
    try:
        from dftlammps.causal_models.structural_equation import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_bayesian_network_demo():
    """运行贝叶斯网络演示"""
    print("\n" + "=" * 70)
    print("【模块7】贝叶斯网络 (Bayesian Network)")
    print("=" * 70)
    
    try:
        from dftlammps.causal_models.bayesian_network import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_counterfactual_demo():
    """运行反事实推理演示"""
    print("\n" + "=" * 70)
    print("【模块8】反事实推理引擎 (Counterfactual Reasoning)")
    print("=" * 70)
    
    try:
        from dftlammps.causal_models.counterfactual import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_intervention_simulator_demo():
    """运行干预模拟器演示"""
    print("\n" + "=" * 70)
    print("【模块9】干预模拟器 (Intervention Simulator)")
    print("=" * 70)
    
    try:
        from dftlammps.causal_models.intervention_simulator import demo
        demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_battery_example():
    """运行电池因果发现示例"""
    print("\n" + "=" * 70)
    print("【案例1】电池性能因果图自动发现")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic_examples.battery_causal_discovery import run_battery_causal_discovery
        run_battery_causal_discovery()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_catalyst_example():
    """运行催化剂机理解释示例"""
    print("\n" + "=" * 70)
    print("【案例2】催化剂机理解释")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic_examples.catalyst_mechanism_explanation import run_catalyst_mechanism_demo
        run_catalyst_mechanism_demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_material_example():
    """运行材料性质预测示例"""
    print("\n" + "=" * 70)
    print("【案例3】材料性质预测器")
    print("=" * 70)
    
    try:
        from dftlammps.neuro_symbolic_examples.material_property_predictor import run_material_property_demo
        run_material_property_demo()
        return True
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_lines():
    """统计代码行数"""
    import os
    
    total_lines = 0
    module_lines = {
        'neuro_symbolic': 0,
        'causal_models': 0,
        'neuro_symbolic_examples': 0
    }
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for module_name in module_lines.keys():
        module_path = os.path.join(base_path, 'dftlammps', module_name)
        if os.path.exists(module_path):
            for filename in os.listdir(module_path):
                if filename.endswith('.py'):
                    filepath = os.path.join(module_path, filename)
                    with open(filepath, 'r') as f:
                        lines = len(f.readlines())
                        module_lines[module_name] += lines
                        total_lines += lines
    
    return total_lines, module_lines


def main():
    """主函数"""
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  神经符号融合与因果发现引擎 - Neuro-Symbolic Fusion and Causal Discovery  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # 统计代码行数
    total_lines, module_lines = count_lines()
    print(f"\n📊 代码统计:")
    print(f"   神经符号模块: {module_lines['neuro_symbolic']:,} 行")
    print(f"   因果模型模块: {module_lines['causal_models']:,} 行")
    print(f"   案例示例模块: {module_lines['neuro_symbolic_examples']:,} 行")
    print(f"   总计: {total_lines:,} 行")
    
    results = {
        'core_modules': {},
        'causal_models': {},
        'examples': {}
    }
    
    # 运行核心模块演示
    print("\n" + "▓" * 70)
    print("核心模块演示 (Core Modules)")
    print("▓" * 70)
    
    results['core_modules']['neural_perception'] = run_neural_perception_demo()
    results['core_modules']['symbolic_reasoning'] = run_symbolic_reasoning_demo()
    results['core_modules']['causal_discovery'] = run_causal_discovery_demo()
    results['core_modules']['neural_symbolic_bridge'] = run_neural_symbolic_bridge_demo()
    results['core_modules']['explainable_ai'] = run_explainable_ai_demo()
    
    # 运行因果模型演示
    print("\n" + "▓" * 70)
    print("因果模型演示 (Causal Models)")
    print("▓" * 70)
    
    results['causal_models']['structural_equation'] = run_structural_equation_demo()
    results['causal_models']['bayesian_network'] = run_bayesian_network_demo()
    results['causal_models']['counterfactual'] = run_counterfactual_demo()
    results['causal_models']['intervention_simulator'] = run_intervention_simulator_demo()
    
    # 运行案例示例
    print("\n" + "▓" * 70)
    print("应用案例演示 (Application Examples)")
    print("▓" * 70)
    
    results['examples']['battery_causal_discovery'] = run_battery_example()
    results['examples']['catalyst_mechanism'] = run_catalyst_example()
    results['examples']['material_property'] = run_material_example()
    
    # 汇总结果
    print("\n" + "█" * 70)
    print("演示结果汇总")
    print("█" * 70)
    
    total_passed = 0
    total_failed = 0
    
    for category, modules in results.items():
        print(f"\n{category.upper()}:")
        for module_name, passed in modules.items():
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"  {module_name}: {status}")
            if passed:
                total_passed += 1
            else:
                total_failed += 1
    
    print(f"\n总计: {total_passed} 通过, {total_failed} 失败")
    
    print("\n" + "█" * 70)
    print("所有演示完成!")
    print("█" * 70 + "\n")
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
