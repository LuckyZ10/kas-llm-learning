"""
DFT+LAMMPS Integration Tests Runner
====================================
集成测试运行器

运行所有测试：
    python run_tests.py

运行特定测试：
    python run_tests.py -k test_config

运行性能测试：
    python run_tests.py --benchmark

生成覆盖率报告：
    python run_tests.py --coverage
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """运行测试"""
    test_dir = Path(__file__).parent
    
    # 构建pytest命令
    cmd = ["python", "-m", "pytest"]
    
    # 添加测试目录
    cmd.append(str(test_dir))
    
    # 添加选项
    if args.verbose:
        cmd.append("-v")
    
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    if args.benchmark:
        cmd.extend(["-m", "benchmark"])
    elif args.regression:
        cmd.extend(["-m", "regression"])
    
    if args.coverage:
        cmd.extend(["--cov=dftlammps.unified", "--cov-report=html", "--cov-report=term"])
    
    if args.failfast:
        cmd.append("-x")
    
    if args.pdb:
        cmd.append("--pdb")
    
    cmd.append("--tb=short")
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


def run_benchmark_suite():
    """运行完整的基准测试套件"""
    print("\n" + "=" * 60)
    print("DFT+LAMMPS Performance Benchmark Suite")
    print("=" * 60 + "\n")
    
    test_dir = Path(__file__).parent
    
    cmd = [
        "python", "-m", "pytest",
        str(test_dir / "test_performance.py"),
        "-v",
        "-m", "benchmark",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def run_regression_suite():
    """运行回归测试套件"""
    print("\n" + "=" * 60)
    print("DFT+LAMMPS Regression Test Suite")
    print("=" * 60 + "\n")
    
    test_dir = Path(__file__).parent
    
    cmd = [
        "python", "-m", "pytest",
        str(test_dir),
        "-v",
        "-m", "regression",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="DFT+LAMMPS Integration Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("-k", "--keyword", help="只运行匹配的测试")
    parser.add_argument("--benchmark", action="store_true", help="只运行基准测试")
    parser.add_argument("--regression", action="store_true", help="只运行回归测试")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("-x", "--failfast", action="store_true", help="遇到失败立即停止")
    parser.add_argument("--pdb", action="store_true", help="失败时进入调试器")
    parser.add_argument("--all", action="store_true", help="运行所有测试")
    
    args = parser.parse_args()
    
    # 如果没有指定，默认运行所有测试
    if not any([args.benchmark, args.regression, args.keyword, args.all]):
        args.all = True
    
    if args.benchmark:
        return run_benchmark_suite()
    elif args.regression:
        return run_regression_suite()
    else:
        return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
