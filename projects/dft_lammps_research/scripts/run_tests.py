#!/usr/bin/env python3
"""
测试运行脚本
Test Runner Script
==================

提供便捷的测试运行和报告功能。

使用方法:
    python scripts/run_tests.py                    # 运行所有测试
    python scripts/run_tests.py unit               # 运行单元测试
    python scripts/run_tests.py integration        # 运行集成测试
    python scripts/run_tests.py regression         # 运行回归测试
    python scripts/run_tests.py performance        # 运行性能测试
    python scripts/run_tests.py --coverage         # 生成覆盖率报告
    python scripts/run_tests.py --html             # 生成HTML报告
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.BLUE}{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}{Colors.RESET}\n")


def print_success(text: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text: str):
    """打印错误消息"""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_warning(text: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def run_command(cmd: list, description: str = "") -> bool:
    """运行命令并返回是否成功"""
    if description:
        print(f"Running: {description}")
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start_time
    
    success = result.returncode == 0
    
    if success:
        print_success(f"Completed in {elapsed:.1f}s")
    else:
        print_error(f"Failed after {elapsed:.1f}s")
    
    return success


def run_tests(test_type: str, coverage: bool = False, html: bool = False, 
              verbose: bool = False, parallel: bool = False) -> bool:
    """运行指定类型的测试"""
    
    cmd = ["pytest"]
    
    # 添加测试路径和标记
    if test_type == "all":
        cmd.append("tests/")
    elif test_type == "unit":
        cmd.extend(["tests/unit", "-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["tests/integration", "-m", "integration"])
    elif test_type == "regression":
        cmd.extend(["tests/regression", "-m", "regression"])
    elif test_type == "performance":
        cmd.extend(["tests/performance", "-m", "performance", "--benchmark-only"])
    elif test_type == "e2e":
        cmd.extend(["tests/e2e", "-m", "e2e"])
    else:
        print_error(f"Unknown test type: {test_type}")
        return False
    
    # 添加选项
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if parallel:
        cmd.extend(["-n", "auto"])
    
    if coverage:
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
        if html:
            cmd.append("--cov-report=html")
    
    if html and not coverage:
        cmd.extend(["--html=test_reports/report.html", "--self-contained-html"])
    
    # 添加通用选项
    cmd.extend(["--tb=short", "--color=yes"])
    
    return run_command(cmd, f"{test_type} tests")


def run_linters() -> bool:
    """运行代码检查"""
    print_header("Code Quality Checks")
    
    success = True
    
    # Black format check
    if not run_command(["black", "--check", "."], "Black format check"):
        success = False
        print_warning("Run 'make format' to fix formatting")
    
    # isort check
    if not run_command(["isort", "--check-only", "."], "Import sorting check"):
        success = False
    
    # flake8
    if not run_command(["flake8", "."], "Linting with flake8"):
        success = False
    
    return success


def run_security_scan() -> bool:
    """运行安全扫描"""
    print_header("Security Scans")
    
    success = True
    
    # Bandit
    if not run_command(["bandit", "-r", ".", "-ll"], "Bandit security scan"):
        success = False
    
    return success


def generate_reports():
    """生成测试报告"""
    print_header("Generating Reports")
    
    reports_dir = Path("test_reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Coverage report
    run_command(
        ["pytest", "tests/", "--cov=.", "--cov-report=html:htmlcov", "-q"],
        "Coverage report"
    )
    
    print_success(f"Reports saved to: {reports_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for DFT-MD-ML Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run all tests
  %(prog)s unit               # Run unit tests only
  %(prog)s integration        # Run integration tests
  %(prog)s --coverage         # Run with coverage
  %(prog)s --lint             # Run linters only
        """
    )
    
    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=["all", "unit", "integration", "regression", "performance", "e2e"],
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "--lint",
        action="store_true",
        help="Run linters only"
    )
    
    parser.add_argument(
        "--security",
        action="store_true",
        help="Run security scans only"
    )
    
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Run full CI pipeline"
    )
    
    args = parser.parse_args()
    
    print_header("DFT-MD-ML Platform Test Runner")
    
    start_time = time.time()
    
    # CI模式
    if args.ci:
        success = True
        if not run_linters():
            success = False
        if not run_security_scan():
            success = False
        if not run_tests("unit", verbose=args.verbose):
            success = False
        if not run_tests("integration", verbose=args.verbose):
            success = False
        
        if success:
            print_success("CI pipeline passed!")
        else:
            print_error("CI pipeline failed!")
        
        return 0 if success else 1
    
    # 仅lint模式
    if args.lint:
        return 0 if run_linters() else 1
    
    # 仅安全扫描模式
    if args.security:
        return 0 if run_security_scan() else 1
    
    # 运行测试
    success = run_tests(
        args.test_type,
        coverage=args.coverage,
        html=args.html,
        verbose=args.verbose,
        parallel=args.parallel
    )
    
    # 生成报告
    if args.html or args.coverage:
        generate_reports()
    
    elapsed = time.time() - start_time
    
    print_header(f"Completed in {elapsed:.1f}s")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
