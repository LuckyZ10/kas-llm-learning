# Makefile for Testing
# ====================

.PHONY: test test-unit test-integration test-regression test-performance test-e2e
.PHONY: coverage coverage-html coverage-xml
.PHONY: lint format security clean
.PHONY: install-test-deps ci

# 默认目标
.DEFAULT_GOAL := help

help:
	@echo "DFT-MD-ML Platform Testing Commands"
	@echo "===================================="
	@echo ""
	@echo "Test Commands:"
	@echo "  make test              - Run all tests"
	@echo "  make test-unit         - Run unit tests only"
	@echo "  make test-integration  - Run integration tests"
	@echo "  make test-regression   - Run regression tests"
	@echo "  make test-performance  - Run performance benchmarks"
	@echo "  make test-e2e          - Run end-to-end tests"
	@echo "  make test-quick        - Run quick tests (no slow)"
	@echo ""
	@echo "Coverage Commands:"
	@echo "  make coverage          - Generate coverage report"
	@echo "  make coverage-html     - Generate HTML coverage report"
	@echo "  make coverage-xml      - Generate XML coverage report"
	@echo ""
	@echo "Quality Commands:"
	@echo "  make lint              - Run all linters"
	@echo "  make format            - Format code with black/isort"
	@echo "  make format-check      - Check code formatting"
	@echo "  make security          - Run security scans"
	@echo "  make type-check        - Run mypy type checking"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make install-test-deps - Install test dependencies"
	@echo "  make clean             - Clean test artifacts"
	@echo "  make ci                - Run full CI pipeline locally"
	@echo ""

# 测试命令
# --------

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit -v -m unit --tb=short

test-integration:
	pytest tests/integration -v -m integration --tb=short

test-regression:
	pytest tests/regression -v -m regression --tb=short --timeout=600

test-performance:
	pytest tests/performance -v -m performance --benchmark-only

test-e2e:
	pytest tests/e2e -v -m e2e --tb=short --timeout=300

test-quick:
	pytest tests/ -v -m "not slow" --tb=short

test-dft:
	pytest tests/ -v -m dft --tb=short

test-md:
	pytest tests/ -v -m md --tb=short

test-ml:
	pytest tests/ -v -m ml --tb=short

test-hpc:
	pytest tests/ -v -m hpc --tb=short

# 覆盖率命令
# ----------

coverage:
	pytest tests/ --cov=. --cov-report=term-missing

coverage-html:
	pytest tests/ --cov=. --cov-report=html
	@echo "HTML report generated at: htmlcov/index.html"

coverage-xml:
	pytest tests/ --cov=. --cov-report=xml
	@echo "XML report generated at: coverage.xml"

coverage-badge:
	pytest tests/ --cov=. --cov-report=term-missing
	coverage-badge -o coverage.svg

# 代码质量命令
# ------------

lint:
	@echo "Running flake8..."
	flake8 . --count --statistics --exclude=.git,__pycache__,venv,.venv
	@echo "Running pylint..."
	pylint dft_to_lammps_bridge nep_training_pipeline hpc_scheduler --disable=all --enable=E,W || true

format:
	@echo "Formatting with black..."
	black .
	@echo "Sorting imports with isort..."
	isort .

format-check:
	@echo "Checking black formatting..."
	black --check --diff .
	@echo "Checking isort..."
	isort --check-only --diff .

type-check:
	mypy dft_to_lammps_bridge nep_training_pipeline hpc_scheduler --ignore-missing-imports || true

security:
	@echo "Running bandit..."
	bandit -r . -ll || true
	@echo "Running safety check..."
	safety check --full-report || true

# CI 命令
# -------

ci: format-check lint type-check security test-unit test-integration
	@echo "Full CI pipeline completed"

# 依赖安装
# --------

install-test-deps:
	pip install -r requirements.txt || true
	pip install -r requirements-test.txt

# 清理命令
# --------

clean:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -f coverage.xml
	rm -f junit-*.xml
	rm -rf test_reports
	rm -f .benchmarks
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# 报告命令
# --------

report-test:
	pytest tests/ --html=test_reports/report.html --self-contained-html || true

report-benchmark:
	pytest tests/performance --benchmark-only --benchmark-json=test_reports/benchmark.json

# 开发辅助
# --------

watch-test:
	# 需要安装 pytest-watch
	ptw tests/ -v --onpass "echo 'Tests passed'" --onfail "echo 'Tests failed'"

dev-setup:
	pip install -e . || true
	pip install -r requirements-test.txt
	pre-commit install || true
