"""
DFT+LAMMPS Integration Tests - Conftest
=======================================
PyTest配置和共享fixture
"""

import pytest
import asyncio
import tempfile
from pathlib import Path

# 统一架构导入
from dftlammps.unified import (
    ConfigManager, ConfigBuilder,
    UnifiedAPIRouter, init_api,
    OrchestratorV2, get_orchestrator,
    init_logging
)


# 初始化测试日志
init_logging(level="DEBUG")


def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line("markers", "benchmark: 性能基准测试")
    config.addinivalue_line("markers", "regression: 回归测试")
    config.addinivalue_line("markers", "integration: 集成测试")
    config.addinivalue_line("markers", "slow: 慢速测试")


@pytest.fixture(scope="session")
def event_loop():
    """创建session级别的事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """提供临时目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """提供示例配置"""
    return ConfigBuilder().with_project(
        "Test Project", "1.0.0"
    ).with_dft(
        calculator="vasp", encut=520.0
    ).with_lammps(
        pair_style="deepmd", timestep=0.001
    ).with_ml_potential(
        framework="deepmd", rcut=6.0
    ).with_hpc(
        cluster_type="local", max_nodes=4
    ).with_logging(
        level="DEBUG"
    ).build()


@pytest.fixture
async def api_router(sample_config):
    """提供API路由器"""
    router = init_api(sample_config)
    yield router
    # 清理
    await router.shutdown()


@pytest.fixture
async def orchestrator(sample_config):
    """提供编排器"""
    orch = get_orchestrator(sample_config)
    yield orch
    # 清理
    await orch.stop()


@pytest.fixture
def config_manager():
    """提供配置管理器"""
    return ConfigManager()


@pytest.fixture
def mock_dft_executor():
    """提供Mock DFT执行器"""
    from dftlammps.unified import PythonFunctionExecutor, Task, TaskResult, ExecutionContext
    
    class MockDFTExecutor(PythonFunctionExecutor):
        """Mock DFT执行器"""
        
        def __init__(self):
            super().__init__()
            self.calculations = []
            
            # 注册mock函数
            self.register("calculate", self.mock_calculate)
            self.register("relax", self.mock_relax)
            self.register("phonon", self.mock_phonon)
        
        def can_execute(self, task: Task) -> bool:
            return task.module == "dft" and super().can_execute(task)
        
        async def execute(self, task: Task, context: ExecutionContext) -> TaskResult:
            self.calculations.append({
                "task_id": task.id,
                "operation": task.operation,
                "params": task.params
            })
            return await super().execute(task, context)
        
        def mock_calculate(self, structure, encut=520):
            """Mock DFT计算"""
            import random
            return {
                "energy": random.uniform(-100, -50),
                "forces": [[0.0, 0.0, 0.0]] * len(structure),
                "stress": [0.0] * 6
            }
        
        def mock_relax(self, structure, steps=100):
            """Mock结构优化"""
            return {
                "final_structure": structure,
                "steps": steps,
                "converged": True
            }
        
        def mock_phonon(self, structure, mesh=[11, 11, 11]):
            """Mock声子计算"""
            return {
                "phonon_bands": [],
                "dos": [],
                "thermal_properties": {}
            }
    
    return MockDFTExecutor()


@pytest.fixture
def mock_lammps_executor():
    """提供Mock LAMMPS执行器"""
    from dftlammps.unified import PythonFunctionExecutor, Task, TaskResult, ExecutionContext
    
    class MockLAMMPSExecutor(PythonFunctionExecutor):
        """Mock LAMMPS执行器"""
        
        def __init__(self):
            super().__init__()
            self.simulations = []
            
            self.register("md_run", self.mock_md_run)
            self.register("minimize", self.mock_minimize)
            self.register("npt", self.mock_npt)
        
        def can_execute(self, task: Task) -> bool:
            return task.module == "lammps" and super().can_execute(task)
        
        async def execute(self, task: Task, context: ExecutionContext) -> TaskResult:
            self.simulations.append({
                "task_id": task.id,
                "operation": task.operation,
                "params": task.params
            })
            return await super().execute(task, context)
        
        def mock_md_run(self, potential, temperature=300, steps=10000):
            """Mock MD模拟"""
            import numpy as np
            return {
                "trajectory": f"traj_{temperature}K.xyz",
                "temperature": temperature,
                "pressure": 1.0,
                "energy_mean": np.random.uniform(-100, -50),
                "steps_completed": steps
            }
        
        def mock_minimize(self, potential, max_iter=1000):
            """Mock能量最小化"""
            return {
                "converged": True,
                "iterations": max_iter // 2,
                "final_energy": -150.5
            }
        
        def mock_npt(self, potential, temperature=300, pressure=1.0, steps=100000):
            """Mock NPT系综模拟"""
            return {
                "density": 2.5,
                "temperature_mean": temperature,
                "pressure_mean": pressure,
                "volume_fluctuation": 0.01
            }
    
    return MockLAMMPSExecutor()


@pytest.fixture
def mock_ml_executor():
    """提供Mock ML执行器"""
    from dftlammps.unified import PythonFunctionExecutor, Task, TaskResult, ExecutionContext
    
    class MockMLExecutor(PythonFunctionExecutor):
        """Mock ML执行器"""
        
        def __init__(self):
            super().__init__()
            self.trainings = []
            
            self.register("train", self.mock_train)
            self.register("predict", self.mock_predict)
            self.register("evaluate", self.mock_evaluate)
        
        def can_execute(self, task: Task) -> bool:
            return task.module in ["ml", "ml_potential"] and super().can_execute(task)
        
        async def execute(self, task: Task, context: ExecutionContext) -> TaskResult:
            if task.operation == "train":
                self.trainings.append({
                    "task_id": task.id,
                    "params": task.params
                })
            return await super().execute(task, context)
        
        def mock_train(self, dataset, epochs=1000, batch_size=32):
            """Mock训练"""
            import random
            return {
                "model_path": f"models/trained_model_{random.randint(1000, 9999)}.pt",
                "final_loss": random.uniform(0.001, 0.1),
                "training_time": epochs * 0.01,
                "epochs_completed": epochs
            }
        
        def mock_predict(self, model_path, structures):
            """Mock预测"""
            import random
            return {
                "energies": [random.uniform(-100, -50) for _ in structures],
                "forces": [[[0.0, 0.0, 0.0]] for _ in structures]
            }
        
        def mock_evaluate(self, model_path, test_dataset):
            """Mock评估"""
            return {
                "mae_energy": 0.05,
                "rmse_force": 0.1,
                "r2_score": 0.98
            }
    
    return MockMLExecutor()
