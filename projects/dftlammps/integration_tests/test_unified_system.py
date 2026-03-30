"""
DFT+LAMMPS Integration Tests
============================
集成测试套件 - 端到端测试、性能基准、回归测试
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# 统一架构导入
from dftlammps.unified import (
    # 配置系统
    ConfigManager, ConfigBuilder, GlobalConfig,
    DFTConfig, LAMMPSConfig, MLPotentialConfig,
    load_config, create_default_config,
    ConfigValidationError, ConfigNotFoundError,
    
    # API系统
    UnifiedAPIRouter, ModuleInterface, RouteInfo,
    APIRequest, APIResponse, HTTPMethod,
    Middleware, LoggingMiddleware,
    get_router, init_api, call_api,
    
    # 编排器
    OrchestratorV2, WorkflowBuilder,
    Workflow, Task, TaskResult, TaskStatus, TaskPriority,
    ExecutionMode, ResourceRequirements,
    get_orchestrator, run_workflow,
    
    # 通用工具
    get_logger, DFTLAMMPSError, ValidationError,
    retry, log_execution, timing
)

# 测试日志
logger = get_logger("integration_tests")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """临时目录fixture"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_manager():
    """配置管理器fixture"""
    return ConfigManager()


@pytest.fixture
def sample_config():
    """示例配置fixture"""
    return ConfigBuilder().with_project(
        "Test Project", "1.0.0"
    ).with_dft(
        calculator="vasp", encut=520.0
    ).with_lammps(
        pair_style="deepmd", timestep=0.001
    ).with_ml_potential(
        framework="deepmd", rcut=6.0
    ).build()


@pytest.fixture
def api_router(sample_config):
    """API路由器fixture"""
    return init_api(sample_config)


@pytest.fixture
def orchestrator(sample_config):
    """编排器fixture"""
    return get_orchestrator(sample_config)


@pytest.fixture
def event_loop():
    """事件循环fixture"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# 配置系统测试
# =============================================================================

class TestConfigSystem:
    """配置系统测试"""
    
    def test_config_builder(self):
        """测试配置构建器"""
        config = (ConfigBuilder()
            .with_project("Test Project", "1.0.0")
            .with_dft(calculator="vasp", encut=600.0)
            .with_lammps(timestep=0.0005)
            .with_ml_potential(framework="mace")
            .with_hpc(cluster_type="slurm", max_nodes=8)
            .build())
        
        assert config.project_name == "Test Project"
        assert config.project_version == "1.0.0"
        assert config.dft.calculator == "vasp"
        assert config.dft.encut == 600.0
        assert config.lammps.timestep == 0.0005
        assert config.ml_potential.framework == "mace"
        assert config.hpc.cluster_type == "slurm"
        assert config.hpc.max_nodes == 8
    
    def test_config_save_load(self, temp_dir):
        """测试配置保存和加载"""
        config_path = temp_dir / "test_config.yaml"
        
        # 创建并保存配置
        manager = ConfigManager()
        manager.create_default_config(config_path)
        
        assert config_path.exists()
        
        # 加载配置
        new_manager = ConfigManager()
        config = new_manager.load_config(config_path)
        
        assert config.project_name == "My DFT+LAMMPS Project"
        assert config.dft.calculator == "vasp"
    
    def test_env_override(self, monkeypatch):
        """测试环境变量覆盖"""
        monkeypatch.setenv("DFTLAMMPS_DFT_ENCUT", "800")
        monkeypatch.setenv("DFTLAMMPS_LAMMPS_TIMESTEP", "0.0005")
        monkeypatch.setenv("DFTLAMMPS_DEBUG_MODE", "true")
        
        manager = ConfigManager()
        manager.load_from_env()
        
        assert manager.global_config.dft.encut == 800
        assert manager.global_config.lammps.timestep == 0.0005
        assert manager.global_config.debug_mode == True
    
    def test_config_validation(self, config_manager):
        """测试配置验证"""
        # 注册验证模式
        from dftlammps.unified import ConfigSchema
        
        config_manager.register_schema("dft", [
            ConfigSchema("encut", float, True, validator=lambda x: x > 0),
            ConfigSchema("nelm", int, True, validator=lambda x: x > 0)
        ])
        
        # 设置有效值
        config_manager.global_config.dft.encut = 520.0
        config_manager.global_config.dft.nelm = 100
        
        errors = config_manager.validate_config("dft")
        assert len(errors) == 0
        
        # 设置无效值
        config_manager.global_config.dft.encut = -100
        errors = config_manager.validate_config("dft")
        assert len(errors) > 0


# =============================================================================
# API系统测试
# =============================================================================

class TestUnifiedAPI:
    """统一API测试"""
    
    @pytest.mark.asyncio
    async def test_api_routing(self, api_router):
        """测试API路由"""
        # 测试健康检查端点
        request = APIRequest(
            path="/health",
            method=HTTPMethod.GET
        )
        
        response = await api_router.route(request)
        
        assert response.status.value == "success"
        assert response.data is not None
        assert "modules" in response.data
    
    @pytest.mark.asyncio
    async def test_api_docs_generation(self, api_router):
        """测试API文档生成"""
        request = APIRequest(
            path="/api/docs",
            method=HTTPMethod.GET
        )
        
        response = await api_router.route(request)
        
        assert response.status.value == "success"
        assert "routes" in response.data
        assert len(response.data["routes"]) > 0
    
    @pytest.mark.asyncio
    async def test_route_not_found(self, api_router):
        """测试路由未找到"""
        request = APIRequest(
            path="/nonexistent/path",
            method=HTTPMethod.GET
        )
        
        response = await api_router.route(request)
        
        assert response.status.value == "error"
        assert "ROUTE_NOT_FOUND" in response.error.get("code", "")
    
    def test_custom_module_registration(self, sample_config):
        """测试自定义模块注册"""
        
        class TestModule(ModuleInterface):
            name = "test_module"
            version = "1.0.0"
            description = "Test module"
            
            async def initialize(self):
                self._initialized = True
            
            async def shutdown(self):
                self._initialized = False
            
            def get_routes(self):
                return [
                    RouteInfo(
                        path="/test/hello",
                        method=HTTPMethod.GET,
                        handler=self.hello,
                        name="test_hello"
                    )
                ]
            
            async def hello(self, request):
                return {"message": "Hello from test module"}
        
        router = UnifiedAPIRouter(sample_config)
        test_module = TestModule(sample_config)
        
        router.register_module(test_module)
        
        assert "test_module" in router.registry.get_all_modules()


# =============================================================================
# 编排器测试
# =============================================================================

class TestOrchestrator:
    """编排器测试"""
    
    @pytest.mark.asyncio
    async def test_task_creation(self, orchestrator):
        """测试任务创建"""
        workflow = orchestrator.create_workflow("test_workflow")
        
        task = Task(
            name="test_task",
            module="dft",
            operation="calculate"
        )
        
        orchestrator.add_task_to_workflow(workflow.id, task)
        
        assert task.id in workflow.tasks
        assert workflow.tasks[task.id].name == "test_task"
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator):
        """测试工作流执行"""
        # 注册Python函数执行器
        from dftlammps.unified import PythonFunctionExecutor
        
        executor = PythonFunctionExecutor()
        executor.register("add", lambda a, b: a + b)
        executor.register("multiply", lambda a, b: a * b)
        
        orchestrator.add_executor(executor)
        
        # 创建工作流
        workflow = orchestrator.create_workflow("calc_workflow")
        
        task1 = Task(
            name="add_task",
            module="python",
            operation="add",
            params={"a": 2, "b": 3}
        )
        
        task2 = Task(
            name="multiply_task",
            module="python",
            operation="multiply",
            params={"a": 10, "b": 5},
            dependencies=[task1.id]
        )
        
        orchestrator.add_task_to_workflow(workflow.id, task1)
        orchestrator.add_task_to_workflow(workflow.id, task2)
        
        # 启动编排器
        await orchestrator.start()
        
        try:
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert "add_task" in result["results"]
            assert "multiply_task" in result["results"]
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator):
        """测试并行执行"""
        from dftlammps.unified import PythonFunctionExecutor
        
        executor = PythonFunctionExecutor()
        
        async def slow_task(duration):
            await asyncio.sleep(duration)
            return {"duration": duration}
        
        executor.register("slow", slow_task)
        orchestrator.add_executor(executor)
        
        workflow = orchestrator.create_workflow("parallel_test")
        
        # 添加并行任务
        tasks = []
        for i in range(3):
            task = Task(
                name=f"task_{i}",
                module="python",
                operation="slow",
                params={"duration": 0.1},
                execution_mode=ExecutionMode.PARALLEL
            )
            orchestrator.add_task_to_workflow(workflow.id, task)
            tasks.append(task)
        
        await orchestrator.start()
        
        try:
            start_time = time.time()
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            elapsed = time.time() - start_time
            
            # 并行执行应该比顺序执行快
            assert elapsed < 0.5  # 3个0.1秒的任务并行应该小于0.5秒
            assert result["all_success"] == True
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_workflow_builder(self, orchestrator):
        """测试工作流构建器"""
        from dftlammps.unified import PythonFunctionExecutor
        
        executor = PythonFunctionExecutor()
        executor.register("square", lambda x: x * x)
        executor.register("cube", lambda x: x ** 3)
        
        orchestrator.add_executor(executor)
        
        builder = WorkflowBuilder("builder_test", orchestrator)
        
        builder.add_task(
            name="square_task",
            module="python",
            operation="square",
            params={"x": 5}
        ).add_task(
            name="cube_task",
            module="python",
            operation="cube",
            params={"x": 3}
        )
        
        await orchestrator.start()
        
        try:
            result = await builder.execute()
            
            assert result["all_success"] == True
            assert result["results"]["square_task"]["data"] == 25
            assert result["results"]["cube_task"]["data"] == 27
        finally:
            await orchestrator.stop()
    
    def test_dependency_validation(self, orchestrator):
        """测试依赖验证"""
        workflow = orchestrator.create_workflow("dependency_test")
        
        task1 = Task(name="task1", module="test", operation="op1")
        task2 = Task(
            name="task2",
            module="test",
            operation="op2",
            dependencies=[task1.id]
        )
        task3 = Task(
            name="task3",
            module="test",
            operation="op3",
            dependencies=[task2.id]
        )
        
        # 创建循环依赖
        task1.dependencies = [task3.id]
        
        orchestrator.add_task_to_workflow(workflow.id, task1)
        orchestrator.add_task_to_workflow(workflow.id, task2)
        orchestrator.add_task_to_workflow(workflow.id, task3)
        
        errors = workflow.validate()
        
        assert len(errors) > 0
        assert "Circular dependency" in errors[0]


# =============================================================================
# 端到端测试
# =============================================================================

class TestEndToEnd:
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_simulation_workflow(self):
        """测试完整模拟工作流"""
        # 初始化系统
        from dftlammps.unified import initialize_unified_system
        
        config_manager, api_router, orchestrator = initialize_unified_system()
        
        # 注册执行器
        from dftlammps.unified import PythonFunctionExecutor
        
        executor = PythonFunctionExecutor()
        
        # 模拟DFT计算
        def mock_dft_calculator(structure, encut):
            time.sleep(0.01)  # 模拟计算时间
            return {"energy": -100.5, "forces": [[0, 0, 0]]}
        
        # 模拟ML训练
        def mock_ml_training(dataset, epochs):
            time.sleep(0.01)
            return {"model_path": "/path/to/model", "loss": 0.001}
        
        # 模拟LAMMPS模拟
        def mock_lammps_simulation(potential, temperature):
            time.sleep(0.01)
            return {"trajectory": "traj.xyz", "temperature": temperature}
        
        executor.register("dft_calculate", mock_dft_calculator)
        executor.register("ml_train", mock_ml_training)
        executor.register("lammps_run", mock_lammps_simulation)
        
        orchestrator.add_executor(executor)
        
        # 创建完整工作流
        workflow = orchestrator.create_workflow("full_simulation")
        
        # DFT计算任务
        dft_task = Task(
            name="dft_calculation",
            module="python",
            operation="dft_calculate",
            params={"structure": "Li3PS4", "encut": 520},
            priority=TaskPriority.HIGH
        )
        
        # ML训练任务（依赖DFT数据）
        ml_task = Task(
            name="ml_training",
            module="python",
            operation="ml_train",
            params={"dataset": "dft_results", "epochs": 100},
            dependencies=[dft_task.id],
            priority=TaskPriority.HIGH
        )
        
        # LAMMPS模拟任务（依赖ML势）
        lammps_task = Task(
            name="lammps_simulation",
            module="python",
            operation="lammps_run",
            params={"potential": "ml_potential", "temperature": 300},
            dependencies=[ml_task.id],
            priority=TaskPriority.NORMAL
        )
        
        orchestrator.add_task_to_workflow(workflow.id, dft_task)
        orchestrator.add_task_to_workflow(workflow.id, ml_task)
        orchestrator.add_task_to_workflow(workflow.id, lammps_task)
        
        # 执行工作流
        await orchestrator.start()
        
        try:
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert len(result["results"]) == 3
            assert "dft_calculation" in result["results"]
            assert "ml_training" in result["results"]
            assert "lammps_simulation" in result["results"]
            
            logger.info(f"Full workflow completed in {result['execution_time']:.2f}s")
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        from dftlammps.unified import initialize_unified_system, PythonFunctionExecutor
        
        config_manager, api_router, orchestrator = initialize_unified_system()
        
        executor = PythonFunctionExecutor()
        
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Simulated error {call_count}")
            return "success"
        
        executor.register("flaky", flaky_function)
        orchestrator.add_executor(executor)
        
        workflow = orchestrator.create_workflow("error_test")
        
        task = Task(
            name="flaky_task",
            module="python",
            operation="flaky",
            max_retries=3  # 允许3次重试
        )
        
        orchestrator.add_task_to_workflow(workflow.id, task)
        
        await orchestrator.start()
        
        try:
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert call_count == 3  # 应该重试3次
        finally:
            await orchestrator.stop()


# =============================================================================
# 性能基准测试
# =============================================================================

class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_scheduler_throughput(self):
        """测试调度器吞吐量"""
        from dftlammps.unified import initialize_unified_system, PythonFunctionExecutor
        
        config_manager, api_router, orchestrator = initialize_unified_system()
        
        executor = PythonFunctionExecutor()
        executor.register("noop", lambda: None)
        orchestrator.add_executor(executor)
        
        # 创建大量任务
        num_tasks = 100
        workflow = orchestrator.create_workflow("throughput_test")
        
        for i in range(num_tasks):
            task = Task(
                name=f"task_{i}",
                module="python",
                operation="noop"
            )
            orchestrator.add_task_to_workflow(workflow.id, task)
        
        await orchestrator.start()
        
        try:
            start_time = time.time()
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            elapsed = time.time() - start_time
            
            throughput = num_tasks / elapsed
            
            logger.info(f"Processed {num_tasks} tasks in {elapsed:.2f}s ({throughput:.1f} tasks/s)")
            
            assert result["all_success"] == True
            assert throughput > 10  # 至少每秒10个任务
        finally:
            await orchestrator.stop()
    
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_parallel_scaling(self):
        """测试并行扩展性"""
        from dftlammps.unified import initialize_unified_system, PythonFunctionExecutor
        
        config_manager, api_router, orchestrator = initialize_unified_system()
        
        executor = PythonFunctionExecutor()
        
        async def cpu_bound_task():
            # CPU密集型任务
            total = 0
            for i in range(1000000):
                total += i
            return total
        
        executor.register("cpu_task", cpu_bound_task)
        orchestrator.add_executor(executor)
        
        workflow = orchestrator.create_workflow("scaling_test")
        
        # 添加8个并行任务
        num_parallel = 8
        for i in range(num_parallel):
            task = Task(
                name=f"cpu_task_{i}",
                module="python",
                operation="cpu_task",
                execution_mode=ExecutionMode.PARALLEL
            )
            orchestrator.add_task_to_workflow(workflow.id, task)
        
        await orchestrator.start()
        
        try:
            start_time = time.time()
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            elapsed = time.time() - start_time
            
            logger.info(f"Parallel execution of {num_parallel} tasks: {elapsed:.2f}s")
            
            assert result["all_success"] == True
            # 并行执行应该比顺序执行快
        finally:
            await orchestrator.stop()
    
    @pytest.mark.benchmark
    def test_config_loading_performance(self, temp_dir):
        """测试配置加载性能"""
        config_path = temp_dir / "perf_config.yaml"
        
        # 创建大型配置文件
        config_data = {
            "project_name": "Performance Test",
            "custom_modules": {
                f"module_{i}": {
                    "param1": f"value_{i}",
                    "param2": i * 100,
                    "nested": {"a": 1, "b": 2}
                }
                for i in range(100)
            }
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # 性能测试
        times = []
        for _ in range(10):
            start = time.perf_counter()
            manager = ConfigManager()
            manager.load_config(config_path)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        logger.info(f"Config loading average time: {avg_time*1000:.2f}ms")
        
        assert avg_time < 0.1  # 应该在100ms内完成


# =============================================================================
# 回归测试
# =============================================================================

class TestRegression:
    """回归测试"""
    
    @pytest.mark.regression
    def test_error_handling_consistency(self):
        """测试错误处理一致性"""
        # 测试所有异常类都有正确的属性
        from dftlammps.unified import (
            DFTLAMMPSError, ConfigurationError, ValidationError,
            FileSystemError, WorkflowError, APIError
        )
        
        exceptions = [
            DFTLAMMPSError("test"),
            ConfigurationError("test config"),
            ValidationError("test validation", field="test_field"),
            FileSystemError("test file", path="/test/path"),
            WorkflowError("test workflow", step="test_step"),
            APIError("test api", status_code=500)
        ]
        
        for exc in exceptions:
            # 验证基本属性
            assert hasattr(exc, 'message')
            assert hasattr(exc, 'error_code')
            assert hasattr(exc, 'timestamp')
            assert hasattr(exc, 'to_dict')
            
            # 验证to_dict方法
            data = exc.to_dict()
            assert 'error_type' in data
            assert 'error_code' in data
            assert 'message' in data
            assert 'timestamp' in data
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_task_state_transitions(self, orchestrator):
        """测试任务状态转换"""
        from dftlammps.unified import PythonFunctionExecutor
        
        executor = PythonFunctionExecutor()
        executor.register("simple", lambda: "done")
        orchestrator.add_executor(executor)
        
        workflow = orchestrator.create_workflow("state_test")
        
        task = Task(
            name="state_task",
            module="python",
            operation="simple"
        )
        
        orchestrator.add_task_to_workflow(workflow.id, task)
        
        # 验证初始状态
        assert task.status == TaskStatus.PENDING
        
        await orchestrator.start()
        
        try:
            # 执行工作流
            await orchestrator.execute_workflow(workflow.id, wait=True)
            
            # 验证最终状态
            assert task.status == TaskStatus.COMPLETED
            assert task.result is not None
            assert task.result.success == True
            assert task.started_at is not None
            assert task.completed_at is not None
            assert task.completed_at >= task.started_at
        finally:
            await orchestrator.stop()
    
    @pytest.mark.regression
    def test_config_immutability(self, sample_config):
        """测试配置不可变性（浅拷贝保护）"""
        import copy
        
        # 深度拷贝应该创建独立副本
        config_copy = copy.deepcopy(sample_config)
        
        # 修改副本不应影响原配置
        config_copy.dft.encut = 999
        assert sample_config.dft.encut != 999
    
    @pytest.mark.regression
    @pytest.mark.asyncio
    async def test_concurrent_workflow_isolation(self):
        """测试并发工作流隔离"""
        from dftlammps.unified import initialize_unified_system, PythonFunctionExecutor
        
        config_manager, api_router, orchestrator = initialize_unified_system()
        
        executor = PythonFunctionExecutor()
        
        results = {}
        
        def store_result(workflow_id, value):
            results[workflow_id] = value
            return value
        
        executor.register("store", lambda wid, val: store_result(wid, val))
        orchestrator.add_executor(executor)
        
        await orchestrator.start()
        
        try:
            # 创建多个并发工作流
            workflows = []
            for i in range(5):
                workflow = orchestrator.create_workflow(f"concurrent_{i}")
                task = Task(
                    name="store_task",
                    module="python",
                    operation="store",
                    params={"wid": workflow.id, "val": i}
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
                workflows.append(workflow)
            
            # 并发执行
            await asyncio.gather(*[
                orchestrator.execute_workflow(w.id, wait=True)
                for w in workflows
            ])
            
            # 验证隔离性
            for i, workflow in enumerate(workflows):
                assert results.get(workflow.id) == i
        finally:
            await orchestrator.stop()


# =============================================================================
# 测试运行器
# =============================================================================

if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x"
    ])
