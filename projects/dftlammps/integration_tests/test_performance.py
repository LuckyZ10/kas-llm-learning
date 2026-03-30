"""
DFT+LAMMPS Integration Tests - Performance Benchmarks
=====================================================
性能基准测试套件
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
import numpy as np

from dftlammps.unified import (
    OrchestratorV2, Workflow, Task, TaskResult,
    TaskStatus, TaskPriority, ExecutionMode,
    ResourceRequirements, PythonFunctionExecutor,
    get_orchestrator
)


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, float] = {}
    
    def start(self, name: str):
        """开始计时"""
        self.timestamps[name] = time.perf_counter()
    
    def end(self, name: str):
        """结束计时"""
        if name in self.timestamps:
            elapsed = time.perf_counter() - self.timestamps[name]
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append(elapsed)
            del self.timestamps[name]
            return elapsed
        return None
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """获取统计信息"""
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "total": sum(values)
        }
    
    def report(self) -> str:
        """生成报告"""
        lines = ["\n=== Performance Report ==="]
        for name in sorted(self.metrics.keys()):
            stats = self.get_stats(name)
            lines.append(f"\n{name}:")
            lines.append(f"  Count: {stats['count']}")
            lines.append(f"  Mean: {stats['mean']*1000:.2f}ms")
            lines.append(f"  Median: {stats['median']*1000:.2f}ms")
            lines.append(f"  StdDev: {stats['stdev']*1000:.2f}ms")
            lines.append(f"  Min: {stats['min']*1000:.2f}ms")
            lines.append(f"  Max: {stats['max']*1000:.2f}ms")
            lines.append(f"  Total: {stats['total']:.2f}s")
        return "\n".join(lines)


@pytest.mark.benchmark
class TestSchedulerPerformance:
    """调度器性能测试"""
    
    @pytest.mark.asyncio
    async def test_task_submission_throughput(self):
        """测试任务提交吞吐量"""
        orchestrator = get_orchestrator()
        
        executor = PythonFunctionExecutor()
        executor.register("noop", lambda: None)
        orchestrator.add_executor(executor)
        
        await orchestrator.start()
        
        try:
            metrics = PerformanceMetrics()
            
            # 测试不同规模的任务集
            task_counts = [10, 50, 100, 200]
            
            for count in task_counts:
                workflow = orchestrator.create_workflow(f"throughput_{count}")
                
                metrics.start(f"submit_{count}")
                for i in range(count):
                    task = Task(
                        name=f"task_{i}",
                        module="python",
                        operation="noop"
                    )
                    orchestrator.add_task_to_workflow(workflow.id, task)
                metrics.end(f"submit_{count}")
                
                # 执行并计时
                metrics.start(f"execute_{count}")
                await orchestrator.execute_workflow(workflow.id, wait=True)
                metrics.end(f"execute_{count}")
            
            # 输出报告
            print(metrics.report())
            
            # 验证性能要求
            stats_100 = metrics.get_stats("execute_100")
            throughput_100 = 100 / stats_100["mean"] if stats_100 else 0
            print(f"\nThroughput with 100 tasks: {throughput_100:.1f} tasks/s")
            assert throughput_100 > 20  # 至少20任务/秒
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_parallel_scaling_efficiency(self):
        """测试并行扩展效率"""
        orchestrator = get_orchestrator()
        
        executor = PythonFunctionExecutor()
        
        def cpu_task(duration=0.01):
            """CPU任务"""
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                pass  # 忙等待
            return "done"
        
        executor.register("cpu_task", cpu_task)
        orchestrator.add_executor(executor)
        
        await orchestrator.start()
        
        try:
            parallel_levels = [1, 2, 4, 8]
            results = {}
            
            for level in parallel_levels:
                workflow = orchestrator.create_workflow(f"parallel_{level}")
                
                # 创建并行任务
                for i in range(level):
                    task = Task(
                        name=f"task_{i}",
                        module="python",
                        operation="cpu_task",
                        params={"duration": 0.1},
                        execution_mode=ExecutionMode.PARALLEL
                    )
                    orchestrator.add_task_to_workflow(workflow.id, task)
                
                start = time.perf_counter()
                await orchestrator.execute_workflow(workflow.id, wait=True)
                elapsed = time.perf_counter() - start
                
                results[level] = elapsed
                print(f"Parallel level {level}: {elapsed:.3f}s")
            
            # 计算加速比
            baseline = results[1]
            for level, elapsed in results.items():
                speedup = baseline / elapsed
                efficiency = speedup / level
                print(f"Level {level}: Speedup={speedup:.2f}x, Efficiency={efficiency*100:.1f}%")
                
                # 验证扩展效率
                if level <= 4:  # 小规模应该接近线性
                    assert speedup >= level * 0.7  # 至少70%效率
            
        finally:
            await orchestrator.stop()


@pytest.mark.benchmark
class TestMemoryPerformance:
    """内存性能测试"""
    
    @pytest.mark.asyncio
    async def test_large_workflow_memory(self):
        """测试大型工作流内存使用"""
        orchestrator = get_orchestrator()
        
        executor = PythonFunctionExecutor()
        
        results_store = {}
        
        def store_result(key, size_mb):
            """存储大结果"""
            data = "x" * (size_mb * 1024 * 1024 // 10)  # 模拟数据
            results_store[key] = len(data)
            return {"stored": key, "size": size_mb}
        
        executor.register("store", store_result)
        orchestrator.add_executor(executor)
        
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("memory_test")
            
            # 创建产生大结果的任务
            for i in range(10):
                task = Task(
                    name=f"large_task_{i}",
                    module="python",
                    operation="store",
                    params={"key": f"result_{i}", "size_mb": 1}
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
            
            start = time.perf_counter()
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            elapsed = time.perf_counter() - start
            
            print(f"\nLarge workflow completed in {elapsed:.2f}s")
            assert result["all_success"] == True
            
        finally:
            await orchestrator.stop()


@pytest.mark.benchmark
class TestDependencyResolutionPerformance:
    """依赖解析性能测试"""
    
    def test_dependency_graph_resolution(self):
        """测试依赖图解析性能"""
        from dftlammps.unified import Workflow, Task
        
        metrics = PerformanceMetrics()
        
        # 测试不同复杂度的依赖图
        sizes = [10, 50, 100, 200]
        
        for size in sizes:
            workflow = Workflow(name=f"dep_test_{size}")
            
            # 创建链式依赖
            prev_task = None
            for i in range(size):
                task = Task(
                    name=f"task_{i}",
                    module="test",
                    operation="op"
                )
                if prev_task:
                    task.dependencies = [prev_task.id]
                workflow.add_task(task)
                prev_task = task
            
            metrics.start(f"resolve_chain_{size}")
            ready = workflow.get_ready_tasks()
            metrics.end(f"resolve_chain_{size}")
            
            # 只有第一个任务应该就绪
            assert len(ready) == 1
        
        print(metrics.report())
        
        # 验证性能 - 200个任务的依赖解析应该在1ms内
        stats = metrics.get_stats("resolve_chain_200")
        assert stats["mean"] < 0.001
    
    def test_complex_dependency_resolution(self):
        """测试复杂依赖图"""
        from dftlammps.unified import Workflow, Task
        
        workflow = Workflow(name="complex_deps")
        
        # 创建复杂的依赖图结构
        # A, B -> C -> D, E -> F
        
        task_a = Task(name="A", module="test", operation="op")
        task_b = Task(name="B", module="test", operation="op")
        workflow.add_task(task_a)
        workflow.add_task(task_b)
        
        task_c = Task(
            name="C",
            module="test",
            operation="op",
            dependencies=[task_a.id, task_b.id]
        )
        workflow.add_task(task_c)
        
        task_d = Task(
            name="D",
            module="test",
            operation="op",
            dependencies=[task_c.id]
        )
        task_e = Task(
            name="E",
            module="test",
            operation="op",
            dependencies=[task_c.id]
        )
        workflow.add_task(task_d)
        workflow.add_task(task_e)
        
        task_f = Task(
            name="F",
            module="test",
            operation="op",
            dependencies=[task_d.id, task_e.id]
        )
        workflow.add_task(task_f)
        
        # 验证初始就绪任务
        ready = workflow.get_ready_tasks()
        ready_names = {t.name for t in ready}
        assert ready_names == {"A", "B"}
        
        # 验证依赖图
        graph = workflow.get_dependency_graph()
        assert len(graph) > 0


@pytest.mark.benchmark
class TestAPILatency:
    """API延迟测试"""
    
    @pytest.mark.asyncio
    async def test_api_endpoint_latency(self):
        """测试API端点延迟"""
        from dftlammps.unified import init_api, APIRequest, HTTPMethod
        
        router = init_api()
        
        metrics = PerformanceMetrics()
        
        # 测试健康检查端点
        for i in range(100):
            request = APIRequest(
                path="/health",
                method=HTTPMethod.GET
            )
            
            metrics.start("health_check")
            response = await router.route(request)
            metrics.end("health_check")
            
            assert response.status.value == "success"
        
        print(metrics.report())
        
        # 验证延迟要求 - P99应该小于10ms
        stats = metrics.get_stats("health_check")
        p99 = np.percentile(metrics.metrics["health_check"], 99)
        print(f"\nP99 latency: {p99*1000:.2f}ms")
        assert p99 < 0.01  # 10ms


@pytest.mark.benchmark
class TestConfigurationPerformance:
    """配置系统性能测试"""
    
    def test_config_loading_performance(self, temp_dir):
        """测试配置加载性能"""
        from dftlammps.unified import ConfigManager
        import yaml
        
        config_path = temp_dir / "perf_config.yaml"
        
        # 创建大型配置
        config_data = {
            "project_name": "Performance Test",
            "custom_modules": {
                f"module_{i}": {
                    "params": {f"param_{j}": j for j in range(100)}
                }
                for i in range(100)
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # 性能测试
        times = []
        for _ in range(50):
            start = time.perf_counter()
            manager = ConfigManager()
            manager.load_config(config_path)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        mean_time = statistics.mean(times)
        print(f"\nConfig loading mean time: {mean_time*1000:.2f}ms")
        
        # 配置加载应该很快
        assert mean_time < 0.05  # 50ms
    
    def test_config_validation_performance(self):
        """测试配置验证性能"""
        from dftlammps.unified import ConfigManager, ConfigSchema
        
        manager = ConfigManager()
        
        # 注册大量验证规则
        schemas = [
            ConfigSchema(f"field_{i}", float, True, validator=lambda x: x > 0)
            for i in range(100)
        ]
        manager.register_schema("test", schemas)
        
        # 设置有效值
        for i in range(100):
            manager.global_config.custom_modules["test"] = {
                f"field_{i}": i + 1
            }
        
        # 性能测试
        times = []
        for _ in range(100):
            start = time.perf_counter()
            errors = manager.validate_config("test")
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        mean_time = statistics.mean(times)
        print(f"\nConfig validation mean time: {mean_time*1000:.2f}ms")
        
        assert mean_time < 0.001  # 1ms


@pytest.mark.benchmark
class TestResourceManagement:
    """资源管理性能测试"""
    
    @pytest.mark.asyncio
    async def test_resource_allocation_efficiency(self):
        """测试资源分配效率"""
        from dftlammps.unified import SmartScheduler, Task, ResourceRequirements
        
        scheduler = SmartScheduler(max_workers=4)
        
        metrics = PerformanceMetrics()
        
        # 创建具有不同资源需求的任务
        resource_profiles = [
            ResourceRequirements(cpu_cores=1, memory_gb=2),
            ResourceRequirements(cpu_cores=2, memory_gb=4),
            ResourceRequirements(cpu_cores=4, memory_gb=8),
        ]
        
        for i, profile in enumerate(resource_profiles):
            task = Task(
                name=f"res_task_{i}",
                module="test",
                operation="op",
                resources=profile
            )
            
            metrics.start(f"submit_res_{i}")
            await scheduler.submit(task)
            metrics.end(f"submit_res_{i}")
        
        print(metrics.report())
        
        # 验证调度器状态
        stats = scheduler.get_stats()
        assert stats["tasks_submitted"] == 3
