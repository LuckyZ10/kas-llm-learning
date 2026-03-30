"""
DFT+LAMMPS Integration Tests - Workflow Tests
=============================================
工作流集成测试
"""

import pytest
import asyncio
import time
from dftlammps.unified import (
    OrchestratorV2, WorkflowBuilder, Workflow,
    Task, TaskResult, TaskStatus, TaskPriority, ExecutionMode,
    ResourceRequirements, PythonFunctionExecutor,
    get_orchestrator, run_workflow
)


@pytest.mark.integration
class TestDFTWorkflows:
    """DFT工作流测试"""
    
    @pytest.mark.asyncio
    async def test_dft_relaxation_workflow(self, orchestrator, mock_dft_executor):
        """测试DFT结构优化工作流"""
        orchestrator.add_executor(mock_dft_executor)
        await orchestrator.start()
        
        try:
            builder = WorkflowBuilder("dft_relaxation", orchestrator)
            
            builder.add_task(
                name="scf_calc",
                module="dft",
                operation="calculate",
                params={"structure": "Li3PS4", "encut": 520},
                priority=TaskPriority.HIGH
            ).add_task(
                name="relaxation",
                module="dft",
                operation="relax",
                params={"structure": "Li3PS4", "steps": 100},
                priority=TaskPriority.HIGH
            ).add_task(
                name="final_scf",
                module="dft",
                operation="calculate",
                params={"structure": "relaxed_structure", "encut": 600},
                priority=TaskPriority.NORMAL
            )
            
            result = await builder.execute()
            
            assert result["all_success"] == True
            assert len(result["results"]) == 3
            assert len(mock_dft_executor.calculations) == 3
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_phonon_workflow(self, orchestrator, mock_dft_executor):
        """测试声子计算工作流"""
        orchestrator.add_executor(mock_dft_executor)
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("phonon_calc")
            
            # 结构优化
            relax_task = Task(
                name="relax",
                module="dft",
                operation="relax",
                params={"structure": "input_structure"}
            )
            orchestrator.add_task_to_workflow(workflow.id, relax_task)
            
            # 声子计算（依赖结构优化）
            phonon_task = Task(
                name="phonon",
                module="dft",
                operation="phonon",
                params={"structure": "relaxed", "mesh": [11, 11, 11]},
                dependencies=[relax_task.id]
            )
            orchestrator.add_task_to_workflow(workflow.id, phonon_task)
            
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert "relax" in result["results"]
            assert "phonon" in result["results"]
        finally:
            await orchestrator.stop()


@pytest.mark.integration
class TestMLWorkflows:
    """机器学习工作流测试"""
    
    @pytest.mark.asyncio
    async def test_ml_training_pipeline(self, orchestrator, mock_ml_executor, mock_dft_executor):
        """测试ML训练流程"""
        orchestrator.add_executor(mock_ml_executor)
        orchestrator.add_executor(mock_dft_executor)
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("ml_training_pipeline")
            
            # 生成训练数据
            data_tasks = []
            for i in range(5):
                task = Task(
                    name=f"dft_calc_{i}",
                    module="dft",
                    operation="calculate",
                    params={"structure": f"structure_{i}", "encut": 520}
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
                data_tasks.append(task)
            
            # 训练模型（依赖所有DFT计算）
            train_task = Task(
                name="train_model",
                module="ml",
                operation="train",
                params={"dataset": "dft_results", "epochs": 1000, "batch_size": 32},
                dependencies=[t.id for t in data_tasks],
                resources=ResourceRequirements(gpu_count=1, memory_gb=16)
            )
            orchestrator.add_task_to_workflow(workflow.id, train_task)
            
            # 评估模型
            eval_task = Task(
                name="evaluate_model",
                module="ml",
                operation="evaluate",
                params={"model_path": "trained_model", "test_dataset": "test_data"},
                dependencies=[train_task.id]
            )
            orchestrator.add_task_to_workflow(workflow.id, eval_task)
            
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert len(mock_dft_executor.calculations) == 5
            assert len(mock_ml_executor.trainings) == 1
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_active_learning_loop(self, orchestrator, mock_ml_executor, mock_dft_executor):
        """测试主动学习循环"""
        orchestrator.add_executor(mock_ml_executor)
        orchestrator.add_executor(mock_dft_executor)
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("active_learning")
            
            # 初始训练
            initial_train = Task(
                name="initial_train",
                module="ml",
                operation="train",
                params={"dataset": "initial", "epochs": 500}
            )
            orchestrator.add_task_to_workflow(workflow.id, initial_train)
            
            # 迭代循环（简化版）
            prev_task = initial_train
            for iteration in range(3):
                # 预测
                predict_task = Task(
                    name=f"predict_{iteration}",
                    module="ml",
                    operation="predict",
                    params={"model_path": "current_model", "structures": ["s1", "s2", "s3"]},
                    dependencies=[prev_task.id]
                )
                orchestrator.add_task_to_workflow(workflow.id, predict_task)
                
                # DFT验证
                dft_task = Task(
                    name=f"dft_validate_{iteration}",
                    module="dft",
                    operation="calculate",
                    params={"structure": f"selected_{iteration}"},
                    dependencies=[predict_task.id]
                )
                orchestrator.add_task_to_workflow(workflow.id, dft_task)
                
                # 重训练
                retrain_task = Task(
                    name=f"retrain_{iteration}",
                    module="ml",
                    operation="train",
                    params={"dataset": f"expanded_{iteration}", "epochs": 300},
                    dependencies=[dft_task.id]
                )
                orchestrator.add_task_to_workflow(workflow.id, retrain_task)
                
                prev_task = retrain_task
            
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            # 初始训练 + 3次重训练 = 4次训练
            assert len(mock_ml_executor.trainings) == 4
        finally:
            await orchestrator.stop()


@pytest.mark.integration
class TestLAMMPSWorkflows:
    """LAMMPS工作流测试"""
    
    @pytest.mark.asyncio
    async def test_md_simulation_workflow(self, orchestrator, mock_lammps_executor):
        """测试MD模拟工作流"""
        orchestrator.add_executor(mock_lammps_executor)
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("md_simulation")
            
            # 能量最小化
            min_task = Task(
                name="minimization",
                module="lammps",
                operation="minimize",
                params={"potential": "deepmd_model", "max_iter": 1000}
            )
            orchestrator.add_task_to_workflow(workflow.id, min_task)
            
            # 升温
            heat_task = Task(
                name="heating",
                module="lammps",
                operation="md_run",
                params={"potential": "deepmd_model", "temperature": 300, "steps": 50000},
                dependencies=[min_task.id]
            )
            orchestrator.add_task_to_workflow(workflow.id, heat_task)
            
            # 平衡
            equil_task = Task(
                name="equilibration",
                module="lammps",
                operation="npt",
                params={"potential": "deepmd_model", "temperature": 300, "pressure": 1.0, "steps": 100000},
                dependencies=[heat_task.id]
            )
            orchestrator.add_task_to_workflow(workflow.id, equil_task)
            
            # 生产运行
            prod_task = Task(
                name="production",
                module="lammps",
                operation="md_run",
                params={"potential": "deepmd_model", "temperature": 300, "steps": 500000},
                dependencies=[equil_task.id]
            )
            orchestrator.add_task_to_workflow(workflow.id, prod_task)
            
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            assert result["all_success"] == True
            assert len(result["results"]) == 4
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_multi_temperature_sampling(self, orchestrator, mock_lammps_executor):
        """测试多温度并行采样"""
        orchestrator.add_executor(mock_lammps_executor)
        await orchestrator.start()
        
        try:
            builder = WorkflowBuilder("multi_temp", orchestrator)
            
            # 先进行能量最小化
            builder.add_task(
                name="minimize",
                module="lammps",
                operation="minimize",
                params={"potential": "model"}
            )
            
            # 并行运行不同温度
            temps = [100, 300, 500, 700, 900]
            temp_tasks = []
            for temp in temps:
                task = Task(
                    name=f"md_{temp}K",
                    module="lammps",
                    operation="md_run",
                    params={"potential": "model", "temperature": temp, "steps": 10000},
                    execution_mode=ExecutionMode.PARALLEL
                )
                temp_tasks.append(task)
            
            builder.parallel(*temp_tasks)
            
            result = await builder.execute()
            
            assert result["all_success"] == True
            assert len(result["results"]) == 1 + len(temps)  # minimize + temp runs
        finally:
            await orchestrator.stop()


@pytest.mark.integration
class TestEndToEndWorkflows:
    """端到端工作流测试"""
    
    @pytest.mark.asyncio
    async def test_complete_dft_ml_md_pipeline(self, orchestrator, 
                                               mock_dft_executor,
                                               mock_ml_executor,
                                               mock_lammps_executor):
        """测试完整DFT-ML-MD流程"""
        orchestrator.add_executor(mock_dft_executor)
        orchestrator.add_executor(mock_ml_executor)
        orchestrator.add_executor(mock_lammps_executor)
        await orchestrator.start()
        
        try:
            workflow = orchestrator.create_workflow("complete_pipeline")
            
            # 1. DFT数据生成
            dft_tasks = []
            structures = ["Li3PS4_bulk", "Li3PS4_surface", "Li3PS4_defect"]
            for struct in structures:
                task = Task(
                    name=f"dft_{struct}",
                    module="dft",
                    operation="calculate",
                    params={"structure": struct}
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
                dft_tasks.append(task)
            
            # 2. ML势训练
            ml_train_task = Task(
                name="train_potential",
                module="ml",
                operation="train",
                params={"dataset": "dft_data", "epochs": 10000},
                dependencies=[t.id for t in dft_tasks]
            )
            orchestrator.add_task_to_workflow(workflow.id, ml_train_task)
            
            # 3. ML势验证（并行）
            validation_tasks = []
            for struct in structures:
                task = Task(
                    name=f"validate_{struct}",
                    module="ml",
                    operation="predict",
                    params={"model_path": "trained_model", "structures": [struct]},
                    dependencies=[ml_train_task.id],
                    execution_mode=ExecutionMode.PARALLEL
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
                validation_tasks.append(task)
            
            # 4. 大规模MD模拟（并行）
            md_tasks = []
            conditions = [
                ("nvt_300K", {"temperature": 300, "ensemble": "nvt"}),
                ("npt_300K", {"temperature": 300, "pressure": 1.0, "ensemble": "npt"}),
                ("nvt_500K", {"temperature": 500, "ensemble": "nvt"}),
            ]
            for name, params in conditions:
                task = Task(
                    name=f"md_{name}",
                    module="lammps",
                    operation="md_run",
                    params={"potential": "trained_model", **params, "steps": 100000},
                    dependencies=[ml_train_task.id],
                    execution_mode=ExecutionMode.PARALLEL
                )
                orchestrator.add_task_to_workflow(workflow.id, task)
                md_tasks.append(task)
            
            # 5. 结果分析（依赖所有MD完成）
            all_md_ids = [t.id for t in md_tasks]
            analysis_task = Task(
                name="analyze_results",
                module="python",
                operation="analyze",
                params={"md_results": "all_trajectories"},
                dependencies=all_md_ids
            )
            # 使用Python执行器
            py_exec = PythonFunctionExecutor()
            py_exec.register("analyze", lambda md_results: {
                "diffusion_coefficients": {"Li": 1e-5, "P": 1e-7, "S": 1e-6},
                "conductivity": 0.5,
                "analysis_complete": True
            })
            orchestrator.add_executor(py_exec)
            orchestrator.add_task_to_workflow(workflow.id, analysis_task)
            
            result = await orchestrator.execute_workflow(workflow.id, wait=True)
            
            # 验证
            assert result["all_success"] == True
            assert len(result["results"]) == (
                len(structures) +  # DFT计算
                1 +  # ML训练
                len(structures) +  # 验证
                len(conditions) +  # MD模拟
                1  # 分析
            )
            
            # 验证依赖关系正确
            assert "dft_Li3PS4_bulk" in result["results"]
            assert "train_potential" in result["results"]
            assert "md_nvt_300K" in result["results"]
            assert "analyze_results" in result["results"]
            
        finally:
            await orchestrator.stop()
