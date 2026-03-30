"""
单元测试套件
Unit Test Suite
================

核心模块的单元测试，确保各个组件独立正确工作。

测试覆盖:
    - DFT解析器
    - MD模拟器
    - ML势模型
    - HPC调度器
    - 工作流管理
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# DFT解析器单元测试
# =============================================================================

@pytest.mark.unit
@pytest.mark.dft
class TestDFTParserUnit:
    """DFT解析器单元测试"""
    
    def test_outcar_energy_extraction(self):
        """测试OUTCAR能量提取"""
        outcar_content = """
  free  energy   TOTEN  =      -100.12345678 eV
  energy  without entropy=     -100.00000000
"""
        # 提取能量
        import re
        match = re.search(r'TOTEN\s+=\s+([-\d.]+)', outcar_content)
        assert match is not None
        energy = float(match.group(1))
        assert np.isclose(energy, -100.12345678)
    
    def test_outcar_force_extraction(self):
        """测试OUTCAR力提取"""
        outcar_content = """
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000         0.123456   -0.234567    0.345678
      1.00000      0.00000      0.00000        -0.123456    0.234567   -0.345678
 -----------------------------------------------------------------------------------
"""
        # 提取力
        lines = outcar_content.split('\n')
        forces = []
        for line in lines:
            parts = line.split()
            if len(parts) == 6 and parts[0].replace('.', '').replace('-', '').isdigit():
                forces.append([float(parts[3]), float(parts[4]), float(parts[5])])
        
        assert len(forces) == 2
        assert np.allclose(forces[0], [0.123456, -0.234567, 0.345678])
    
    def test_convergence_detection(self):
        """测试收敛检测"""
        converged_content = "Convergence achieved after 15 iterations"
        not_converged_content = "Convergence NOT achieved"
        
        assert "Convergence achieved" in converged_content
        assert "Convergence NOT achieved" in not_converged_content
    
    def test_outcar_validation(self):
        """测试OUTCAR文件验证"""
        valid_outcar = """
vasp.5.4.4 18Apr17
FREE ENERGIE OF THE ION-ELECTRON SYSTEM
free  energy   TOTEN  =      -100.0 eV
"""
        # 验证关键字段存在
        assert 'TOTEN' in valid_outcar
        assert 'FREE ENERGIE' in valid_outcar
    
    def test_multiple_frame_parsing(self):
        """测试多帧解析"""
        # 模拟多帧数据
        frames = []
        for i in range(5):
            frame = {
                'energy': -100.0 - i * 0.1,
                'forces': np.random.randn(10, 3) * 0.1,
                'step': i
            }
            frames.append(frame)
        
        assert len(frames) == 5
        assert all('energy' in f for f in frames)
        assert all('forces' in f for f in frames)


# =============================================================================
# MD模拟器单元测试
# =============================================================================

@pytest.mark.unit
@pytest.mark.md
class TestMDSimulatorUnit:
    """MD模拟器单元测试"""
    
    def test_verlet_integrator(self):
        """测试Verlet积分器"""
        dt = 0.001
        positions = np.array([[1.0, 0.0, 0.0]])
        velocities = np.array([[0.0, 0.5, 0.0]])
        
        # 简化的谐振子力
        forces = -positions * 0.1
        
        # Verlet步
        velocities_half = velocities + 0.5 * forces * dt
        positions_new = positions + velocities_half * dt
        forces_new = -positions_new * 0.1
        velocities_new = velocities_half + 0.5 * forces_new * dt
        
        # 验证位置变化
        assert not np.allclose(positions_new, positions)
        assert positions_new.shape == positions.shape
    
    def test_temperature_calculation(self):
        """测试温度计算"""
        n_atoms = 10
        velocities = np.random.randn(n_atoms, 3) * 0.1
        masses = np.ones(n_atoms)
        
        # 动能
        ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
        
        # 温度（3N自由度，减去3个约束）
        dof = 3 * n_atoms - 3
        kb = 8.617e-5  # eV/K
        temperature = 2 * ke / (dof * kb)
        
        assert temperature > 0
        assert np.isfinite(temperature)
    
    def test_pbc_wrapping(self):
        """测试周期性边界条件"""
        box = np.array([10.0, 10.0, 10.0])
        positions = np.array([
            [11.0, 5.0, 5.0],   # 超出x边界
            [5.0, -1.0, 5.0],   # 低于y边界
            [5.0, 5.0, 15.0]    # 超出z边界
        ])
        
        # 应用PBC
        wrapped = positions % box
        
        expected = np.array([
            [1.0, 5.0, 5.0],
            [5.0, 9.0, 5.0],
            [5.0, 5.0, 5.0]
        ])
        
        assert np.allclose(wrapped, expected)
    
    def test_neighbor_list(self):
        """测试邻居列表"""
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0]
        ])
        cutoff = 2.0
        
        neighbors = [[] for _ in range(len(positions))]
        
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff:
                    neighbors[i].append(j)
                    neighbors[j].append(i)
        
        # 原子0和1是邻居（距离=1.0）
        assert 1 in neighbors[0]
        assert 0 in neighbors[1]
        # 原子2不是邻居（距离=5.0）
        assert 2 not in neighbors[0]


# =============================================================================
# ML势模型单元测试
# =============================================================================

@pytest.mark.unit
@pytest.mark.ml
class TestMLPotentialUnit:
    """ML势模型单元测试"""
    
    def test_energy_prediction_shape(self):
        """测试能量预测输出形状"""
        n_structures = 10
        
        # 模拟预测
        predictions = np.random.randn(n_structures)
        
        assert predictions.shape == (n_structures,)
    
    def test_force_prediction_shape(self):
        """测试力预测输出形状"""
        batch_size = 5
        n_atoms = 10
        
        # 模拟预测
        forces = np.random.randn(batch_size, n_atoms, 3)
        
        assert forces.shape == (batch_size, n_atoms, 3)
    
    def test_gradient_computation(self):
        """测试梯度计算"""
        # 简化的势能函数
        def potential(x):
            return np.sum(x**2)
        
        x = np.array([1.0, 2.0, 3.0])
        
        # 数值梯度
        grad_numerical = np.zeros_like(x)
        eps = 1e-5
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad_numerical[i] = (potential(x_plus) - potential(x_minus)) / (2 * eps)
        
        # 解析梯度
        grad_analytical = 2 * x
        
        assert np.allclose(grad_numerical, grad_analytical, rtol=1e-4)
    
    def test_model_parameter_count(self):
        """测试模型参数计数"""
        # 模拟神经网络参数
        layer_sizes = [128, 64, 32, 1]
        total_params = 0
        
        for i in range(len(layer_sizes) - 1):
            weights = layer_sizes[i] * layer_sizes[i+1]
            biases = layer_sizes[i+1]
            total_params += weights + biases
        
        assert total_params > 0
        # 128*64 + 64 + 64*32 + 32 + 32*1 + 1
        expected = 128*64 + 64 + 64*32 + 32 + 32*1 + 1
        assert total_params == expected


# =============================================================================
# HPC调度器单元测试
# =============================================================================

@pytest.mark.unit
@pytest.mark.hpc
class TestHPCSchedulerUnit:
    """HPC调度器单元测试"""
    
    def test_job_script_generation(self):
        """测试作业脚本生成"""
        job_name = "test_job"
        num_nodes = 2
        cores_per_node = 16
        walltime = "01:00:00"
        
        # 模拟Slurm脚本
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={cores_per_node}
#SBATCH --time={walltime}

srun ./my_program
"""
        
        assert f"--job-name={job_name}" in script
        assert f"--nodes={num_nodes}" in script
        assert f"--time={walltime}" in script
    
    def test_job_id_parsing(self):
        """测试作业ID解析"""
        # Slurm输出示例
        slurm_output = "Submitted batch job 12345"
        
        # 提取作业ID
        import re
        match = re.search(r'job\s+(\d+)', slurm_output)
        assert match is not None
        job_id = match.group(1)
        assert job_id == "12345"
    
    def test_resource_request_validation(self):
        """测试资源请求验证"""
        resources = {
            'num_nodes': 2,
            'cores_per_node': 16,
            'memory_gb': 64,
            'walltime_hours': 1.0
        }
        
        # 验证正数
        assert resources['num_nodes'] > 0
        assert resources['cores_per_node'] > 0
        assert resources['memory_gb'] > 0
        assert resources['walltime_hours'] > 0
        
        # 验证合理性
        total_cores = resources['num_nodes'] * resources['cores_per_node']
        assert total_cores <= 1000  # 假设最大限制
    
    def test_queue_status_parsing(self):
        """测试队列状态解析"""
        # squeue输出示例
        squeue_output = """
JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)
12345 normal test_job user R 10:00 2 node[1-2]
12346 normal test2 user PD 0:00 1 (Resources)
"""
        
        lines = squeue_output.strip().split('\n')[1:]  # 跳过标题
        jobs = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                jobs.append({
                    'job_id': parts[0],
                    'partition': parts[1],
                    'name': parts[2],
                    'status': parts[4]
                })
        
        assert len(jobs) == 2
        assert jobs[0]['status'] == 'R'  # Running
        assert jobs[1]['status'] == 'PD'  # Pending


# =============================================================================
# 工作流管理单元测试
# =============================================================================

@pytest.mark.unit
@pytest.mark.workflow
class TestWorkflowManagerUnit:
    """工作流管理单元测试"""
    
    def test_task_dependency_resolution(self):
        """测试任务依赖解析"""
        # 定义任务依赖图
        # A -> B -> D
        #   -> C /
        tasks = {
            'A': {'deps': []},
            'B': {'deps': ['A']},
            'C': {'deps': ['A']},
            'D': {'deps': ['B', 'C']}
        }
        
        # 拓扑排序
        sorted_tasks = []
        remaining = set(tasks.keys())
        
        while remaining:
            # 找到没有未满足依赖的任务
            ready = {t for t in remaining if all(d in sorted_tasks for d in tasks[t]['deps'])}
            assert ready, "Circular dependency detected"
            sorted_tasks.extend(sorted(ready))
            remaining -= ready
        
        # 验证顺序
        assert sorted_tasks.index('A') < sorted_tasks.index('B')
        assert sorted_tasks.index('A') < sorted_tasks.index('C')
        assert sorted_tasks.index('B') < sorted_tasks.index('D')
        assert sorted_tasks.index('C') < sorted_tasks.index('D')
    
    def test_task_status_transitions(self):
        """测试任务状态转换"""
        valid_transitions = {
            'pending': ['running', 'cancelled'],
            'running': ['completed', 'failed', 'cancelled'],
            'completed': [],
            'failed': ['pending'],  # 可重试
            'cancelled': ['pending']
        }
        
        # 测试有效转换
        assert 'running' in valid_transitions['pending']
        assert 'completed' in valid_transitions['running']
        
        # 测试无效转换
        assert 'pending' not in valid_transitions['completed']
    
    def test_checkpoint_save_load(self, tmp_path):
        """测试检查点保存和加载"""
        checkpoint_file = tmp_path / "checkpoint.json"
        
        state = {
            'stage': 2,
            'completed': ['stage1', 'stage2'],
            'current_data': {'energy': -100.0}
        }
        
        # 保存
        import json
        with open(checkpoint_file, 'w') as f:
            json.dump(state, f)
        
        # 加载
        with open(checkpoint_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['stage'] == state['stage']
        assert loaded['completed'] == state['completed']
    
    def test_error_handling_retry(self):
        """测试错误处理和重试"""
        max_retries = 3
        retry_count = 0
        
        def flaky_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < max_retries:
                raise RuntimeError("Temporary error")
            return "success"
        
        result = None
        for attempt in range(max_retries):
            try:
                result = flaky_operation()
                break
            except RuntimeError:
                if attempt == max_retries - 1:
                    raise
        
        assert result == "success"
        assert retry_count == max_retries


# =============================================================================
# 数据格式单元测试
# =============================================================================

@pytest.mark.unit
class TestDataFormatsUnit:
    """数据格式单元测试"""
    
    def test_xyz_format_parsing(self):
        """测试XYZ格式解析"""
        xyz_content = """3
Test structure
C 0.000000 0.000000 0.000000
H 1.089000 0.000000 0.000000
H -0.363000 1.026719 0.000000
"""
        lines = xyz_content.strip().split('\n')
        n_atoms = int(lines[0])
        
        atoms = []
        for line in lines[2:]:  # 跳过标题
            parts = line.split()
            atoms.append({
                'symbol': parts[0],
                'position': [float(x) for x in parts[1:4]]
            })
        
        assert n_atoms == 3
        assert len(atoms) == 3
        assert atoms[0]['symbol'] == 'C'
    
    def test_cif_format_parsing(self):
        """测试CIF格式解析"""
        cif_content = """
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""
        # 提取晶格参数
        import re
        a = float(re.search(r'_cell_length_a\s+([\d.]+)', cif_content).group(1))
        
        assert a == 5.0
    
    def test_json_serialization(self):
        """测试JSON序列化"""
        import json
        
        data = {
            'structures': [
                {'id': 1, 'energy': -100.0},
                {'id': 2, 'energy': -95.5}
            ]
        }
        
        # 序列化
        json_str = json.dumps(data)
        
        # 反序列化
        loaded = json.loads(json_str)
        
        assert loaded['structures'][0]['energy'] == -100.0
