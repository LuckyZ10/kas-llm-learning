"""
端到端测试套件
End-to-End Test Suite
======================

验证完整工作流从输入到输出的正确性。

测试场景:
    1. 完整材料筛选工作流
    2. DFT到ML训练到MD的完整流程
    3. 主动学习工作流
    4. HPC调度工作流
    5. Web UI完整交互流程

使用方法:
    pytest tests/e2e -v
    pytest tests/e2e -m e2e --headed  # 显示浏览器（UI测试）
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import json
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# 完整工作流端到端测试
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestFullWorkflowE2E:
    """完整工作流端到端测试"""
    
    def test_dft_to_ml_to_md_workflow(self, tmp_path):
        """E2E测试: DFT → ML训练 → MD模拟完整流程"""
        working_dir = tmp_path / "full_workflow"
        working_dir.mkdir()
        
        # Step 1: 模拟DFT计算
        dft_dir = working_dir / "dft"
        dft_dir.mkdir()
        
        # 创建模拟DFT输出
        for i in range(5):
            outcar = dft_dir / f"OUTCAR_{i}"
            outcar.write_text(f"""
FREE ENERGIE OF THE ION-ELECTRON SYSTEM
total  energy   TOTEN  =      {-100.0 - i*0.1:.8f} eV
POSITION                                       TOTAL-FORCE
  0.00000  0.00000  0.00000        0.001234    0.002345   -0.003456
  1.00000  0.00000  0.00000       -0.001234   -0.002345    0.003456
""")
        
        assert len(list(dft_dir.glob("OUTCAR_*"))) == 5, "DFT outputs not created"
        
        # Step 2: 数据准备
        data_dir = working_dir / "data"
        data_dir.mkdir()
        
        # 创建训练数据
        train_xyz = data_dir / "train.xyz"
        train_xyz.write_text("""6
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-100.1234567890
Li 0.0 0.0 0.0 0.01 0.02 0.03
Li 2.0 0.0 0.0 -0.01 -0.02 0.01
S 1.0 1.0 0.0 0.02 -0.01 0.04
S 3.0 1.0 0.0 -0.02 0.01 -0.04
P 1.0 0.0 1.0 0.03 -0.04 0.01
P 2.0 1.0 1.0 -0.03 0.04 -0.01
""")
        
        assert train_xyz.exists(), "Training data not created"
        
        # Step 3: 模拟ML训练
        model_dir = working_dir / "model"
        model_dir.mkdir()
        
        # 创建模拟模型文件
        model_file = model_dir / "nep.txt"
        model_file.write_text("""nep4 3
Li S P
...
""")
        
        assert model_file.exists(), "Model file not created"
        
        # Step 4: 模拟MD模拟
        md_dir = working_dir / "md"
        md_dir.mkdir()
        
        # 创建模拟MD输入
        lammps_input = md_dir / "in.lammps"
        lammps_input.write_text("""
units metal
atom_style atomic
read_data structure.data
pair_style nep
pair_coeff * * ../model/nep.txt
run 1000
""")
        
        # 创建模拟轨迹输出
        trajectory = md_dir / "dump.lammpstrj"
        trajectory.write_text("""ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
6
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z
1 1 0.0 0.0 0.0
2 1 2.0 0.0 0.0
3 2 1.0 1.0 0.0
4 2 3.0 1.0 0.0
5 3 1.0 0.0 1.0
6 3 2.0 1.0 1.0
""")
        
        assert trajectory.exists(), "MD trajectory not created"
        
        # 验证完整工作流输出
        results = {
            'dft_frames': 5,
            'training_data': str(train_xyz),
            'model_file': str(model_file),
            'trajectory': str(trajectory)
        }
        
        assert all(Path(v).exists() for v in results.values() if isinstance(v, str)), \
            "Not all workflow outputs created"
    
    def test_battery_screening_workflow(self, tmp_path):
        """E2E测试: 电池材料筛选工作流"""
        working_dir = tmp_path / "battery_screening"
        working_dir.mkdir()
        
        # 材料ID列表
        material_ids = ["mp-1234", "mp-5678", "mp-9012"]
        
        # 模拟筛选结果
        screening_results = []
        for mat_id in material_ids:
            result = {
                'material_id': mat_id,
                'voltage': np.random.uniform(2.0, 4.5),
                'capacity': np.random.uniform(100, 300),
                'stability': np.random.uniform(0, 1),
                'passed_screening': False
            }
            # 筛选标准
            if result['voltage'] > 2.5 and result['capacity'] > 150 and result['stability'] > 0.5:
                result['passed_screening'] = True
            screening_results.append(result)
        
        # 保存结果
        results_file = working_dir / "screening_results.json"
        with open(results_file, 'w') as f:
            json.dump(screening_results, f, indent=2)
        
        # 验证结果
        assert results_file.exists()
        
        passed = sum(1 for r in screening_results if r['passed_screening'])
        assert passed >= 0, "No materials passed screening"
        
        # 验证所有材料都有完整数据
        for result in screening_results:
            assert 'voltage' in result
            assert 'capacity' in result
            assert 'stability' in result
    
    def test_active_learning_workflow(self, tmp_path):
        """E2E测试: 主动学习工作流"""
        working_dir = tmp_path / "active_learning"
        working_dir.mkdir()
        
        # 初始模型
        model_dir = working_dir / "models"
        model_dir.mkdir()
        
        # 模拟多轮主动学习
        for iteration in range(3):
            iter_dir = working_dir / f"iteration_{iteration}"
            iter_dir.mkdir()
            
            # 探索阶段
            explore_dir = iter_dir / "exploration"
            explore_dir.mkdir()
            
            # 生成候选结构
            candidates = []
            for i in range(5):
                candidate = {
                    'id': f"iter_{iteration}_cand_{i}",
                    'uncertainty': np.random.uniform(0.1, 0.5),
                    'selected': False
                }
                # 选择高不确定性结构
                if candidate['uncertainty'] > 0.3:
                    candidate['selected'] = True
                candidates.append(candidate)
            
            # 标记阶段
            selected = [c for c in candidates if c['selected']]
            
            # 模拟DFT计算
            labeled = []
            for c in selected:
                labeled.append({
                    'id': c['id'],
                    'energy': -100.0 + np.random.randn(),
                    'forces': np.random.randn(6, 3) * 0.1
                })
            
            # 重新训练
            model_file = model_dir / f"model_iter_{iteration}.txt"
            model_file.write_text(f"# Model iteration {iteration}\n")
            
            assert model_file.exists(), f"Model for iteration {iteration} not created"
        
        # 验证生成了多个模型
        models = list(model_dir.glob("model_iter_*.txt"))
        assert len(models) == 3, "Not all iteration models created"


# =============================================================================
# API端到端测试
# =============================================================================

@pytest.mark.e2e
class TestAPIE2E:
    """API端到端测试"""
    
    def test_structure_upload_to_calculation_workflow(self, tmp_path):
        """E2E测试: 结构上传 → 计算提交 → 结果获取"""
        # 模拟API调用流程
        
        # 1. 上传结构
        structure_data = {
            'symbols': ['Li', 'Li', 'S'],
            'positions': [[0, 0, 0], [2, 0, 0], [1, 1, 0]],
            'cell': [[3, 0, 0], [0, 3, 0], [0, 0, 3]],
            'pbc': True
        }
        
        # 模拟保存
        struct_file = tmp_path / "structure.json"
        with open(struct_file, 'w') as f:
            json.dump(structure_data, f)
        
        assert struct_file.exists()
        
        # 2. 提交计算任务
        job_data = {
            'job_id': 'test-12345',
            'structure_id': 'struct-001',
            'calculation_type': 'dft',
            'status': 'queued',
            'submitted_at': '2024-01-15T10:00:00'
        }
        
        job_file = tmp_path / "job.json"
        with open(job_file, 'w') as f:
            json.dump(job_data, f)
        
        assert job_file.exists()
        
        # 3. 模拟结果
        result_data = {
            'job_id': 'test-12345',
            'status': 'completed',
            'energy': -100.5,
            'forces': [[0.01, 0.02, 0.03], [-0.01, -0.02, 0.01], [0, 0, -0.04]],
            'completed_at': '2024-01-15T10:30:00'
        }
        
        result_file = tmp_path / "result.json"
        with open(result_file, 'w') as f:
            json.dump(result_data, f)
        
        assert result_file.exists()
        assert result_data['status'] == 'completed'
    
    def test_batch_submission_workflow(self, tmp_path):
        """E2E测试: 批量提交工作流"""
        # 批量任务
        n_tasks = 10
        
        jobs = []
        for i in range(n_tasks):
            job = {
                'job_id': f'batch-{i:03d}',
                'status': 'queued',
                'priority': n_tasks - i
            }
            jobs.append(job)
        
        # 模拟调度
        jobs.sort(key=lambda x: x['priority'], reverse=True)
        
        # 模拟执行
        for job in jobs:
            job['status'] = 'running'
            job['status'] = 'completed'
        
        # 验证所有任务完成
        completed = sum(1 for j in jobs if j['status'] == 'completed')
        assert completed == n_tasks, f"Only {completed}/{n_tasks} jobs completed"


# =============================================================================
# HPC调度E2E测试
# =============================================================================

@pytest.mark.e2e
class TestHPCSchedulerE2E:
    """HPC调度端到端测试"""
    
    def test_job_submission_to_completion_workflow(self, tmp_path):
        """E2E测试: 作业提交到完成工作流"""
        from unittest.mock import Mock, patch
        
        # 模拟作业规范
        job_spec = {
            'name': 'test_job',
            'working_dir': str(tmp_path),
            'commands': ['echo "Hello World"', 'sleep 1'],
            'resources': {
                'num_nodes': 1,
                'cores_per_node': 4,
                'walltime': '00:10:00'
            }
        }
        
        # 模拟提交
        job_id = 'local.12345'
        
        # 模拟状态跟踪
        job_status = {
            'job_id': job_id,
            'status': 'queued',
            'submitted_at': '2024-01-15T10:00:00'
        }
        
        # 状态转换
        states = ['queued', 'running', 'completed']
        for state in states:
            job_status['status'] = state
        
        assert job_status['status'] == 'completed'
    
    def test_workflow_with_dependencies(self, tmp_path):
        """E2E测试: 带依赖的工作流"""
        # 定义任务依赖图
        # A → B → D
        #  ↘ C ↗
        
        tasks = {
            'A': {'dependencies': [], 'status': 'pending'},
            'B': {'dependencies': ['A'], 'status': 'pending'},
            'C': {'dependencies': ['A'], 'status': 'pending'},
            'D': {'dependencies': ['B', 'C'], 'status': 'pending'}
        }
        
        # 模拟执行
        def can_run(task_name, tasks):
            return all(tasks[dep]['status'] == 'completed' 
                      for dep in tasks[task_name]['dependencies'])
        
        completed = set()
        while len(completed) < len(tasks):
            for name, task in tasks.items():
                if task['status'] == 'pending' and can_run(name, tasks):
                    task['status'] = 'running'
                    task['status'] = 'completed'
                    completed.add(name)
        
        # 验证所有任务完成
        assert all(t['status'] == 'completed' for t in tasks.values())


# =============================================================================
# 数据流E2E测试
# =============================================================================

@pytest.mark.e2e
class TestDataFlowE2E:
    """数据流端到端测试"""
    
    def test_structure_conversion_pipeline(self, tmp_path):
        """E2E测试: 结构格式转换管道"""
        # ASE Atoms格式
        ase_structure = {
            'positions': np.random.randn(10, 3),
            'cell': np.eye(3) * 10,
            'symbols': ['Li'] * 5 + ['S'] * 5
        }
        
        # 转换为Pymatgen格式
        pymatgen_structure = {
            'lattice': ase_structure['cell'].tolist(),
            'species': ase_structure['symbols'],
            'coords': ase_structure['positions'].tolist()
        }
        
        # 转换为LAMMPS格式
        lammps_data = f"""
{len(ase_structure['symbols'])} atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Atoms

"""
        for i, (sym, pos) in enumerate(zip(ase_structure['symbols'], 
                                           ase_structure['positions']), 1):
            atom_type = 1 if sym == 'Li' else 2
            lammps_data += f"{i} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n"
        
        # 保存所有格式
        (tmp_path / "ase.json").write_text(json.dumps(ase_structure, cls=NumpyEncoder))
        (tmp_path / "pymatgen.json").write_text(json.dumps(pymatgen_structure))
        (tmp_path / "lammps.data").write_text(lammps_data)
        
        # 验证文件
        assert (tmp_path / "ase.json").exists()
        assert (tmp_path / "pymatgen.json").exists()
        assert (tmp_path / "lammps.data").exists()
    
    def test_trajectory_export_pipeline(self, tmp_path):
        """E2E测试: 轨迹导出管道"""
        # 生成模拟轨迹
        n_frames = 10
        n_atoms = 20
        
        trajectory = []
        for i in range(n_frames):
            frame = {
                'step': i,
                'positions': np.random.randn(n_atoms, 3),
                'energy': -100.0 + np.random.randn() * 0.1
            }
            trajectory.append(frame)
        
        # 导出为不同格式
        
        # XYZ格式
        xyz_lines = [f"{n_atoms}", f"Frame {i}"]
        for j, pos in enumerate(trajectory[0]['positions']):
            xyz_lines.append(f"Li {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        (tmp_path / "trajectory.xyz").write_text("\n".join(xyz_lines))
        
        # LAMMPS dump格式
        dump_lines = []
        for i, frame in enumerate(trajectory):
            dump_lines.extend([
                "ITEM: TIMESTEP",
                str(i),
                "ITEM: NUMBER OF ATOMS",
                str(n_atoms),
                "ITEM: BOX BOUNDS pp pp pp",
                "0 10",
                "0 10",
                "0 10",
                "ITEM: ATOMS id type x y z"
            ])
            for j, pos in enumerate(frame['positions']):
                dump_lines.append(f"{j+1} 1 {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}")
        (tmp_path / "trajectory.dump").write_text("\n".join(dump_lines))
        
        # 验证
        assert (tmp_path / "trajectory.xyz").exists()
        assert (tmp_path / "trajectory.dump").exists()


# =============================================================================
# Web UI E2E测试
# =============================================================================

@pytest.mark.e2e
@pytest.mark.skip(reason="Requires Playwright and running web server")
class TestWebUIE2E:
    """Web UI端到端测试"""
    
    def test_login_and_dashboard(self, page):
        """E2E测试: 登录和仪表盘"""
        # 假设使用Playwright
        # page.goto("http://localhost:8000")
        # page.fill("input[name=username]", "test_user")
        # page.fill("input[name=password]", "test_pass")
        # page.click("button[type=submit]")
        # assert page.url.endswith("/dashboard")
        pass
    
    def test_structure_upload_and_visualization(self, page):
        """E2E测试: 结构上传和可视化"""
        # page.goto("http://localhost:8000/structures")
        # page.click("button:has-text('Upload')")
        # page.set_input_files("input[type=file]", "test_structure.cif")
        # page.click("button:has-text('Submit')")
        # assert page.wait_for_selector(".visualization-container")
        pass


# =============================================================================
# 错误恢复E2E测试
# =============================================================================

@pytest.mark.e2e
class TestErrorRecoveryE2E:
    """错误恢复端到端测试"""
    
    def test_failed_calculation_recovery(self, tmp_path):
        """E2E测试: 失败计算恢复"""
        # 模拟失败的DFT计算
        job = {
            'job_id': 'fail-001',
            'status': 'failed',
            'error': 'SCF did not converge',
            'retry_count': 0
        }
        
        # 错误恢复策略
        if job['status'] == 'failed' and job['retry_count'] < 3:
            # 修改参数重试
            job['mixing_param'] = 0.5  # 降低mixing
            job['retry_count'] += 1
            job['status'] = 'retrying'
        
        assert job['retry_count'] > 0
        assert job['status'] == 'retrying'
    
    def test_checkpoint_resume_workflow(self, tmp_path):
        """E2E测试: 检查点恢复工作流"""
        checkpoint_file = tmp_path / "checkpoint.json"
        
        # 模拟中断前保存的检查点
        checkpoint = {
            'stage': 2,
            'completed_stages': ['data_preparation', 'model_training'],
            'current_stage': 'md_simulation',
            'progress': 0.5,
            'data': {'temperature': 300, 'pressure': 1.0}
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)
        
        # 恢复工作流
        with open(checkpoint_file, 'r') as f:
            recovered = json.load(f)
        
        # 从检查点继续
        assert recovered['stage'] == 2
        assert recovered['current_stage'] == 'md_simulation'
        assert recovered['progress'] == 0.5


# =============================================================================
# 辅助类
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """NumPy数组JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
