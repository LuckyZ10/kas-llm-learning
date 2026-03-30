"""
测试工具库
Testing Utilities Library
==========================

提供测试辅助函数和模拟数据生成器。

包含:
    - 模拟数据生成器
    - 文件 fixtures 管理
    - 性能测量工具
    - 数值比较工具
    - 测试报告生成器
"""

import os
import json
import pickle
import hashlib
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time
import warnings


# =============================================================================
# 模拟数据生成器
# =============================================================================

class MockDataGenerator:
    """模拟数据生成器"""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_structure(self, 
                          n_atoms: int = 10,
                          species: Optional[List[str]] = None,
                          cell_size: float = 10.0) -> Dict:
        """生成模拟原子结构"""
        if species is None:
            species = ['Li'] * (n_atoms // 2) + ['S'] * (n_atoms - n_atoms // 2)
        
        positions = self.rng.rand(n_atoms, 3) * cell_size
        
        return {
            'positions': positions,
            'species': species[:n_atoms],
            'cell': np.eye(3) * cell_size,
            'pbc': True
        }
    
    def generate_trajectory(self,
                           n_frames: int = 100,
                           n_atoms: int = 10,
                           timestep: float = 1.0) -> List[Dict]:
        """生成模拟轨迹"""
        frames = []
        positions = self.rng.rand(n_atoms, 3) * 10
        
        for i in range(n_frames):
            # 添加随机位移
            positions += self.rng.randn(n_atoms, 3) * 0.01
            
            frame = {
                'step': i,
                'time': i * timestep,
                'positions': positions.copy(),
                'energy': -100.0 + self.rng.randn() * 0.1,
                'temperature': 300.0 + self.rng.randn() * 10
            }
            frames.append(frame)
        
        return frames
    
    def generate_dft_output(self,
                           n_atoms: int = 10,
                           converged: bool = True) -> Dict:
        """生成模拟DFT输出"""
        forces = self.rng.randn(n_atoms, 3) * 0.1
        
        return {
            'energy': -100.0 + self.rng.randn(),
            'forces': forces,
            'stress': self.rng.randn(6) * 0.01,
            'converged': converged,
            'n_steps': self.rng.randint(10, 100) if converged else 100,
            'fermi_energy': -2.5 + self.rng.randn() * 0.1
        }
    
    def generate_ml_predictions(self,
                               n_structures: int = 10,
                               with_uncertainty: bool = True) -> List[Dict]:
        """生成模拟ML预测"""
        predictions = []
        
        for i in range(n_structures):
            pred = {
                'structure_id': f'struct_{i:04d}',
                'energy': -100.0 + self.rng.randn() * 5,
                'forces': self.rng.randn(10, 3) * 0.5
            }
            
            if with_uncertainty:
                pred['energy_uncertainty'] = abs(self.rng.randn()) * 0.1
                pred['force_uncertainty'] = abs(self.rng.randn(10, 3)) * 0.05
            
            predictions.append(pred)
        
        return predictions
    
    def generate_workflow_status(self,
                                n_tasks: int = 5) -> Dict:
        """生成模拟工作流状态"""
        statuses = ['pending', 'running', 'completed', 'failed']
        
        tasks = []
        for i in range(n_tasks):
            task = {
                'task_id': f'task_{i:03d}',
                'status': self.rng.choice(statuses),
                'progress': self.rng.uniform(0, 100),
                'started_at': None,
                'completed_at': None
            }
            
            if task['status'] in ['running', 'completed', 'failed']:
                task['started_at'] = '2024-01-15T10:00:00'
            if task['status'] in ['completed', 'failed']:
                task['completed_at'] = '2024-01-15T10:30:00'
            
            tasks.append(task)
        
        return {
            'workflow_id': 'wf-001',
            'status': 'running' if any(t['status'] == 'running' for t in tasks) else 'completed',
            'tasks': tasks,
            'progress': np.mean([t['progress'] for t in tasks])
        }


# =============================================================================
# Fixtures 管理器
# =============================================================================

class FixtureManager:
    """测试fixtures管理器"""
    
    def __init__(self, fixtures_dir: Optional[Path] = None):
        if fixtures_dir is None:
            fixtures_dir = Path(__file__).parent.parent / "fixtures"
        self.fixtures_dir = Path(fixtures_dir)
        self.data_dir = self.fixtures_dir / "data"
        self.models_dir = self.fixtures_dir / "models"
        self.configs_dir = self.fixtures_dir / "configs"
        
        for d in [self.data_dir, self.models_dir, self.configs_dir]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_fixture(self, data: Any, name: str, category: str = 'data'):
        """保存fixture数据"""
        if category == 'data':
            save_dir = self.data_dir
        elif category == 'models':
            save_dir = self.models_dir
        elif category == 'configs':
            save_dir = self.configs_dir
        else:
            raise ValueError(f"Unknown category: {category}")
        
        filepath = save_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load_fixture(self, name: str, category: str = 'data') -> Any:
        """加载fixture数据"""
        if category == 'data':
            load_dir = self.data_dir
        elif category == 'models':
            load_dir = self.models_dir
        elif category == 'configs':
            load_dir = self.configs_dir
        else:
            raise ValueError(f"Unknown category: {category}")
        
        filepath = load_dir / f"{name}.pkl"
        
        if not filepath.exists():
            raise FileNotFoundError(f"Fixture not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def create_temp_fixture(self, suffix: str = '.tmp') -> Path:
        """创建临时fixture文件"""
        return Path(tempfile.mktemp(suffix=suffix, dir=self.data_dir))
    
    def list_fixtures(self, category: str = 'data') -> List[str]:
        """列出所有fixtures"""
        if category == 'data':
            list_dir = self.data_dir
        elif category == 'models':
            list_dir = self.models_dir
        elif category == 'configs':
            list_dir = self.configs_dir
        else:
            return []
        
        return [f.stem for f in list_dir.glob("*.pkl")]
    
    def cleanup_fixtures(self, pattern: str = "*.tmp"):
        """清理临时fixtures"""
        for d in [self.data_dir, self.models_dir, self.configs_dir]:
            for f in d.glob(pattern):
                f.unlink()


# =============================================================================
# 性能测量工具
# =============================================================================

@dataclass
class PerformanceResult:
    """性能测试结果"""
    name: str
    execution_time: float
    memory_used: float  # MB
    cpu_percent: float
    iterations: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.results: List[PerformanceResult] = []
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
        except ImportError:
            self.psutil_available = False
    
    @contextmanager
    def measure(self, name: str):
        """上下文管理器用于测量性能"""
        start_time = time.time()
        start_memory = self._get_memory()
        
        yield
        
        end_time = time.time()
        end_memory = self._get_memory()
        
        result = PerformanceResult(
            name=name,
            execution_time=end_time - start_time,
            memory_used=end_memory - start_memory,
            cpu_percent=self._get_cpu_percent(),
            iterations=1
        )
        
        self.results.append(result)
    
    def _get_memory(self) -> float:
        """获取当前内存使用（MB）"""
        if self.psutil_available:
            return self.process.memory_info().rss / 1024 / 1024
        return 0.0
    
    def _get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        if self.psutil_available:
            return self.process.cpu_percent()
        return 0.0
    
    def get_summary(self) -> Dict:
        """获取性能摘要"""
        if not self.results:
            return {}
        
        total_time = sum(r.execution_time for r in self.results)
        total_memory = sum(r.memory_used for r in self.results)
        
        return {
            'total_tests': len(self.results),
            'total_time': total_time,
            'avg_time': total_time / len(self.results),
            'total_memory_mb': total_memory,
            'tests': [r.to_dict() for r in self.results]
        }
    
    def save_report(self, filepath: Path):
        """保存性能报告"""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# 数值比较工具
# =============================================================================

class NumericalAssertions:
    """数值断言工具类"""
    
    TOLERANCE = {
        'strict': {'rtol': 1e-8, 'atol': 1e-10},
        'normal': {'rtol': 1e-5, 'atol': 1e-8},
        'relaxed': {'rtol': 1e-3, 'atol': 1e-6},
    }
    
    @classmethod
    def assert_allclose(cls, a, b, tolerance='normal', msg=''):
        """断言两个数组近似相等"""
        tol = cls.TOLERANCE.get(tolerance, cls.TOLERANCE['normal'])
        
        a = np.asarray(a)
        b = np.asarray(b)
        
        if a.shape != b.shape:
            raise AssertionError(f"Shape mismatch: {a.shape} vs {b.shape}")
        
        if not np.allclose(a, b, **tol):
            diff = np.abs(a - b)
            max_diff = np.max(diff)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            
            raise AssertionError(
                f"Arrays not close{msg}: max_diff={max_diff} at {max_idx}\n"
                f"a[{max_idx}]={a[max_idx]}, b[{max_idx}]={b[max_idx]}"
            )
    
    @classmethod
    def assert_energy_conservation(cls, energies, max_drift=1e-6):
        """断言能量守恒"""
        energies = np.asarray(energies)
        drift = np.max(energies) - np.min(energies)
        mean_energy = np.mean(energies)
        relative_drift = drift / abs(mean_energy) if mean_energy != 0 else drift
        
        if relative_drift > max_drift:
            raise AssertionError(
                f"Energy not conserved: relative_drift={relative_drift} > {max_drift}"
            )
    
    @classmethod
    def assert_forces_vanish(cls, forces, threshold=0.01):
        """断言力接近于零（平衡位置）"""
        forces = np.asarray(forces)
        max_force = np.max(np.linalg.norm(forces, axis=-1))
        
        if max_force > threshold:
            raise AssertionError(f"Forces not vanishing: max_force={max_force} > {threshold}")
    
    @classmethod
    def assert_symmetric_matrix(cls, matrix, msg=''):
        """断言矩阵对称"""
        matrix = np.asarray(matrix)
        if not np.allclose(matrix, matrix.T):
            raise AssertionError(f"Matrix not symmetric{msg}")


# =============================================================================
# 测试报告生成器
# =============================================================================

class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
    
    def add_result(self, test_name: str, passed: bool, 
                   duration: float, message: str = ''):
        """添加测试结果"""
        self.results.append({
            'test_name': test_name,
            'passed': passed,
            'duration': duration,
            'message': message,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
    
    def generate_html_report(self) -> Path:
        """生成HTML报告"""
        passed = sum(1 for r in self.results if r['passed'])
        failed = len(self.results) - passed
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #333; color: white; padding: 10px; }}
                .summary {{ margin: 20px 0; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f2f2f2; }}
                tr:nth-child(even) {{ background: #f9f9f9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Report</h1>
                <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total: {len(self.results)} | 
                   <span class="passed">Passed: {passed}</span> | 
                   <span class="failed">Failed: {failed}</span></p>
            </div>
            
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>Message</th>
                </tr>
        """
        
        for result in self.results:
            status_class = 'passed' if result['passed'] else 'failed'
            status_text = '✓ PASS' if result['passed'] else '✗ FAIL'
            
            html += f"""
                <tr>
                    <td>{result['test_name']}</td>
                    <td class="{status_class}">{status_text}</td>
                    <td>{result['duration']:.3f}</td>
                    <td>{result['message']}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        report_file = self.output_dir / f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.html"
        report_file.write_text(html)
        
        return report_file
    
    def generate_json_report(self) -> Path:
        """生成JSON报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total': len(self.results),
                'passed': sum(1 for r in self.results if r['passed']),
                'failed': sum(1 for r in self.results if not r['passed'])
            },
            'results': self.results
        }
        
        report_file = self.output_dir / f"test_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report_file


# =============================================================================
# 辅助装饰器
# =============================================================================

def slow_test(max_seconds: float = 60.0):
    """标记慢速测试的装饰器"""
    def decorator(func):
        func.is_slow = True
        func.max_seconds = max_seconds
        return func
    return decorator


def requires_external(command: str):
    """标记需要外部命令的测试的装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import shutil
            if not shutil.which(command):
                pytest.skip(f"External command '{command}' not available")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def flaky_test(max_runs: int = 3, min_passes: int = 2):
    """标记不稳定测试的装饰器（允许重试）"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            passes = 0
            for i in range(max_runs):
                try:
                    func(*args, **kwargs)
                    passes += 1
                    if passes >= min_passes:
                        return
                except AssertionError:
                    if i == max_runs - 1 and passes < min_passes:
                        raise
        return wrapper
    return decorator


# =============================================================================
# 便利函数
# =============================================================================

def create_mock_outcar(filepath: Path, 
                       n_atoms: int = 10,
                       converged: bool = True,
                       energy: float = -100.0):
    """创建模拟OUTCAR文件"""
    content = f""" vasp.5.4.4 18Apr17 complex 
 executed on date 2024.01.15

 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
"""
    for i in range(n_atoms):
        pos = np.random.rand(3) * 10
        force = np.random.randn(3) * 0.1
        content += f"    {pos[0]:8.5f}    {pos[1]:8.5f}    {pos[2]:8.5f}       {force[0]:10.6f}   {force[1]:10.6f}   {force[2]:10.6f}\n"
    
    content += f""" -----------------------------------------------------------------------------------

  free  energy   TOTEN  =      {energy:.8f} eV

  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  ---------------------------------------------------
  free  energy   TOTEN  =      {energy:.8f} eV
  ---------------------------------------------------

"""
    
    if converged:
        content += "    Convergence achieved after    15 iterations\n"
    else:
        content += "    Convergence NOT achieved\n"
    
    filepath.write_text(content)


def create_mock_lammps_input(filepath: Path,
                             potential_file: Optional[str] = None):
    """创建模拟LAMMPS输入文件"""
    content = """# LAMMPS input file
units metal
atom_style atomic
boundary p p p

read_data structure.data
"""
    
    if potential_file:
        content += f"""
pair_style nep
pair_coeff * * {potential_file}
"""
    
    content += """
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes

velocity all create 300 12345

fix 1 all nvt temp 300 300 0.1

timestep 0.001
dump traj all custom 100 dump.lammpstrj id type x y z
thermo 100
run 1000
"""
    
    filepath.write_text(content)


def create_mock_nep_xyz(filepath: Path, 
                       n_frames: int = 10,
                       n_atoms: int = 10):
    """创建模拟NEP XYZ文件"""
    content = ""
    
    for frame in range(n_frames):
        content += f"{n_atoms}\n"
        energy = -100.0 + np.random.randn() * 0.5
        content += f'Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy={energy:.10f}\n'
        
        for i in range(n_atoms):
            sym = 'Li' if i % 2 == 0 else 'S'
            pos = np.random.rand(3) * 10
            force = np.random.randn(3) * 0.1
            content += f" {sym}  {pos[0]:8.6f}  {pos[1]:8.6f}  {pos[2]:8.6f}  {force[0]:8.6f}  {force[1]:8.6f}  {force[2]:8.6f}\n"
    
    filepath.write_text(content)


# 导出所有公共API
__all__ = [
    'MockDataGenerator',
    'FixtureManager',
    'PerformanceMonitor',
    'PerformanceResult',
    'NumericalAssertions',
    'TestReportGenerator',
    'slow_test',
    'requires_external',
    'flaky_test',
    'create_mock_outcar',
    'create_mock_lammps_input',
    'create_mock_nep_xyz',
]
