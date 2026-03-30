#!/usr/bin/env python3
"""
examples/workflow_examples.py
===========================
工作流示例 - 展示复杂HPC工作流

包含：
- 高通量筛选工作流
- 主动学习工作流
- 多尺度模拟工作流
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict
from hpc_integration import create_hpc_client
from hpc_integration.job_submitter import JobTemplate, JobArrayBuilder, JobSpec


class HighThroughputWorkflow:
    """高通量筛选工作流"""
    
    def __init__(self, client):
        self.client = client
        self.results = []
    
    def run_structure_screening(
        self,
        structures: List[str],
        work_dir: Path,
        calc_type: str = "vasp"
    ) -> Dict:
        """
        运行结构筛选
        
        Args:
            structures: 结构文件列表
            work_dir: 工作目录
            calc_type: 计算类型
        
        Returns:
            工作流结果
        """
        print(f"\n高通量筛选: {len(structures)} 个结构")
        
        # 阶段1: 结构松弛
        relax_template = self._create_relax_template(calc_type)
        
        # 阶段2: 性质计算
        prop_template = self._create_property_template(calc_type)
        
        workflow_results = {
            "structures": structures,
            "stages": []
        }
        
        # 提交批量计算
        for i, structure in enumerate(structures):
            struct_work_dir = work_dir / f"struct_{i:04d}"
            struct_work_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  准备: {structure}")
            
            # 创建作业
            spec = JobSpec(
                template=relax_template,
                working_dir=struct_work_dir
            )
            
            # 提交（示例中不实际提交）
            # job = self.client.submit_job(spec)
        
        return workflow_results
    
    def _create_relax_template(self, calc_type: str) -> JobTemplate:
        """创建结构松弛模板"""
        if calc_type == "vasp":
            return JobTemplate.for_vasp(
                name="struct_relax",
                nodes=1,
                cores_per_node=16,
                walltime_hours=12.0
            )
        elif calc_type == "lammps":
            return JobTemplate.for_lammps(
                name="struct_relax",
                nodes=1,
                cores_per_node=8,
                walltime_hours=6.0
            )
        else:
            raise ValueError(f"Unknown calc_type: {calc_type}")
    
    def _create_property_template(self, calc_type: str) -> JobTemplate:
        """创建性质计算模板"""
        return JobTemplate(
            name="property_calc",
            calculation_type=JobTemplate.from_dict({'name': 'custom', 'calculation_type': 'custom'}).calculation_type,
            nodes=1,
            cores_per_node=8,
            walltime_hours=4.0
        )


class ActiveLearningWorkflow:
    """主动学习工作流"""
    
    def __init__(self, client):
        self.client = client
        self.iteration = 0
    
    def run_active_learning_loop(
        self,
        initial_structures: List[str],
        work_dir: Path,
        max_iterations: int = 10,
        convergence_threshold: float = 0.01
    ) -> Dict:
        """
        运行主动学习循环
        
        Args:
            initial_structures: 初始结构
            work_dir: 工作目录
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
        
        Returns:
            训练结果
        """
        print(f"\n主动学习工作流")
        print(f"  最大迭代: {max_iterations}")
        print(f"  收敛阈值: {convergence_threshold}")
        
        results = {
            "iterations": [],
            "final_model": None
        }
        
        current_structures = initial_structures.copy()
        
        for iteration in range(max_iterations):
            print(f"\n  迭代 {iteration + 1}/{max_iterations}")
            
            iter_dir = work_dir / f"iter_{iteration:03d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            
            # 阶段1: DFT计算（新结构）
            dft_results = self._run_dft_calculations(
                current_structures,
                iter_dir / "dft"
            )
            
            # 阶段2: 训练ML势
            training_result = self._train_ml_potential(
                iter_dir / "training",
                iteration
            )
            
            # 阶段3: 探索与不确定性量化
            new_structures = self._explore_structures(
                iter_dir / "exploration"
            )
            
            # 检查收敛
            if len(new_structures) == 0:
                print(f"  已收敛!")
                break
            
            current_structures = new_structures
            
            results["iterations"].append({
                "iteration": iteration,
                "structures": len(current_structures),
                "training_error": training_result.get("error", 0.0)
            })
        
        return results
    
    def _run_dft_calculations(
        self,
        structures: List[str],
        work_dir: Path
    ) -> List[Dict]:
        """运行DFT计算"""
        print(f"    DFT计算: {len(structures)} 个结构")
        
        template = JobTemplate.for_vasp(
            name="dft_calc",
            nodes=1,
            cores_per_node=16,
            walltime_hours=8.0
        )
        
        return [{"structure": s, "energy": 0.0} for s in structures]
    
    def _train_ml_potential(self, work_dir: Path, iteration: int) -> Dict:
        """训练ML势"""
        print(f"    ML训练: 迭代 {iteration}")
        
        template = JobTemplate.for_nep(
            name=f"nep_train_iter{iteration}",
            num_gpus=1,
            walltime_hours=24.0
        )
        
        return {
            "model_path": str(work_dir / "model.ckpt"),
            "error": 0.01 / (iteration + 1)
        }
    
    def _explore_structures(self, work_dir: Path) -> List[str]:
        """探索新结构"""
        print(f"    结构探索...")
        
        # 模拟发现新结构
        template = JobTemplate.for_lammps(
            name="exploration",
            nodes=1,
            cores_per_node=4,
            walltime_hours=4.0
        )
        
        return ["new_struct_1", "new_struct_2"]


class MultiscaleWorkflow:
    """多尺度模拟工作流"""
    
    def __init__(self, client):
        self.client = client
    
    def run_multiscale_simulation(
        self,
        system_config: Dict,
        work_dir: Path
    ) -> Dict:
        """
        运行多尺度模拟
        
        流程:
        1. DFT计算获得基本参数
        2. 拟合ML势
        3. MD模拟（小尺度）
        4. MD模拟（大尺度）
        5. 有限元分析
        
        Args:
            system_config: 系统配置
            work_dir: 工作目录
        
        Returns:
            模拟结果
        """
        print(f"\n多尺度模拟工作流")
        
        work_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # 阶段1: DFT计算
        print("  阶段1: DFT计算")
        dft_results = self._run_stage1_dft(system_config, work_dir / "dft")
        results["dft"] = dft_results
        
        # 阶段2: ML势训练（依赖于DFT）
        print("  阶段2: ML势训练")
        ml_results = self._run_stage2_ml_training(
            dft_results,
            work_dir / "ml_training"
        )
        results["ml_training"] = ml_results
        
        # 阶段3: 小尺度MD（验证）
        print("  阶段3: 小尺度MD验证")
        small_md = self._run_stage3_small_md(
            ml_results,
            work_dir / "small_md"
        )
        results["small_md"] = small_md
        
        # 阶段4: 大尺度MD
        print("  阶段4: 大尺度MD")
        large_md = self._run_stage4_large_md(
            ml_results,
            work_dir / "large_md"
        )
        results["large_md"] = large_md
        
        # 阶段5: 有限元分析（可选）
        print("  阶段5: 有限元分析")
        fem_results = self._run_stage5_fem(
            large_md,
            work_dir / "fem"
        )
        results["fem"] = fem_results
        
        return results
    
    def _run_stage1_dft(self, config: Dict, work_dir: Path) -> Dict:
        """阶段1: DFT计算"""
        template = JobTemplate.for_vasp(
            name="dft_reference",
            nodes=2,
            cores_per_node=32,
            walltime_hours=24.0
        )
        
        return {
            "energies": [],
            "forces": [],
            "dataset_path": str(work_dir / "training_data")
        }
    
    def _run_stage2_ml_training(self, dft_results: Dict, work_dir: Path) -> Dict:
        """阶段2: ML势训练"""
        template = JobTemplate.for_nep(
            name="nep_fitting",
            num_gpus=2,
            walltime_hours=72.0
        )
        
        return {
            "model_path": str(work_dir / "model.com"),
            "rmse_energy": 0.005,
            "rmse_force": 0.1
        }
    
    def _run_stage3_small_md(self, ml_results: Dict, work_dir: Path) -> Dict:
        """阶段3: 小尺度MD"""
        template = JobTemplate.for_lammps(
            name="md_validation",
            nodes=1,
            cores_per_node=8,
            walltime_hours=12.0
        )
        
        return {
            "temperature": 300,
            "pressure": 1.0,
            "trajectory": str(work_dir / "validation.dump")
        }
    
    def _run_stage4_large_md(self, ml_results: Dict, work_dir: Path) -> Dict:
        """阶段4: 大尺度MD"""
        template = JobTemplate.for_lammps(
            name="large_scale_md",
            nodes=4,
            cores_per_node=32,
            walltime_hours=48.0
        )
        
        return {
            "natoms": 1000000,
            "trajectory": str(work_dir / "production.dump")
        }
    
    def _run_stage5_fem(self, md_results: Dict, work_dir: Path) -> Dict:
        """阶段5: 有限元分析"""
        template = JobTemplate(
            name="fem_analysis",
            calculation_type=JobTemplate.from_dict({'name': 'custom', 'calculation_type': 'custom'}).calculation_type,
            nodes=2,
            cores_per_node=16,
            walltime_hours=8.0,
            executable="abaqus"
        )
        
        return {
            "stress_distribution": str(work_dir / "stress.vtk"),
            "strain_distribution": str(work_dir / "strain.vtk")
        }


def run_all_examples():
    """运行所有示例"""
    print("=" * 60)
    print("HPC工作流示例")
    print("=" * 60)
    
    # 创建模拟客户端
    client = None
    
    # 高通量筛选
    print("\n" + "=" * 60)
    print("示例1: 高通量筛选")
    print("=" * 60)
    ht_workflow = HighThroughputWorkflow(client)
    structures = [f"POSCAR_{i}" for i in range(10)]
    ht_results = ht_workflow.run_structure_screening(
        structures,
        Path("./high_throughput_example")
    )
    print(f"完成: {len(ht_results['structures'])} 个结构")
    
    # 主动学习
    print("\n" + "=" * 60)
    print("示例2: 主动学习")
    print("=" * 60)
    al_workflow = ActiveLearningWorkflow(client)
    al_results = al_workflow.run_active_learning_loop(
        ["struct_1", "struct_2", "struct_3"],
        Path("./active_learning_example"),
        max_iterations=3
    )
    print(f"完成: {len(al_results['iterations'])} 次迭代")
    
    # 多尺度模拟
    print("\n" + "=" * 60)
    print("示例3: 多尺度模拟")
    print("=" * 60)
    ms_workflow = MultiscaleWorkflow(client)
    ms_results = ms_workflow.run_multiscale_simulation(
        {"material": "LiCoO2", "size": "10nm"},
        Path("./multiscale_example")
    )
    print(f"完成: {len(ms_results)} 个阶段")
    
    print("\n" + "=" * 60)
    print("所有工作流示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
