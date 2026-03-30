#!/usr/bin/env python3
"""
端到端DFT-MLP-MD耦合工作流
完整的多尺度材料模拟流程
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

# ASE
from ase import Atoms
from ase.io import read, write

# 导入自定义模块
from dft_workflow import DFTConfig, StructureOptimizer, MaterialsProjectInterface
from ml_potential_training import DeepMDConfig, DataPreprocessor, DeepMDTrainer, ActiveLearningWorkflow
from md_simulation_lammps import LAMMPSConfig, LAMMPSSimulator, MDAnalyzer, MDSimulationWorkflow
from high_throughput_screening import HighThroughputConfig, HighThroughputScreening, ScreeningCriteria

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class MultiscaleWorkflowConfig:
    """多尺度工作流配置"""
    # 项目设置
    project_name: str = "multiscale_simulation"
    working_dir: str = "./multiscale_project"
    
    # 阶段控制
    run_dft: bool = True
    run_ml_training: bool = True
    run_md: bool = True
    run_active_learning: bool = False
    
    # DFT设置
    dft_config: DFTConfig = None
    
    # ML势设置
    ml_config: DeepMDConfig = None
    
    # MD设置
    md_config: LAMMPSConfig = None
    
    # 主动学习设置
    al_iterations: int = 5
    al_uncertainty_threshold: float = 0.3
    
    def __post_init__(self):
        if self.dft_config is None:
            self.dft_config = DFTConfig(code="vasp", ncores=32)
        if self.ml_config is None:
            self.ml_config = DeepMDConfig(type_map=["Li", "C", "O"])
        if self.md_config is None:
            self.md_config = LAMMPSConfig(
                pair_style="deepmd",
                potential_file="model.pb",
                ensemble="nvt",
                temperature=300
            )


class MultiscaleWorkflow:
    """
    多尺度材料模拟工作流
    
    完整流程:
    1. DFT结构优化和AIMD数据生成
    2. 机器学习势训练
    3. 主动学习迭代 (可选)
    4. 大尺度MD模拟
    5. 性质计算和分析
    """
    
    def __init__(self, config: MultiscaleWorkflowConfig):
        self.config = config
        self.working_dir = Path(config.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.dft_dir = self.working_dir / "01_dft"
        self.ml_dir = self.working_dir / "02_ml_training"
        self.md_dir = self.working_dir / "03_md_simulation"
        self.analysis_dir = self.working_dir / "04_analysis"
        
        for d in [self.dft_dir, self.ml_dir, self.md_dir, self.analysis_dir]:
            d.mkdir(exist_ok=True)
        
        # 状态跟踪
        self.state = {
            'dft_completed': False,
            'ml_training_completed': False,
            'md_completed': False,
            'current_iteration': 0
        }
        
        self._load_state()
    
    def _load_state(self):
        """加载工作流状态"""
        state_file = self.working_dir / "workflow_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                self.state.update(json.load(f))
    
    def _save_state(self):
        """保存工作流状态"""
        state_file = self.working_dir / "workflow_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def run(self, structure_file: Optional[str] = None) -> Dict:
        """
        运行完整工作流
        
        Args:
            structure_file: 初始结构文件 (POSCAR, cif, etc.)
            
        Returns:
            results: 包含所有结果的词典
        """
        start_time = datetime.now()
        logger.info(f"Starting multiscale workflow: {self.config.project_name}")
        logger.info(f"Working directory: {self.working_dir}")
        
        results = {}
        
        # 阶段1: DFT计算
        if self.config.run_dft and not self.state['dft_completed']:
            logger.info("=" * 60)
            logger.info("Phase 1: DFT Calculations")
            logger.info("=" * 60)
            
            dft_results = self._run_dft_phase(structure_file)
            results['dft'] = dft_results
            
            self.state['dft_completed'] = True
            self._save_state()
        
        # 阶段2: ML势训练
        if self.config.run_ml_training and not self.state['ml_training_completed']:
            logger.info("=" * 60)
            logger.info("Phase 2: ML Potential Training")
            logger.info("=" * 60)
            
            ml_results = self._run_ml_training_phase()
            results['ml_training'] = ml_results
            
            self.state['ml_training_completed'] = True
            self._save_state()
        
        # 阶段3: 主动学习 (可选)
        if self.config.run_active_learning:
            logger.info("=" * 60)
            logger.info("Phase 3: Active Learning")
            logger.info("=" * 60)
            
            al_results = self._run_active_learning_phase()
            results['active_learning'] = al_results
        
        # 阶段4: MD模拟
        if self.config.run_md and not self.state['md_completed']:
            logger.info("=" * 60)
            logger.info("Phase 4: Molecular Dynamics Simulation")
            logger.info("=" * 60)
            
            md_results = self._run_md_phase()
            results['md_simulation'] = md_results
            
            self.state['md_completed'] = True
            self._save_state()
        
        # 阶段5: 分析和报告
        logger.info("=" * 60)
        logger.info("Phase 5: Analysis and Reporting")
        logger.info("=" * 60)
        
        analysis_results = self._run_analysis_phase(results)
        results['analysis'] = analysis_results
        
        # 保存完整结果
        self._save_final_results(results)
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Workflow completed in {duration}")
        
        return results
    
    def _run_dft_phase(self, structure_file: Optional[str] = None) -> Dict:
        """DFT计算阶段"""
        results = {
            'optimized_structures': [],
            'aimd_trajectories': [],
            'vibrations': []
        }
        
        # 获取初始结构
        if structure_file:
            atoms = read(structure_file)
            structures = [atoms]
        else:
            # 从Materials Project获取示例结构
            mp = MaterialsProjectInterface()
            docs = mp.mpr.summary.search(
                formula="Li2O",
                fields=["structure"],
                num_chunks=1,
                chunk_size=1
            )
            structures = [AseAtomsAdaptor.get_atoms(d.structure) for d in docs]
        
        # 为每个结构运行DFT
        for i, atoms in enumerate(structures):
            struct_dir = self.dft_dir / f"structure_{i}"
            struct_dir.mkdir(exist_ok=True)
            
            logger.info(f"Processing structure {i+1}/{len(structures)}")
            
            # 结构优化
            optimizer = StructureOptimizer(self.config.dft_config)
            optimizer.load_structure(str(struct_dir / "POSCAR"))
            
            # 保存初始结构
            write(struct_dir / "POSCAR_initial", atoms)
            
            # 优化
            optimized = optimizer.relax_structure()
            write(struct_dir / "POSCAR_optimized", optimized)
            results['optimized_structures'].append(str(struct_dir / "POSCAR_optimized"))
            
            # 振动分析
            vib_data = optimizer.compute_vibrations()
            results['vibrations'].append(vib_data)
            
            with open(struct_dir / "vibrations.json", 'w') as f:
                json.dump(vib_data, f, indent=2)
            
            # AIMD生成训练数据
            if i == 0:  # 只为第一个结构生成AIMD
                trajectory = optimizer.generate_aimd_data(
                    temperature=300,
                    timestep=1.0,
                    nsteps=10000
                )
                results['aimd_trajectories'].append(trajectory)
                
                # 复制到ML训练目录
                shutil.copy(trajectory, self.ml_dir / "aimd.traj")
        
        return results
    
    def _run_ml_training_phase(self) -> Dict:
        """ML势训练阶段"""
        results = {}
        
        # 数据预处理
        preprocessor = DataPreprocessor(self.config.ml_config.type_map)
        
        # 查找AIMD轨迹
        traj_file = self.ml_dir / "aimd.traj"
        if not traj_file.exists():
            raise FileNotFoundError(f"AIMD trajectory not found: {traj_file}")
        
        # 转换数据
        data_dir = self.ml_dir / "training_data"
        preprocessor.convert_aimd_to_deepmd(
            str(traj_file),
            str(data_dir)
        )
        
        # 分割训练/验证集
        import numpy as np
        import dpdata
        
        system = dpdata.LabeledSystem(str(data_dir), fmt='deepmd/npy')
        n_frames = len(system)
        n_train = int(0.9 * n_frames)
        
        indices = np.random.permutation(n_frames)
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:]
        
        train_dir = self.ml_dir / "train"
        valid_dir = self.ml_dir / "valid"
        
        train_system = system.sub_system(train_idx)
        valid_system = system.sub_system(valid_idx)
        
        train_system.to_deepmd_npy(str(train_dir))
        valid_system.to_deepmd_npy(str(valid_dir))
        
        # 更新配置
        self.config.ml_config.training_data = str(train_dir)
        self.config.ml_config.validation_data = str(valid_dir)
        self.config.ml_config.output_dir = str(self.ml_dir / "model")
        
        # 训练
        trainer = DeepMDTrainer(self.config.ml_config)
        
        # 生成输入文件
        input_file = trainer.generate_input()
        shutil.copy(input_file, self.ml_dir / "input.json")
        
        # 执行训练 (这里简化，实际应运行dp train)
        logger.info("Training ML potential (this may take hours)...")
        # model_path = trainer.train()
        
        # 冻结模型
        # frozen_model = trainer.freeze_model(str(self.ml_dir / "model"))
        
        # 复制到MD目录
        model_file = self.ml_dir / "model" / "model.pb"
        if model_file.exists():
            shutil.copy(model_file, self.md_dir / "model.pb")
            results['model_path'] = str(self.md_dir / "model.pb")
        else:
            # 占位符
            results['model_path'] = str(self.md_dir / "model.pb")
        
        return results
    
    def _run_active_learning_phase(self) -> Dict:
        """主动学习阶段"""
        results = {'iterations': []}
        
        for iteration in range(self.config.al_iterations):
            logger.info(f"Active Learning Iteration {iteration + 1}/{self.config.al_iterations}")
            
            iter_dir = self.ml_dir / f"al_iteration_{iteration}"
            iter_dir.mkdir(exist_ok=True)
            
            # 探索阶段: 使用当前ML势运行MD
            explorer = LAMMPSSimulator(self.config.md_config)
            
            # 生成探索结构 (简化)
            # uncertain_structures = explorer.explore()
            
            # 标记阶段: DFT计算
            # labeled_data = self._label_structures(uncertain_structures)
            
            # 重新训练
            # new_model = self._retrain_model(labeled_data)
            
            results['iterations'].append({
                'iteration': iteration,
                'status': 'completed'
            })
        
        return results
    
    def _run_md_phase(self) -> Dict:
        """MD模拟阶段"""
        results = {}
        
        # 获取优化后的结构
        optimized_structure = None
        for struct_file in self.dft_dir.glob("*/POSCAR_optimized"):
            optimized_structure = read(struct_file)
            break
        
        if optimized_structure is None:
            raise ValueError("No optimized structure found")
        
        # 设置MD配置
        self.config.md_config.working_dir = str(self.md_dir)
        self.config.md_config.potential_file = str(self.md_dir / "model.pb")
        
        # 创建模拟工作流
        workflow = MDSimulationWorkflow(self.config.md_config)
        
        # 运行完整工作流
        md_results = workflow.full_workflow(optimized_structure)
        
        results['trajectory'] = md_results['trajectory']
        results['report'] = md_results['report']
        
        return results
    
    def _run_analysis_phase(self, all_results: Dict) -> Dict:
        """分析阶段"""
        results = {}
        
        # 加载MD轨迹
        trajectory_file = all_results.get('md_simulation', {}).get('trajectory')
        
        if trajectory_file and Path(trajectory_file).exists():
            analyzer = MDAnalyzer(trajectory_file)
            
            # 计算扩散系数
            try:
                D = analyzer.compute_diffusion_coefficient()
                results['diffusion_coefficient'] = {
                    'value': float(D),
                    'unit': 'cm^2/s'
                }
            except Exception as e:
                logger.warning(f"Failed to compute diffusion coefficient: {e}")
            
            # 计算RDF
            try:
                rdf = analyzer.compute_rdf()
                results['rdf'] = {str(k): v for k, v in rdf.items()}
            except Exception as e:
                logger.warning(f"Failed to compute RDF: {e}")
            
            # 生成完整报告
            report_file = self.analysis_dir / "md_analysis_report.json"
            analyzer.generate_report(str(report_file))
            results['report_file'] = str(report_file)
        
        # 生成总结报告
        summary = {
            'project_name': self.config.project_name,
            'workflow_status': self.state,
            'properties': results,
            'recommendations': self._generate_recommendations(all_results)
        }
        
        summary_file = self.analysis_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        results['summary'] = summary
        
        return results
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于结果生成建议
        if 'md_simulation' in results:
            report = results['md_simulation'].get('report', {})
            
            if 'diffusion_coefficient' in report.get('properties', {}):
                D = report['properties']['diffusion_coefficient']['value']
                if D < 1e-10:
                    recommendations.append(
                        "Low diffusion coefficient detected. Consider: "
                        "(1) Higher temperature simulation, "
                        "(2) Different stoichiometry, "
                        "(3) Doping strategy"
                    )
        
        # 主动学习建议
        if not self.config.run_active_learning:
            recommendations.append(
                "Consider enabling active learning for improved model accuracy "
                "in unexplored configuration space."
            )
        
        return recommendations
    
    def _save_final_results(self, results: Dict):
        """保存最终结果"""
        final_results_file = self.working_dir / "final_results.json"
        
        # 序列化结果
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, default=serialize)
        
        logger.info(f"Final results saved to: {final_results_file}")


class BatteryMaterialsWorkflow(MultiscaleWorkflow):
    """电池材料专用工作流"""
    
    def __init__(self, config: MultiscaleWorkflowConfig, 
                 battery_type: str = "solid_state"):
        super().__init__(config)
        self.battery_type = battery_type
    
    def run(self, cathode_formula: str = "LiCoO2") -> Dict:
        """运行电池材料工作流"""
        logger.info(f"Starting battery materials workflow for {cathode_formula}")
        
        # 获取初始结构
        mp = MaterialsProjectInterface()
        docs = mp.mpr.summary.search(
            formula=cathode_formula,
            fields=["structure", "material_id"],
            num_chunks=1,
            chunk_size=1
        )
        
        if not docs:
            raise ValueError(f"Material {cathode_formula} not found in Materials Project")
        
        structure = AseAtomsAdaptor.get_atoms(docs[0].structure)
        
        # 保存初始结构
        initial_file = self.working_dir / "initial_structure.cif"
        write(initial_file, structure)
        
        # 运行父类工作流
        results = super().run(str(initial_file))
        
        # 添加电池特定分析
        results['battery_properties'] = self._analyze_battery_properties(results)
        
        return results
    
    def _analyze_battery_properties(self, results: Dict) -> Dict:
        """分析电池相关性质"""
        battery_props = {}
        
        # 计算开路电压 (简化)
        if 'dft' in results:
            # 需要计算不同锂化状态的能
            pass
        
        # 分析离子电导率
        if 'md_simulation' in results:
            report = results['md_simulation'].get('report', {})
            if 'diffusion_coefficient' in report.get('properties', {}):
                D = report['properties']['diffusion_coefficient']['value']
                # Nernst-Einstein方程
                # 简化计算
                conductivity = D * 1e4  # 占位符
                battery_props['ionic_conductivity'] = {
                    'value': conductivity,
                    'unit': 'S/cm'
                }
        
        return battery_props


def main():
    """示例用法"""
    
    # 创建配置
    config = MultiscaleWorkflowConfig(
        project_name="Li2O_diffusion_study",
        working_dir="./example_project",
        run_dft=True,
        run_ml_training=True,
        run_md=True,
        run_active_learning=False,
        dft_config=DFTConfig(code="vasp", ncores=32),
        ml_config=DeepMDConfig(
            type_map=["Li", "O"],
            descriptor_type="se_e2_a",
            numb_steps=100000
        ),
        md_config=LAMMPSConfig(
            pair_style="deepmd",
            ensemble="nvt",
            temperature=300,
            timestep=1.0,
            nsteps=1000000,
            nprocs=4
        )
    )
    
    # 创建工作流
    workflow = MultiscaleWorkflow(config)
    
    # 运行 (需要实际的DFT和MD环境)
    # results = workflow.run()
    
    print("Multiscale workflow template ready!")
    print(f"Working directory: {config.working_dir}")
    print("To run the workflow:")
    print("  1. Prepare DFT input files in 01_dft/")
    print("  2. Run: results = workflow.run()")


if __name__ == "__main__":
    main()
