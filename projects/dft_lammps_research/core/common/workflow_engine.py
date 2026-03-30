#!/usr/bin/env python3
"""
Integrated Materials Workflow - DFT + LAMMPS + ML Potentials
===========================================================

统一材料计算工作流，整合以下模块：
1. dft_to_lammps_bridge.py - DFT/LAMMPS桥接、经典势拟合
2. nep_training_pipeline.py - NEP训练流程
3. active_learning_workflow.py - 主动学习工作流
4. battery_screening_pipeline.py - 高通量筛选

功能特点:
- 端到端工作流: 结构获取 → DFT计算 → ML势训练 → MD模拟 → 性能预测
- 模块化设计: 各组件可独立使用或组合
- 错误处理和日志记录
- 进度监控和报告生成
- 支持命令行和API调用

Author: DFT+LAMMPS Integration Expert
Version: 1.0.0
"""

import os
import sys
import json
import time
import pickle
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import threading
import queue

import numpy as np
import pandas as pd

# ASE
from ase import Atoms
from ase.io import read, write
from ase.io.vasp import read_vasp_out
from ase.units import fs, kB, GPa

# Pymatgen
from pymatgen.core import Structure, Composition, Element
from pymatgen.io.ase import AseAtomsAdaptor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('integrated_workflow.log')
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class WorkflowStage:
    """工作流阶段配置"""
    name: str
    enabled: bool = True
    depends_on: List[str] = field(default_factory=list)
    retry_count: int = 3
    timeout: int = 3600  # seconds


@dataclass
class MaterialsProjectConfig:
    """Materials Project配置"""
    api_key: Optional[str] = None
    query_criteria: Dict[str, Any] = field(default_factory=dict)
    max_entries: int = 100


@dataclass
class DFTStageConfig:
    """DFT计算阶段配置"""
    code: str = "vasp"  # vasp, espresso, abacus
    functional: str = "PBE"
    encut: float = 520  # eV
    kpoints_density: float = 0.25
    ediff: float = 1e-6
    ncores: int = 32
    max_steps: int = 200
    fmax: float = 0.01  # eV/Å


@dataclass
class MLPotentialConfig:
    """ML势训练配置"""
    framework: str = "deepmd"  # deepmd, nep, mace
    preset: str = "fast"  # fast, accurate, light
    num_models: int = 4  # for ensemble
    max_iterations: int = 10  # active learning iterations
    uncertainty_threshold: float = 0.15


@dataclass
class MDStageConfig:
    """MD模拟阶段配置"""
    ensemble: str = "nvt"
    temperatures: List[float] = field(default_factory=lambda: [300, 500, 700, 900])
    timestep: float = 1.0  # fs
    nsteps_equil: int = 50000
    nsteps_prod: int = 500000
    nprocs: int = 4


@dataclass
class AnalysisConfig:
    """分析阶段配置"""
    compute_diffusion: bool = True
    compute_conductivity: bool = True
    compute_activation_energy: bool = True
    compute_vibration: bool = False


@dataclass
class IntegratedWorkflowConfig:
    """统一工作流配置"""
    # 基本信息
    workflow_name: str = "materials_workflow"
    working_dir: str = "./workflow_output"
    
    # 各阶段配置
    mp_config: MaterialsProjectConfig = field(default_factory=MaterialsProjectConfig)
    dft_config: DFTStageConfig = field(default_factory=DFTStageConfig)
    ml_config: MLPotentialConfig = field(default_factory=MLPotentialConfig)
    md_config: MDStageConfig = field(default_factory=MDStageConfig)
    analysis_config: AnalysisConfig = field(default_factory=AnalysisConfig)
    
    # 阶段控制
    stages: Dict[str, WorkflowStage] = field(default_factory=lambda: {
        "fetch_structure": WorkflowStage("fetch_structure", enabled=True),
        "dft_calculation": WorkflowStage("dft_calculation", enabled=True, 
                                         depends_on=["fetch_structure"]),
        "ml_training": WorkflowStage("ml_training", enabled=True,
                                     depends_on=["dft_calculation"]),
        "md_simulation": WorkflowStage("md_simulation", enabled=True,
                                       depends_on=["ml_training"]),
        "analysis": WorkflowStage("analysis", enabled=True,
                                  depends_on=["md_simulation"]),
    })
    
    # 系统设置
    max_parallel: int = 4
    save_intermediate: bool = True
    generate_report: bool = True
    
    def __post_init__(self):
        # Create working directory
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        
        # Get API key from environment if not provided
        if self.mp_config.api_key is None:
            self.mp_config.api_key = os.environ.get("MP_API_KEY")


# =============================================================================
# Progress Monitor
# =============================================================================

class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self, total_stages: int):
        self.total_stages = total_stages
        self.current_stage = 0
        self.stage_name = ""
        self.stage_progress = 0.0
        self.start_time = time.time()
        self.stage_start_time = None
        self.history = []
        self._lock = threading.Lock()
    
    def start_stage(self, name: str):
        """开始新阶段"""
        with self._lock:
            self.current_stage += 1
            self.stage_name = name
            self.stage_progress = 0.0
            self.stage_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage {self.current_stage}/{self.total_stages}: {name}")
        logger.info(f"{'='*60}")
    
    def update(self, progress: float, message: str = ""):
        """更新进度"""
        with self._lock:
            self.stage_progress = min(100.0, max(0.0, progress))
        if message:
            logger.info(f"  [{self.stage_progress:.1f}%] {message}")
    
    def finish_stage(self, status: str = "success", message: str = ""):
        """完成当前阶段"""
        with self._lock:
            elapsed = time.time() - self.stage_start_time
            self.history.append({
                'stage': self.stage_name,
                'status': status,
                'elapsed': elapsed,
                'message': message
            })
        logger.info(f"  Stage completed: {status} (took {elapsed:.1f}s)")
        if message:
            logger.info(f"  {message}")
    
    def get_summary(self) -> Dict:
        """获取进度摘要"""
        total_elapsed = time.time() - self.start_time
        return {
            'total_stages': self.total_stages,
            'completed_stages': self.current_stage,
            'overall_progress': (self.current_stage / self.total_stages) * 100,
            'total_elapsed': total_elapsed,
            'history': self.history
        }


# =============================================================================
# Error Handler
# =============================================================================

class WorkflowError(Exception):
    """工作流错误"""
    pass


class StageError(WorkflowError):
    """阶段执行错误"""
    pass


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self, config: IntegratedWorkflowConfig):
        self.config = config
        self.errors = []
        self.warnings = []
    
    def handle_error(self, error: Exception, stage: str, context: Dict = None) -> bool:
        """
        处理错误，返回是否应该重试
        
        Returns:
            should_retry: 是否应该重试
        """
        error_info = {
            'stage': stage,
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
            'context': context or {}
        }
        self.errors.append(error_info)
        
        logger.error(f"Error in stage '{stage}': {error}")
        logger.debug(f"Traceback:\n{error_info['traceback']}")
        
        # 检查是否应该重试
        stage_config = self.config.stages.get(stage)
        if stage_config:
            retry_count = context.get('retry_count', 0) if context else 0
            if retry_count < stage_config.retry_count:
                logger.info(f"Will retry stage '{stage}' (attempt {retry_count + 1}/{stage_config.retry_count})")
                return True
        
        return False
    
    def add_warning(self, message: str, stage: str = ""):
        """添加警告"""
        self.warnings.append({
            'stage': stage,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        logger.warning(f"[{stage}] {message}")
    
    def get_report(self) -> Dict:
        """获取错误报告"""
        return {
            'n_errors': len(self.errors),
            'n_warnings': len(self.warnings),
            'errors': self.errors,
            'warnings': self.warnings
        }


# =============================================================================
# Stage 1: Structure Fetching
# =============================================================================

class StructureFetcher:
    """结构获取器 - 从Materials Project或文件"""
    
    def __init__(self, config: MaterialsProjectConfig, monitor: ProgressMonitor, 
                 error_handler: ErrorHandler):
        self.config = config
        self.monitor = monitor
        self.error_handler = error_handler
        self.mpr = None
        
    def _init_mp(self):
        """初始化Materials Project连接"""
        try:
            from mp_api.client import MPRester
            if self.config.api_key:
                self.mpr = MPRester(self.config.api_key)
            else:
                self.mpr = MPRester()
        except Exception as e:
            self.error_handler.add_warning(f"Failed to initialize MP API: {e}")
            self.mpr = None
    
    def fetch_from_mp(self, material_id: Optional[str] = None, 
                      formula: Optional[str] = None,
                      chemsys: Optional[str] = None) -> List[Structure]:
        """从Materials Project获取结构"""
        self.monitor.start_stage("fetch_structure")
        self._init_mp()
        
        if self.mpr is None:
            raise StageError("Materials Project API not available")
        
        structures = []
        
        try:
            if material_id:
                self.monitor.update(10, f"Fetching {material_id}")
                doc = self.mpr.summary.search(material_ids=[material_id],
                                             fields=["structure", "formula_pretty"])[0]
                structures.append(doc.structure)
                logger.info(f"Fetched {material_id}: {doc.formula_pretty}")
                
            elif formula:
                self.monitor.update(10, f"Searching for formula: {formula}")
                docs = self.mpr.summary.search(formula=formula,
                                              fields=["structure", "formula_pretty", "material_id"],
                                              num_chunks=1, chunk_size=self.config.max_entries)
                for doc in docs:
                    structures.append(doc.structure)
                logger.info(f"Found {len(structures)} structures for {formula}")
                
            elif chemsys:
                self.monitor.update(10, f"Searching chemical system: {chemsys}")
                docs = self.mpr.summary.search(chemsys=chemsys,
                                              fields=["structure", "formula_pretty"],
                                              num_chunks=1, chunk_size=self.config.max_entries)
                for doc in docs:
                    structures.append(doc.structure)
                logger.info(f"Found {len(structures)} structures in {chemsys}")
                
            else:
                # 使用筛选条件查询
                self.monitor.update(10, "Querying with criteria")
                criteria = self.config.query_criteria
                docs = self.mpr.summary.search(**criteria,
                                              fields=["structure", "formula_pretty"],
                                              num_chunks=1, chunk_size=self.config.max_entries)
                for doc in docs:
                    structures.append(doc.structure)
                logger.info(f"Found {len(structures)} structures matching criteria")
            
            self.monitor.update(100, f"Retrieved {len(structures)} structures")
            self.monitor.finish_stage("success", f"Fetched {len(structures)} structures")
            
        except Exception as e:
            self.error_handler.handle_error(e, "fetch_structure")
            raise StageError(f"Failed to fetch structures: {e}")
        
        return structures
    
    def fetch_from_file(self, file_path: str) -> List[Structure]:
        """从文件加载结构"""
        self.monitor.start_stage("fetch_structure")
        
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            self.monitor.update(50, f"Loading from {path}")
            
            # 使用ASE读取
            ase_atoms = read(str(path))
            if isinstance(ase_atoms, list):
                structures = []
                for atoms in ase_atoms:
                    adaptor = AseAtomsAdaptor()
                    structures.append(adaptor.get_structure(atoms))
            else:
                adaptor = AseAtomsAdaptor()
                structures = [adaptor.get_structure(ase_atoms)]
            
            self.monitor.update(100, f"Loaded {len(structures)} structures")
            self.monitor.finish_stage("success", f"Loaded from {path}")
            
            return structures
            
        except Exception as e:
            self.error_handler.handle_error(e, "fetch_structure")
            raise StageError(f"Failed to load structure from file: {e}")


# =============================================================================
# Stage 2: DFT Calculation
# =============================================================================

class DFTStage:
    """DFT计算阶段"""
    
    def __init__(self, config: DFTStageConfig, monitor: ProgressMonitor,
                 error_handler: ErrorHandler):
        self.config = config
        self.monitor = monitor
        self.error_handler = error_handler
        self.results = {}
    
    def run_relaxation(self, structure: Structure, output_dir: str) -> Dict:
        """运行结构优化"""
        self.monitor.start_stage("dft_calculation")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert to ASE
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            
            self.monitor.update(10, "Setting up DFT calculator")
            
            # Setup calculator
            if self.config.code == "vasp":
                from ase.calculators.vasp import Vasp
                calc = Vasp(
                    xc=self.config.functional,
                    encut=self.config.encut,
                    ediff=self.config.ediff,
                    ediffg=-self.config.fmax,
                    ibrion=2,
                    nsw=self.config.max_steps,
                    isif=3,
                    ismear=0,
                    sigma=0.05,
                    prec="Accurate",
                    lreal="Auto",
                    lwave=True,
                    lcharg=True,
                    ncore=self.config.ncores,
                    kpts=self._get_kpoints(atoms),
                )
            elif self.config.code == "espresso":
                from ase.calculators.espresso import Espresso
                pseudopotentials = self._get_pseudopotentials(atoms)
                calc = Espresso(
                    pseudopotentials=pseudopotentials,
                    input_data={
                        'control': {'calculation': 'vc-relax'},
                        'system': {'ecutwfc': self.config.encut / 13.6},
                    },
                    kpts=self._get_kpoints(atoms),
                )
            else:
                raise ValueError(f"Unsupported DFT code: {self.config.code}")
            
            atoms.calc = calc
            
            self.monitor.update(30, "Running structure relaxation")
            
            # Run optimization
            from ase.optimize import FIRE
            optimizer = FIRE(atoms, logfile=os.path.join(output_dir, 'relax.log'))
            optimizer.run(fmax=self.config.fmax)
            
            self.monitor.update(80, "Extracting results")
            
            # Get results
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            
            # Save results
            results = {
                'energy': float(energy),
                'energy_per_atom': float(energy / len(atoms)),
                'forces': forces.tolist(),
                'stress': stress.tolist() if stress is not None else None,
                'cell': atoms.get_cell().tolist(),
                'positions': atoms.get_positions().tolist(),
                'success': True
            }
            
            # Write output files
            write(os.path.join(output_dir, 'CONTCAR'), atoms)
            with open(os.path.join(output_dir, 'results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            self.monitor.update(100, "DFT calculation completed")
            self.monitor.finish_stage("success", f"E = {results['energy_per_atom']:.4f} eV/atom")
            
            return results
            
        except Exception as e:
            should_retry = self.error_handler.handle_error(e, "dft_calculation")
            if should_retry:
                raise  # Let caller handle retry
            self.monitor.finish_stage("failed", str(e))
            return {'success': False, 'error': str(e)}
    
    def _get_kpoints(self, atoms) -> tuple:
        """自动计算k点网格"""
        cell = atoms.get_cell()
        kpoints = []
        for i in range(3):
            k = int(np.ceil(2 * np.pi / (self.config.kpoints_density * np.linalg.norm(cell[i]))))
            kpoints.append(max(1, k))
        return tuple(kpoints)
    
    def _get_pseudopotentials(self, atoms) -> Dict[str, str]:
        """获取伪势映射"""
        pps = {}
        for symbol in set(atoms.get_chemical_symbols()):
            pps[symbol] = f"{symbol}.upf"
        return pps
    
    def run_aimd(self, structure: Structure, temperature: float, 
                 nsteps: int, output_dir: str) -> str:
        """运行AIMD生成训练数据"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            adaptor = AseAtomsAdaptor()
            atoms = adaptor.get_atoms(structure)
            
            # Setup calculator
            if self.config.code == "vasp":
                from ase.calculators.vasp import Vasp
                calc = Vasp(
                    xc=self.config.functional,
                    encut=self.config.encut,
                    ibrion=0,
                    nsw=nsteps,
                    ismear=0,
                    sigma=0.05,
                    mdalgo=2,
                    tebeg=temperature,
                    teend=temperature,
                    potim=1.0,
                    ncore=self.config.ncores,
                    kpts=(1, 1, 1),  # Gamma only for MD
                )
                atoms.calc = calc
                
                # Run MD
                atoms.get_potential_energy()
                
            return output_dir
            
        except Exception as e:
            self.error_handler.handle_error(e, "dft_aimd")
            raise


# =============================================================================
# Stage 3: ML Potential Training
# =============================================================================

class MLTrainingStage:
    """ML势训练阶段"""
    
    def __init__(self, config: MLPotentialConfig, monitor: ProgressMonitor,
                 error_handler: ErrorHandler):
        self.config = config
        self.monitor = monitor
        self.error_handler = error_handler
        self.model_paths = []
    
    def prepare_data(self, dft_output_dirs: List[str], output_dir: str) -> Tuple[str, str]:
        """准备训练数据"""
        self.monitor.update(10, "Preparing training data")
        
        try:
            import dpdata
            
            all_systems = []
            for dft_dir in dft_output_dirs:
                outcar_path = Path(dft_dir) / "OUTCAR"
                if outcar_path.exists():
                    try:
                        system = dpdata.LabeledSystem(str(outcar_path), fmt='vasp/outcar')
                        all_systems.append(system)
                        logger.info(f"Loaded {len(system)} frames from {dft_dir}")
                    except Exception as e:
                        self.error_handler.add_warning(f"Failed to load {dft_dir}: {e}")
            
            if not all_systems:
                raise StageError("No valid DFT data found for training")
            
            # Merge and split
            multi_systems = dpdata.MultiSystems(*all_systems)
            
            train_dir = Path(output_dir) / "training"
            valid_dir = Path(output_dir) / "validation"
            train_dir.mkdir(parents=True, exist_ok=True)
            valid_dir.mkdir(parents=True, exist_ok=True)
            
            for name, system in multi_systems.systems.items():
                n_frames = len(system)
                n_train = int(n_frames * 0.9)
                
                indices = np.random.permutation(n_frames)
                train_idx = indices[:n_train]
                valid_idx = indices[n_train:]
                
                train_system = system.sub_system(train_idx)
                valid_system = system.sub_system(valid_idx)
                
                train_system.to_deepmd_npy(str(train_dir / name))
                valid_system.to_deepmd_npy(str(valid_dir / name))
            
            self.monitor.update(30, f"Training data prepared: {train_dir}")
            return str(train_dir), str(valid_dir)
            
        except Exception as e:
            self.error_handler.handle_error(e, "ml_training_data_prep")
            raise
    
    def train_deepmd(self, train_dir: str, valid_dir: str, 
                     type_map: List[str], output_dir: str) -> List[str]:
        """训练DeepMD模型"""
        self.monitor.start_stage("ml_training")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate input file
            input_dict = {
                "model": {
                    "type_map": type_map,
                    "descriptor": {
                        "type": "se_e2_a",
                        "rcut": 6.0,
                        "rcut_smth": 0.5,
                        "sel": [50] * len(type_map),
                        "neuron": [25, 50, 100],
                        "axis_neuron": 16,
                        "seed": 1,
                    },
                    "fitting_net": {
                        "neuron": [240, 240, 240],
                        "resnet_dt": True,
                        "seed": 1,
                    }
                },
                "learning_rate": {
                    "type": "exp",
                    "decay_steps": 5000,
                    "start_lr": 0.001,
                    "stop_lr": 3.51e-8,
                },
                "loss": {
                    "type": "ener",
                    "start_pref_e": 0.02,
                    "limit_pref_e": 1.0,
                    "start_pref_f": 1000.0,
                    "limit_pref_f": 1.0,
                },
                "training": {
                    "training_data": {"systems": [train_dir], "batch_size": "auto"},
                    "validation_data": {"systems": [valid_dir], "batch_size": "auto"},
                    "numb_steps": 1000000 if self.config.preset == "accurate" else 100000,
                    "seed": 1,
                    "disp_file": "lcurve.out",
                    "save_freq": 10000,
                }
            }
            
            # Train ensemble of models
            model_paths = []
            
            for i in range(self.config.num_models):
                self.monitor.update(30 + i * 50 // self.config.num_models, 
                                   f"Training model {i+1}/{self.config.num_models}")
                
                model_dir = Path(output_dir) / f"model_{i}"
                model_dir.mkdir(exist_ok=True)
                
                # Modify seed for each model
                input_dict["model"]["descriptor"]["seed"] = i * 100 + 1
                input_dict["model"]["fitting_net"]["seed"] = i * 100 + 1
                input_dict["training"]["seed"] = i * 100 + 1
                
                input_file = model_dir / "input.json"
                with open(input_file, 'w') as f:
                    json.dump(input_dict, f, indent=2)
                
                # Run training
                import subprocess
                result = subprocess.run(
                    ["dp", "train", str(input_file)],
                    cwd=str(model_dir),
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    logger.error(f"Training failed: {result.stderr}")
                    continue
                
                # Freeze model
                result = subprocess.run(
                    ["dp", "freeze", "-o", "graph.pb"],
                    cwd=str(model_dir),
                    capture_output=True
                )
                
                frozen_path = model_dir / "graph.pb"
                if frozen_path.exists():
                    model_paths.append(str(frozen_path))
            
            self.model_paths = model_paths
            self.monitor.update(100, f"Trained {len(model_paths)} models")
            self.monitor.finish_stage("success", f"Models: {model_paths}")
            
            return model_paths
            
        except Exception as e:
            self.error_handler.handle_error(e, "ml_training")
            raise
    
    def train_nep(self, train_xyz: str, test_xyz: str, 
                  type_list: List[str], output_dir: str) -> str:
        """训练NEP模型 (GPUMD)"""
        self.monitor.start_stage("ml_training")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create nep.in
            nep_config = f"""type {' '.join(type_list)}
version 4
cutoff 6.0 4.0
n_max 4 4
basis_size 8 8
l_max 4
neuron 30
population 50
generation 100000
batch 1000
"""
            
            with open(Path(output_dir) / "nep.in", 'w') as f:
                f.write(nep_config)
            
            # Copy training data
            import shutil
            shutil.copy(train_xyz, Path(output_dir) / "train.xyz")
            shutil.copy(test_xyz, Path(output_dir) / "test.xyz")
            
            # Run NEP training
            import subprocess
            result = subprocess.run(
                ["nep"],
                cwd=output_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"NEP training failed: {result.stderr}")
            
            model_path = Path(output_dir) / "nep.txt"
            self.monitor.finish_stage("success", f"NEP model: {model_path}")
            
            return str(model_path)
            
        except Exception as e:
            self.error_handler.handle_error(e, "ml_training")
            raise


# =============================================================================
# Stage 4: MD Simulation
# =============================================================================

class MDSimulationStage:
    """MD模拟阶段"""
    
    def __init__(self, config: MDStageConfig, monitor: ProgressMonitor,
                 error_handler: ErrorHandler):
        self.config = config
        self.monitor = monitor
        self.error_handler = error_handler
    
    def run_lammps_md(self, structure: Structure, model_path: str,
                      temperature: float, output_dir: str) -> str:
        """运行LAMMPS MD"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to ASE
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        
        # Write LAMMPS data file
        from ase.io.lammpsdata import write_lammps_data
        data_file = Path(output_dir) / "structure.data"
        write_lammps_data(data_file, atoms, atom_style='atomic')
        
        # Determine pair style
        if model_path.endswith('.pb'):
            pair_style = "deepmd"
        elif model_path.endswith('.txt'):
            pair_style = "nep"
        else:
            pair_style = "snap"
        
        # Generate LAMMPS input
        input_content = f"""# LAMMPS MD simulation
units metal
atom_style atomic
boundary p p p
read_data structure.data

# Potential
pair_style {pair_style} {model_path}
pair_coeff * *

neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes

# Temperature initialization
velocity all create {temperature} 12345 dist gaussian

# Output
thermo 100
thermo_style custom step temp pe ke etotal press vol density
dump traj all custom 100 dump.lammpstrj id type x y z vx vy vz

# Ensemble
timestep {self.config.timestep / 1000}
"""
        
        # Add ensemble fix
        if self.config.ensemble == "nvt":
            input_content += f"fix ensemble all nvt temp {temperature} {temperature} 0.1\n"
        elif self.config.ensemble == "npt":
            input_content += f"fix ensemble all npt temp {temperature} {temperature} 0.1 iso 1.0 1.0 1.0\n"
        else:  # nve
            input_content += "fix ensemble all nve\n"
        
        # Equilibration
        input_content += f"""
# Equilibration
run {self.config.nsteps_equil}

# Production
run {self.config.nsteps_prod}

write_data final.data
"""
        
        # Write input file
        input_file = Path(output_dir) / "in.lammps"
        with open(input_file, 'w') as f:
            f.write(input_content)
        
        # Run LAMMPS
        import subprocess
        result = subprocess.run(
            ["mpirun", "-np", str(self.config.nprocs), "lmp", "-in", "in.lammps"],
            cwd=output_dir,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"LAMMPS failed: {result.stderr}")
            raise RuntimeError("LAMMPS simulation failed")
        
        dump_file = Path(output_dir) / "dump.lammpstrj"
        return str(dump_file)
    
    def run_multi_temperature(self, structure: Structure, model_path: str,
                              output_base_dir: str) -> Dict[float, str]:
        """在多个温度下运行MD"""
        self.monitor.start_stage("md_simulation")
        
        trajectories = {}
        
        for i, T in enumerate(self.config.temperatures):
            self.monitor.update((i / len(self.config.temperatures)) * 100, 
                               f"Running MD at {T}K")
            
            output_dir = Path(output_base_dir) / f"T{T}"
            
            try:
                traj_path = self.run_lammps_md(structure, model_path, T, str(output_dir))
                trajectories[T] = traj_path
            except Exception as e:
                self.error_handler.handle_error(e, f"md_simulation_T{T}")
        
        self.monitor.finish_stage("success", f"Generated {len(trajectories)} trajectories")
        
        return trajectories


# =============================================================================
# Stage 5: Analysis
# =============================================================================

class AnalysisStage:
    """分析阶段"""
    
    def __init__(self, config: AnalysisConfig, monitor: ProgressMonitor,
                 error_handler: ErrorHandler):
        self.config = config
        self.monitor = monitor
        self.error_handler = error_handler
    
    def analyze_diffusion(self, trajectory_file: str, atom_type: str,
                         timestep: float = 1.0) -> float:
        """计算扩散系数"""
        try:
            from ase.io.lammpsrun import read_lammps_dump_text
            from ase.io import read
            
            # Read trajectory
            frames = read(trajectory_file, index=':', format='lammps-dump-text')
            
            if len(frames) < 10:
                logger.warning("Not enough frames for diffusion analysis")
                return 0.0
            
            # Get positions of target atoms
            target_indices = [i for i, sym in enumerate(frames[0].get_chemical_symbols()) 
                            if sym == atom_type]
            
            if not target_indices:
                logger.warning(f"No {atom_type} atoms found")
                return 0.0
            
            # Calculate MSD
            positions_0 = frames[0].get_positions()[target_indices]
            msd_values = []
            times = []
            
            for i, frame in enumerate(frames[::10]):  # Sample every 10 frames
                positions_t = frame.get_positions()[target_indices]
                # Handle periodic boundary conditions
                msd = np.mean(np.sum((positions_t - positions_0)**2, axis=1))
                msd_values.append(msd)
                times.append(i * timestep * 10)
            
            # Linear fit to get D
            times = np.array(times)
            msd_values = np.array(msd_values)
            
            # Use middle 50% of data for fitting
            start_idx = len(times) // 4
            end_idx = 3 * len(times) // 4
            
            slope = np.polyfit(times[start_idx:end_idx], msd_values[start_idx:end_idx], 1)[0]
            D = slope / 6 * 1e-16  # Convert to cm^2/s (assuming positions in Å, time in fs)
            
            return D
            
        except Exception as e:
            self.error_handler.handle_error(e, "diffusion_analysis")
            return 0.0
    
    def compute_conductivity(self, D: float, structure: Structure, 
                            temperature: float, ion_type: str = "Li") -> float:
        """使用Nernst-Einstein计算离子电导率"""
        try:
            # Number density of ions
            comp = structure.composition
            n_ions = comp[ion_type]
            volume_cm3 = structure.volume * 1e-24  # Å^3 to cm^3
            n_density = n_ions / volume_cm3
            
            # Charge (assuming +1)
            q = 1.0 * 1.602e-19  # C
            
            # Boltzmann constant
            kB = 1.380649e-23  # J/K
            
            # Nernst-Einstein: σ = n * q² * D / (kB * T)
            sigma = n_density * q**2 * D / (kB * temperature)  # S/cm
            
            return sigma
            
        except Exception as e:
            self.error_handler.handle_error(e, "conductivity_calculation")
            return 0.0
    
    def fit_arrhenius(self, temperatures: List[float], 
                     diffusion_coeffs: List[float]) -> Tuple[float, float]:
        """拟合Arrhenius方程得到活化能"""
        try:
            temps = np.array(temperatures)
            diffs = np.array(diffusion_coeffs)
            
            # Remove invalid data
            valid = diffs > 0
            if np.sum(valid) < 2:
                return 0.0, 0.0
            
            temps = temps[valid]
            diffs = diffs[valid]
            
            # Arrhenius: ln(D) = ln(D0) - Ea/(kB*T)
            ln_D = np.log(diffs)
            inv_T = 1 / temps
            kB = 8.617e-5  # eV/K
            
            slope, intercept = np.polyfit(inv_T, ln_D, 1)
            Ea = -slope * kB  # eV
            D0 = np.exp(intercept)
            
            return Ea, D0
            
        except Exception as e:
            self.error_handler.handle_error(e, "arrhenius_fit")
            return 0.0, 0.0
    
    def run_full_analysis(self, trajectories: Dict[float, str], 
                         structure: Structure, output_dir: str) -> Dict:
        """运行完整分析"""
        self.monitor.start_stage("analysis")
        
        results = {
            'diffusion_coefficients': {},
            'conductivities': {},
            'activation_energy': None,
        }
        
        temperatures = sorted(trajectories.keys())
        diffusion_coeffs = []
        
        for i, T in enumerate(temperatures):
            self.monitor.update((i / len(temperatures)) * 50, 
                               f"Analyzing T={T}K")
            
            traj_file = trajectories[T]
            
            # Compute diffusion
            D = self.analyze_diffusion(traj_file, "Li", timestep=1.0)
            results['diffusion_coefficients'][T] = D
            diffusion_coeffs.append(D)
            
            # Compute conductivity
            sigma = self.compute_conductivity(D, structure, T, "Li")
            results['conductivities'][T] = sigma
        
        # Arrhenius fit
        if len(temperatures) >= 2:
            self.monitor.update(75, "Fitting Arrhenius equation")
            Ea, D0 = self.fit_arrhenius(temperatures, diffusion_coeffs)
            results['activation_energy'] = Ea
            results['pre_exponential'] = D0
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(Path(output_dir) / "analysis_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        self.monitor.update(100, "Analysis completed")
        self.monitor.finish_stage("success", f"Ea = {results['activation_energy']:.3f} eV")
        
        return results


# =============================================================================
# Main Workflow Orchestrator
# =============================================================================

class IntegratedMaterialsWorkflow:
    """
    统一材料计算工作流主类
    
    集成所有阶段: 结构获取 → DFT计算 → ML势训练 → MD模拟 → 分析
    """
    
    def __init__(self, config: IntegratedWorkflowConfig):
        self.config = config
        self.monitor = ProgressMonitor(len([s for s in config.stages.values() if s.enabled]))
        self.error_handler = ErrorHandler(config)
        
        # Initialize stages
        self.fetcher = StructureFetcher(config.mp_config, self.monitor, self.error_handler)
        self.dft_stage = DFTStage(config.dft_config, self.monitor, self.error_handler)
        self.ml_stage = MLTrainingStage(config.ml_config, self.monitor, self.error_handler)
        self.md_stage = MDSimulationStage(config.md_config, self.monitor, self.error_handler)
        self.analysis_stage = AnalysisStage(config.analysis_config, self.monitor, self.error_handler)
        
        self.results = {}
        self.working_dir = Path(config.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, 
            material_id: Optional[str] = None,
            structure_file: Optional[str] = None,
            formula: Optional[str] = None) -> Dict:
        """
        运行完整工作流
        
        Args:
            material_id: Materials Project ID (e.g., "mp-1234")
            structure_file: 本地结构文件路径
            formula: 化学式 (e.g., "Li3PS4")
            
        Returns:
            results: 包含所有阶段结果的字典
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"Starting Integrated Materials Workflow: {self.config.workflow_name}")
        logger.info(f"{'#'*60}\n")
        
        start_time = time.time()
        
        try:
            # Stage 1: Fetch Structure
            if self.config.stages["fetch_structure"].enabled:
                if structure_file:
                    structures = self.fetcher.fetch_from_file(structure_file)
                elif material_id or formula:
                    structures = self.fetcher.fetch_from_mp(
                        material_id=material_id,
                        formula=formula
                    )
                else:
                    raise WorkflowError("No input structure specified")
                
                structure = structures[0]  # Use first structure
                self.results['structure'] = structure
                self.results['formula'] = structure.formula
                
                # Save structure
                adaptor = AseAtomsAdaptor()
                atoms = adaptor.get_atoms(structure)
                write(self.working_dir / "initial_structure.vasp", atoms)
            else:
                structure = self.results.get('structure')
            
            # Stage 2: DFT Calculation
            if self.config.stages["dft_calculation"].enabled and structure is not None:
                dft_output_dir = self.working_dir / "dft_results"
                dft_results = self.dft_stage.run_relaxation(structure, str(dft_output_dir))
                self.results['dft'] = dft_results
                
                if self.config.save_intermediate:
                    with open(self.working_dir / "dft_results.json", 'w') as f:
                        json.dump(dft_results, f, indent=2)
            
            # Stage 3: ML Training
            if self.config.stages["ml_training"].enabled:
                dft_dirs = [str(self.working_dir / "dft_results")]
                ml_output_dir = self.working_dir / "ml_models"
                
                # Prepare data
                train_dir, valid_dir = self.ml_stage.prepare_data(
                    dft_dirs, 
                    str(self.working_dir / "training_data")
                )
                
                # Train models
                type_map = list(set(structure.symbol_set))
                
                if self.config.ml_config.framework == "deepmd":
                    model_paths = self.ml_stage.train_deepmd(
                        train_dir, valid_dir, type_map, str(ml_output_dir)
                    )
                elif self.config.ml_config.framework == "nep":
                    # Need to convert to XYZ first
                    model_paths = [self.ml_stage.train_nep(
                        train_dir + "/train.xyz",
                        valid_dir + "/test.xyz",
                        type_map,
                        str(ml_output_dir)
                    )]
                else:
                    raise ValueError(f"Unknown framework: {self.config.ml_config.framework}")
                
                self.results['ml_models'] = model_paths
            
            # Stage 4: MD Simulation
            if self.config.stages["md_simulation"].enabled:
                model_path = self.results['ml_models'][0]  # Use first model
                md_output_dir = self.working_dir / "md_results"
                
                trajectories = self.md_stage.run_multi_temperature(
                    structure, model_path, str(md_output_dir)
                )
                
                self.results['trajectories'] = trajectories
            
            # Stage 5: Analysis
            if self.config.stages["analysis"].enabled:
                analysis_output_dir = self.working_dir / "analysis"
                
                analysis_results = self.analysis_stage.run_full_analysis(
                    self.results['trajectories'],
                    structure,
                    str(analysis_output_dir)
                )
                
                self.results['analysis'] = analysis_results
            
            # Generate final report
            if self.config.generate_report:
                self._generate_report()
            
            total_time = time.time() - start_time
            logger.info(f"\n{'#'*60}")
            logger.info(f"Workflow completed in {total_time:.1f}s")
            logger.info(f"{'#'*60}\n")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            self.error_handler.handle_error(e, "workflow")
            raise
    
    def _generate_report(self):
        """生成最终报告"""
        report = {
            'workflow_name': self.config.workflow_name,
            'timestamp': datetime.now().isoformat(),
            'results': {
                'formula': self.results.get('formula'),
                'dft_energy': self.results.get('dft', {}).get('energy_per_atom'),
                'ml_models': self.results.get('ml_models'),
                'analysis': self.results.get('analysis'),
            },
            'progress': self.monitor.get_summary(),
            'errors': self.error_handler.get_report(),
        }
        
        # Convert numpy types
        report = self._convert_numpy(report)
        
        with open(self.working_dir / "workflow_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {self.working_dir / 'workflow_report.json'}")
    
    def _convert_numpy(self, obj):
        """递归转换numpy类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy(item) for item in obj]
        return obj


# =============================================================================
# Command Line Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Integrated Materials Workflow - DFT + ML + MD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full workflow from Materials Project ID
  python integrated_materials_workflow.py --mp-id mp-1234 -o ./output
  
  # Run from structure file
  python integrated_materials_workflow.py --structure POSCAR -o ./output
  
  # Run specific formula
  python integrated_materials_workflow.py --formula "Li3PS4" -o ./output
  
  # Disable DFT stage (use existing data)
  python integrated_materials_workflow.py --mp-id mp-1234 --skip-dft
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--mp-id', type=str, help='Materials Project ID')
    input_group.add_argument('--structure', type=str, help='Structure file path')
    input_group.add_argument('--formula', type=str, help='Chemical formula')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, default='./workflow_output',
                       help='Output directory')
    parser.add_argument('-n', '--name', type=str, default='materials_workflow',
                       help='Workflow name')
    
    # Stage control
    parser.add_argument('--skip-fetch', action='store_true',
                       help='Skip structure fetching (use existing)')
    parser.add_argument('--skip-dft', action='store_true',
                       help='Skip DFT calculation')
    parser.add_argument('--skip-ml', action='store_true',
                       help='Skip ML training')
    parser.add_argument('--skip-md', action='store_true',
                       help='Skip MD simulation')
    parser.add_argument('--skip-analysis', action='store_true',
                       help='Skip analysis')
    
    # Configuration options
    parser.add_argument('--dft-code', type=str, default='vasp',
                       choices=['vasp', 'espresso', 'abacus'],
                       help='DFT code')
    parser.add_argument('--ml-framework', type=str, default='deepmd',
                       choices=['deepmd', 'nep', 'mace'],
                       help='ML potential framework')
    parser.add_argument('--md-temps', type=float, nargs='+',
                       default=[300, 500, 700, 900],
                       help='MD temperatures (K)')
    parser.add_argument('--max-parallel', type=int, default=4,
                       help='Maximum parallel jobs')
    
    # Other options
    parser.add_argument('--api-key', type=str,
                       help='Materials Project API key')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    return parser


def main():
    """命令行入口"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = IntegratedWorkflowConfig(
        workflow_name=args.name,
        working_dir=args.output,
        mp_config=MaterialsProjectConfig(
            api_key=args.api_key,
        ),
        dft_config=DFTStageConfig(
            code=args.dft_code,
        ),
        ml_config=MLPotentialConfig(
            framework=args.ml_framework,
        ),
        md_config=MDStageConfig(
            temperatures=args.md_temps,
        ),
        max_parallel=args.max_parallel,
    )
    
    # Configure stages
    config.stages["fetch_structure"].enabled = not args.skip_fetch
    config.stages["dft_calculation"].enabled = not args.skip_dft
    config.stages["ml_training"].enabled = not args.skip_ml
    config.stages["md_simulation"].enabled = not args.skip_md
    config.stages["analysis"].enabled = not args.skip_analysis
    
    # Create and run workflow
    workflow = IntegratedMaterialsWorkflow(config)
    
    try:
        results = workflow.run(
            material_id=args.mp_id,
            structure_file=args.structure,
            formula=args.formula
        )
        
        # Print summary
        print("\n" + "="*60)
        print("WORKFLOW SUMMARY")
        print("="*60)
        print(f"Formula: {results.get('formula')}")
        print(f"DFT Energy: {results.get('dft', {}).get('energy_per_atom', 'N/A')} eV/atom")
        print(f"ML Models: {results.get('ml_models')}")
        
        if 'analysis' in results:
            analysis = results['analysis']
            print(f"\nDiffusion Coefficients:")
            for T, D in analysis.get('diffusion_coefficients', {}).items():
                print(f"  {T}K: {D:.2e} cm²/s")
            print(f"\nActivation Energy: {analysis.get('activation_energy', 'N/A'):.3f} eV")
        
        print(f"\nOutput directory: {args.output}")
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
