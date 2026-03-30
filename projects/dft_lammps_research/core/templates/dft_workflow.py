#!/usr/bin/env python3
"""
DFT计算自动化工作流
支持VASP和Quantum ESPRESSO
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

# ASE
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.constraints import FixAtoms, UnitCellFilter
from ase.vibrations import Vibrations
from ase.phonons import Phonons
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.calculators.vasp import Vasp
from ase.calculators.espresso import Espresso
from ase.units import fs, kB

# Pymatgen
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.ext.matproj import MPRester

# FireWorks
from fireworks import Firework, Workflow, FWorker, LaunchPad
from fireworks.core.rocket_launcher import rapidfire

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DFTConfig:
    """DFT计算配置"""
    code: str = "vasp"  # "vasp" or "espresso"
    functional: str = "PBE"
    encut: float = 520  # eV
    kpoints_density: float = 0.25  # Å^-1
    ediff: float = 1e-6
    ncores: int = 32
    queue: str = "normal"
    walltime: int = 86400  # seconds
    
    # VASP specific
    vasp_cmd: str = "vasp_std"
    vasp_pp_path: str = "/path/to/pseudopotentials"
    
    # QE specific
    qe_cmd: str = "pw.x"
    qe_pp_path: str = "/path/to/qe/pseudopotentials"


class StructureOptimizer:
    """结构优化器"""
    
    def __init__(self, config: DFTConfig):
        self.config = config
        self.atoms = None
        
    def load_structure(self, input_file: str) -> Atoms:
        """加载结构文件"""
        self.atoms = read(input_file)
        logger.info(f"Loaded structure with {len(self.atoms)} atoms")
        return self.atoms
    
    def setup_calculator(self, atoms: Atoms, calc_type: str = "relax"):
        """设置DFT计算器"""
        if self.config.code == "vasp":
            return self._setup_vasp(atoms, calc_type)
        elif self.config.code == "espresso":
            return self._setup_espresso(atoms, calc_type)
        else:
            raise ValueError(f"Unsupported code: {self.config.code}")
    
    def _setup_vasp(self, atoms: Atoms, calc_type: str) -> Vasp:
        """配置VASP计算器"""
        calc = Vasp(
            xc=self.config.functional,
            encut=self.config.encut,
            ediff=self.config.ediff,
            ediffg=-0.01,
            ibrion=2,
            nsw=200,
            isif=3 if calc_type == "relax" else 2,
            ismear=0,
            sigma=0.05,
            prec="Accurate",
            lreal="Auto",
            lwave=False,
            lcharg=True,
            ncore=self.config.ncores,
            kpts=self._get_kpoints(atoms),
            command=f"mpirun -np {self.config.ncores} {self.config.vasp_cmd}",
        )
        return calc
    
    def _setup_espresso(self, atoms: Atoms, calc_type: str) -> Espresso:
        """配置Quantum ESPRESSO计算器"""
        pseudopotentials = self._get_pseudopotentials(atoms)
        
        input_data = {
            'control': {
                'calculation': 'vc-relax' if calc_type == "relax" else 'relax',
                'restart_mode': 'from_scratch',
                'prefix': 'struct',
                'outdir': './tmp',
                'tstress': True,
                'tprnfor': True,
            },
            'system': {
                'ecutwfc': self.config.encut / 13.6,  # Ry
                'ecutrho': 4 * self.config.encut / 13.6,
                'occupations': 'smearing',
                'smearing': 'gaussian',
                'degauss': 0.01,
            },
            'electrons': {
                'conv_thr': 1e-8,
                'mixing_beta': 0.7,
            },
            'ions': {
                'ion_dynamics': 'bfgs',
            },
            'cell': {
                'cell_dynamics': 'bfgs',
            } if calc_type == "relax" else {}
        }
        
        calc = Espresso(
            command=f"mpirun -np {self.config.ncores} {self.config.qe_cmd} -in PREFIX.pwi > PREFIX.pwo",
            pseudopotentials=pseudopotentials,
            input_data=input_data,
            kpts=self._get_kpoints(atoms),
        )
        return calc
    
    def _get_kpoints(self, atoms: Atoms) -> Tuple[int, int, int]:
        """基于k点密度自动计算k点网格"""
        cell = atoms.get_cell()
        kpoints = []
        for i in range(3):
            k = int(np.ceil(2 * np.pi / (self.config.kpoints_density * np.linalg.norm(cell[i]))))
            kpoints.append(max(1, k))
        return tuple(kpoints)
    
    def _get_pseudopotentials(self, atoms: Atoms) -> Dict[str, str]:
        """获取伪势文件映射"""
        # 简化版本，实际应从数据库读取
        pps = {}
        for symbol in set(atoms.get_chemical_symbols()):
            pps[symbol] = f"{symbol}.upf"
        return pps
    
    def relax_structure(self, fmax: float = 0.01) -> Atoms:
        """执行结构优化"""
        if self.atoms is None:
            raise ValueError("No structure loaded. Call load_structure first.")
        
        logger.info("Starting structure relaxation...")
        
        # 设置计算器
        calc = self.setup_calculator(self.atoms, calc_type="relax")
        self.atoms.calc = calc
        
        # 使用FIRE优化器 (比BFGS更稳定)
        optimizer = FIRE(self.atoms, logfile='relax.log')
        optimizer.run(fmax=fmax)
        
        logger.info(f"Relaxation completed. Final energy: {self.atoms.get_potential_energy():.4f} eV")
        
        return self.atoms
    
    def compute_vibrations(self, delta: float = 0.01) -> Dict:
        """计算振动频率"""
        logger.info("Computing vibrational frequencies...")
        
        vib = Vibrations(self.atoms, delta=delta)
        vib.run()
        
        frequencies = vib.get_frequencies()
        
        # 检查虚频
        imaginary_modes = [f for f in frequencies if f < 0]
        is_stable = len(imaginary_modes) == 0
        
        results = {
            'frequencies': frequencies.tolist(),
            'imaginary_modes': len(imaginary_modes),
            'is_stable': is_stable,
        }
        
        logger.info(f"Vibrational analysis: {len(imaginary_modes)} imaginary modes")
        
        return results
    
    def generate_aimd_data(self, 
                          temperature: float = 300,
                          timestep: float = 1.0,  # fs
                          nsteps: int = 10000,
                          output_interval: int = 50) -> str:
        """生成AIMD训练数据"""
        logger.info(f"Starting AIMD at {temperature}K for {nsteps} steps...")
        
        # 设置MD计算器
        calc = self.setup_calculator(self.atoms, calc_type="md")
        self.atoms.calc = calc
        
        # 设置速度分布
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        
        # 创建MD运行器
        dyn = Langevin(self.atoms, timestep * fs, temperature_K=temperature, friction=0.01)
        
        # 设置输出
        trajectory_file = "aimd.traj"
        from ase.io import Trajectory
        traj = Trajectory(trajectory_file, 'w', self.atoms)
        
        energies = []
        forces = []
        
        def write_frame():
            traj.write()
            energies.append(self.atoms.get_potential_energy())
            forces.append(self.atoms.get_forces().copy())
        
        dyn.attach(write_frame, interval=output_interval)
        
        # 运行MD
        dyn.run(nsteps)
        traj.close()
        
        logger.info(f"AIMD completed. Data saved to {trajectory_file}")
        
        return trajectory_file


class BatchDFTCalculator:
    """批量DFT计算器"""
    
    def __init__(self, config: DFTConfig):
        self.config = config
        self.optimizer = StructureOptimizer(config)
        
    def run_batch_relaxation(self, 
                            input_files: List[str],
                            output_dir: str = "./results") -> List[Dict]:
        """批量结构优化"""
        results = []
        
        for i, input_file in enumerate(input_files):
            logger.info(f"Processing structure {i+1}/{len(input_files)}: {input_file}")
            
            try:
                # 加载并优化
                atoms = self.optimizer.load_structure(input_file)
                optimized = self.optimizer.relax_structure()
                
                # 计算振动
                vib_data = self.optimizer.compute_vibrations()
                
                # 保存结果
                result = {
                    'input_file': input_file,
                    'energy': optimized.get_potential_energy(),
                    'forces': optimized.get_forces().tolist(),
                    'stress': optimized.get_stress().tolist(),
                    'cell': optimized.get_cell().tolist(),
                    'positions': optimized.get_positions().tolist(),
                    'vibrations': vib_data,
                    'status': 'success'
                }
                
                # 保存优化后的结构
                output_file = Path(output_dir) / f"relaxed_{i}.vasp"
                write(output_file, optimized)
                
            except Exception as e:
                result = {
                    'input_file': input_file,
                    'status': 'failed',
                    'error': str(e)
                }
                logger.error(f"Failed to process {input_file}: {e}")
            
            results.append(result)
        
        # 保存汇总结果
        with open(Path(output_dir) / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        return results


class MaterialsProjectInterface:
    """Materials Project接口"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        self.mpr = MPRester(self.api_key) if self.api_key else None
    
    def query_by_formula(self, formula: str) -> List[Dict]:
        """按化学式查询材料"""
        if self.mpr is None:
            raise ValueError("MP API key not provided")
        
        docs = self.mpr.summary.search(formula=formula)
        return [doc.dict() for doc in docs]
    
    def download_structure(self, material_id: str) -> Atoms:
        """下载结构并转换为ASE Atoms"""
        structure = self.mpr.get_structure_by_material_id(material_id)
        return AseAtomsAdaptor.get_atoms(structure)
    
    def get_bulk_modulus_candidates(self, 
                                   elements: List[str],
                                   max_entries: int = 10) -> List[Dict]:
        """获取体模量计算候选材料"""
        docs = self.mpr.summary.search(
            elements=elements,
            fields=["material_id", "formula_pretty", "structure"],
            num_chunks=1,
            chunk_size=max_entries
        )
        return docs


def main():
    """示例用法"""
    
    # 配置
    config = DFTConfig(
        code="vasp",
        functional="PBE",
        encut=520,
        ncores=32
    )
    
    # 创建优化器
    optimizer = StructureOptimizer(config)
    
    # 加载结构 (假设存在POSCAR文件)
    # optimizer.load_structure("POSCAR")
    
    # 结构优化
    # optimized = optimizer.relax_structure()
    
    # 计算振动
    # vib_results = optimizer.compute_vibrations()
    
    # 生成AIMD数据
    # trajectory = optimizer.generate_aimd_data(
    #     temperature=300,
    #     timestep=1.0,
    #     nsteps=10000
    # )
    
    print("DFT Workflow template ready!")


if __name__ == "__main__":
    main()
