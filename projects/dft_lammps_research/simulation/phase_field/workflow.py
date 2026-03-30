"""
Phase Field Workflow Module
===========================
相场-DFT多尺度工作流模块

整合DFT/MD计算与相场模拟的完整工作流。
提供从参数提取到结果验证的自动化流程。
"""

import numpy as np
from typing import Dict, Optional, List, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class WorkflowConfig:
    """工作流配置"""
    # 工作流名称
    name: str = "phase_field_dft_workflow"
    
    # 工作目录
    work_dir: str = "./pf_workflow"
    
    # 计算阶段
    run_dft: bool = True
    run_md: bool = False
    run_phase_field: bool = True
    
    # DFT参数
    dft_output_path: Optional[str] = None
    
    # MD参数
    md_trajectory_path: Optional[str] = None
    
    # 相场参数
    pf_config: Optional[Dict] = None
    pf_model_type: str = "cahn_hilliard"  # cahn_hilliard, electrochemical, mechanochemical
    
    # 耦合参数
    coupling_mode: str = "one_way"  # one_way, two_way, iterative
    
    # 验证
    validate_results: bool = True
    compare_with_experiment: bool = False
    experimental_data: Optional[Dict] = None
    
    # 输出
    save_intermediate: bool = True
    generate_report: bool = True


class PhaseFieldWorkflow:
    """
    相场-DFT多尺度工作流
    
    自动化执行从DFT到相场的完整模拟流程:
    1. 从DFT/MD提取参数
    2. 参数传递和转换
    3. 相场模拟
    4. 结果验证和反馈
    """
    
    def __init__(self, config: Optional[WorkflowConfig] = None):
        """
        初始化工作流
        
        Args:
            config: 工作流配置
        """
        self.config = config or WorkflowConfig()
        
        # 创建工作目录
        self.work_dir = Path(self.config.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.results = {
            'dft_params': None,
            'md_params': None,
            'transferred_params': None,
            'phase_field_result': None,
            'validation': None
        }
        
        # 日志
        self.log_file = self.work_dir / "workflow.log"
        
        logger.info(f"Phase field workflow initialized: {self.config.name}")
    
    def run(self, callbacks: Optional[Dict[str, Callable]] = None) -> Dict:
        """
        运行完整工作流
        
        Args:
            callbacks: 阶段回调函数字典
            
        Returns:
            results: 工作流结果
        """
        self._log("=" * 60)
        self._log(f"Starting workflow: {self.config.name}")
        self._log(f"Timestamp: {datetime.now().isoformat()}")
        self._log("=" * 60)
        
        callbacks = callbacks or {}
        
        # 步骤1: DFT参数提取
        if self.config.run_dft and self.config.dft_output_path:
            self._log("\n[Step 1] Extracting parameters from DFT...")
            dft_params = self._extract_dft_parameters()
            self.results['dft_params'] = dft_params
            
            if 'on_dft_complete' in callbacks:
                callbacks['on_dft_complete'](dft_params)
            
            if self.config.save_intermediate:
                self._save_json(dft_params, "dft_parameters.json")
        
        # 步骤2: MD参数提取
        if self.config.run_md and self.config.md_trajectory_path:
            self._log("\n[Step 2] Extracting parameters from MD...")
            md_params = self._extract_md_parameters()
            self.results['md_params'] = md_params
            
            if 'on_md_complete' in callbacks:
                callbacks['on_md_complete'](md_params)
            
            if self.config.save_intermediate:
                self._save_json(md_params, "md_parameters.json")
        
        # 步骤3: 参数传递
        self._log("\n[Step 3] Transferring parameters...")
        transferred = self._transfer_parameters()
        self.results['transferred_params'] = transferred
        
        if 'on_transfer_complete' in callbacks:
            callbacks['on_transfer_complete'](transferred)
        
        if self.config.save_intermediate:
            self._save_json(transferred, "transferred_parameters.json")
        
        # 步骤4: 相场模拟
        if self.config.run_phase_field:
            self._log("\n[Step 4] Running phase field simulation...")
            pf_result = self._run_phase_field_simulation()
            self.results['phase_field_result'] = pf_result
            
            if 'on_pf_complete' in callbacks:
                callbacks['on_pf_complete'](pf_result)
            
            if self.config.save_intermediate:
                self._save_json(pf_result, "phase_field_result.json")
        
        # 步骤5: 验证
        if self.config.validate_results:
            self._log("\n[Step 5] Validating results...")
            validation = self._validate_results()
            self.results['validation'] = validation
            
            if self.config.save_intermediate:
                self._save_json(validation, "validation.json")
        
        # 步骤6: 反馈到DFT (双向耦合)
        if self.config.coupling_mode in ['two_way', 'iterative']:
            self._log("\n[Step 6] Feedback to DFT...")
            feedback = self._feedback_to_dft()
            self.results['feedback'] = feedback
        
        # 生成报告
        if self.config.generate_report:
            self._generate_report()
        
        self._log("\n" + "=" * 60)
        self._log("Workflow completed!")
        self._log("=" * 60)
        
        return self.results
    
    def _extract_dft_parameters(self) -> Dict:
        """从DFT输出提取参数"""
        from ..coupling.dft_coupling import DFTCoupling, DFTCouplingConfig
        
        config = DFTCouplingConfig(
            dft_code="vasp",
            extract_thermodynamic_params=True,
            extract_kinetic_params=True
        )
        
        coupling = DFTCoupling(config)
        
        # 提取参数
        params = coupling.extract_from_dft_output(self.config.dft_output_path)
        
        # 生成相场参数
        pf_params = coupling.generate_phase_field_params()
        
        return pf_params
    
    def _extract_md_parameters(self) -> Dict:
        """从MD轨迹提取参数"""
        from ..coupling.md_coupling import MDCoupling, MDCouplingConfig
        
        config = MDCouplingConfig()
        coupling = MDCoupling(config)
        
        # 提取参数
        params = coupling.extract_from_trajectory(self.config.md_trajectory_path)
        
        # 生成相场参数
        pf_params = coupling.generate_phase_field_params()
        
        return pf_params
    
    def _transfer_parameters(self) -> Dict:
        """参数传递和转换"""
        from ..coupling.parameter_transfer import ParameterTransfer, TransferConfig
        
        config = TransferConfig()
        transfer = ParameterTransfer(config)
        
        # 合并DFT和MD参数
        dft_params = self.results.get('dft_params', {})
        md_params = self.results.get('md_params', {})
        
        transferred = {}
        
        if dft_params:
            transferred_dft = transfer.transfer_from_dft(dft_params)
            transferred.update(transferred_dft)
        
        if md_params:
            transferred_md = transfer.transfer_from_md(md_params)
            transferred.update(transferred_md)
        
        if dft_params and md_params:
            merged = transfer.merge_parameters(dft_params, md_params)
            transferred = merged
        
        # 保存参数文件
        transfer.transferred_params = {'merged': transferred}
        transfer.generate_parameter_file(str(self.work_dir / "phase_field_parameters.json"))
        
        return transferred
    
    def _run_phase_field_simulation(self) -> Dict:
        """运行相场模拟"""
        params = self.results.get('transferred_params', {})
        
        # 根据模型类型选择求解器
        if self.config.pf_model_type == "cahn_hilliard":
            from ..core.cahn_hilliard import CahnHilliardSolver, CahnHilliardConfig
            
            config = CahnHilliardConfig(**self.config.pf_config or {})
            # 应用传递的参数
            if 'M' in params:
                config.M = params['M']
            if 'kappa' in params:
                config.kappa = params['kappa']
            
            solver = CahnHilliardSolver(config)
            solver.initialize_fields()
            
        elif self.config.pf_model_type == "electrochemical":
            from ..core.electrochemical import ElectrochemicalPhaseField, ElectrochemicalConfig
            
            config = ElectrochemicalConfig(**self.config.pf_config or {})
            solver = ElectrochemicalPhaseField(config)
            solver.initialize_fields()
            
        elif self.config.pf_model_type == "mechanochemical":
            from ..core.mechanochemistry import MechanoChemicalSolver, MechanoChemicalConfig
            
            config = MechanoChemicalConfig(**self.config.pf_config or {})
            # 应用弹性常数
            if 'elastic' in params:
                elastic = params['elastic']
                config.E = elastic.get('C11', 100)
            
            solver = MechanoChemicalSolver(config)
            solver.initialize_fields()
        else:
            raise ValueError(f"Unknown model type: {self.config.pf_model_type}")
        
        # 运行模拟
        self._log(f"Running {self.config.pf_model_type} simulation...")
        result = solver.run()
        
        # 收集结果
        pf_result = {
            'model_type': self.config.pf_model_type,
            'final_energy': result.get('final_energy', 0),
            'total_steps': result.get('total_steps', 0),
            'converged': result.get('converged', False),
            'parameters_used': {
                'M': getattr(config, 'M', 1.0),
                'kappa': getattr(config, 'kappa', 1.0),
            }
        }
        
        # 添加特定结果
        if hasattr(solver, 'get_domain_size'):
            pf_result['domain_size'] = solver.get_domain_size()
        
        return pf_result
    
    def _validate_results(self) -> Dict:
        """验证结果"""
        validation = {
            'status': 'passed',
            'checks': []
        }
        
        pf_result = self.results.get('phase_field_result', {})
        
        # 检查收敛性
        if pf_result.get('converged'):
            validation['checks'].append({
                'name': 'convergence',
                'status': 'passed',
                'message': 'Simulation converged'
            })
        else:
            validation['checks'].append({
                'name': 'convergence',
                'status': 'warning',
                'message': 'Simulation did not converge within max steps'
            })
        
        # 与实验对比
        if self.config.compare_with_experiment and self.config.experimental_data:
            exp_data = self.config.experimental_data
            
            if 'domain_size' in pf_result and 'domain_size' in exp_data:
                calc_size = pf_result['domain_size']
                exp_size = exp_data['domain_size']
                error = abs(calc_size - exp_size) / exp_size * 100
                
                validation['checks'].append({
                    'name': 'domain_size_comparison',
                    'status': 'passed' if error < 20 else 'warning',
                    'calculated': calc_size,
                    'experimental': exp_size,
                    'error_percent': error
                })
        
        return validation
    
    def _feedback_to_dft(self) -> Dict:
        """将相场结果反馈到DFT"""
        from ..coupling.dft_coupling import DFTCoupling
        
        pf_result = self.results.get('phase_field_result', {})
        
        # 准备反馈数据
        feedback_data = {
            'interface_structure': pf_result.get('interface_profile'),
            'defect_structures': pf_result.get('defects', []),
            'domain_size': pf_result.get('domain_size')
        }
        
        # 生成DFT输入
        coupling = DFTCoupling()
        dft_inputs = coupling.feedback_to_dft(feedback_data, 
                                             str(self.work_dir / "dft_feedback"))
        
        return {
            'dft_input_files': [str(f) for f in dft_inputs],
            'suggested_calculations': ['interface_energy', 'defect_formation_energy']
        }
    
    def _generate_report(self):
        """生成工作报告"""
        report_file = self.work_dir / "workflow_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Phase Field - DFT Workflow Report\n\n")
            f.write(f"**Workflow Name:** {self.config.name}\n\n")
            f.write(f"**Date:** {datetime.now().isoformat()}\n\n")
            f.write("---\n\n")
            
            # DFT参数
            if self.results.get('dft_params'):
                f.write("## DFT Parameters\n\n")
                f.write(f"```json\n{json.dumps(self.results['dft_params'], indent=2)}\n```\n\n")
            
            # 相场结果
            if self.results.get('phase_field_result'):
                f.write("## Phase Field Results\n\n")
                pf_result = self.results['phase_field_result']
                f.write(f"- **Model Type:** {pf_result.get('model_type')}\n")
                f.write(f"- **Total Steps:** {pf_result.get('total_steps')}\n")
                f.write(f"- **Converged:** {pf_result.get('converged')}\n")
                f.write(f"- **Final Energy:** {pf_result.get('final_energy', 0):.4f}\n\n")
            
            # 验证结果
            if self.results.get('validation'):
                f.write("## Validation\n\n")
                validation = self.results['validation']
                f.write(f"**Status:** {validation.get('status')}\n\n")
                for check in validation.get('checks', []):
                    f.write(f"- {check['name']}: {check['status']}\n")
            
            f.write("\n---\n\n")
            f.write("*Generated by Phase Field - DFT Workflow Module*\n")
        
        logger.info(f"Report generated: {report_file}")
    
    def _log(self, message: str):
        """记录日志"""
        logger.info(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
    
    def _save_json(self, data: Dict, filename: str):
        """保存JSON文件"""
        filepath = self.work_dir / filename
        
        # 转换不可序列化的对象
        def serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if callable(obj):
                return 'Callable'
            return str(obj)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=serialize)
        
        logger.info(f"Saved: {filepath}")
