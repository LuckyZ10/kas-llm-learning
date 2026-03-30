"""
重构示例：展示如何使用新的统一系统简化代码
=============================================

此文件演示如何将旧的分散代码重构为使用新架构的版本。

对比:
- 旧代码：分散的dataclass、重复的日志配置、硬编码路径
- 新代码：统一配置、基类继承、工厂模式
"""

# =============================================================================
# 旧代码风格（分散、重复）
# =============================================================================

OLD_STYLE_CODE = '''
# 文件1: dft_config.py
from dataclasses import dataclass
import logging

@dataclass
class DFTConfig:
    code: str = "vasp"
    encut: float = 520.0
    functional: str = "PBE"

# 独立的日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 文件2: vasp_calculator.py  
import logging
from dataclasses import dataclass

# 重复的dataclass
@dataclass
class VaspConfig:
    encut: float = 520.0
    functional: str = "PBE"
    
# 又一个独立的日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class VaspCalculator:
    def __init__(self, config=None):
        self.config = config or VaspConfig()
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, atoms):
        self.logger.info("Starting calculation")
        # ... 硬编码路径和配置
        return {"energy": 0.0}
'''


# =============================================================================
# 新代码风格（统一、简洁）
# =============================================================================

# 只需要导入 core 模块
from core import (
    get_config,           # 统一配置
    get_logger,           # 统一日志
    DFTCalculator,        # 抽象基类
    CalculationResult,    # 统一数据结构
    CalculatorFactory,    # 工厂模式
)


# 继承基类，实现特定功能
class VaspCalculator(DFTCalculator):
    """VASP计算器 - 遵循统一接口"""
    
    @property
    def code_name(self) -> str:
        return "vasp"
    
    def setup(self, atoms, working_dir):
        """实现基类要求的接口"""
        from pathlib import Path
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用统一配置
        config = self.config or get_config().dft.vasp
        
        # 写INCAR文件
        incar_content = f"""ENCUT = {config.encut}
EDIFF = {config.ediff}
ISMEAR = {config.ismear}
SIGMA = {config.sigma}
"""
        (self.working_dir / "INCAR").write_text(incar_content)
        self.logger.info(f"INCAR written to {self.working_dir}")
    
    def calculate(self, atoms):
        """实现基类要求的接口"""
        self.setup(atoms, "./vasp_calc")
        
        # 模拟计算...
        self.logger.info("Running VASP calculation")
        
        # 返回统一数据结构
        return CalculationResult(
            energy=-100.5,
            forces=None,
            success=True,
            computation_time=120.0
        )
    
    def relax_structure(self, atoms, fmax=0.01, max_steps=200):
        """实现基类要求的接口"""
        result = self.calculate(atoms)
        return atoms, result
    
    def read_results(self, working_dir):
        """实现基类要求的接口"""
        # 解析OUTCAR...
        return CalculationResult(energy=-100.5, success=True)


# 注册到工厂
CalculatorFactory.register_dft("vasp", VaspCalculator)


# =============================================================================
# 使用示例
# =============================================================================

def example_usage():
    """展示如何使用新的统一系统"""
    
    # 1. 初始化配置和日志（一次即可）
    from core import initialize_logging, get_config
    
    initialize_logging(
        level="INFO",
        log_dir="./logs",
        file="workflow.log",
        colors=True
    )
    
    # 2. 获取统一配置
    config = get_config("config.yaml")  # 或 get_config() 使用默认值
    
    print(f"VASP ENCUT: {config.dft.vasp.encut}")
    print(f"NEP Neuron: {config.ml.nep.neuron}")
    
    # 3. 使用工厂创建计算器
    calc = CalculatorFactory.create_dft("vasp", config.dft)
    
    # 4. 获取统一日志记录器
    logger = get_logger(__name__)
    logger.info("Starting materials workflow")
    
    # 5. 执行计算
    from ase import Atoms
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    
    result = calc.calculate(atoms)
    logger.info(f"Calculation completed: energy={result.energy:.3f} eV")


# =============================================================================
# 更多重构示例
# =============================================================================

# 示例：将分散的日志配置统一

def refactor_logging_example():
    """日志重构示例"""
    
    # 旧代码：每个文件都有
    # logging.basicConfig(level=logging.INFO)
    # logger = logging.getLogger(__name__)
    
    # 新代码：统一入口
    from core import get_logger
    
    # 自动继承全局配置
    logger = get_logger(__name__)
    logger.info("This uses the unified logging configuration")


# 示例：将分散的dataclass统一

def refactor_config_example():
    """配置重构示例"""
    
    # 旧代码：每个模块定义自己的配置类
    # @dataclass
    # class MyDFTConfig:
    #     encut: float = 520
    
    # 新代码：使用统一配置
    from core import get_config
    
    config = get_config()
    
    # 访问层级化配置
    encut = config.dft.vasp.encut
    temperature = config.md.lammps.temperature
    
    # 支持环境变量覆盖
    # $ export DFT_VASP_ENCUT=600
    # $ python script.py  # 自动使用600


# 示例：将硬编码的实例化改为工厂模式

def refactor_factory_example():
    """工厂模式重构示例"""
    
    # 旧代码：硬编码实例化
    # if code == "vasp":
    #     calc = VaspCalculator(config)
    # elif code == "espresso":
    #     calc = EspressoCalculator(config)
    
    # 新代码：工厂模式
    from core import CalculatorFactory
    
    calc = CalculatorFactory.create_dft("vasp")  # 或 "espresso"
    
    # 同样适用于MD和ML
    md = CalculatorFactory.create_md("lammps")
    ml = CalculatorFactory.create_ml("nep")


# 示例：使用结构化日志

def refactor_structured_logging():
    """结构化日志示例"""
    
    from core import log_structured, get_logger
    import logging
    
    logger = get_logger(__name__)
    
    # 普通日志
    logger.info("Calculation started")
    
    # 结构化日志（便于后续分析）
    log_structured(
        logger, logging.INFO,
        "DFT calculation completed",
        structure_id="mp-12345",
        energy=-123.456,
        n_iterations=45,
        converged=True
    )


# 示例：使用装饰器记录函数执行

def refactor_decorators():
    """装饰器重构示例"""
    
    from core import log_execution, log_time, get_logger
    
    logger = get_logger(__name__)
    
    @log_execution(logger_name=__name__)
    @log_time(logger_name=__name__)
    def expensive_calculation(n):
        """自动记录开始、完成和执行时间"""
        import time
        time.sleep(0.1)
        return sum(range(n))
    
    result = expensive_calculation(1000000)


# =============================================================================
# 完整工作流示例
# =============================================================================

def complete_workflow_example():
    """展示完整的重构后工作流"""
    
    from core import (
        get_config, get_logger, initialize_logging,
        CalculatorFactory, CalculationResult
    )
    from ase import Atoms
    
    # 1. 初始化
    initialize_logging(level="INFO", log_dir="./logs")
    config = get_config()
    logger = get_logger("workflow")
    
    # 2. 准备结构
    logger.info("Preparing structures")
    structures = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]]),
        Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
    ]
    
    # 3. 创建计算器（统一接口）
    dft_calc = CalculatorFactory.create_dft(config.dft.code, config.dft)
    
    # 4. 批量计算
    results = []
    for i, atoms in enumerate(structures):
        logger.info(f"Calculating structure {i+1}/{len(structures)}")
        result = dft_calc.calculate(atoms)
        results.append(result)
        
        if result.success:
            logger.info(f"  Energy: {result.energy:.3f} eV")
        else:
            logger.error(f"  Failed: {result.error_message}")
    
    # 5. 汇总
    logger.info(f"Completed {len([r for r in results if r.success])}/{len(results)} calculations")
    
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("重构示例：展示新的统一架构如何使用")
    print("=" * 60)
    
    # 运行示例
    example_usage()
    
    print("\n" + "=" * 60)
    print("重构完成！查看上方的代码对比。")
    print("=" * 60)
