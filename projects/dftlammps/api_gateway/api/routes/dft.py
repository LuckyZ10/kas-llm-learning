"""
DFT计算路由
"""

from typing import Optional
from fastapi import APIRouter, Depends, Request

from ...auth import (
    User,
    get_current_user_or_api_key,
    Permission,
    has_permission,
)
from ...models.schemas import (
    DFTCalculationRequest,
    DFTCalculationResponse,
    TaskResponse,
    TaskPriority,
    TaskStatus,
    APIResponse,
)
from ...tasks import run_dft_calculation, calculate_band_structure, calculate_dos, relax_structure
from ...utils import ForbiddenException, ValidationException
from ...monitoring import api_metrics

router = APIRouter()


@router.post("/calculate", response_model=DFTCalculationResponse)
async def submit_dft_calculation(
    request: Request,
    calc_request: DFTCalculationRequest,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    提交DFT计算任务
    
    支持多种DFT代码：VASP、Quantum ESPRESSO、ABACUS、CP2K
    支持多种计算类型：SCF、结构优化、能带、态密度、声子、NEB等
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    # 提交Celery任务
    task = run_dft_calculation.delay({
        "calculation_type": calc_request.calculation_type.value,
        "code": calc_request.code.value,
        "structure": calc_request.structure,
        "parameters": calc_request.parameters,
        "kpoints": calc_request.kpoints,
        "pseudopotentials": calc_request.pseudopotentials,
    })
    
    api_metrics.record_task_submission("dft_calculation", str(current_user.id))
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message=f"DFT calculation queued with {calc_request.code.value}"
    )


@router.post("/bands", response_model=DFTCalculationResponse)
async def calculate_band_structure_endpoint(
    request: Request,
    structure: dict,
    k_path: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    计算能带结构
    
    - **structure**: 晶体结构
    - **k_path**: K点路径
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    task = calculate_band_structure.delay(structure, k_path)
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="Band structure calculation queued"
    )


@router.post("/dos", response_model=DFTCalculationResponse)
async def calculate_dos_endpoint(
    request: Request,
    structure: dict,
    energy_range: tuple = (-20, 20),
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    计算态密度
    
    - **structure**: 晶体结构
    - **energy_range**: 能量范围 (eV)
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    task = calculate_dos.delay(structure, energy_range)
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="DOS calculation queued"
    )


@router.post("/relax", response_model=DFTCalculationResponse)
async def relax_structure_endpoint(
    request: Request,
    structure: dict,
    parameters: dict = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    结构优化
    
    - **structure**: 初始结构
    - **parameters**: 优化参数
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    task = relax_structure.delay(structure, parameters or {})
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="Structure relaxation queued"
    )


@router.post("/phonon", response_model=DFTCalculationResponse)
async def calculate_phonon(
    request: Request,
    structure: dict,
    parameters: dict = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    计算声子谱
    
    - **structure**: 晶体结构
    - **parameters**: 计算参数
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    from ...tasks import calculate_phonon as calc_phonon
    
    task = calc_phonon.delay(structure, parameters or {})
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="Phonon calculation queued"
    )


@router.post("/neb", response_model=DFTCalculationResponse)
async def run_neb(
    request: Request,
    images: list,  # NEB图像
    parameters: dict = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    运行NEB计算（过渡态搜索）
    
    - **images**: NEB图像列表
    - **parameters**: NEB参数
    """
    if not has_permission(current_user.role, Permission.DFT_CALCULATE):
        raise ForbiddenException("Permission denied")
    
    from ...tasks import run_neb_calculation
    
    task = run_neb_calculation.delay(images, parameters or {})
    
    return DFTCalculationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="NEB calculation queued"
    )


@router.get("/codes")
async def list_dft_codes(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取支持的DFT代码列表"""
    return {
        "codes": [
            {
                "code": "vasp",
                "name": "VASP",
                "description": "Vienna Ab initio Simulation Package",
                "supported_calculations": ["scf", "relax", "bands", "dos", "phonon", "neb", "md"],
            },
            {
                "code": "quantum_espresso",
                "name": "Quantum ESPRESSO",
                "description": "Open-source DFT package",
                "supported_calculations": ["scf", "relax", "bands", "dos", "phonon", "md"],
            },
            {
                "code": "abacus",
                "name": "ABACUS",
                "description": "Atomic-orbital Based Ab-initio Computation at UStc",
                "supported_calculations": ["scf", "relax", "bands", "dos"],
            },
            {
                "code": "cp2k",
                "name": "CP2K",
                "description": "Open-source DFT and MD package",
                "supported_calculations": ["scf", "relax", "md"],
            },
        ]
    }


@router.get("/calculations")
async def list_dft_calculations(
    code: Optional[str] = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取DFT计算类型列表"""
    calculations = [
        {"type": "scf", "name": "Self-Consistent Field", "description": "单点能计算"},
        {"type": "relax", "name": "Structure Relaxation", "description": "结构优化"},
        {"type": "bands", "name": "Band Structure", "description": "能带结构计算"},
        {"type": "dos", "name": "Density of States", "description": "态密度计算"},
        {"type": "phonon", "name": "Phonon", "description": "声子谱计算"},
        {"type": "neb", "name": "Nudged Elastic Band", "description": "过渡态搜索"},
        {"type": "md", "name": "Molecular Dynamics", "description": "从头算分子动力学"},
    ]
    
    return {"calculations": calculations}
