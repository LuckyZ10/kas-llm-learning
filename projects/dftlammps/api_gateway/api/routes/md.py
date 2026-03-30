"""
MD模拟路由
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, Request

from ...auth import (
    User,
    get_current_user_or_api_key,
    Permission,
    has_permission,
)
from ...models.schemas import (
    MDSimulationRequest,
    MDSimulationResponse,
    TaskStatus,
    APIResponse,
)
from ...tasks import (
    run_md_simulation,
    calculate_rdf,
    calculate_msd,
    analyze_trajectory,
    run_metadynamics,
    equilibrate_system,
)
from ...utils import ForbiddenException
from ...monitoring import api_metrics

router = APIRouter()


@router.post("/simulate", response_model=MDSimulationResponse)
async def submit_md_simulation(
    request: Request,
    sim_request: MDSimulationRequest,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    提交MD模拟任务
    
    支持多种ML势函数：NEP、DeepMD、ReaxFF、EAM等
    支持多种系综：NVT、NPT、NVE、Langevin等
    """
    if not has_permission(current_user.role, Permission.MD_SIMULATE):
        raise ForbiddenException("Permission denied")
    
    # 提交Celery任务
    task = run_md_simulation.delay({
        "simulation_type": sim_request.simulation_type.value,
        "potential": sim_request.potential.value,
        "structure": sim_request.structure,
        "temperature": sim_request.temperature,
        "pressure": sim_request.pressure,
        "n_steps": sim_request.n_steps,
        "time_step": sim_request.time_step,
        "ensemble": sim_request.ensemble,
        "constraints": sim_request.constraints,
    })
    
    api_metrics.record_task_submission("md_simulation", str(current_user.id))
    
    return MDSimulationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message=f"MD simulation queued with {sim_request.potential.value} potential"
    )


@router.post("/equilibrate", response_model=MDSimulationResponse)
async def equilibrate(
    request: Request,
    structure: dict,
    parameters: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    系统平衡
    
    将系统平衡到目标温度和压力
    """
    if not has_permission(current_user.role, Permission.MD_SIMULATE):
        raise ForbiddenException("Permission denied")
    
    task = equilibrate_system.delay(structure, parameters)
    
    return MDSimulationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="Equilibration queued"
    )


@router.post("/analysis/rdf")
async def calculate_rdf_endpoint(
    request: Request,
    trajectory_file: str,
    r_cut: float = 10.0,
    n_bins: int = 200,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    计算径向分布函数 (RDF)
    
    - **trajectory_file**: 轨迹文件路径
    - **r_cut**: 截断半径 (Å)
    - **n_bins**: 分箱数量
    """
    if not has_permission(current_user.role, Permission.MD_READ):
        raise ForbiddenException("Permission denied")
    
    task = calculate_rdf.delay(trajectory_file, r_cut, n_bins)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "RDF calculation queued"
    }


@router.post("/analysis/msd")
async def calculate_msd_endpoint(
    request: Request,
    trajectory_file: str,
    atom_types: Optional[List[str]] = None,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    计算均方位移 (MSD) 和扩散系数
    
    - **trajectory_file**: 轨迹文件路径
    - **atom_types**: 原子类型列表（可选）
    """
    if not has_permission(current_user.role, Permission.MD_READ):
        raise ForbiddenException("Permission denied")
    
    task = calculate_msd.delay(trajectory_file, atom_types)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "MSD calculation queued"
    }


@router.post("/analysis/trajectory")
async def analyze_trajectory_endpoint(
    request: Request,
    trajectory_file: str,
    analysis_types: List[str],
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    轨迹综合分析
    
    支持多种分析：RDF、MSD、振动谱等
    
    - **trajectory_file**: 轨迹文件路径
    - **analysis_types**: 分析类型列表 ["rdf", "msd", "vibrational_spectrum"]
    """
    if not has_permission(current_user.role, Permission.MD_READ):
        raise ForbiddenException("Permission denied")
    
    task = analyze_trajectory.delay(trajectory_file, analysis_types)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Trajectory analysis queued"
    }


@router.post("/metadynamics", response_model=MDSimulationResponse)
async def run_metadynamics_endpoint(
    request: Request,
    structure: dict,
    parameters: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    运行MetaDynamics模拟
    
    用于自由能面计算和增强采样
    
    - **structure**: 初始结构
    - **parameters**: 元动力学参数
    """
    if not has_permission(current_user.role, Permission.MD_SIMULATE):
        raise ForbiddenException("Permission denied")
    
    task = run_metadynamics.delay(structure, parameters)
    
    return MDSimulationResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message="Metadynamics simulation queued"
    )


@router.get("/potentials")
async def list_potentials(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取支持的势能列表"""
    return {
        "potentials": [
            {
                "name": "nep",
                "description": "Neuroevolution Potential - 高精度神经网络势",
                "type": "machine_learning",
                "supports": ["Li", "Na", "Mg", "Si", "O", "S", "P", "Cl"],
            },
            {
                "name": "deepmd",
                "description": "Deep Potential Molecular Dynamics",
                "type": "machine_learning",
                "supports": ["all_elements"],
            },
            {
                "name": "reaxff",
                "description": "Reactive Force Field - 反应力场",
                "type": "reactive",
                "supports": ["C", "H", "O", "N", "S", "P"],
            },
            {
                "name": "eam",
                "description": "Embedded Atom Method - 嵌入原子方法",
                "type": "empirical",
                "supports": ["metals"],
            },
            {
                "name": "meam",
                "description": "Modified EAM - 改进的EAM",
                "type": "empirical",
                "supports": ["metals", "alloys"],
            },
        ]
    }


@router.get("/ensembles")
async def list_ensembles(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取支持的系综列表"""
    return {
        "ensembles": [
            {"name": "NVT", "description": "Constant Number, Volume, Temperature"},
            {"name": "NPT", "description": "Constant Number, Pressure, Temperature"},
            {"name": "NVE", "description": "Constant Number, Volume, Energy"},
            {"name": "Langevin", "description": "Langevin dynamics with thermostat"},
            {"name": "Metadynamics", "description": "Enhanced sampling method"},
        ]
    }
