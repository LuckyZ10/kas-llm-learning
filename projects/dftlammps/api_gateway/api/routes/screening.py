"""
高通量筛选路由
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
    ScreeningRequest,
    ScreeningResponse,
    TaskStatus,
    APIResponse,
)
from ...tasks import (
    run_screening,
    batch_calculate_properties,
    generate_candidates,
    optimize_composition,
)
from ...utils import ForbiddenException
from ...monitoring import api_metrics

router = APIRouter()


@router.post("/run", response_model=ScreeningResponse)
async def submit_screening(
    request: Request,
    screening_request: ScreeningRequest,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    提交高通量筛选任务
    
    支持多种筛选策略：多目标优化、Pareto前沿、聚类分析
    """
    if not has_permission(current_user.role, Permission.SCREENING_RUN):
        raise ForbiddenException("Permission denied")
    
    # 提交Celery任务
    task = run_screening.delay({
        "dataset": screening_request.dataset,
        "criteria": [c.dict() for c in screening_request.criteria],
        "method": screening_request.method,
        "top_k": screening_request.top_k,
    })
    
    api_metrics.record_task_submission("screening", str(current_user.id))
    
    return ScreeningResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message=f"Screening queued using {screening_request.method} method"
    )


@router.post("/batch-properties")
async def batch_properties(
    request: Request,
    structures: List[dict],
    properties: List[str],
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    批量计算性质
    
    对多个结构批量计算指定性质
    
    - **structures**: 结构列表
    - **properties**: 性质列表 ["energy", "band_gap", "bulk_modulus"]
    """
    if not has_permission(current_user.role, Permission.SCREENING_RUN):
        raise ForbiddenException("Permission denied")
    
    task = batch_calculate_properties.delay(structures, properties)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": f"Batch property calculation queued for {len(structures)} structures"
    }


@router.post("/generate-candidates")
async def generate_candidates_endpoint(
    request: Request,
    template: dict,
    variations: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    生成候选结构
    
    基于模板结构生成变体
    
    - **template**: 模板结构
    - **variations**: 变体策略
    """
    if not has_permission(current_user.role, Permission.SCREENING_RUN):
        raise ForbiddenException("Permission denied")
    
    task = generate_candidates.delay(template, variations)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Candidate generation queued"
    }


@router.post("/optimize-composition")
async def optimize_composition_endpoint(
    request: Request,
    target_property: str,
    constraints: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    优化材料组分
    
    寻找最优组分以达到目标性质
    
    - **target_property**: 目标性质
    - **constraints**: 约束条件
    """
    if not has_permission(current_user.role, Permission.SCREENING_RUN):
        raise ForbiddenException("Permission denied")
    
    task = optimize_composition.delay(target_property, constraints)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": f"Composition optimization queued for {target_property}"
    }


@router.get("/methods")
async def list_screening_methods(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取支持的筛选方法"""
    return {
        "methods": [
            {
                "name": "multi_objective",
                "description": "多目标优化，加权组合多个目标函数",
                "parameters": ["weights", "objectives"],
            },
            {
                "name": "pareto",
                "description": "Pareto前沿搜索，寻找非支配解集",
                "parameters": ["objectives", "front_size"],
            },
            {
                "name": "clustering",
                "description": "聚类分析，探索化学空间多样性",
                "parameters": ["n_clusters", "features"],
            },
        ]
    }


@router.get("/properties")
async def list_screenable_properties(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取可筛选的性质列表"""
    return {
        "properties": [
            {
                "name": "energy",
                "unit": "eV",
                "description": "Formation energy",
                "type": "minimize",
            },
            {
                "name": "band_gap",
                "unit": "eV",
                "description": "Electronic band gap",
                "type": "target",
            },
            {
                "name": "bulk_modulus",
                "unit": "GPa",
                "description": "Bulk modulus",
                "type": "maximize",
            },
            {
                "name": "ionic_conductivity",
                "unit": "mS/cm",
                "description": "Ionic conductivity",
                "type": "maximize",
            },
            {
                "name": "diffusion_coefficient",
                "unit": "cm^2/s",
                "description": "Diffusion coefficient",
                "type": "maximize",
            },
            {
                "name": "volume_change",
                "unit": "%",
                "description": "Volume change during cycling",
                "type": "minimize",
            },
        ]
    }
