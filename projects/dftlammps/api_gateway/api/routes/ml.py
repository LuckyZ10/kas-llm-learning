"""
ML训练路由
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
    MLTrainingRequest,
    MLTrainingResponse,
    TaskStatus,
    APIResponse,
)
from ...tasks import (
    train_ml_model,
    evaluate_model,
    validate_model,
    hyperparameter_search,
    active_learning_iteration,
)
from ...utils import ForbiddenException
from ...monitoring import api_metrics

router = APIRouter()


@router.post("/train", response_model=MLTrainingResponse)
async def submit_ml_training(
    request: Request,
    training_request: MLTrainingRequest,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    提交ML模型训练任务
    
    支持多种模型：NEP、DeepMD、GAP、ACE等
    """
    if not has_permission(current_user.role, Permission.ML_TRAIN):
        raise ForbiddenException("Permission denied")
    
    # 提交Celery任务
    task = train_ml_model.delay({
        "model_type": training_request.model_type.value,
        "training_data": training_request.training_data,
        "validation_data": training_request.validation_data,
        "hyperparameters": training_request.hyperparameters,
        "epochs": training_request.epochs,
        "batch_size": training_request.batch_size,
        "device": training_request.device,
    })
    
    api_metrics.record_task_submission("ml_training", str(current_user.id))
    
    return MLTrainingResponse(
        task_id=task.id,
        status=TaskStatus.QUEUED,
        message=f"ML training queued for {training_request.model_type.value} model"
    )


@router.post("/evaluate")
async def evaluate_ml_model(
    request: Request,
    model_file: str,
    test_data: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    评估ML模型性能
    
    - **model_file**: 模型文件路径
    - **test_data**: 测试数据集
    """
    if not has_permission(current_user.role, Permission.ML_READ):
        raise ForbiddenException("Permission denied")
    
    task = evaluate_model.delay(model_file, test_data)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Model evaluation queued"
    }


@router.post("/validate")
async def validate_ml_model(
    request: Request,
    model_file: str,
    validation_config: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    验证ML模型
    
    检查能量、力、声子等性质
    
    - **model_file**: 模型文件路径
    - **validation_config**: 验证配置
    """
    if not has_permission(current_user.role, Permission.ML_READ):
        raise ForbiddenException("Permission denied")
    
    task = validate_model.delay(model_file, validation_config)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Model validation queued"
    }


@router.post("/hyperparameter-search")
async def hyperparameter_search_endpoint(
    request: Request,
    training_data: dict,
    search_space: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    超参数搜索
    
    自动寻找最优超参数组合
    
    - **training_data**: 训练数据
    - **search_space**: 搜索空间
    """
    if not has_permission(current_user.role, Permission.ML_TRAIN):
        raise ForbiddenException("Permission denied")
    
    task = hyperparameter_search.delay(training_data, search_space)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Hyperparameter search queued"
    }


@router.post("/active-learning")
async def active_learning(
    request: Request,
    config: dict,
    current_user: User = Depends(get_current_user_or_api_key)
):
    """
    主动学习迭代
    
    选择不确定性高的构型进行DFT计算，迭代改进模型
    
    - **config**: 主动学习配置
    """
    if not has_permission(current_user.role, Permission.ML_TRAIN):
        raise ForbiddenException("Permission denied")
    
    task = active_learning_iteration.delay(config)
    
    return {
        "task_id": task.id,
        "status": TaskStatus.QUEUED,
        "message": "Active learning iteration queued"
    }


@router.get("/models")
async def list_ml_models(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取支持的ML模型列表"""
    return {
        "models": [
            {
                "name": "nep",
                "full_name": "Neuroevolution Potential",
                "type": "neural_network",
                "description": "高精度神经网络势能",
                "training_speed": "fast",
                "inference_speed": "very_fast",
            },
            {
                "name": "deepmd",
                "full_name": "Deep Potential MD",
                "type": "neural_network",
                "description": "深度势能分子动力学",
                "training_speed": "medium",
                "inference_speed": "fast",
            },
            {
                "name": "gap",
                "full_name": "Gaussian Approximation Potential",
                "type": "kernel",
                "description": "高斯近似势能",
                "training_speed": "slow",
                "inference_speed": "medium",
            },
            {
                "name": "ace",
                "full_name": "Atomic Cluster Expansion",
                "type": "polynomial",
                "description": "原子簇展开",
                "training_speed": "fast",
                "inference_speed": "very_fast",
            },
        ]
    }


@router.get("/datasets")
async def list_datasets(
    current_user: User = Depends(get_current_user_or_api_key)
):
    """获取可用的训练数据集"""
    return {
        "datasets": [
            {
                "id": "li3ps4",
                "name": "Li3PS4",
                "description": "Lithium thiophosphate superionic conductor",
                "n_structures": 10000,
                "elements": ["Li", "P", "S"],
            },
            {
                "id": "llzo",
                "name": "LLZO",
                "description": "Li7La3Zr2O12 garnet electrolyte",
                "n_structures": 5000,
                "elements": ["Li", "La", "Zr", "O"],
            },
        ]
    }
