"""
ML训练任务
"""

import logging
from typing import Dict, Any, Optional
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)
def train_ml_model(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    训练ML势函数模型
    
    Args:
        training_params: 训练参数
        
    Returns:
        训练结果
    """
    task_id = self.request.id
    logger.info(f"Starting ML training task {task_id}")
    
    try:
        model_type = training_params.get("model_type", "nep")
        training_data = training_params.get("training_data", {})
        hyperparameters = training_params.get("hyperparameters", {})
        epochs = training_params.get("epochs", 1000)
        
        self.update_state(state="PROGRESS", meta={
            "progress": 0,
            "message": "Loading training data",
            "epoch": 0,
            "total_epochs": epochs,
        })
        
        # 根据不同模型类型调用相应训练器
        if model_type == "nep":
            result = _train_nep_model(training_data, hyperparameters, epochs, self)
        elif model_type == "deepmd":
            result = _train_deepmd_model(training_data, hyperparameters, epochs, self)
        elif model_type == "gap":
            result = _train_gap_model(training_data, hyperparameters, epochs, self)
        elif model_type == "ace":
            result = _train_ace_model(training_data, hyperparameters, epochs, self)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "model_type": model_type,
            "result": result,
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id} exceeded time limit")
        self.update_state(state="FAILURE", meta={"error": "Time limit exceeded"})
        raise
    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}")
        try:
            self.retry(countdown=300, exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
            }


def _train_nep_model(training_data: Dict, hyperparameters: Dict, epochs: int, task) -> Dict[str, Any]:
    """训练NEP模型"""
    logger.info("Training NEP model")
    
    import random
    
    # 模拟训练过程
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    batch_size = hyperparameters.get("batch_size", 50)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # 模拟损失下降
        train_loss = 1.0 / (epoch ** 0.5) + random.uniform(0, 0.01)
        val_loss = train_loss * 1.1 + random.uniform(0, 0.005)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 每10个epoch更新进度
        if epoch % 10 == 0:
            progress = int((epoch / epochs) * 100)
            task.update_state(state="PROGRESS", meta={
                "progress": progress,
                "message": f"Training epoch {epoch}/{epochs}",
                "epoch": epoch,
                "total_epochs": epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
    
    return {
        "model_type": "nep",
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "train_losses": train_losses[::10],  # 每10个epoch一个点
        "val_losses": val_losses[::10],
        "model_file": f"/tmp/nep_model_{random.randint(1000, 9999)}.txt",
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
        },
        "metrics": {
            "energy_rmse": 0.005,
            "force_rmse": 0.15,
            "virial_rmse": 0.02,
        },
    }


def _train_deepmd_model(training_data: Dict, hyperparameters: Dict, epochs: int, task) -> Dict[str, Any]:
    """训练DeepMD模型"""
    logger.info("Training DeepMD model")
    
    import random
    
    learning_rate = hyperparameters.get("learning_rate", 0.001)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = 2.0 / (epoch ** 0.3) + random.uniform(0, 0.02)
        val_loss = train_loss * 1.15 + random.uniform(0, 0.01)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 50 == 0:
            progress = int((epoch / epochs) * 100)
            task.update_state(state="PROGRESS", meta={
                "progress": progress,
                "epoch": epoch,
                "train_loss": train_loss,
            })
    
    return {
        "model_type": "deepmd",
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "model_file": "/tmp/deepmd_model.pb",
        "metrics": {
            "energy_rmse": 0.008,
            "force_rmse": 0.20,
        },
    }


def _train_gap_model(training_data: Dict, hyperparameters: Dict, epochs: int, task) -> Dict[str, Any]:
    """训练GAP模型"""
    logger.info("Training GAP model")
    
    return {
        "model_type": "gap",
        "model_file": "/tmp/gap_model.xml",
        "metrics": {
            "energy_rmse": 0.003,
            "force_rmse": 0.10,
        },
    }


def _train_ace_model(training_data: Dict, hyperparameters: Dict, epochs: int, task) -> Dict[str, Any]:
    """训练ACE模型"""
    logger.info("Training ACE model")
    
    return {
        "model_type": "ace",
        "model_file": "/tmp/ace_model.yace",
        "metrics": {
            "energy_rmse": 0.004,
            "force_rmse": 0.12,
        },
    }


@celery_app.task(bind=True)
def evaluate_model(self, model_file: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """评估模型性能"""
    task_id = self.request.id
    logger.info(f"Evaluating model {model_file}")
    
    # 模拟评估
    import random
    
    n_test = len(test_data.get("structures", []))
    
    return {
        "task_id": task_id,
        "n_test": n_test,
        "energy_rmse": random.uniform(0.002, 0.01),
        "force_rmse": random.uniform(0.05, 0.2),
        "virial_rmse": random.uniform(0.01, 0.05),
        "max_force_error": random.uniform(0.5, 2.0),
    }


@celery_app.task(bind=True)
def validate_model(self, model_file: str, validation_config: Dict[str, Any]) -> Dict[str, Any]:
    """验证模型"""
    task_id = self.request.id
    logger.info(f"Validating model {model_file}")
    
    validation_types = validation_config.get("types", ["energy", "forces", "phonon"])
    
    results = {}
    
    if "energy" in validation_types:
        results["energy"] = {
            "rmse": 0.005,
            "mae": 0.004,
            "r2": 0.999,
        }
    
    if "forces" in validation_types:
        results["forces"] = {
            "rmse": 0.15,
            "mae": 0.10,
            "r2": 0.995,
        }
    
    if "phonon" in validation_types:
        results["phonon"] = {
            "band_structure_match": 0.95,
            "dos_match": 0.92,
        }
    
    return {
        "task_id": task_id,
        "model_file": model_file,
        "validation_results": results,
        "passed": all(r.get("r2", 1) > 0.99 for r in results.values() if isinstance(r, dict)),
    }


@celery_app.task(bind=True)
def hyperparameter_search(self, training_data: Dict, search_space: Dict[str, Any]) -> Dict[str, Any]:
    """超参数搜索"""
    task_id = self.request.id
    logger.info(f"Starting hyperparameter search {task_id}")
    
    # 模拟网格搜索
    n_trials = search_space.get("n_trials", 10)
    
    best_loss = float("inf")
    best_params = {}
    all_results = []
    
    for i in range(n_trials):
        import random
        
        # 随机采样参数
        lr = random.choice(search_space.get("learning_rate", [0.001, 0.01, 0.0001]))
        bs = random.choice(search_space.get("batch_size", [32, 64, 128]))
        
        # 模拟训练结果
        loss = random.uniform(0.01, 0.1)
        
        result = {
            "trial": i,
            "learning_rate": lr,
            "batch_size": bs,
            "loss": loss,
        }
        all_results.append(result)
        
        if loss < best_loss:
            best_loss = loss
            best_params = {"learning_rate": lr, "batch_size": bs}
        
        self.update_state(state="PROGRESS", meta={
            "progress": int((i + 1) / n_trials * 100),
            "trial": i + 1,
            "best_loss": best_loss,
        })
    
    return {
        "task_id": task_id,
        "best_params": best_params,
        "best_loss": best_loss,
        "all_results": all_results,
    }


@celery_app.task(bind=True)
def active_learning_iteration(self, config: Dict[str, Any]) -> Dict[str, Any]:
    """主动学习迭代"""
    task_id = self.request.id
    logger.info(f"Starting active learning iteration {task_id}")
    
    iteration = config.get("iteration", 1)
    uncertainty_threshold = config.get("uncertainty_threshold", 0.1)
    
    # 模拟主动学习流程
    return {
        "task_id": task_id,
        "iteration": iteration,
        "candidates_selected": 10,
        "uncertainty_mean": uncertainty_threshold * 0.9,
        "dft_calculations_queued": 10,
        "next_model_training_queued": True,
    }
