"""
筛选任务
"""

import logging
from typing import Dict, Any, List
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def run_screening(self, screening_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    运行高通量筛选任务
    
    Args:
        screening_params: 筛选参数
        
    Returns:
        筛选结果
    """
    task_id = self.request.id
    logger.info(f"Starting screening task {task_id}")
    
    try:
        dataset = screening_params.get("dataset", {})
        criteria = screening_params.get("criteria", [])
        method = screening_params.get("method", "multi_objective")
        top_k = screening_params.get("top_k", 10)
        
        self.update_state(state="PROGRESS", meta={
            "progress": 10,
            "message": "Loading dataset",
        })
        
        # 执行筛选
        if method == "multi_objective":
            result = _multi_objective_screening(dataset, criteria, top_k, self)
        elif method == "pareto":
            result = _pareto_screening(dataset, criteria, top_k, self)
        elif method == "clustering":
            result = _clustering_screening(dataset, criteria, top_k, self)
        else:
            result = _simple_screening(dataset, criteria, top_k, self)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "method": method,
            "result": result,
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Task {task_id} exceeded time limit")
        self.update_state(state="FAILURE", meta={"error": "Time limit exceeded"})
        raise
    except Exception as exc:
        logger.error(f"Task {task_id} failed: {exc}")
        try:
            self.retry(countdown=60, exc=exc)
        except self.MaxRetriesExceededError:
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(exc),
            }


def _multi_objective_screening(dataset: Dict, criteria: List[Dict], top_k: int, task) -> Dict[str, Any]:
    """多目标筛选"""
    logger.info("Running multi-objective screening")
    
    import random
    
    structures = dataset.get("structures", [])
    n_structures = len(structures) if structures else 100
    
    results = []
    
    for i in range(min(top_k * 3, n_structures)):
        scores = {}
        for criterion in criteria:
            prop_name = criterion.get("property_name", "score")
            weight = criterion.get("weight", 1.0)
            scores[prop_name] = random.uniform(0, 10) * weight
        
        # 计算加权总分
        total_score = sum(scores.values()) / len(scores) if scores else random.uniform(0, 10)
        
        results.append({
            "id": f"structure_{i}",
            "scores": scores,
            "total_score": total_score,
            "rank": i,
        })
    
    # 排序并选择top_k
    results.sort(key=lambda x: x["total_score"], reverse=True)
    
    task.update_state(state="PROGRESS", meta={
        "progress": 80,
        "message": "Sorting results",
    })
    
    return {
        "candidates": results[:top_k],
        "n_evaluated": n_structures,
        "method": "multi_objective",
        "criteria_used": [c.get("property_name") for c in criteria],
    }


def _pareto_screening(dataset: Dict, criteria: List[Dict], top_k: int, task) -> Dict[str, Any]:
    """Pareto前沿筛选"""
    logger.info("Running Pareto screening")
    
    import random
    
    # 模拟Pareto前沿
    pareto_front = []
    for i in range(min(top_k, 20)):
        pareto_front.append({
            "id": f"pareto_{i}",
            "objectives": {
                "property_1": random.uniform(5, 10),
                "property_2": random.uniform(5, 10),
            },
            "dominates": random.randint(0, 50),
        })
    
    task.update_state(state="PROGRESS", meta={
        "progress": 70,
        "message": "Computing Pareto front",
    })
    
    return {
        "pareto_front": pareto_front,
        "n_pareto": len(pareto_front),
        "method": "pareto",
    }


def _clustering_screening(dataset: Dict, criteria: List[Dict], top_k: int, task) -> Dict[str, Any]:
    """聚类筛选"""
    logger.info("Running clustering screening")
    
    import random
    
    n_clusters = min(5, top_k)
    
    clusters = []
    for i in range(n_clusters):
        cluster_size = random.randint(5, 20)
        representative = {
            "id": f"cluster_{i}_rep",
            "cluster_id": i,
            "cluster_size": cluster_size,
            "properties": {c.get("property_name", "score"): random.uniform(0, 10) for c in criteria},
        }
        clusters.append(representative)
    
    task.update_state(state="PROGRESS", meta={
        "progress": 75,
        "message": "Clustering structures",
    })
    
    return {
        "clusters": clusters,
        "n_clusters": n_clusters,
        "method": "clustering",
    }


def _simple_screening(dataset: Dict, criteria: List[Dict], top_k: int, task) -> Dict[str, Any]:
    """简单筛选"""
    logger.info("Running simple screening")
    
    import random
    
    results = []
    for i in range(top_k):
        results.append({
            "id": f"candidate_{i}",
            "score": random.uniform(7, 10),
            "properties": {c.get("property_name"): random.uniform(0, 10) for c in criteria},
        })
    
    return {
        "candidates": results,
        "n_evaluated": top_k * 10,
        "method": "simple",
    }


@celery_app.task(bind=True)
def batch_calculate_properties(self, structures: List[Dict], properties: List[str]) -> Dict[str, Any]:
    """批量计算性质"""
    task_id = self.request.id
    logger.info(f"Batch calculating properties for {len(structures)} structures")
    
    results = []
    
    for i, structure in enumerate(structures):
        # 更新进度
        if i % max(1, len(structures) // 10) == 0:
            progress = int((i / len(structures)) * 100)
            self.update_state(state="PROGRESS", meta={
                "progress": progress,
                "message": f"Calculated {i}/{len(structures)} structures",
            })
        
        # 模拟计算
        import random
        result = {
            "id": structure.get("id", f"struct_{i}"),
            "properties": {},
        }
        
        for prop in properties:
            result["properties"][prop] = random.uniform(-100, 100)
        
        results.append(result)
    
    return {
        "task_id": task_id,
        "n_calculated": len(structures),
        "results": results,
    }


@celery_app.task(bind=True)
def generate_candidates(self, template: Dict, variations: Dict[str, Any]) -> Dict[str, Any]:
    """生成候选结构"""
    task_id = self.request.id
    logger.info(f"Generating candidate structures {task_id}")
    
    import random
    
    n_candidates = variations.get("n_candidates", 100)
    strategy = variations.get("strategy", "random")
    
    candidates = []
    
    for i in range(n_candidates):
        candidate = template.copy()
        candidate["id"] = f"candidate_{i}"
        
        if strategy == "random":
            # 随机扰动
            candidate["perturbation"] = random.uniform(-0.1, 0.1)
        elif strategy == "substitution":
            # 元素替换
            candidate["substitution"] = random.choice(["Li", "Na", "K"])
        elif strategy == "strain":
            # 应变
            candidate["strain"] = random.uniform(-0.05, 0.05)
        
        candidates.append(candidate)
    
    return {
        "task_id": task_id,
        "n_generated": n_candidates,
        "strategy": strategy,
        "candidates": candidates[:50],  # 只返回部分
    }


@celery_app.task(bind=True)
def optimize_composition(self, target_property: str, constraints: Dict[str, Any]) -> Dict[str, Any]:
    """优化组分"""
    task_id = self.request.id
    logger.info(f"Optimizing composition for {target_property}")
    
    import random
    
    # 模拟优化过程
    best_composition = {
        "Li": random.uniform(0.3, 0.5),
        "P": random.uniform(0.2, 0.3),
        "S": random.uniform(0.3, 0.5),
    }
    
    return {
        "task_id": task_id,
        "best_composition": best_composition,
        "target_property": target_property,
        "predicted_value": random.uniform(8, 10),
        "optimization_steps": 50,
    }
