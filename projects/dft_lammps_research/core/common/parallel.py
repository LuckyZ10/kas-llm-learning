#!/usr/bin/env python3
"""
parallel_optimizer.py
====================
并行优化模块 - 支持DFT、ML和MD的并行化

功能：
1. DFT计算并行化（多个结构同时提交）
2. ML训练分布式支持
3. MD轨迹分析多进程加速
4. 动态负载均衡
5. 结果聚合和错误处理

作者: HPC & Performance Optimization Expert
日期: 2026-03-09
"""

import os
import sys
import json
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Pool, Manager, Queue as MPQueue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Iterator
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import queue
import threading
import functools
import traceback
from abc import ABC, abstractmethod
import hashlib

import numpy as np

# 尝试导入分布式训练库
try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from mpi4py import MPI
    MPI4PY_AVAILABLE = True
except ImportError:
    MPI4PY_AVAILABLE = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class ParallelConfig:
    """并行计算配置"""
    # 并行度
    num_workers: int = -1  # -1 表示使用所有可用核心
    max_concurrent_jobs: int = 10
    
    # 执行模式
    mode: str = "process"  # process, thread, mpi, distributed
    
    # 批处理
    batch_size: int = 1
    chunk_size: int = 10
    
    # 容错
    retry_attempts: int = 3
    retry_delay: float = 1.0
    fail_fast: bool = False
    
    # 进度跟踪
    progress_interval: int = 10
    save_intermediate: bool = True
    
    # 资源限制
    memory_limit_gb: Optional[float] = None
    time_limit_seconds: Optional[float] = None
    
    def __post_init__(self):
        if self.num_workers == -1:
            self.num_workers = min(os.cpu_count() or 1, 32)


@dataclass
class TaskResult:
    """任务执行结果"""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    retries: int = 0
    worker_id: int = 0
    memory_used_mb: float = 0.0


@dataclass
class BatchStats:
    """批处理统计"""
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    total_time: float = 0.0
    avg_time_per_task: float = 0.0
    throughput_per_second: float = 0.0
    memory_peak_gb: float = 0.0


# =============================================================================
# 基础并行执行器
# =============================================================================

class ParallelExecutor(ABC):
    """并行执行器基类"""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self._stats = BatchStats()
    
    @abstractmethod
    def map(self, func: Callable, items: List[Any]) -> Iterator[TaskResult]:
        """并行执行函数"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """关闭执行器"""
        pass
    
    def get_stats(self) -> BatchStats:
        """获取执行统计"""
        return self._stats


class ProcessPoolExecutor(ParallelExecutor):
    """进程池执行器"""
    
    def __init__(self, config: ParallelConfig):
        super().__init__(config)
        self._executor = ProcessPoolExecutor(
            max_workers=config.num_workers
        )
    
    def map(self, func: Callable, items: List[Any]) -> Iterator[TaskResult]:
        """使用进程池并行执行"""
        start_time = time.time()
        self._stats.total_tasks = len(items)
        
        # 包装函数以捕获执行信息
        wrapped_func = functools.partial(self._execute_with_retry, func, self.config)
        
        futures = {
            self._executor.submit(wrapped_func, i, item): (i, item) 
            for i, item in enumerate(items)
        }
        
        completed = 0
        for future in as_completed(futures):
            task_id, item = futures[future]
            try:
                result = future.result()
                completed += 1
                if completed % self.config.progress_interval == 0:
                    logger.info(f"Progress: {completed}/{len(items)} tasks completed")
                yield result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                yield TaskResult(
                    task_id=str(task_id),
                    success=False,
                    error=str(e)
                )
        
        self._stats.completed = completed
        self._stats.total_time = time.time() - start_time
        if completed > 0:
            self._stats.avg_time_per_task = self._stats.total_time / completed
            self._stats.throughput_per_second = completed / self._stats.total_time
    
    @staticmethod
    def _execute_with_retry(func: Callable, config: ParallelConfig, 
                           task_id: int, item: Any) -> TaskResult:
        """带重试机制的任务执行"""
        start_time = time.time()
        last_error = None
        
        for attempt in range(config.retry_attempts):
            try:
                result = func(item)
                return TaskResult(
                    task_id=str(task_id),
                    success=True,
                    result=result,
                    execution_time=time.time() - start_time,
                    retries=attempt
                )
            except Exception as e:
                last_error = e
                if attempt < config.retry_attempts - 1:
                    time.sleep(config.retry_delay * (attempt + 1))
        
        return TaskResult(
            task_id=str(task_id),
            success=False,
            error=f"{str(last_error)}\n{traceback.format_exc()}",
            execution_time=time.time() - start_time,
            retries=config.retry_attempts
        )
    
    def shutdown(self):
        """关闭进程池"""
        self._executor.shutdown(wait=True)


class ThreadPoolExecutor(ParallelExecutor):
    """线程池执行器（适用于I/O密集型任务）"""
    
    def __init__(self, config: ParallelConfig):
        super().__init__(config)
        self._executor = ThreadPoolExecutor(
            max_workers=config.num_workers
        )
    
    def map(self, func: Callable, items: List[Any]) -> Iterator[TaskResult]:
        """使用线程池并行执行"""
        start_time = time.time()
        self._stats.total_tasks = len(items)
        
        futures = {
            self._executor.submit(func, item): (i, item) 
            for i, item in enumerate(items)
        }
        
        completed = 0
        for future in as_completed(futures):
            task_id, item = futures[future]
            task_start = time.time()
            try:
                result = future.result()
                completed += 1
                yield TaskResult(
                    task_id=str(task_id),
                    success=True,
                    result=result,
                    execution_time=time.time() - task_start
                )
            except Exception as e:
                yield TaskResult(
                    task_id=str(task_id),
                    success=False,
                    error=str(e)
                )
        
        self._stats.completed = completed
        self._stats.total_time = time.time() - start_time
    
    def shutdown(self):
        """关闭线程池"""
        self._executor.shutdown(wait=True)


class MPIExecutor(ParallelExecutor):
    """MPI分布式执行器"""
    
    def __init__(self, config: ParallelConfig):
        super().__init__(config)
        if not MPI4PY_AVAILABLE:
            raise ImportError("mpi4py is required for MPI execution")
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def map(self, func: Callable, items: List[Any]) -> Iterator[TaskResult]:
        """使用MPI并行执行"""
        # 仅rank 0分发任务
        if self.rank == 0:
            self._stats.total_tasks = len(items)
            # 分块
            chunks = [items[i::self.size] for i in range(self.size)]
        else:
            chunks = None
        
        # 广播分块
        local_chunk = self.comm.scatter(chunks, root=0)
        
        # 本地执行
        results = []
        for i, item in enumerate(local_chunk):
            task_id = f"{self.rank}_{i}"
            start_time = time.time()
            try:
                result = func(item)
                results.append(TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=time.time() - start_time,
                    worker_id=self.rank
                ))
            except Exception as e:
                results.append(TaskResult(
                    task_id=task_id,
                    success=False,
                    error=str(e),
                    worker_id=self.rank
                ))
        
        # 收集结果
        all_results = self.comm.gather(results, root=0)
        
        if self.rank == 0:
            for worker_results in all_results:
                for result in worker_results:
                    yield result
    
    def shutdown(self):
        """MPI不需要显式关闭"""
        pass


def create_executor(config: ParallelConfig) -> ParallelExecutor:
    """创建并行执行器"""
    if config.mode == "mpi" and MPI4PY_AVAILABLE:
        return MPIExecutor(config)
    elif config.mode == "thread":
        return ThreadPoolExecutor(config)
    else:
        return ProcessPoolExecutor(config)


# =============================================================================
# DFT计算并行化
# =============================================================================

class DFTBatchProcessor:
    """
    DFT批量处理器
    
    支持VASP、Quantum ESPRESSO等DFT代码的并行执行
    """
    
    def __init__(self, config: ParallelConfig, scheduler=None):
        self.config = config
        self.scheduler = scheduler
        self._executor = create_executor(config)
    
    def process_structures(
        self,
        structures: List[Union[str, Dict]],
        calc_type: str = "relaxation",
        calculator: str = "vasp",
        template_dir: Optional[str] = None,
        output_dir: str = "./dft_results"
    ) -> List[TaskResult]:
        """
        并行处理多个结构的DFT计算
        
        Args:
            structures: 结构列表（文件路径或结构数据字典）
            calc_type: 计算类型 (relaxation, static, md, phonon)
            calculator: 计算程序 (vasp, espresso, abacus)
            template_dir: 输入模板目录
            output_dir: 输出目录
        
        Returns:
            任务结果列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备任务
        tasks = []
        for i, structure in enumerate(structures):
            task = {
                "task_id": f"{calc_type}_{i}",
                "structure": structure,
                "calc_type": calc_type,
                "calculator": calculator,
                "template_dir": template_dir,
                "output_dir": output_dir / f"calc_{i:04d}",
                "index": i
            }
            tasks.append(task)
        
        logger.info(f"Processing {len(tasks)} DFT calculations with {self.config.num_workers} workers")
        
        # 并行执行
        results = list(self._executor.map(self._run_single_calculation, tasks))
        
        # 汇总结果
        self._summarize_results(results, output_dir)
        
        return results
    
    def _run_single_calculation(self, task: Dict) -> Dict:
        """执行单个DFT计算"""
        task_id = task["task_id"]
        output_dir = Path(task["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        calculator = task["calculator"]
        calc_type = task["calc_type"]
        
        logger.debug(f"Starting {calculator} calculation: {task_id}")
        
        try:
            # 准备输入文件
            self._prepare_input_files(task, output_dir)
            
            # 执行计算
            if calculator == "vasp":
                result = self._run_vasp(output_dir, calc_type)
            elif calculator == "espresso":
                result = self._run_espresso(output_dir, calc_type)
            elif calculator == "abacus":
                result = self._run_abacus(output_dir, calc_type)
            else:
                raise ValueError(f"Unsupported calculator: {calculator}")
            
            # 解析结果
            result["task_id"] = task_id
            result["output_dir"] = str(output_dir)
            
            return result
            
        except Exception as e:
            logger.error(f"Calculation {task_id} failed: {e}")
            return {
                "task_id": task_id,
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _prepare_input_files(self, task: Dict, output_dir: Path):
        """准备DFT输入文件"""
        structure = task["structure"]
        template_dir = task.get("template_dir")
        
        # 写入结构文件
        if isinstance(structure, str) and Path(structure).exists():
            # 复制结构文件
            import shutil
            shutil.copy(structure, output_dir / "POSCAR")
        elif isinstance(structure, dict):
            # 从字典生成结构文件
            self._write_structure_from_dict(structure, output_dir)
        
        # 复制模板文件
        if template_dir and Path(template_dir).exists():
            for f in Path(template_dir).glob("*"):
                if f.is_file():
                    import shutil
                    shutil.copy(f, output_dir / f.name)
    
    def _write_structure_from_dict(self, structure: Dict, output_dir: Path):
        """从字典写入结构"""
        # 这里可以根据实际格式实现（POSCAR, cif等）
        # 简化版本：假设为POSCAR格式
        poscar_lines = structure.get("poscar_lines", [])
        with open(output_dir / "POSCAR", "w") as f:
            f.write("\n".join(poscar_lines))
    
    def _run_vasp(self, output_dir: Path, calc_type: str) -> Dict:
        """运行VASP计算"""
        import subprocess
        
        # 根据计算类型选择VASP可执行文件
        vasp_exec = "vasp_std"
        if calc_type == "gamma":
            vasp_exec = "vasp_gam"
        elif calc_type == "ncl":
            vasp_exec = "vasp_ncl"
        
        cmd = f"cd {output_dir} && mpirun -np {self.config.num_workers} {vasp_exec}"
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.config.time_limit_seconds
        )
        
        # 检查是否成功
        outcar_path = output_dir / "OUTCAR"
        success = outcar_path.exists() and result.returncode == 0
        
        return {
            "success": success,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,
            "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
        }
    
    def _run_espresso(self, output_dir: Path, calc_type: str) -> Dict:
        """运行Quantum ESPRESSO计算"""
        import subprocess
        
        cmd = f"cd {output_dir} && mpirun -np {self.config.num_workers} pw.x -in pw.in"
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.config.time_limit_seconds
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode
        }
    
    def _run_abacus(self, output_dir: Path, calc_type: str) -> Dict:
        """运行ABACUS计算"""
        import subprocess
        
        cmd = f"cd {output_dir} && mpirun -np {self.config.num_workers} abacus"
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=self.config.time_limit_seconds
        )
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode
        }
    
    def _summarize_results(self, results: List[TaskResult], output_dir: Path):
        """汇总计算结果"""
        success_count = sum(1 for r in results if r.success)
        failed_count = len(results) - success_count
        
        summary = {
            "total": len(results),
            "success": success_count,
            "failed": failed_count,
            "success_rate": success_count / len(results) if results else 0,
            "results": [asdict(r) for r in results]
        }
        
        with open(output_dir / "dft_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"DFT batch complete: {success_count}/{len(results)} successful")
    
    def shutdown(self):
        """关闭处理器"""
        self._executor.shutdown()


# =============================================================================
# ML训练分布式支持
# =============================================================================

@dataclass
class DistributedTrainingConfig:
    """分布式训练配置"""
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"
    use_amp: bool = True  # 自动混合精度
    gradient_accumulation_steps: int = 1
    
    # 数据并行
    num_workers_dataloader: int = 4
    pin_memory: bool = True
    
    # 模型并行
    model_parallel_size: int = 1
    pipeline_parallel_size: int = 1


class DistributedTrainer:
    """
    分布式训练器
    
    支持PyTorch DDP和DeepSpeed等分布式训练框架
    """
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self._setup_distributed()
    
    def _setup_distributed(self):
        """初始化分布式环境"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, running in single-node mode")
            return
        
        if self.config.world_size > 1:
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = self.config.master_port
            
            if not dist.is_initialized():
                dist.init_process_group(
                    backend=self.config.backend,
                    rank=self.config.rank,
                    world_size=self.config.world_size
                )
            
            torch.cuda.set_device(self.config.local_rank)
            logger.info(f"Initialized rank {self.config.rank}/{self.config.world_size}")
    
    def prepare_model(self, model: "torch.nn.Module") -> "torch.nn.Module":
        """准备分布式模型"""
        if not TORCH_AVAILABLE or self.config.world_size == 1:
            return model
        
        device = torch.device(f"cuda:{self.config.local_rank}")
        model = model.to(device)
        
        # 使用DDP包装
        model = DDP(
            model,
            device_ids=[self.config.local_rank],
            output_device=self.config.local_rank,
            find_unused_parameters=False
        )
        
        return model
    
    def prepare_dataloader(
        self, 
        dataset: "torch.utils.data.Dataset",
        batch_size: int,
        shuffle: bool = True
    ) -> "torch.utils.data.DataLoader":
        """准备分布式数据加载器"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        sampler = None
        if self.config.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers_dataloader,
            pin_memory=self.config.pin_memory,
            drop_last=True
        )
        
        return loader
    
    def train_step(self, model, batch, optimizer, scaler=None):
        """单步训练"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        model.train()
        
        # 前向传播
        with torch.cuda.amp.autocast(enabled=self.config.use_amp and scaler is not None):
            loss = model(batch)
            if isinstance(loss, dict):
                loss = loss["loss"]
            loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度更新
        if self.should_step():
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        return loss.item()
    
    def should_step(self) -> bool:
        """是否应该执行优化器步骤"""
        return True  # 简化版，实际需要考虑梯度累积
    
    def all_reduce(self, tensor: "torch.Tensor", op: str = "sum"):
        """跨进程聚合"""
        if not TORCH_AVAILABLE or self.config.world_size == 1:
            return tensor
        
        if op == "sum":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self.config.world_size
        elif op == "max":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        
        return tensor
    
    def barrier(self):
        """进程同步"""
        if TORCH_AVAILABLE and self.config.world_size > 1:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """是否为主进程"""
        return self.config.rank == 0
    
    def save_checkpoint(self, model, optimizer, epoch: int, path: str):
        """保存检查点（仅主进程）"""
        if not self.is_main_process():
            return
        
        if not TORCH_AVAILABLE:
            return
        
        # 获取原始模型（从DDP解包）
        if isinstance(model, DDP):
            model = model.module
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(self.config)
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")
    
    def cleanup(self):
        """清理分布式环境"""
        if TORCH_AVAILABLE and dist.is_initialized():
            dist.destroy_process_group()


# =============================================================================
# MD轨迹分析多进程加速
# =============================================================================

class MDTrajectoryAnalyzer:
    """
    MD轨迹分析器
    
    支持多进程并行分析大规模轨迹文件
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self._executor = create_executor(config)
    
    def analyze_trajectory(
        self,
        trajectory_file: str,
        analysis_funcs: Dict[str, Callable],
        frame_range: Optional[Tuple[int, int, int]] = None,
        chunk_size: int = 100
    ) -> Dict[str, Any]:
        """
        并行分析MD轨迹
        
        Args:
            trajectory_file: 轨迹文件路径（LAMMPS dump, XYZ等）
            analysis_funcs: 分析函数字典
            frame_range: 帧范围 (start, stop, step)
            chunk_size: 每块处理的帧数
        
        Returns:
            分析结果字典
        """
        # 获取总帧数
        total_frames = self._count_frames(trajectory_file)
        
        if frame_range is None:
            frame_range = (0, total_frames, 1)
        
        start, stop, step = frame_range
        frame_indices = list(range(start, min(stop, total_frames), step))
        
        logger.info(f"Analyzing {len(frame_indices)} frames with {len(analysis_funcs)} metrics")
        
        # 分块处理
        chunks = [
            frame_indices[i:i + chunk_size] 
            for i in range(0, len(frame_indices), chunk_size)
        ]
        
        # 准备任务
        tasks = [
            {
                "trajectory_file": trajectory_file,
                "frame_indices": chunk,
                "analysis_funcs": analysis_funcs,
                "chunk_id": i
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # 并行执行
        results = list(self._executor.map(self._analyze_chunk, tasks))
        
        # 合并结果
        merged_results = self._merge_results(results, analysis_funcs.keys())
        
        return merged_results
    
    def _count_frames(self, trajectory_file: str) -> int:
        """统计轨迹文件帧数"""
        # 简化实现：计算文件中的"ITEM: TIMESTEP"数量
        count = 0
        with open(trajectory_file, 'r') as f:
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    count += 1
        return count
    
    def _analyze_chunk(self, task: Dict) -> Dict:
        """分析单个块"""
        trajectory_file = task["trajectory_file"]
        frame_indices = task["frame_indices"]
        analysis_funcs = task["analysis_funcs"]
        chunk_id = task["chunk_id"]
        
        results = {name: [] for name in analysis_funcs.keys()}
        
        try:
            # 读取指定帧
            frames = self._read_frames(trajectory_file, frame_indices)
            
            for frame_idx, frame_data in zip(frame_indices, frames):
                for name, func in analysis_funcs.items():
                    try:
                        value = func(frame_data)
                        results[name].append({"frame": frame_idx, "value": value})
                    except Exception as e:
                        logger.warning(f"Analysis {name} failed for frame {frame_idx}: {e}")
            
            return {
                "chunk_id": chunk_id,
                "success": True,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Chunk {chunk_id} analysis failed: {e}")
            return {
                "chunk_id": chunk_id,
                "success": False,
                "error": str(e)
            }
    
    def _read_frames(self, trajectory_file: str, frame_indices: List[int]) -> List[Dict]:
        """读取指定帧数据"""
        frames = []
        current_frame = -1
        current_data = []
        reading_frame = False
        
        with open(trajectory_file, 'r') as f:
            for line in f:
                if "ITEM: TIMESTEP" in line:
                    # 保存上一帧
                    if current_frame in frame_indices and current_data:
                        frames.append(self._parse_frame(current_data))
                    
                    current_frame += 1
                    current_data = [line]
                    reading_frame = current_frame in frame_indices
                    
                    # 如果已经超过需要的帧，提前退出
                    if current_frame > max(frame_indices):
                        break
                
                elif reading_frame:
                    current_data.append(line)
            
            # 处理最后一帧
            if current_frame in frame_indices and current_data:
                frames.append(self._parse_frame(current_data))
        
        return frames
    
    def _parse_frame(self, lines: List[str]) -> Dict:
        """解析单帧数据（LAMMPS dump格式）"""
        data = {
            "timestep": None,
            "num_atoms": 0,
            "box": {},
            "atoms": []
        }
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if line == "ITEM: TIMESTEP":
                data["timestep"] = int(lines[i + 1].strip())
                i += 2
            elif line == "ITEM: NUMBER OF ATOMS":
                data["num_atoms"] = int(lines[i + 1].strip())
                i += 2
            elif line.startswith("ITEM: BOX BOUNDS"):
                # 读取盒子边界
                bounds = []
                for j in range(3):
                    parts = lines[i + 1 + j].split()
                    bounds.append([float(parts[0]), float(parts[1])])
                data["box"]["bounds"] = bounds
                i += 4
            elif line.startswith("ITEM: ATOMS"):
                # 读取原子数据
                header = line.replace("ITEM: ATOMS ", "").split()
                for j in range(data["num_atoms"]):
                    parts = lines[i + 1 + j].split()
                    atom = {h: float(v) if self._is_number(v) else v 
                           for h, v in zip(header, parts)}
                    data["atoms"].append(atom)
                i += 1 + data["num_atoms"]
            else:
                i += 1
        
        return data
    
    def _is_number(self, s: str) -> bool:
        """检查字符串是否为数字"""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _merge_results(self, chunk_results: List[Dict], analysis_names: List[str]) -> Dict:
        """合并各块结果"""
        merged = {name: [] for name in analysis_names}
        
        for chunk_result in chunk_results:
            if not chunk_result.get("success", False):
                continue
            
            for name in analysis_names:
                if name in chunk_result.get("results", {}):
                    merged[name].extend(chunk_result["results"][name])
        
        # 按帧排序
        for name in analysis_names:
            merged[name].sort(key=lambda x: x["frame"])
        
        return merged
    
    def compute_rdf(
        self,
        trajectory_file: str,
        bin_width: float = 0.1,
        r_max: Optional[float] = None,
        pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        计算径向分布函数（并行版本）
        
        Args:
            trajectory_file: 轨迹文件
            bin_width: 径向距离分辨率
            r_max: 最大距离
            pairs: 原子对列表
        
        Returns:
            RDF数据
        """
        # 首先确定系统参数
        first_frame = self._read_frames(trajectory_file, [0])[0]
        box = first_frame["box"]["bounds"]
        
        if r_max is None:
            # 取盒子最小边的一半
            box_sizes = [b[1] - b[0] for b in box]
            r_max = min(box_sizes) / 2
        
        num_bins = int(r_max / bin_width)
        
        # 定义RDF计算函数
        def calc_frame_rdf(frame_data):
            atoms = frame_data["atoms"]
            positions = np.array([[a.get("x", 0), a.get("y", 0), a.get("z", 0)] for a in atoms])
            types = [a.get("type", a.get("element", "X")) for a in atoms]
            
            # 计算距离
            from scipy.spatial.distance import cdist
            distances = cdist(positions, positions)
            
            # 计算直方图
            hist, bin_edges = np.histogram(
                distances[distances > 0],
                bins=num_bins,
                range=(0, r_max)
            )
            
            return {"hist": hist, "edges": bin_edges}
        
        # 并行分析
        results = self.analyze_trajectory(
            trajectory_file,
            {"rdf": calc_frame_rdf},
            chunk_size=50
        )
        
        # 合并RDF
        total_hist = np.zeros(num_bins)
        for r in results["rdf"]:
            total_hist += r["value"]["hist"]
        
        # 归一化
        bin_edges = results["rdf"][0]["value"]["edges"]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 体积归一化
        volume = np.prod([b[1] - b[0] for b in box])
        num_atoms = len(first_frame["atoms"])
        density = num_atoms / volume
        
        shell_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
        normalization = density * shell_volumes * num_atoms
        
        g_r = total_hist / normalization / len(results["rdf"])
        
        return {
            "r": bin_centers,
            "g_r": g_r,
            "n_frames": len(results["rdf"])
        }
    
    def shutdown(self):
        """关闭分析器"""
        self._executor.shutdown()


# =============================================================================
# 动态负载均衡器
# =============================================================================

class DynamicLoadBalancer:
    """
    动态负载均衡器
    
    根据任务执行时间动态调整任务分配
    """
    
    def __init__(self, num_workers: int, adapt_interval: int = 10):
        self.num_workers = num_workers
        self.adapt_interval = adapt_interval
        self._task_times: Dict[str, List[float]] = defaultdict(list)
        self._task_count = 0
    
    def estimate_time(self, task_type: str) -> float:
        """估计任务类型执行时间"""
        times = self._task_times.get(task_type, [])
        if not times:
            return 1.0
        return np.median(times[-10:])  # 使用中位数，对异常值鲁棒
    
    def record_time(self, task_type: str, execution_time: float):
        """记录任务执行时间"""
        self._task_times[task_type].append(execution_time)
    
    def create_batches(
        self, 
        tasks: List[Tuple[str, Any]], 
        target_batch_time: float = 60.0
    ) -> List[List[Tuple[str, Any]]]:
        """
        创建平衡的任务批次
        
        Args:
            tasks: (task_type, task_data) 列表
            target_batch_time: 目标批次执行时间（秒）
        
        Returns:
            批次列表
        """
        # 按估计时间排序任务（最长任务优先）
        task_with_estimates = [
            (task_type, task_data, self.estimate_time(task_type))
            for task_type, task_data in tasks
        ]
        task_with_estimates.sort(key=lambda x: x[2], reverse=True)
        
        # 使用最长处理时间优先（LPT）算法
        batches: List[List[Tuple[str, Any]]] = [[] for _ in range(self.num_workers)]
        batch_times = [0.0] * self.num_workers
        
        for task_type, task_data, est_time in task_with_estimates:
            # 分配给当前负载最小的worker
            min_worker = np.argmin(batch_times)
            batches[min_worker].append((task_type, task_data))
            batch_times[min_worker] += est_time
        
        # 合并小批次
        merged_batches = []
        current_batch = []
        current_time = 0.0
        
        for worker_tasks in batches:
            for task_type, task_data in worker_tasks:
                est_time = self.estimate_time(task_type)
                
                if current_time + est_time > target_batch_time and current_batch:
                    merged_batches.append(current_batch)
                    current_batch = [(task_type, task_data)]
                    current_time = est_time
                else:
                    current_batch.append((task_type, task_data))
                    current_time += est_time
        
        if current_batch:
            merged_batches.append(current_batch)
        
        return merged_batches


# =============================================================================
# 高级API
# =============================================================================

def parallel_map(
    func: Callable[[Any], Any],
    items: List[Any],
    num_workers: int = -1,
    mode: str = "process",
    retry_attempts: int = 3,
    progress_bar: bool = True
) -> List[Any]:
    """
    并行map函数
    
    Args:
        func: 要执行的函数
        items: 输入列表
        num_workers: worker数量
        mode: 执行模式 (process, thread, mpi)
        retry_attempts: 重试次数
        progress_bar: 是否显示进度
    
    Returns:
        结果列表
    """
    config = ParallelConfig(
        num_workers=num_workers,
        mode=mode,
        retry_attempts=retry_attempts
    )
    
    executor = create_executor(config)
    
    try:
        results = list(executor.map(func, items))
        return [r.result if r.success else None for r in results]
    finally:
        executor.shutdown()


def parallel_dft_calculations(
    structures: List[Union[str, Dict]],
    calc_type: str = "relaxation",
    calculator: str = "vasp",
    num_workers: int = 4,
    output_dir: str = "./dft_results",
    **kwargs
) -> List[TaskResult]:
    """
    并行DFT计算便捷函数
    
    Args:
        structures: 结构列表
        calc_type: 计算类型
        calculator: 计算程序
        num_workers: 并行worker数
        output_dir: 输出目录
        **kwargs: 其他配置
    
    Returns:
        任务结果列表
    """
    config = ParallelConfig(num_workers=num_workers, **kwargs)
    processor = DFTBatchProcessor(config)
    
    try:
        results = processor.process_structures(
            structures=structures,
            calc_type=calc_type,
            calculator=calculator,
            output_dir=output_dir
        )
        return results
    finally:
        processor.shutdown()


def parallel_md_analysis(
    trajectory_file: str,
    analysis_funcs: Dict[str, Callable],
    num_workers: int = 4,
    chunk_size: int = 100,
    **kwargs
) -> Dict[str, Any]:
    """
    并行MD轨迹分析便捷函数
    
    Args:
        trajectory_file: 轨迹文件路径
        analysis_funcs: 分析函数字典
        num_workers: 并行worker数
        chunk_size: 每块处理的帧数
        **kwargs: 其他配置
    
    Returns:
        分析结果
    """
    config = ParallelConfig(num_workers=num_workers, **kwargs)
    analyzer = MDTrajectoryAnalyzer(config)
    
    try:
        results = analyzer.analyze_trajectory(
            trajectory_file=trajectory_file,
            analysis_funcs=analysis_funcs,
            chunk_size=chunk_size
        )
        return results
    finally:
        analyzer.shutdown()


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 示例1: 简单的并行map
    def square(x):
        return x ** 2
    
    numbers = list(range(100))
    # results = parallel_map(square, numbers, num_workers=4)
    # print(f"Processed {len(results)} items")
    
    # 示例2: DFT批量计算
    structures = ["POSCAR_1", "POSCAR_2", "POSCAR_3"]
    # results = parallel_dft_calculations(
    #     structures=structures,
    #     calc_type="relaxation",
    #     calculator="vasp",
    #     num_workers=4
    # )
    
    # 示例3: MD轨迹分析
    def calc_energy(frame_data):
        # 简化的能量计算示例
        atoms = frame_data.get("atoms", [])
        return sum(a.get("c_pe", 0) for a in atoms) if atoms else 0
    
    def calc_temperature(frame_data):
        # 简化的温度计算示例
        atoms = frame_data.get("atoms", [])
        if not atoms:
            return 0
        ke = sum(a.get("c_ke", 0) for a in atoms)
        return ke / len(atoms) * 2 / 3  # 简化公式
    
    # results = parallel_md_analysis(
    #     trajectory_file="trajectory.dump",
    #     analysis_funcs={
    #         "energy": calc_energy,
    #         "temperature": calc_temperature
    #     },
    #     num_workers=4
    # )
    
    print("parallel_optimizer module loaded successfully")
