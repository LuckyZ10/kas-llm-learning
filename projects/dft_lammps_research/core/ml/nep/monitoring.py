"""
nep_training/monitoring.py
==========================
训练监控与可视化模块

包含:
- 实时训练监控
- 指标追踪
- TensorBoard集成
- Wandb集成
- WebSocket实时推送
"""

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
import queue

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """指标快照"""
    timestamp: float
    generation: int
    train_loss: float
    val_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch_time: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    custom_metrics: Dict[str, float] = field(default_factory=dict)


class MetricsTracker:
    """
    指标追踪器
    
    记录和追踪训练过程中的各种指标
    """
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.history: deque = deque(maxlen=max_history)
        self.best_metrics: Dict[str, Any] = {}
        self.start_time = time.time()
    
    def log(self, 
           generation: int,
           train_loss: float,
           val_loss: Optional[float] = None,
           learning_rate: Optional[float] = None,
           **kwargs):
        """记录指标"""
        snapshot = MetricsSnapshot(
            timestamp=time.time() - self.start_time,
            generation=generation,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            custom_metrics=kwargs
        )
        
        self.history.append(snapshot)
        
        # 更新最佳指标
        if val_loss is not None:
            if 'best_val_loss' not in self.best_metrics or val_loss < self.best_metrics['best_val_loss']:
                self.best_metrics['best_val_loss'] = val_loss
                self.best_metrics['best_generation'] = generation
    
    def get_latest(self, n: int = 1) -> List[MetricsSnapshot]:
        """获取最近的n个指标"""
        return list(self.history)[-n:]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        
        return {
            'current_generation': latest.generation,
            'current_train_loss': latest.train_loss,
            'current_val_loss': latest.val_loss,
            'best_metrics': self.best_metrics,
            'total_time': time.time() - self.start_time,
            'samples_per_second': len(self.history) / (time.time() - self.start_time) if self.history else 0
        }
    
    def export_csv(self, output_path: str):
        """导出为CSV"""
        import pandas as pd
        
        data = []
        for snap in self.history:
            row = {
                'timestamp': snap.timestamp,
                'generation': snap.generation,
                'train_loss': snap.train_loss,
                'val_loss': snap.val_loss,
                'learning_rate': snap.learning_rate,
                **snap.custom_metrics
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def plot_training_curves(self, output_path: str):
        """绘制训练曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return
        
        if not self.history:
            return
        
        generations = [s.generation for s in self.history]
        train_losses = [s.train_loss for s in self.history]
        val_losses = [s.val_loss for s in self.history if s.val_loss is not None]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 训练损失
        axes[0].semilogy(generations, train_losses, label='Train')
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 验证损失
        if val_losses:
            val_gens = [s.generation for s in self.history if s.val_loss is not None]
            axes[1].semilogy(val_gens, val_losses, label='Validation')
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Validation Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        logger.info(f"Training curves saved to {output_path}")


class TrainingMonitor:
    """
    训练监控器
    
    统一接口整合多种监控后端
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 enable_tensorboard: bool = False,
                 enable_wandb: bool = False,
                 enable_websocket: bool = False,
                 websocket_url: Optional[str] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_tracker = MetricsTracker()
        self.backends: List[Any] = []
        
        # 初始化后端
        if enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
                self.backends.append('tensorboard')
                logger.info("TensorBoard enabled")
            except ImportError:
                logger.warning("TensorBoard not available")
        
        if enable_wandb:
            try:
                import wandb
                self.wandb = wandb
                self.backends.append('wandb')
                logger.info("Wandb enabled")
            except ImportError:
                logger.warning("Wandb not available")
        
        if enable_websocket:
            self.websocket_logger = WebSocketLogger(websocket_url)
            self.backends.append('websocket')
    
    def start_run(self, config: Dict[str, Any]):
        """开始新的训练运行"""
        # TensorBoard
        if 'tensorboard' in self.backends:
            for key, value in config.items():
                self.tensorboard_writer.add_text(f"config/{key}", str(value))
        
        # Wandb
        if 'wandb' in self.backends:
            self.wandb.init(
                project="nep-training",
                config=config,
                dir=str(self.log_dir)
            )
        
        # WebSocket
        if 'websocket' in self.backends:
            self.websocket_logger.send({
                'event': 'run_start',
                'config': config,
                'timestamp': datetime.now().isoformat()
            })
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """记录指标"""
        # 更新本地追踪器
        self.metrics_tracker.log(
            generation=step,
            train_loss=metrics.get('train_loss', 0),
            val_loss=metrics.get('val_loss'),
            learning_rate=metrics.get('learning_rate'),
            **{k: v for k, v in metrics.items() if k not in ['train_loss', 'val_loss', 'learning_rate']}
        )
        
        # TensorBoard
        if 'tensorboard' in self.backends:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.tensorboard_writer.add_scalar(key, value, step)
        
        # Wandb
        if 'wandb' in self.backends:
            self.wandb.log(metrics, step=step)
        
        # WebSocket
        if 'websocket' in self.backends:
            self.websocket_logger.send({
                'event': 'metrics',
                'step': step,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            })
    
    def log_artifacts(self, artifact_path: str, artifact_type: str = "model"):
        """记录模型文件等产物"""
        # Wandb
        if 'wandb' in self.backends:
            artifact = self.wandb.Artifact(artifact_type, type=artifact_type)
            artifact.add_file(artifact_path)
            self.wandb.log_artifact(artifact)
    
    def finish_run(self):
        """结束训练运行"""
        # TensorBoard
        if 'tensorboard' in self.backends:
            self.tensorboard_writer.close()
        
        # Wandb
        if 'wandb' in self.backends:
            self.wandb.finish()
        
        # WebSocket
        if 'websocket' in self.backends:
            self.websocket_logger.send({
                'event': 'run_end',
                'timestamp': datetime.now().isoformat()
            })
    
    def get_status(self) -> Dict[str, Any]:
        """获取当前状态"""
        return {
            'backends': self.backends,
            'metrics_summary': self.metrics_tracker.get_summary()
        }


class TensorBoardLogger:
    """TensorBoard专用日志器"""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        except ImportError:
            logger.error("TensorBoard requires PyTorch")
            raise
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值"""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量"""
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """记录直方图"""
        self.writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model):
        """记录模型结构"""
        # NEP模型结构固定，这里可以记录配置
        pass
    
    def close(self):
        """关闭日志器"""
        self.writer.close()


class WandbLogger:
    """Weights & Biases专用日志器"""
    
    def __init__(self, 
                 project: str = "nep-training",
                 entity: Optional[str] = None,
                 config: Optional[Dict] = None):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            logger.error("Wandb not installed")
            raise
        
        self.run = self.wandb.init(
            project=project,
            entity=entity,
            config=config
        )
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """记录指标"""
        self.wandb.log(metrics, step=step)
    
    def log_artifact(self, artifact_path: str, name: Optional[str] = None):
        """记录产物"""
        artifact = self.wandb.Artifact(name or Path(artifact_path).name, type="model")
        artifact.add_file(artifact_path)
        self.run.log_artifact(artifact)
    
    def finish(self):
        """结束运行"""
        self.wandb.finish()


class WebSocketLogger:
    """
    WebSocket实时日志器
    
    将训练状态实时推送到Web前端
    """
    
    def __init__(self, url: Optional[str] = None, auto_reconnect: bool = True):
        self.url = url or "ws://localhost:8000/ws/training"
        self.auto_reconnect = auto_reconnect
        self.connected = False
        self.ws = None
        self.message_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        
        # 启动发送线程
        self._sender_thread = threading.Thread(target=self._send_loop)
        self._sender_thread.daemon = True
        self._sender_thread.start()
    
    def connect(self):
        """建立WebSocket连接"""
        try:
            import websocket
            self.ws = websocket.create_connection(self.url)
            self.connected = True
            logger.info(f"WebSocket connected to {self.url}")
        except Exception as e:
            logger.warning(f"WebSocket connection failed: {e}")
            self.connected = False
    
    def send(self, data: Dict[str, Any]):
        """发送数据"""
        self.message_queue.put(data)
    
    def _send_loop(self):
        """发送循环"""
        while not self._stop_event.is_set():
            try:
                if not self.connected and self.auto_reconnect:
                    self.connect()
                
                # 获取消息
                data = self.message_queue.get(timeout=1.0)
                
                if self.connected and self.ws:
                    message = json.dumps(data)
                    self.ws.send(message)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.debug(f"WebSocket send error: {e}")
                self.connected = False
    
    def close(self):
        """关闭连接"""
        self._stop_event.set()
        if self.ws:
            self.ws.close()


class RealTimePlotter:
    """
    实时绘图器
    
    在训练过程中实时显示训练曲线
    """
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.metrics_buffer: deque = deque(maxlen=1000)
        self._stop_event = threading.Event()
        self._update_thread: Optional[threading.Thread] = None
        
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("matplotlib not available for real-time plotting")
    
    def start(self):
        """开始实时绘图"""
        if not self.available:
            return
        
        self._stop_event.clear()
        self._update_thread = threading.Thread(target=self._update_loop)
        self._update_thread.daemon = True
        self._update_thread.start()
    
    def update(self, generation: int, train_loss: float, val_loss: Optional[float] = None):
        """更新数据"""
        self.metrics_buffer.append({
            'generation': generation,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    def _update_loop(self):
        """更新循环"""
        import matplotlib.pyplot as plt
        
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        
        while not self._stop_event.is_set():
            if len(self.metrics_buffer) > 0:
                self._draw(ax)
                plt.pause(self.update_interval)
            else:
                time.sleep(0.1)
        
        plt.close(fig)
    
    def _draw(self, ax):
        """绘制图表"""
        if not self.metrics_buffer:
            return
        
        ax.clear()
        
        generations = [m['generation'] for m in self.metrics_buffer]
        train_losses = [m['train_loss'] for m in self.metrics_buffer]
        val_losses = [m['val_loss'] for m in self.metrics_buffer if m['val_loss'] is not None]
        
        ax.semilogy(generations, train_losses, label='Train', alpha=0.8)
        if val_losses:
            val_gens = [m['generation'] for m in self.metrics_buffer if m['val_loss'] is not None]
            ax.semilogy(val_gens, val_losses, label='Validation', alpha=0.8)
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def stop(self):
        """停止实时绘图"""
        if self._update_thread:
            self._stop_event.set()
            self._update_thread.join(timeout=2.0)


class TrainingDashboard:
    """
    训练仪表盘
    
    整合所有监控信息的统一界面
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.monitor = TrainingMonitor(
            enable_tensorboard=True,
            enable_websocket=True
        )
        self.plotter = RealTimePlotter()
    
    def start(self):
        """启动仪表盘"""
        self.plotter.start()
        logger.info(f"Training dashboard started on port {self.port}")
    
    def update(self, generation: int, metrics: Dict[str, float]):
        """更新仪表盘"""
        self.monitor.log_metrics(generation, metrics)
        self.plotter.update(
            generation,
            metrics.get('train_loss', 0),
            metrics.get('val_loss')
        )
    
    def stop(self):
        """停止仪表盘"""
        self.plotter.stop()
        self.monitor.finish_run()
