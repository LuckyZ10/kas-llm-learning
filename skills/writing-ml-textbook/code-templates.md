# 代码架构模板 (Code Architecture Templates)

> 标准化代码结构，确保可读性和可维护性

---

## 模板1：从零实现（NumPy版本）

适用于：理解算法原理，教学演示

```python
"""
[算法名称] - 从零实现（NumPy版本）
目标：不依赖深度学习框架，纯NumPy实现，便于理解原理
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class AlgorithmScratch:
    """
    [算法名称]基础实现
    
    核心思想：
    - [要点1]
    - [要点2]
    - [要点3]
    
    参数说明：
    - param1: 参数1说明
    - param2: 参数2说明
    """
    
    def __init__(self, param1: float = 0.01, param2: int = 100):
        """
        初始化算法
        
        Args:
            param1: 学习率/步长
            param2: 最大迭代次数
        """
        self.param1 = param1
        self.param2 = param2
        
        # 状态变量
        self.weights = None
        self.bias = None
        self.loss_history = []
        
    def _initialize_parameters(self, n_features: int) -> None:
        """
        初始化模型参数
        
        使用Xavier/He初始化，确保初始梯度合理
        """
        np.random.seed(42)
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        
    def _forward(self, X: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            X: 输入特征 (n_samples, n_features)
            
        Returns:
            predictions: 预测结果 (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        计算损失函数
        
        Args:
            y_true: 真实标签
            y_pred: 预测结果
            
        Returns:
            loss: 标量损失值
        """
        # 实现具体损失函数
        pass
    
    def _backward(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        反向传播，计算梯度
        
        Args:
            X: 输入特征
            y_true: 真实标签
            y_pred: 预测结果
            
        Returns:
            dw: 权重梯度
            db: 偏置梯度
        """
        n_samples = X.shape[0]
        
        # 计算梯度
        dw = np.dot(X.T, (y_pred - y_true)) / n_samples
        db = np.mean(y_pred - y_true)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'AlgorithmScratch':
        """
        训练模型
        
        Args:
            X: 训练数据 (n_samples, n_features)
            y: 训练标签 (n_samples,)
            verbose: 是否打印训练过程
            
        Returns:
            self: 返回自身，支持链式调用
        """
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        
        for iteration in range(self.param2):
            # 前向传播
            y_pred = self._forward(X)
            
            # 计算损失
            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # 反向传播
            dw, db = self._backward(X, y, y_pred)
            
            # 参数更新
            self.weights -= self.param1 * dw
            self.bias -= self.param1 * db
            
            # 打印进度
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入数据
            
        Returns:
            predictions: 预测结果
        """
        return self._forward(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        评估模型
        
        Args:
            X: 测试数据
            y: 测试标签
            
        Returns:
            metrics: 评估指标字典
        """
        y_pred = self.predict(X)
        loss = self._compute_loss(y, y_pred)
        
        return {
            'loss': loss,
            # 添加其他指标
        }
    
    def visualize_training(self) -> None:
        """
        可视化训练过程
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()


def demo():
    """
    完整演示流程
    """
    # 1. 生成示例数据
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # 2. 划分训练集和测试集
    split_idx = 80
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 3. 创建并训练模型
    model = AlgorithmScratch(param1=0.01, param2=1000)
    model.fit(X_train, y_train)
    
    # 4. 评估模型
    metrics = model.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")
    
    # 5. 可视化
    model.visualize_training()


if __name__ == "__main__":
    demo()
```

---

## 模板2：框架实现（PyTorch版本）

适用于：生产级代码，GPU加速

```python
"""
[算法名称] - PyTorch框架实现
目标：使用现代深度学习框架，支持GPU加速
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


class AlgorithmModel(nn.Module):
    """
    [算法名称] PyTorch实现
    
    继承nn.Module，支持：
    - GPU加速
    - 自动求导
    - 模型保存/加载
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        """
        初始化模型
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super(AlgorithmModel, self).__init__()
        
        # 定义网络结构
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self) -> None:
        """
        使用He初始化权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, input_dim)
            
        Returns:
            output: 输出张量 (batch_size, output_dim)
        """
        return self.network(x)


class AlgorithmTrainer:
    """
    训练器类
    
    封装训练流程，支持：
    - 早停
    - 学习率调度
    - 日志记录
    - 检查点保存
    """
    
    def __init__(
        self,
        model: AlgorithmModel,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            learning_rate: 学习率
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # 根据任务修改
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            dataloader: 训练数据加载器
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader: DataLoader) -> float:
        """
        验证
        
        Args:
            dataloader: 验证数据加载器
            
        Returns:
            avg_loss: 平均损失
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        patience: int = 10
    ) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 训练轮数
            patience: 早停耐心值
            
        Returns:
            history: 训练历史
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def predict(self, dataloader: DataLoader) -> torch.Tensor:
        """
        预测
        
        Args:
            dataloader: 数据加载器
            
        Returns:
            predictions: 预测结果
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def visualize_training(self) -> None:
        """
        可视化训练过程
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)
        plt.show()


def demo_pytorch():
    """
    PyTorch版本完整演示
    """
    # 1. 准备数据
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 标准化
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 2. 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # 3. 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 4. 创建模型和训练器
    model = AlgorithmModel(input_dim=10, hidden_dim=64, output_dim=1)
    trainer = AlgorithmTrainer(model, learning_rate=1e-3)
    
    # 5. 训练
    history = trainer.fit(train_loader, test_loader, epochs=100, patience=15)
    
    # 6. 可视化
    trainer.visualize_training()


if __name__ == "__main__":
    demo_pytorch()
```

---

## 模板3：完整实验Pipeline

适用于：端到端实验，可复现

```python
"""
[实验名称] - 完整实验Pipeline
目标：一键运行完整实验，确保可复现性
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, Any


class Experiment:
    """
    完整实验管理类
    
    功能：
    - 配置管理
    - 数据准备
    - 模型训练
    - 结果评估
    - 实验记录
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化实验
        
        Args:
            config: 实验配置字典
        """
        self.config = config
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiments/exp_{self.exp_id}"
        
        # 创建实验目录
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 保存配置
        with open(f"{self.exp_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # 设置随机种子
        self._set_seed(config.get('seed', 42))
    
    def _set_seed(self, seed: int) -> None:
        """
        设置随机种子确保可复现
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def prepare_data(self) -> tuple:
        """
        准备数据
        
        Returns:
            train_data, test_data: 训练和测试数据
        """
        # 实现数据准备逻辑
        pass
    
    def build_model(self):
        """
        构建模型
        
        Returns:
            model: 模型实例
        """
        # 实现模型构建逻辑
        pass
    
    def train(self, model, train_data) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
            training_history: 训练历史
        """
        # 实现训练逻辑
        pass
    
    def evaluate(self, model, test_data) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            metrics: 评估指标
        """
        # 实现评估逻辑
        pass
    
    def run(self) -> Dict[str, Any]:
        """
        运行完整实验
        
        Returns:
            results: 实验结果
        """
        print(f"Starting experiment {self.exp_id}")
        print(f"Configuration: {self.config}")
        
        # 1. 准备数据
        train_data, test_data = self.prepare_data()
        
        # 2. 构建模型
        model = self.build_model()
        
        # 3. 训练
        history = self.train(model, train_data)
        
        # 4. 评估
        metrics = self.evaluate(model, test_data)
        
        # 5. 保存结果
        results = {
            'exp_id': self.exp_id,
            'config': self.config,
            'history': history,
            'metrics': metrics
        }
        
        with open(f"{self.exp_dir}/results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # 6. 保存模型
        torch.save(model.state_dict(), f"{self.exp_dir}/model.pth")
        
        print(f"Experiment completed. Results saved to {self.exp_dir}")
        
        return results


if __name__ == "__main__":
    # 定义配置
    config = {
        'seed': 42,
        'learning_rate': 1e-3,
        'batch_size': 32,
        'epochs': 100,
        'hidden_dim': 128
    }
    
    # 运行实验
    exp = Experiment(config)
    results = exp.run()
    
    print(f"Final metrics: {results['metrics']}")
```

---

## 使用指南

### 选择模板
- **教学演示** → 模板1（NumPy从零实现）
- **生产代码** → 模板2（PyTorch框架实现）
- **论文复现** → 模板3（完整实验Pipeline）

### 代码规范
1. **注释**：每个函数必须有docstring
2. **类型提示**：使用typing模块标注类型
3. **错误处理**：关键步骤添加try-except
4. **日志**：使用print或logging记录进度
5. **可视化**：提供训练过程可视化工具

### 测试要求
- 单元测试：每个方法单独测试
- 集成测试：完整流程跑通
- 边界测试：处理异常情况

---

*根据具体章节选择合适的模板，保持风格统一*
