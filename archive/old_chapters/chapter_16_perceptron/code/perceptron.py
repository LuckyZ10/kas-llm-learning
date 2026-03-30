"""
第十六章：感知机 - 完整实现
============================

本模块实现了感知机算法的完整功能，包括：
1. 基础感知机（原始形式）
2. 感知机对偶形式
3. 可视化工具
4. 逻辑门测试
5. 收敛性分析

作者：机器学习教程
日期：2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import ListedColormap

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False


class Perceptron:
    """
    感知机分类器（原始形式）
    
    使用感知机学习规则进行二分类。感知机是最早的神经网络模型，
    由Frank Rosenblatt于1957年提出。
    
    数学原理:
    -----------
    感知机的决策函数为: f(x) = sign(w^T * x + b)
    其中 sign(z) = +1 if z >= 0, else -1
    
    学习规则（错误驱动）:
    如果样本(x_i, y_i)被错误分类，则:
        w <- w + η * y_i * x_i
        b <- b + η * y_i
    
    参数:
    -----------
    learning_rate : float, 默认=0.01
        学习率 η，控制权重更新的步长。太大的学习率可能导致震荡，
        太小的学习率会导致收敛缓慢。
    
    n_iterations : int, 默认=1000
        最大迭代次数。如果数据线性可分，算法通常会在达到此限制前收敛。
    
    random_state : int, 默认=None
        随机种子，用于初始化权重的可重复性。
    
    属性:
    -----------
    weights_ : ndarray, shape = [n_features]
        训练后的权重向量
    
    bias_ : float
        训练后的偏置项
    
    errors_ : list
        每次迭代的误分类样本数量列表
    
    n_updates_ : int
        总的权重更新次数
    
    示例:
    -----------
    >>> from perceptron import Perceptron
    >>> import numpy as np
    >>> # AND门数据
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = np.array([-1, -1, -1, 1])  # AND逻辑
    >>> p = Perceptron(learning_rate=0.1, n_iterations=100)
    >>> p.fit(X, y)
    >>> print(p.predict(X))  # 输出: [-1 -1 -1  1]
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights_ = None
        self.bias_ = None
        self.errors_ = []
        self.n_updates_ = 0
        
    def fit(self, X, y):
        """
        使用感知机学习规则训练模型
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
            训练数据特征矩阵。每行代表一个样本，每列代表一个特征。
        
        y : array-like, shape = [n_samples]
            目标标签。必须是 +1 或 -1（二分类）。
            
        返回:
        -----------
        self : object
            返回训练后的感知机实例
            
        注意:
        -----------
        如果数据不是线性可分的，算法将不会收敛，会在n_iterations次迭代后停止。
        可以通过检查 errors_ 列表的最后一个元素是否大于0来判断是否收敛。
        """
        rgen = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        
        # 初始化权重和偏置
        # 使用小的随机值初始化，打破对称性
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=n_features)
        self.bias_ = 0.0
        self.errors_ = []
        self.n_updates_ = 0
        
        # 训练循环
        for iteration in range(self.n_iterations):
            errors = 0
            
            # 遍历所有训练样本
            for xi, target in zip(X, y):
                # 计算预测值
                output = self._predict_single(xi)
                
                # 如果预测错误（y * (w^T * x + b) <= 0），更新权重
                if target * output <= 0:
                    update = self.learning_rate * target
                    self.weights_ += update * xi
                    self.bias_ += update
                    errors += 1
                    self.n_updates_ += 1
            
            self.errors_.append(errors)
            
            # 如果所有样本都正确分类，提前停止（收敛）
            if errors == 0:
                print(f"✓ 感知机在迭代 {iteration + 1} 时收敛")
                break
        else:
            # 未收敛
            print(f"✗ 感知机在 {self.n_iterations} 次迭代后未收敛")
            print(f"  最终错误数: {errors}")
        
        return self
    
    def _predict_single(self, x):
        """
        预测单个样本的类别（内部方法）
        
        参数:
        -----------
        x : array-like, shape = [n_features]
            单个样本的特征向量
            
        返回:
        -----------
        int : +1 或 -1
            预测的类别标签
        """
        activation = np.dot(x, self.weights_) + self.bias_
        return 1 if activation >= 0 else -1
    
    def predict(self, X):
        """
        预测多个样本的类别
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
            样本特征矩阵
            
        返回:
        -----------
        array : shape = [n_samples]
            预测标签数组，每个元素为 +1 或 -1
        """
        return np.array([self._predict_single(xi) for xi in X])
    
    def decision_function(self, X):
        """
        计算决策函数值（到超平面的有符号距离，未归一化）
        
        决策函数定义为: f(x) = w^T * x + b
        正值表示预测为 +1 类，负值表示预测为 -1 类。
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
            样本特征矩阵
            
        返回:
        -----------
        array : shape = [n_samples]
            决策函数值
        """
        return np.dot(X, self.weights_) + self.bias_
    
    def get_params(self):
        """
        获取模型参数
        
        返回:
        -----------
        dict : 包含权重和偏置的字典
        """
        return {
            'weights': self.weights_,
            'bias': self.bias_,
            'n_updates': self.n_updates_
        }


class PerceptronDual:
    """
    感知机分类器（对偶形式）
    
    对偶形式将权重表示为训练样本的线性组合:
        w = sum(alpha_i * y_i * x_i)
        b = sum(alpha_i * y_i)
    
    其中 alpha_i >= 0 表示第i个样本被更新的次数。
    
    对偶形式的优势:
    1. 只需存储Gram矩阵，不需要显式存储权重向量
    2. 当样本数 N << 特征维度时，计算更高效
    3. 为核方法（Kernel Methods）奠定基础
    
    参数:
    -----------
    learning_rate : float, 默认=0.01
        学习率 η
    
    n_iterations : int, 默认=1000
        最大迭代次数
    
    属性:
    -----------
    alpha_ : ndarray, shape = [n_samples]
        每个样本的更新次数（对偶变量）
    
    b_ : float
        偏置项
    
    G_ : ndarray, shape = [n_samples, n_samples]
        Gram矩阵，G[i,j] = x_i^T * x_j
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.alpha_ = None
        self.b_ = 0
        self.X_train_ = None
        self.y_train_ = None
        self.G_ = None
        
    def fit(self, X, y):
        """
        使用对偶形式训练感知机
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
        y : array-like, shape = [n_samples]
        """
        n_samples = X.shape[0]
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        # 初始化对偶变量alpha
        self.alpha_ = np.zeros(n_samples)
        self.b_ = 0
        
        # 预计算Gram矩阵
        print("预计算Gram矩阵...")
        self.G_ = np.dot(X, X.T)
        print(f"Gram矩阵形状: {self.G_.shape}")
        
        # 训练
        for iteration in range(self.n_iterations):
            errors = 0
            
            for i in range(n_samples):
                # 使用对偶形式计算预测
                # y_pred = sign(sum(alpha_j * y_j * (x_j^T * x_i)) + b)
                prediction = np.sum(self.alpha_ * self.y_train_ * self.G_[:, i]) + self.b_
                prediction = 1 if prediction >= 0 else -1
                
                # 如果预测错误，更新alpha和b
                if y[i] * prediction <= 0:
                    self.alpha_[i] += self.learning_rate
                    self.b_ += self.learning_rate * y[i]
                    errors += 1
            
            if errors == 0:
                print(f"✓ 对偶形式收敛于迭代 {iteration + 1}")
                break
        else:
            print(f"✗ 对偶形式在 {self.n_iterations} 次迭代后未收敛")
        
        # 计算原始形式的权重（用于可视化等）
        self.weights_ = np.dot(self.alpha_ * self.y_train_, self.X_train_)
        
        return self
    
    def predict(self, X):
        """
        预测新样本
        
        参数:
        -----------
        X : array-like, shape = [n_samples, n_features]
        
        返回:
        -----------
        array : 预测标签
        """
        # 计算测试样本与训练样本的Gram矩阵
        G_test = np.dot(X, self.X_train_.T)
        predictions = np.dot(G_test, self.alpha_ * self.y_train_) + self.b_
        return np.where(predictions >= 0, 1, -1)


# ==================== 可视化工具 ====================

def plot_decision_boundary(X, y, classifier, title="感知机决策边界", 
                           save_path=None, show_margin=True):
    """
    可视化感知机的决策边界
    
    参数:
    -----------
    X : array-like, shape = [n_samples, 2]
        二维特征数据
    y : array-like, shape = [n_samples]
        标签 (+1 或 -1)
    classifier : Perceptron
        训练好的感知机
    title : str
        图表标题
    save_path : str, optional
        保存图片的路径
    show_margin : bool
        是否显示间隔边界
    """
    plt.figure(figsize=(10, 8))
    
    # 绘制数据点
    positive_idx = y == 1
    negative_idx = y == -1
    
    plt.scatter(X[positive_idx, 0], X[positive_idx, 1], 
                c='#3498db', marker='o', s=150, label='正类 (+1)', 
                edgecolors='k', linewidth=1.5, zorder=5)
    plt.scatter(X[negative_idx, 0], X[negative_idx, 1], 
                c='#e74c3c', marker='x', s=150, label='负类 (-1)', 
                edgecolors='k', linewidth=1.5, zorder=5)
    
    # 绘制决策边界
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_values = np.linspace(x1_min, x1_max, 100)
    
    w1, w2 = classifier.weights_
    b = classifier.bias_
    
    if abs(w2) > 1e-10:  # 避免除以零
        # 决策边界: w1*x1 + w2*x2 + b = 0  =>  x2 = -(w1*x1 + b) / w2
        x2_values = -(w1 * x1_values + b) / w2
        plt.plot(x1_values, x2_values, 'g-', linewidth=2.5, 
                label='决策边界', zorder=3)
        
        if show_margin:
            # 绘制间隔边界（y=+1 和 y=-1 的边界）
            x2_values_pos = -(w1 * x1_values + b - 1) / w2
            x2_values_neg = -(w1 * x1_values + b + 1) / w2
            plt.plot(x1_values, x2_values_pos, 'g--', alpha=0.5, 
                    linewidth=1.5, label='间隔边界')
            plt.plot(x1_values, x2_values_neg, 'g--', alpha=0.5, linewidth=1.5)
    
    # 添加权重向量的可视化
    if show_margin:
        # 在决策边界上取一点，绘制权重向量
        x_mid = (x1_min + x1_max) / 2
        y_mid = -(w1 * x_mid + b) / w2
        scale = 0.3
        plt.arrow(x_mid, y_mid, w1 * scale, w2 * scale, 
                 head_width=0.05, head_length=0.05, fc='purple', ec='purple',
                 linewidth=2, label='权重向量 w', zorder=4)
    
    plt.xlabel('$x_1$', fontsize=14)
    plt.ylabel('$x_2$', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axis('equal')
    
    # 添加权重信息文本
    info_text = f'$w_1={w1:.3f}$, $w_2={w2:.3f}$, $b={b:.3f}$'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()


def plot_learning_curve(classifier, title="感知机学习曲线", save_path=None):
    """
    绘制感知机的学习曲线（误分类数随迭代次数的变化）
    
    参数:
    -----------
    classifier : Perceptron
        训练好的感知机
    title : str
        图表标题
    save_path : str, optional
        保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    
    iterations = range(1, len(classifier.errors_) + 1)
    plt.plot(iterations, classifier.errors_, 'b-o', linewidth=2, 
             markersize=6, markerfacecolor='white', markeredgewidth=2)
    
    plt.xlabel('迭代次数', fontsize=14)
    plt.ylabel('误分类样本数', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加收敛信息
    final_errors = classifier.errors_[-1]
    if final_errors == 0:
        convergence_text = f'✓ 已收敛 (迭代次数: {len(classifier.errors_)})'
        color = 'green'
    else:
        convergence_text = f'✗ 未收敛 (最终错误: {final_errors})'
        color = 'red'
    
    plt.text(0.98, 0.98, convergence_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             color=color, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()


def plot_xor_problem(save_path=None):
    """
    可视化XOR问题，展示其线性不可分性
    
    参数:
    -----------
    save_path : str, optional
        保存图片的路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    # AND门
    y_and = np.array([-1, -1, -1, 1])
    ax = axes[0]
    _plot_logic_gate(ax, X, y_and, "AND门（线性可分）", "#3498db")
    ax.plot([0.5, 0.5], [-0.2, 1.2], 'g--', linewidth=2, label='决策边界')
    ax.legend()
    
    # OR门
    y_or = np.array([-1, 1, 1, 1])
    ax = axes[1]
    _plot_logic_gate(ax, X, y_or, "OR门（线性可分）", "#2ecc71")
    ax.plot([-0.2, 1.2], [0.5, 0.5], 'g--', linewidth=2, label='决策边界')
    ax.legend()
    
    # XOR门
    y_xor = np.array([-1, 1, 1, -1])
    ax = axes[2]
    _plot_logic_gate(ax, X, y_xor, "XOR门（线性不可分）", "#e74c3c")
    
    # 尝试绘制两条决策边界
    ax.plot([0.5, 0.5], [-0.2, 1.2], 'g--', linewidth=2, alpha=0.5)
    ax.plot([-0.2, 1.2], [0.5, 0.5], 'g--', linewidth=2, alpha=0.5)
    
    # 添加说明文字
    ax.text(0.5, 0.5, '无法找到\n单一直线\n分隔两类', 
            ha='center', va='center', fontsize=12, color='purple',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()


def _plot_logic_gate(ax, X, y, title, color):
    """辅助函数：绘制逻辑门的散点图"""
    positive_idx = y == 1
    negative_idx = y == -1
    
    ax.scatter(X[positive_idx, 0], X[positive_idx, 1], 
               c=color, marker='o', s=300, label='输出=1', 
               edgecolors='k', linewidth=2, zorder=5)
    ax.scatter(X[negative_idx, 0], X[negative_idx, 1], 
               c='white', marker='o', s=300, label='输出=0', 
               edgecolors='k', linewidth=2, zorder=5)
    
    # 添加标签
    for i, (x, y_val) in enumerate(zip(X, y)):
        label = '1' if y_val == 1 else '0'
        ax.annotate(label, (x[0], x[1]), textcoords="offset points",
                   xytext=(0, -15), ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_perceptron_architecture(save_path=None):
    """
    绘制感知机的结构图
    
    参数:
    -----------
    save_path : str, optional
        保存图片的路径
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 标题
    ax.text(5, 7.5, '感知机结构 (Perceptron Architecture)', 
            ha='center', fontsize=16, fontweight='bold')
    
    # 输入节点
    input_y = [2, 4, 6]
    input_labels = ['$x_1$', '$x_2$', '$x_n$']
    for y, label in zip(input_y, input_labels):
        circle = plt.Circle((1, y), 0.3, color='#3498db', ec='k', linewidth=2)
        ax.add_patch(circle)
        ax.text(1, y, label, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 省略号
    ax.text(1, 3, '...', ha='center', fontsize=20)
    
    # 权重标签
    weight_labels = ['$w_1$', '$w_2$', '$w_n$']
    for y, label in zip(input_y, weight_labels):
        ax.text(2.5, y, label, ha='center', fontsize=11, color='#e74c3c')
        ax.arrow(1.4, y, 1.2, 0, head_width=0.15, head_length=0.1, 
                fc='gray', ec='gray', linewidth=1.5)
    
    # 求和节点
    circle = plt.Circle((5, 4), 0.6, color='#f39c12', ec='k', linewidth=2)
    ax.add_patch(circle)
    ax.text(5, 4, '$\\Sigma$', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # 偏置
    ax.text(4.2, 5.5, '偏置 $b$', ha='center', fontsize=11, color='#9b59b6')
    ax.arrow(4.5, 5.2, 0.3, -0.8, head_width=0.1, head_length=0.1,
            fc='#9b59b6', ec='#9b59b6', linewidth=1.5)
    
    # 激活函数
    ax.arrow(5.7, 4, 0.8, 0, head_width=0.15, head_length=0.1,
            fc='gray', ec='gray', linewidth=2)
    
    # 激活函数框
    rect = plt.Rectangle((6.6, 3.2), 1.8, 1.6, fill=True, 
                         facecolor='#2ecc71', edgecolor='k', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.5, 4.5, '激活函数', ha='center', fontsize=10)
    ax.text(7.5, 3.8, '$\\phi(z)$', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 3.3, 'sign/step', ha='center', fontsize=9, style='italic')
    
    # 输出
    ax.arrow(8.5, 4, 0.6, 0, head_width=0.15, head_length=0.1,
            fc='gray', ec='gray', linewidth=2)
    circle = plt.Circle((9.5, 4), 0.3, color='#e74c3c', ec='k', linewidth=2)
    ax.add_patch(circle)
    ax.text(9.5, 4, '$y$', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 公式
    formula = r'$y = \phi(\mathbf{w}^T \mathbf{x} + b) = sign(\sum_{i=1}^{n} w_i x_i + b)$'
    ax.text(5, 1, formula, ha='center', fontsize=13,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()


# ==================== 测试和演示函数 ====================

def test_logic_gates():
    """
    测试感知机在逻辑门上的表现
    
    测试AND、OR、NAND（可学习）和XOR（不可学习）
    """
    print("=" * 70)
    print("感知机逻辑门测试")
    print("=" * 70)
    
    # 定义输入
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    gates = [
        ('AND', np.array([-1, -1, -1, 1])),
        ('OR', np.array([-1, 1, 1, 1])),
        ('NAND', np.array([1, 1, 1, -1])),
        ('XOR', np.array([-1, 1, 1, -1]))
    ]
    
    for gate_name, y in gates:
        print(f"\n{'='*50}")
        print(f"测试 {gate_name} 门")
        print(f"{'='*50}")
        
        p = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
        p.fit(X, y)
        
        predictions = p.predict(X)
        accuracy = np.mean(predictions == y) * 100
        
        print(f"权重: w1={p.weights_[0]:.4f}, w2={p.weights_[1]:.4f}, b={p.bias_:.4f}")
        print(f"预测: {predictions}")
        print(f"真实: {y}")
        print(f"正确率: {accuracy:.0f}%")
        
        if gate_name == 'XOR' and accuracy < 100:
            print("  → 预期行为：XOR不是线性可分的，单层感知机无法解决")


def demo_and_gate():
    """
    AND门的完整演示，包括训练和可视化
    """
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n" + "="*70)
    print("AND门完整演示")
    print("="*70)
    
    # 数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])
    
    # 训练
    p = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
    p.fit(X, y)
    
    # 可视化
    plot_learning_curve(p, "AND门 - 感知机学习曲线", "output/and_learning_curve.png")
    plot_decision_boundary(X, y, p, "AND门 - 决策边界", "output/and_decision_boundary.png")
    
    print(f"\n最终参数: w=[{p.weights_[0]:.4f}, {p.weights_[1]:.4f}], b={p.bias_:.4f}")


def demo_xor_failure():
    """
    演示感知机无法解决XOR问题
    """
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n" + "="*70)
    print("XOR问题演示 - 展示感知机的局限性")
    print("="*70)
    
    # 数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, 1, 1, -1])
    
    # 训练
    p = Perceptron(learning_rate=0.1, n_iterations=100, random_state=42)
    p.fit(X, y)
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(p.errors_) + 1), p.errors_, 'r-o', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('误分类数', fontsize=12)
    plt.title('XOR - 无法收敛', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    positive_idx = y == 1
    negative_idx = y == -1
    plt.scatter(X[positive_idx, 0], X[positive_idx, 1], 
                c='blue', marker='o', s=200, label='输出=1', edgecolors='k')
    plt.scatter(X[negative_idx, 0], X[negative_idx, 1], 
                c='red', marker='x', s=200, label='输出=0', edgecolors='k')
    
    # 尝试绘制错误的决策边界
    w1, w2 = p.weights_
    b = p.bias_
    x1_values = np.linspace(-0.5, 1.5, 100)
    if abs(w2) > 1e-10:
        x2_values = -(w1 * x1_values + b) / w2
        plt.plot(x1_values, x2_values, 'g--', linewidth=2, label='感知机找到的边界')
    
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.xlabel('$x_1$', fontsize=12)
    plt.ylabel('$x_2$', fontsize=12)
    plt.title('XOR - 线性不可分', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/xor_failure.png', dpi=150)
    plt.show()
    
    print("\n结论: XOR问题需要非线性决策边界，单层感知机无法解决")


def demo_iris_classification():
    """
    使用感知机对鸢尾花数据集进行分类
    """
    from sklearn.datasets import load_iris
    
    print("\n" + "="*70)
    print("鸢尾花数据集分类演示")
    print("="*70)
    
    # 加载数据
    iris = load_iris()
    X = iris.data[:100, :2]  # 只取前两个类别和前两个特征
    y = iris.target[:100]
    y = np.where(y == 0, -1, 1)  # 转换为 +1/-1
    
    print(f"数据形状: {X.shape}")
    print(f"类别分布: {np.sum(y == -1)} 个Setosa, {np.sum(y == 1)} 个Versicolor")
    
    # 训练
    p = Perceptron(learning_rate=0.01, n_iterations=100, random_state=42)
    p.fit(X, y)
    
    # 评估
    predictions = p.predict(X)
    accuracy = np.mean(predictions == y) * 100
    print(f"\n训练集准确率: {accuracy:.2f}%")
    
    # 可视化
    plot_learning_curve(p, "鸢尾花分类 - 学习曲线")
    plot_decision_boundary(X, y, p, "鸢尾花分类 - 决策边界")


def compare_primal_dual():
    """
    比较感知机的原始形式和对偶形式
    """
    print("\n" + "="*70)
    print("原始形式 vs 对偶形式 对比")
    print("="*70)
    
    # 生成线性可分数据
    np.random.seed(42)
    X = np.random.randn(100, 10)  # 100个样本，10维特征
    y = np.where(np.sum(X[:, :3], axis=1) > 0, 1, -1)
    
    print(f"数据: {X.shape[0]} 个样本, {X.shape[1]} 维特征")
    
    # 原始形式
    import time
    p1 = Perceptron(learning_rate=0.01, n_iterations=100)
    t0 = time.time()
    p1.fit(X, y)
    t1 = time.time()
    print(f"\n原始形式:")
    print(f"  训练时间: {t1-t0:.4f} 秒")
    print(f"  更新次数: {p1.n_updates_}")
    print(f"  准确率: {np.mean(p1.predict(X) == y)*100:.2f}%")
    
    # 对偶形式
    p2 = PerceptronDual(learning_rate=0.01, n_iterations=100)
    t0 = time.time()
    p2.fit(X, y)
    t1 = time.time()
    print(f"\n对偶形式:")
    print(f"  训练时间: {t1-t0:.4f} 秒")
    print(f"  非零alpha数: {np.sum(p2.alpha_ > 0)}")
    print(f"  准确率: {np.mean(p2.predict(X) == y)*100:.2f}%")


# ==================== 主程序 ====================

if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n" + "="*70)
    print("第十六章: 感知机 - 完整演示")
    print("="*70)
    
    # 1. 测试逻辑门
    test_logic_gates()
    
    # 2. AND门详细演示
    demo_and_gate()
    
    # 3. XOR失败演示
    demo_xor_failure()
    
    # 4. 可视化XOR问题
    plot_xor_problem("output/xor_problem.png")
    
    # 5. 可视化感知机结构
    plot_perceptron_architecture("output/perceptron_architecture.png")
    
    print("\n" + "="*70)
    print("所有演示完成！输出文件保存在 output/ 目录")
    print("="*70)
