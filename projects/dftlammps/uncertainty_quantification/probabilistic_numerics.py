"""
概率数值方法 - Probabilistic Numerical Methods

实现概率视角下的数值计算：
- 概率PDE求解（概率有限元）
- 概率线性代数（贝叶斯共轭梯度）
- 贝叶斯积分/ODE求解

核心特性:
- 将数值计算视为推断问题
- 量化离散化误差
- 自适应精度控制
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

try:
    from scipy import sparse
    from scipy.sparse import linalg as sparse_linalg
    from scipy.linalg import cholesky, solve_triangular
    from scipy.integrate import solve_ivp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ==================== 数据结构 ====================

@dataclass
class ProbabilityDistribution:
    """概率分布基类"""
    mean: np.ndarray
    covariance: np.ndarray
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """从分布采样"""
        n_dim = len(self.mean)
        white_noise = np.random.randn(n_dim, n_samples)
        L = cholesky(self.covariance, lower=True)
        return self.mean[:, None] + L @ white_noise
    
    def marginal_variance(self) -> np.ndarray:
        """获取边缘方差"""
        return np.diag(self.covariance)
    
    def confidence_region(self, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """计算置信区域"""
        alpha = 1 - confidence
        z = 1.96  # 近似95%置信区间
        std = np.sqrt(self.marginal_variance())
        lower = self.mean - z * std
        upper = self.mean + z * std
        return lower, upper


@dataclass
class GaussianProcessPrior:
    """高斯过程先验"""
    mean_function: Callable
    kernel_function: Callable
    
    def evaluate_prior(self, x: np.ndarray) -> ProbabilityDistribution:
        """评估先验分布"""
        mean = self.mean_function(x)
        cov = self.kernel_function(x, x)
        return ProbabilityDistribution(mean, cov)
    
    def posterior(self,
                  x_obs: np.ndarray,
                  y_obs: np.ndarray,
                  noise_var: float = 1e-6) -> Callable:
        """
        计算后验分布
        
        Returns:
            返回后验分布的函数
        """
        K_xx = self.kernel_function(x_obs, x_obs)
        K_xx += noise_var * np.eye(len(x_obs))
        
        K_inv = np.linalg.inv(K_xx)
        
        def posterior_distribution(x_query: np.ndarray) -> ProbabilityDistribution:
            K_xs_x = self.kernel_function(x_query, x_obs)
            K_xs_xs = self.kernel_function(x_query, x_query)
            
            mean = self.mean_function(x_query) + K_xs_x @ K_inv @ (y_obs - self.mean_function(x_obs))
            cov = K_xs_xs - K_xs_x @ K_inv @ K_xs_x.T
            
            return ProbabilityDistribution(mean, cov)
        
        return posterior_distribution


@dataclass
class LinearOperatorUncertainty:
    """线性算子不确定性"""
    A_mean: np.ndarray  # 矩阵均值
    A_cov: Optional[np.ndarray] = None  # 矩阵不确定性（可选）
    
    def apply(self, x: np.ndarray) -> ProbabilityDistribution:
        """应用不确定的线性算子"""
        mean = self.A_mean @ x
        
        if self.A_cov is not None:
            # 计算输出协方差
            n = len(mean)
            cov = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    cov[i, j] = x @ self.A_cov[i*n:(i+1)*n, j*n:(j+1)*n] @ x
        else:
            cov = np.zeros((n, n))
        
        return ProbabilityDistribution(mean, cov)


@dataclass
class DiscretizationError:
    """离散化误差估计"""
    estimated_error: np.ndarray
    confidence_level: float = 0.95
    refinement_indicator: Optional[np.ndarray] = None
    
    def is_acceptable(self, tolerance: float) -> bool:
        """检查误差是否可接受"""
        return np.all(self.estimated_error < tolerance)
    
    def adaptive_refinement_points(self, n_points: int = 10) -> np.ndarray:
        """确定需要细化的点"""
        if self.refinement_indicator is None:
            return np.array([])
        
        # 返回误差最大的n个点
        return np.argsort(self.refinement_indicator)[-n_points:]


# ==================== 概率PDE求解器 ====================

class ProbabilisticPDESolver(ABC):
    """概率PDE求解器基类"""
    
    @abstractmethod
    def solve(self,
             pde_operator: Callable,
             boundary_conditions: Dict,
             domain: np.ndarray) -> ProbabilityDistribution:
        """求解PDE并返回解的概率分布"""
        pass


class ProbabilisticFEM(ProbabilisticPDESolver):
    """
    概率有限元方法
    
    将有限元求解视为高斯过程推断问题
    """
    
    def __init__(self,
                 mesh_resolution: int = 50,
                 polynomial_degree: int = 1,
                 prior_gp: Optional[GaussianProcessPrior] = None):
        """
        初始化概率FEM
        
        Args:
            mesh_resolution: 网格分辨率
            polynomial_degree: 多项式次数
            prior_gp: GP先验（可选）
        """
        self.mesh_resolution = mesh_resolution
        self.polynomial_degree = polynomial_degree
        
        if prior_gp is None:
            # 默认先验
            def mean_fn(x):
                return np.zeros(len(x) if hasattr(x, '__len__') else 1)
            
            def kernel_fn(x1, x2):
                # RBF核
                if len(x1.shape) == 1:
                    x1 = x1.reshape(-1, 1)
                if len(x2.shape) == 1:
                    x2 = x2.reshape(-1, 1)
                
                sqdist = np.sum(x1**2, 1).reshape(-1, 1) + \
                        np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
                return np.exp(-0.5 * sqdist)
            
            self.prior = GaussianProcessPrior(mean_fn, kernel_fn)
        else:
            self.prior = prior_gp
    
    def solve(self,
             pde_residual: Callable,
             boundary_conditions: Dict,
             domain: Tuple[float, float]) -> Tuple[ProbabilityDistribution, DiscretizationError]:
        """
        求解1D泊松型PDE
        
        Args:
            pde_residual: PDE残差算子 R(u) = f
            boundary_conditions: 边界条件 {'left': val, 'right': val}
            domain: 求解域 [a, b]
        
        Returns:
            (解分布, 离散化误差)
        """
        # 创建网格
        x = np.linspace(domain[0], domain[1], self.mesh_resolution)
        h = x[1] - x[0]
        
        # 简化的1D有限元组装
        n_nodes = len(x)
        K = np.zeros((n_nodes, n_nodes))
        F = np.zeros(n_nodes)
        
        # 组装刚度矩阵（简化）
        for i in range(n_nodes - 1):
            # 单元矩阵
            K_local = np.array([[1, -1], [-1, 1]]) / h
            
            # 组装
            K[i:i+2, i:i+2] += K_local
        
        # 应用边界条件
        K[0, :] = 0
        K[0, 0] = 1
        F[0] = boundary_conditions['left']
        
        K[-1, :] = 0
        K[-1, -1] = 1
        F[-1] = boundary_conditions['right']
        
        # 求解
        try:
            u_mean = np.linalg.solve(K, F)
        except np.linalg.LinAlgError:
            u_mean = np.zeros(n_nodes)
        
        # 估计离散化误差（简化）
        error_estimate = self._estimate_discretization_error(x, u_mean, pde_residual)
        
        # 构建概率解
        # 解的协方差反映离散化不确定性
        cov = np.eye(n_nodes) * error_estimate**2
        solution = ProbabilityDistribution(u_mean, cov)
        
        disc_error = DiscretizationError(
            estimated_error=np.ones(n_nodes) * error_estimate,
            refinement_indicator=np.abs(np.gradient(np.gradient(u_mean)))
        )
        
        return solution, disc_error
    
    def _estimate_discretization_error(self,
                                       x: np.ndarray,
                                       u: np.ndarray,
                                       pde_residual: Callable) -> float:
        """估计离散化误差"""
        # 简化：基于二阶导数估计
        if len(u) > 2:
            second_deriv = np.gradient(np.gradient(u, x), x)
            return np.max(np.abs(second_deriv)) * (x[1] - x[0])**2
        return 0.0


# ==================== 概率线性代数 ====================

class ProbabilisticLinearSolver(ABC):
    """概率线性求解器基类"""
    
    @abstractmethod
    def solve(self,
             A: np.ndarray,
             b: np.ndarray,
             max_iter: Optional[int] = None) -> ProbabilityDistribution:
        """求解 Ax = b 并返回解的概率分布"""
        pass


class BayesianCG(ProbabilisticLinearSolver):
    """
    贝叶斯共轭梯度法
    
    将CG迭代视为高斯过程推断
    """
    
    def __init__(self,
                 prior_covariance: str = 'inverse',
                 noise_variance: float = 1e-10):
        """
        初始化贝叶斯CG
        
        Args:
            prior_covariance: 先验协方差类型
            noise_variance: 观测噪声方差
        """
        self.prior_covariance = prior_covariance
        self.noise_variance = noise_variance
    
    def solve(self,
             A: np.ndarray,
             b: np.ndarray,
             max_iter: Optional[int] = None,
             tol: float = 1e-6) -> Tuple[ProbabilityDistribution, Dict]:
        """
        贝叶斯CG求解
        
        Returns:
            (解分布, 迭代信息)
        """
        n = len(b)
        if max_iter is None:
            max_iter = n
        
        # 初始化
        x = np.zeros(n)
        r = b - A @ x
        p = r.copy()
        
        # 存储迭代历史
        residuals = [np.linalg.norm(r)]
        iterates = [x.copy()]
        
        # 先验
        if self.prior_covariance == 'inverse':
            # 使用A^{-1}作为先验协方差
            Sigma_0 = np.eye(n)  # 简化
        else:
            Sigma_0 = np.eye(n)
        
        x_mean = x.copy()
        x_cov = Sigma_0.copy()
        
        for k in range(max_iter):
            if np.linalg.norm(r) < tol:
                break
            
            Ap = A @ p
            alpha = np.dot(r, r) / np.dot(p, Ap)
            
            # 更新解
            x = x + alpha * p
            r_new = r - alpha * Ap
            
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
            
            residuals.append(np.linalg.norm(r))
            iterates.append(x.copy())
            
            # 更新后验（简化）
            x_mean = x.copy()
            # 协方差随迭代减小
            x_cov = Sigma_0 * (np.linalg.norm(r) / residuals[0])
        
        solution = ProbabilityDistribution(x_mean, x_cov)
        
        info = {
            'iterations': len(residuals) - 1,
            'residuals': residuals,
            'final_residual': residuals[-1],
            'iterates': np.array(iterates)
        }
        
        return solution, info
    
    def solve_batch(self,
                   A: np.ndarray,
                   B: np.ndarray,
                   max_iter: Optional[int] = None) -> List[ProbabilityDistribution]:
        """批量求解多个右端项"""
        n_rhs = B.shape[1]
        solutions = []
        
        for i in range(n_rhs):
            sol, _ = self.solve(A, B[:, i], max_iter)
            solutions.append(sol)
        
        return solutions


class ProbabilisticEigenvalueSolver:
    """概率特征值求解器"""
    
    def __init__(self, n_power_iterations: int = 100):
        self.n_power_iterations = n_power_iterations
    
    def solve_largest_eigenvalue(self,
                                 A: np.ndarray) -> Tuple[float, ProbabilityDistribution]:
        """
        求解最大特征值
        
        Returns:
            (特征值, 特征向量分布)
        """
        n = A.shape[0]
        
        # 幂迭代
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)
        
        for _ in range(self.n_power_iterations):
            v_new = A @ v
            v_new = v_new / np.linalg.norm(v_new)
            v = v_new
        
        eigenvalue = v @ A @ v
        
        # 特征向量的概率分布
        # 不确定性随迭代次数减小
        covariance = np.eye(n) * (1.0 / self.n_power_iterations)
        eigenvector_dist = ProbabilityDistribution(v, covariance)
        
        return eigenvalue, eigenvector_dist


# ==================== 概率ODE求解器 ====================

class ProbabilisticODE:
    """
    概率ODE求解器
    
    使用滤波/平滑框架进行概率ODE求解
    """
    
    def __init__(self,
                 order: int = 2,
                 step_size: float = 0.01):
        """
        初始化概率ODE求解器
        
        Args:
            order: 积分器阶数
            step_size: 步长
        """
        self.order = order
        self.step_size = step_size
    
    def solve(self,
             ode_func: Callable,
             y0: np.ndarray,
             t_span: Tuple[float, float],
             method: str = 'ekf') -> Dict:
        """
        求解ODE
        
        Args:
            ode_func: dy/dt = f(t, y)
            y0: 初始条件
            t_span: 时间范围
            method: 求解方法 ('ekf', 'ukf')
        
        Returns:
            包含解和不确定性的字典
        """
        t0, tf = t_span
        n_steps = int((tf - t0) / self.step_size)
        t = np.linspace(t0, tf, n_steps + 1)
        
        n_dim = len(y0)
        
        # 存储均值和协方差
        means = np.zeros((n_steps + 1, n_dim))
        covs = np.zeros((n_steps + 1, n_dim, n_dim))
        
        means[0] = y0
        covs[0] = np.eye(n_dim) * 1e-6  # 初始小不确定性
        
        for i in range(n_steps):
            # 简化的EKF步
            # 预测
            f = ode_func(t[i], means[i])
            means[i+1] = means[i] + self.step_size * f
            
            # 协方差传播（简化）
            # 使用数值 Jacobian
            J = self._numerical_jacobian(ode_func, t[i], means[i])
            F = np.eye(n_dim) + self.step_size * J
            Q = np.eye(n_dim) * (self.step_size**3)  # 过程噪声
            
            covs[i+1] = F @ covs[i] @ F.T + Q
        
        return {
            't': t,
            'mean': means,
            'covariance': covs,
            'std': np.sqrt(np.array([np.diag(c) for c in covs]))
        }
    
    def _numerical_jacobian(self,
                           f: Callable,
                           t: float,
                           y: np.ndarray,
                           h: float = 1e-6) -> np.ndarray:
        """数值计算Jacobian"""
        n = len(y)
        J = np.zeros((n, n))
        
        f_y = f(t, y)
        
        for i in range(n):
            y_perturbed = y.copy()
            y_perturbed[i] += h
            J[:, i] = (f(t, y_perturbed) - f_y) / h
        
        return J


# ==================== 贝叶斯积分 ====================

class BayesianQuadrature:
    """
    贝叶斯积分（贝叶斯求积）
    
    将数值积分视为高斯过程推断
    """
    
    def __init__(self,
                 kernel: Optional[Callable] = None,
                 mean_function: Optional[Callable] = None):
        """
        初始化贝叶斯求积
        
        Args:
            kernel: GP核函数
            mean_function: 均值函数
        """
        if kernel is None:
            # 默认Matérn 3/2核
            def kernel(x1, x2, length_scale=1.0):
                r = np.abs(x1 - x2) / length_scale
                return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
            
            self.kernel = kernel
        else:
            self.kernel = kernel
        
        if mean_function is None:
            self.mean_function = lambda x: 0.0
        else:
            self.mean_function = mean_function
    
    def integrate(self,
                 f: Callable,
                 domain: Tuple[float, float],
                 n_evals: int = 10) -> ProbabilityDistribution:
        """
        贝叶斯积分
        
        Args:
            f: 被积函数
            domain: 积分域
            n_evals: 评估点数
        
        Returns:
            积分值的概率分布
        """
        a, b = domain
        
        # 选择评估点（简化：均匀分布）
        x_evals = np.linspace(a, b, n_evals)
        
        # 评估函数
        y_evals = np.array([f(x) for x in x_evals])
        
        # 构建GP后验
        K = np.array([[self.kernel(xi, xj) for xj in x_evals] for xi in x_evals])
        K += 1e-6 * np.eye(n_evals)  # 数值稳定性
        
        K_inv = np.linalg.inv(K)
        
        # 积分核（解析积分）
        z = np.zeros(n_evals)
        for i, xi in enumerate(x_evals):
            # ∫ k(x, xi) dx 的解析表达式
            # 对于Matérn 3/2核，有解析解
            z[i] = self._integrate_kernel(xi, a, b)
        
        # 积分均值
        f_mean = y_evals - np.array([self.mean_function(x) for x in x_evals])
        integral_mean = z @ K_inv @ f_mean
        integral_mean += self._integrate_mean(a, b)
        
        # 积分方差
        integral_variance = self._integrate_double_kernel(a, b) - z @ K_inv @ z
        
        return ProbabilityDistribution(
            np.array([integral_mean]),
            np.array([[max(integral_variance, 0)]])
        )
    
    def _integrate_kernel(self, xi: float, a: float, b: float) -> float:
        """积分核函数 ∫ k(x, xi) dx"""
        # 简化的数值积分
        x = np.linspace(a, b, 100)
        k_vals = np.array([self.kernel(xi, xi_x) for xi_x in x])
        return np.trapz(k_vals, x)
    
    def _integrate_mean(self, a: float, b: float) -> float:
        """积分均值函数"""
        x = np.linspace(a, b, 100)
        m_vals = np.array([self.mean_function(xi) for xi in x])
        return np.trapz(m_vals, x)
    
    def _integrate_double_kernel(self, a: float, b: float) -> float:
        """双重积分 ∫∫ k(x, x') dx dx'"""
        # 简化
        return (b - a)**2


# ==================== 示例和测试 ====================

def demo():
    """演示概率数值方法"""
    print("=" * 80)
    print("🔢 概率数值方法演示")
    print("=" * 80)
    
    # 1. 概率PDE求解
    print("\n1. 概率有限元方法 (1D泊松方程)")
    print("   求解: -d²u/dx² = f, 边界条件 u(0)=u(1)=0")
    
    def poisson_residual(u, x):
        """泊松方程残差"""
        d2u = np.gradient(np.gradient(u, x), x)
        f = np.ones_like(x)  # 右端项
        return -d2u - f
    
    pfem = ProbabilisticFEM(mesh_resolution=20)
    
    bc = {'left': 0.0, 'right': 0.0}
    domain = (0.0, 1.0)
    
    solution, error = pfem.solve(poisson_residual, bc, domain)
    
    print(f"   解节点数: {len(solution.mean)}")
    print(f"   最大解值: {np.max(solution.mean):.4f}")
    print(f"   估计离散化误差: {np.mean(error.estimated_error):.6f}")
    
    # 2. 贝叶斯共轭梯度
    print("\n2. 贝叶斯共轭梯度法")
    print("   求解: Ax = b")
    
    # 创建对称正定矩阵
    n = 50
    A = np.diag(2 * np.ones(n)) + np.diag(-1 * np.ones(n-1), 1) + \
        np.diag(-1 * np.ones(n-1), -1)
    b = np.random.randn(n)
    
    bayesian_cg = BayesianCG()
    solution_dist, info = bayesian_cg.solve(A, b, max_iter=20)
    
    print(f"   矩阵维度: {n}x{n}")
    print(f"   迭代次数: {info['iterations']}")
    print(f"   最终残差: {info['final_residual']:.6e}")
    print(f"   解均值范数: {np.linalg.norm(solution_dist.mean):.4f}")
    print(f"   解协方差迹: {np.trace(solution_dist.covariance):.4f}")
    
    # 3. 概率ODE求解
    print("\n3. 概率ODE求解 (简谐振子)")
    print("   求解: d²x/dt² = -ω²x")
    
    def harmonic_oscillator(t, y):
        """简谐振子: y = [x, v]"""
        omega = 2 * np.pi
        x, v = y
        return np.array([v, -omega**2 * x])
    
    y0 = np.array([1.0, 0.0])  # 初始条件: x=1, v=0
    
    prob_ode = ProbabilisticODE(step_size=0.01)
    result = prob_ode.solve(harmonic_oscillator, y0, (0, 2), method='ekf')
    
    print(f"   时间步数: {len(result['t'])}")
    print(f"   最终位置: {result['mean'][-1, 0]:.4f} ± {result['std'][-1, 0]:.4f}")
    print(f"   最终速度: {result['mean'][-1, 1]:.4f} ± {result['std'][-1, 1]:.4f}")
    
    # 4. 贝叶斯积分
    print("\n4. 贝叶斯求积")
    print("   积分: ∫₀¹ exp(x) dx = e - 1 ≈ 1.71828")
    
    def integrand(x):
        return np.exp(x)
    
    bq = BayesianQuadrature()
    integral_dist = bq.integrate(integrand, (0, 1), n_evals=10)
    
    true_value = np.e - 1
    
    print(f"   积分均值: {integral_dist.mean[0]:.6f}")
    print(f"   积分标准差: {np.sqrt(integral_dist.covariance[0,0]):.6f}")
    print(f"   真实值: {true_value:.6f}")
    print(f"   相对误差: {abs(integral_dist.mean[0] - true_value) / true_value * 100:.4f}%")
    
    # 5. 高斯过程先验
    print("\n5. 高斯过程先验与后验")
    
    def mean_fn(x):
        return np.zeros(len(x))
    
    def kernel_fn(x1, x2, l=0.5):
        """RBF核"""
        X1 = x1.reshape(-1, 1) if len(x1.shape) == 1 else x1
        X2 = x2.reshape(-1, 1) if len(x2.shape) == 1 else x2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
                np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-sqdist / (2 * l**2))
    
    gp_prior = GaussianProcessPrior(mean_fn, kernel_fn)
    
    # 观测数据
    x_obs = np.array([0.1, 0.5, 0.9])
    y_obs = np.sin(2 * np.pi * x_obs)
    
    # 后验
    posterior_fn = gp_prior.posterior(x_obs, y_obs, noise_var=0.01)
    
    # 查询
    x_query = np.linspace(0, 1, 50)
    post_dist = posterior_fn(x_query)
    
    print(f"   观测点数: {len(x_obs)}")
    print(f"   查询点数: {len(x_query)}")
    print(f"   后验均值范围: [{np.min(post_dist.mean):.4f}, {np.max(post_dist.mean):.4f}]")
    print(f"   后验标准差范围: [{np.min(np.sqrt(post_dist.marginal_variance())):.4f}, "
          f"{np.max(np.sqrt(post_dist.marginal_variance())):.4f}]")
    
    print("\n" + "=" * 80)
    print("✅ 概率数值方法演示完成")
    print("=" * 80)


if __name__ == "__main__":
    demo()
