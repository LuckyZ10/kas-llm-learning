"""
结构方程模型 - Structural Equation Modeling (SEM)
实现结构方程模型用于因果分析和潜在变量建模
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.optimize import minimize
from scipy.linalg import inv, sqrtm
from scipy import stats
import warnings


@dataclass
class Variable:
    """变量定义"""
    name: str
    is_latent: bool = False
    is_observed: bool = True
    indicators: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.name)


@dataclass
class Path:
    """路径定义"""
    from_var: str
    to_var: str
    coefficient: float = 0.0
    is_fixed: bool = False
    fixed_value: Optional[float] = None
    
    def __hash__(self):
        return hash((self.from_var, self.to_var))


@dataclass
class Measurement:
    """测量模型定义"""
    latent: str
    indicator: str
    loading: float = 1.0
    is_fixed: bool = False
    error_variance: float = 0.1


class StructuralEquationModel:
    """
    结构方程模型
    整合测量模型和结构模型
    """
    
    def __init__(self):
        # 变量
        self.observed_vars: List[Variable] = []
        self.latent_vars: List[Variable] = []
        
        # 模型
        self.measurement_model: Dict[str, List[Measurement]] = defaultdict(list)
        self.structural_model: List[Path] = []
        
        # 参数
        self.params: Dict[str, float] = {}
        self.param_constraints: Dict[str, Tuple[float, float]] = {}
        
        # 拟合结果
        self.fitted = False
        self.fit_statistics: Dict[str, float] = {}
        
    def add_observed_variable(self, name: str):
        """添加观测变量"""
        var = Variable(name, is_latent=False, is_observed=True)
        self.observed_vars.append(var)
    
    def add_latent_variable(self, name: str, indicators: Optional[List[str]] = None):
        """添加潜在变量"""
        var = Variable(name, is_latent=True, is_observed=False, indicators=indicators or [])
        self.latent_vars.append(var)
    
    def add_measurement(
        self,
        latent: str,
        indicator: str,
        loading: float = 1.0,
        is_fixed: bool = False,
        error_variance: float = 0.1
    ):
        """添加测量关系"""
        measurement = Measurement(latent, indicator, loading, is_fixed, error_variance)
        self.measurement_model[latent].append(measurement)
        
        # 确保指标变量被记录
        if indicator not in [v.name for v in self.observed_vars]:
            self.add_observed_variable(indicator)
    
    def add_path(
        self,
        from_var: str,
        to_var: str,
        coefficient: float = 0.0,
        is_fixed: bool = False,
        fixed_value: Optional[float] = None
    ):
        """添加结构路径"""
        path = Path(from_var, to_var, coefficient, is_fixed, fixed_value)
        self.structural_model.append(path)
    
    def fit(self, data: np.ndarray, var_names: List[str], method: str = "ml"):
        """
        拟合模型
        
        Args:
            data: 观测数据 [n_samples, n_observed]
            var_names: 变量名称列表
            method: 估计方法 ("ml": 最大似然, "gls": 广义最小二乘)
        """
        self.observed_names = var_names
        self.data = data
        self.n_samples = len(data)
        
        # 构建参数向量
        initial_params = self._get_initial_params()
        
        # 定义目标函数
        def objective(params):
            return self._objective_function(params, data, method)
        
        # 优化
        result = minimize(
            objective,
            initial_params,
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if result.success:
            self._update_params(result.x)
            self.fitted = True
            self._compute_fit_statistics(data)
        else:
            warnings.warn(f"Optimization failed: {result.message}")
        
        return result.success
    
    def _get_initial_params(self) -> np.ndarray:
        """获取初始参数值"""
        params = []
        
        # 测量载荷
        for latent, measurements in self.measurement_model.items():
            for m in measurements:
                if not m.is_fixed:
                    params.append(m.loading)
        
        # 结构系数
        for path in self.structural_model:
            if not path.is_fixed:
                params.append(path.coefficient)
        
        return np.array(params)
    
    def _update_params(self, params: np.ndarray):
        """更新参数值"""
        idx = 0
        
        # 更新测量载荷
        for latent, measurements in self.measurement_model.items():
            for m in measurements:
                if not m.is_fixed:
                    m.loading = params[idx]
                    idx += 1
        
        # 更新结构系数
        for path in self.structural_model:
            if not path.is_fixed:
                path.coefficient = params[idx]
                idx += 1
    
    def _objective_function(
        self,
        params: np.ndarray,
        data: np.ndarray,
        method: str
    ) -> float:
        """目标函数"""
        self._update_params(params)
        
        # 计算模型隐含的协方差矩阵
        sigma = self._implied_covariance()
        
        # 样本协方差矩阵
        S = np.cov(data.T)
        
        if method == "ml":
            # 最大似然
            try:
                sigma_inv = inv(sigma)
                log_det_sigma = np.log(np.linalg.det(sigma))
                log_det_S = np.log(np.linalg.det(S))
                
                f_ml = log_det_sigma - log_det_S + np.trace(S @ sigma_inv) - len(S)
                return f_ml
            except:
                return 1e10
        
        elif method == "gls":
            # 广义最小二乘
            try:
                S_inv = inv(S)
                diff = S - sigma
                f_gls = 0.5 * np.trace(diff @ S_inv @ diff @ S_inv)
                return f_gls
            except:
                return 1e10
        
        else:
            # 未加权最小二乘
            diff = S - sigma
            return np.sum(diff ** 2)
    
    def _implied_covariance(self) -> np.ndarray:
        """计算模型隐含的协方差矩阵"""
        n_obs = len(self.observed_names)
        
        # 构建结构矩阵
        B = self._build_structural_matrix()
        Lambda = self._build_loading_matrix()
        Theta = self._build_error_matrix()
        
        try:
            # Σ = Λ (I - B)^(-1) Ψ (I - B)^(-T) Λ^T + Θ
            I_minus_B_inv = inv(np.eye(len(B)) - B)
            
            # 假设潜在变量的协方差为单位矩阵（或估计的Ψ）
            Psi = np.eye(len(self.latent_vars))
            
            sigma = Lambda @ I_minus_B_inv @ Psi @ I_minus_B_inv.T @ Lambda.T + Theta
            
            # 确保正定性
            sigma = (sigma + sigma.T) / 2
            min_eig = np.min(np.linalg.eigvalsh(sigma))
            if min_eig < 0:
                sigma += (-min_eig + 1e-6) * np.eye(len(sigma))
            
            return sigma
        except:
            return np.eye(n_obs)
    
    def _build_structural_matrix(self) -> np.ndarray:
        """构建结构系数矩阵B"""
        n_vars = len(self.latent_vars)
        var_names = [v.name for v in self.latent_vars]
        
        B = np.zeros((n_vars, n_vars))
        for path in self.structural_model:
            if path.from_var in var_names and path.to_var in var_names:
                i = var_names.index(path.to_var)
                j = var_names.index(path.from_var)
                B[i, j] = path.coefficient
        
        return B
    
    def _build_loading_matrix(self) -> np.ndarray:
        """构建因子载荷矩阵Λ"""
        n_obs = len(self.observed_names)
        n_latent = len(self.latent_vars)
        
        Lambda = np.zeros((n_obs, n_latent))
        
        for latent_name, measurements in self.measurement_model.items():
            if latent_name in [v.name for v in self.latent_vars]:
                j = [v.name for v in self.latent_vars].index(latent_name)
                
                for m in measurements:
                    if m.indicator in self.observed_names:
                        i = self.observed_names.index(m.indicator)
                        Lambda[i, j] = m.loading
        
        return Lambda
    
    def _build_error_matrix(self) -> np.ndarray:
        """构建误差协方差矩阵Θ"""
        n_obs = len(self.observed_names)
        Theta = np.eye(n_obs) * 0.1  # 假设较小的误差
        
        for latent_name, measurements in self.measurement_model.items():
            for m in measurements:
                if m.indicator in self.observed_names:
                    i = self.observed_names.index(m.indicator)
                    Theta[i, i] = m.error_variance
        
        return Theta
    
    def _compute_fit_statistics(self, data: np.ndarray):
        """计算拟合统计量"""
        S = np.cov(data.T)
        sigma = self._implied_covariance()
        
        # 卡方统计量
        n = len(data)
        p = len(self.observed_names)
        chi_sq = (n - 1) * self._objective_function(self._get_params_array(), data, "ml")
        
        # 自由度
        df = p * (p + 1) // 2 - len(self._get_params_array())
        
        # RMSEA
        if df > 0:
            rmsea = np.sqrt(max(chi_sq - df, 0) / (df * (n - 1)))
        else:
            rmsea = 0
        
        # CFI, TLI等需要更复杂的计算，这里简化
        self.fit_statistics = {
            "chi_square": chi_sq,
            "df": df,
            "rmsea": rmsea,
            "n_observed": n,
            "n_params": len(self._get_params_array())
        }
    
    def _get_params_array(self) -> np.ndarray:
        """获取参数数组"""
        params = []
        
        for measurements in self.measurement_model.values():
            for m in measurements:
                if not m.is_fixed:
                    params.append(m.loading)
        
        for path in self.structural_model:
            if not path.is_fixed:
                params.append(path.coefficient)
        
        return np.array(params)
    
    def predict(self, data: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        预测潜在变量得分
        
        Args:
            data: 观测数据，None则使用拟合数据
        
        Returns:
            潜在变量得分
        """
        if not self.fitted:
            raise ValueError("Model not fitted yet")
        
        if data is None:
            data = self.data
        
        # 构建载荷矩阵
        Lambda = self._build_loading_matrix()
        
        # 使用回归方法估计因子得分
        # F = (Λ'Λ)^(-1) Λ' X
        try:
            Lambda_pinv = inv(Lambda.T @ Lambda) @ Lambda.T
            factor_scores = data @ Lambda_pinv.T
            
            latent_names = [v.name for v in self.latent_vars]
            return {
                name: factor_scores[:, i]
                for i, name in enumerate(latent_names)
            }
        except:
            return {}
    
    def get_path_coefficients(self) -> Dict[str, float]:
        """获取路径系数"""
        return {
            f"{p.from_var} -> {p.to_var}": p.coefficient
            for p in self.structural_model
        }
    
    def get_loadings(self) -> Dict[str, Dict[str, float]]:
        """获取因子载荷"""
        loadings = {}
        for latent, measurements in self.measurement_model.items():
            loadings[latent] = {
                m.indicator: m.loading
                for m in measurements
            }
        return loadings
    
    def summary(self) -> str:
        """模型摘要"""
        lines = []
        lines.append("=" * 60)
        lines.append("结构方程模型摘要")
        lines.append("=" * 60)
        
        lines.append(f"\n观测变量数: {len(self.observed_vars)}")
        lines.append(f"潜在变量数: {len(self.latent_vars)}")
        
        lines.append("\n测量模型:")
        for latent, measurements in self.measurement_model.items():
            lines.append(f"  {latent}:")
            for m in measurements:
                lines.append(f"    {m.indicator}: {m.loading:.3f}")
        
        lines.append("\n结构模型:")
        for path in self.structural_model:
            lines.append(f"  {path.from_var} -> {path.to_var}: {path.coefficient:.3f}")
        
        if self.fitted:
            lines.append("\n拟合统计量:")
            for stat, value in self.fit_statistics.items():
                if isinstance(value, float):
                    lines.append(f"  {stat}: {value:.4f}")
                else:
                    lines.append(f"  {stat}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class MediationAnalysis:
    """
    中介分析
    分析中介效应
    """
    
    def __init__(self, sem: StructuralEquationModel):
        self.sem = sem
    
    def analyze(
        self,
        independent: str,
        mediator: str,
        dependent: str
    ) -> Dict[str, float]:
        """
        分析中介效应
        
        Args:
            independent: 自变量
            mediator: 中介变量
            dependent: 因变量
        
        Returns:
            中介分析结果
        """
        # 获取路径系数
        a = self._get_path_coefficient(independent, mediator)
        b = self._get_path_coefficient(mediator, dependent)
        c = self._get_path_coefficient(independent, dependent)
        
        # 间接效应 (a * b)
        indirect_effect = a * b
        
        # 直接效应 (c')
        direct_effect = c
        
        # 总效应
        total_effect = direct_effect + indirect_effect
        
        # 中介比例
        if total_effect != 0:
            mediation_ratio = indirect_effect / total_effect
        else:
            mediation_ratio = 0
        
        return {
            "a": a,
            "b": b,
            "c_prime": direct_effect,
            "indirect_effect": indirect_effect,
            "total_effect": total_effect,
            "mediation_ratio": mediation_ratio
        }
    
    def _get_path_coefficient(self, from_var: str, to_var: str) -> float:
        """获取路径系数"""
        for path in self.sem.structural_model:
            if path.from_var == from_var and path.to_var == to_var:
                return path.coefficient
        return 0.0


def demo():
    """演示结构方程模型"""
    print("=" * 60)
    print("结构方程模型演示")
    print("=" * 60)
    
    # 创建SEM模型
    sem = StructuralEquationModel()
    
    # 定义潜在变量
    sem.add_latent_variable("Performance", ["efficiency", "durability", "capacity"])
    sem.add_latent_variable("Structure", ["crystallinity", "porosity"])
    sem.add_latent_variable("Process", ["temperature", "time", "pressure"])
    
    # 添加测量模型
    for indicator in ["efficiency", "durability", "capacity"]:
        sem.add_measurement("Performance", indicator, loading=1.0 if indicator == "efficiency" else 0.8)
    
    for indicator in ["crystallinity", "porosity"]:
        sem.add_measurement("Structure", indicator, loading=1.0 if indicator == "crystallinity" else 0.9)
    
    for indicator in ["temperature", "time", "pressure"]:
        sem.add_measurement("Process", indicator, loading=1.0 if indicator == "temperature" else 0.85)
    
    # 添加结构模型
    sem.add_path("Process", "Structure", coefficient=0.7)
    sem.add_path("Structure", "Performance", coefficient=0.6)
    sem.add_path("Process", "Performance", coefficient=0.3)
    
    print("\n模型定义完成:")
    print(f"  观测变量: 8个")
    print(f"  潜在变量: 3个 (Performance, Structure, Process)")
    print(f"  结构路径: 3条")
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 500
    
    # 生成潜在变量
    Process_latent = np.random.randn(n_samples)
    Structure_latent = 0.7 * Process_latent + 0.5 * np.random.randn(n_samples)
    Performance_latent = 0.6 * Structure_latent + 0.3 * Process_latent + 0.4 * np.random.randn(n_samples)
    
    # 生成观测变量
    temperature = Process_latent + 0.3 * np.random.randn(n_samples)
    time = 0.85 * Process_latent + 0.4 * np.random.randn(n_samples)
    pressure = 0.8 * Process_latent + 0.35 * np.random.randn(n_samples)
    
    crystallinity = Structure_latent + 0.3 * np.random.randn(n_samples)
    porosity = 0.9 * Structure_latent + 0.4 * np.random.randn(n_samples)
    
    efficiency = Performance_latent + 0.3 * np.random.randn(n_samples)
    durability = 0.8 * Performance_latent + 0.4 * np.random.randn(n_samples)
    capacity = 0.85 * Performance_latent + 0.35 * np.random.randn(n_samples)
    
    data = np.column_stack([
        temperature, time, pressure,
        crystallinity, porosity,
        efficiency, durability, capacity
    ])
    
    var_names = [
        "temperature", "time", "pressure",
        "crystallinity", "porosity",
        "efficiency", "durability", "capacity"
    ]
    
    print(f"\n数据形状: {data.shape}")
    
    # 拟合模型
    print("\n拟合模型...")
    success = sem.fit(data, var_names, method="ml")
    
    if success:
        print("模型拟合成功!")
        print(sem.summary())
        
        # 中介分析
        print("\n中介分析:")
        mediation = MediationAnalysis(sem)
        result = mediation.analyze("Process", "Structure", "Performance")
        print(f"  间接效应 (Process -> Structure -> Performance): {result['indirect_effect']:.3f}")
        print(f"  直接效应 (Process -> Performance): {result['c_prime']:.3f}")
        print(f"  总效应: {result['total_effect']:.3f}")
        print(f"  中介比例: {result['mediation_ratio']:.1%}")
        
        # 预测潜在变量
        latent_scores = sem.predict()
        if latent_scores:
            print("\n潜在变量得分示例 (前5个样本):")
            for name, scores in latent_scores.items():
                print(f"  {name}: {scores[:5]}")
    else:
        print("模型拟合失败")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
