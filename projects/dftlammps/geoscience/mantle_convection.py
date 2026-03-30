"""
地幔对流模拟模块
===============
地球地幔对流数值模拟，包括：
- 热对流方程求解
- 板块俯冲动力学
- 地幔柱上升
- 热演化历史

应用场景：
- 地球热历史重建
- 板块构造模拟
- 地幔不均一性研究
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundaryCondition(Enum):
    """边界条件类型"""
    FREE_SLIP = "free_slip"
    NO_SLIP = "no_slip"
    ISOTHERMAL = "isothermal"
    INSULATING = "insulating"
    STRESS_FREE = "stress_free"


class RheologyLaw(Enum):
    """流变学定律"""
    NEWTONIAN = "newtonian"
    POWER_LAW = "power_law"
    ARRHENIUS = "arrhenius"
    VISCOPLASTIC = "viscoplastic"
    COMPOSITE = "composite"


class ConvectionMode(Enum):
    """对流模式"""
    STEADY_STATE = "steady_state"
    TIME_DEPENDENT = "time_dependent"
    OSCILLATORY = "oscillatory"
    CHAOTIC = "chaotic"


@dataclass
class PhysicalParameters:
    """物理参数"""
    # 热学参数
    thermal_diffusivity: float = 1e-6  # m^2/s
    thermal_expansion: float = 3e-5    # 1/K
    heat_capacity: float = 1000        # J/(kg·K)
    thermal_conductivity: float = 4.0  # W/(m·K)
    
    # 力学参数
    reference_viscosity: float = 1e21  # Pa·s
    density: float = 3300              # kg/m^3
    gravity: float = 9.8               # m/s^2
    
    # 几何参数
    domain_depth: float = 2890e3       # m (地幔厚度)
    domain_width: float = 10000e3      # m
    
    # 加热参数
    internal_heating: float = 0.0      # W/m^3
    bottom_temperature: float = 3000   # K
    top_temperature: float = 300       # K


@dataclass
class RheologyParameters:
    """流变学参数"""
    law: RheologyLaw
    activation_energy: float = 300e3   # J/mol
    activation_volume: float = 10e-6   # m^3/mol
    preexponential: float = 1e-15      # 1/(Pa^n·s)
    stress_exponent: float = 3.0       # n
    reference_stress: float = 1e5      # Pa
    
    # 粘塑性参数
    yield_stress: float = 100e6        # Pa
    cohesion: float = 10e6             # Pa
    friction_coefficient: float = 0.6


@dataclass
class ConvectionState:
    """对流状态"""
    temperature: np.ndarray
    velocity: Tuple[np.ndarray, np.ndarray]  # (vx, vz)
    pressure: np.ndarray
    viscosity: np.ndarray
    time: float


class MantleConvectionSolver:
    """地幔对流求解器"""
    
    def __init__(self,
                 physical_params: PhysicalParameters = None,
                 rheology_params: RheologyParameters = None,
                 nx: int = 100,
                 nz: int = 50):
        self.phys = physical_params or PhysicalParameters()
        self.rheo = rheology_params or RheologyParameters(RheologyLaw.NEWTONIAN)
        self.nx = nx
        self.nz = nz
        
        # 网格
        self.dx = self.phys.domain_width / nx
        self.dz = self.phys.domain_depth / nz
        
        # 无量纲参数
        self._calculate_dimensionless_numbers()
        
        logger.info(f"初始化地幔对流求解器: {nx}x{nz} 网格")
    
    def _calculate_dimensionless_numbers(self):
        """计算无量纲数"""
        # Rayleigh数
        self.Ra = (self.phys.density * self.phys.gravity * 
                  self.phys.thermal_expansion * 
                  (self.phys.bottom_temperature - self.phys.top_temperature) *
                  self.phys.domain_depth**3 / 
                  (self.phys.thermal_diffusivity * self.phys.reference_viscosity / self.phys.density))
        
        # Prandtl数 (假设很大，粘性主导)
        self.Pr = self.phys.reference_viscosity / (self.phys.density * self.phys.thermal_diffusivity)
        
        # 内部加热参数
        self.H = (self.phys.internal_heating * self.phys.domain_depth**2 / 
                 (self.phys.thermal_conductivity * 
                  (self.phys.bottom_temperature - self.phys.top_temperature)))
        
        logger.info(f"Rayleigh数: {self.Ra:.2e}")
        logger.info(f"Prandtl数: {self.Pr:.2e}")
        logger.info(f"内部加热参数: {self.H:.4f}")
    
    def initialize_temperature(self, mode: str = "linear") -> np.ndarray:
        """
        初始化温度场
        
        Args:
            mode: 初始模式 (linear, conductive, adiabatic, perturbed)
        
        Returns:
            温度场数组
        """
        T = np.zeros((self.nx, self.nz))
        
        if mode == "linear":
            # 线性温度梯度
            for j in range(self.nz):
                T[:, j] = self.phys.top_temperature + \
                         (self.phys.bottom_temperature - self.phys.top_temperature) * j / (self.nz - 1)
        
        elif mode == "conductive":
            # 纯传导解
            for j in range(self.nz):
                z = j * self.dz
                T[:, j] = self.phys.top_temperature + \
                         (self.phys.bottom_temperature - self.phys.top_temperature) * \
                         (z / self.phys.domain_depth + 0.5 * self.H * (z/self.phys.domain_depth) * 
                          (1 - z/self.phys.domain_depth))
        
        elif mode == "perturbed":
            # 有扰动的线性
            for j in range(self.nz):
                T[:, j] = self.phys.top_temperature + \
                         (self.phys.bottom_temperature - self.phys.top_temperature) * j / (self.nz - 1)
            # 添加扰动
            T += 50 * np.random.randn(self.nx, self.nz)
        
        # 边界条件
        T[:, 0] = self.phys.top_temperature
        T[:, -1] = self.phys.bottom_temperature
        
        return T
    
    def calculate_viscosity(self,
                           temperature: np.ndarray,
                           pressure: Optional[np.ndarray] = None,
                           strain_rate: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算粘度场
        
        根据选择的流变学定律计算粘度
        """
        eta = np.ones_like(temperature) * self.phys.reference_viscosity
        
        if self.rheo.law == RheologyLaw.ARRHENIUS:
            # Arrhenius型温度依赖
            # η = η0 * exp(E/RT)
            R = 8.314  # J/(mol·K)
            T_dim = temperature
            
            activation_term = np.exp(
                self.rheo.activation_energy / (R * T_dim)
            )
            
            # 归一化到参考粘度
            eta_ref_factor = np.exp(
                -self.rheo.activation_energy / (R * 1600)
            )
            
            eta = self.phys.reference_viscosity * activation_term * eta_ref_factor
        
        elif self.rheo.law == RheologyLaw.POWER_LAW:
            # 幂律流变
            if strain_rate is not None:
                eta = self.rheo.preexponential ** (-1/self.rheo.stress_exponent) * \
                     (strain_rate + 1e-20) ** ((1 - self.rheo.stress_exponent) / self.rheo.stress_exponent)
        
        elif self.rheo.law == RheologyLaw.VISCOPLASTIC:
            # 粘塑性
            if strain_rate is not None:
                # 粘性分支
                eta_viscous = eta
                
                # 塑性屈服
                eta_plastic = self.rheo.yield_stress / (2 * strain_rate + 1e-20)
                
                # 取最小值
                eta = np.minimum(eta_viscous, eta_plastic)
        
        elif self.rheo.law == RheologyLaw.COMPOSITE:
            # 复合流变 (温度+应力依赖)
            R = 8.314
            temp_factor = np.exp(self.rheo.activation_energy / (R * temperature) - 
                                self.rheo.activation_energy / (R * 1600))
            
            if strain_rate is not None:
                stress_factor = (strain_rate / 1e-15) ** ((1 - self.rheo.stress_exponent) / self.rheo.stress_exponent)
                eta = self.phys.reference_viscosity * temp_factor * stress_factor
            else:
                eta = self.phys.reference_viscosity * temp_factor
        
        # 粘度截断
        eta = np.clip(eta, 1e18, 1e25)
        
        return eta
    
    def solve_stokes_equation(self,
                             viscosity: np.ndarray,
                             temperature: np.ndarray,
                             max_iter: int = 1000) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        求解Stokes方程 (简化的2D求解)
        
        ∇·(2η ε̇) - ∇p = Ra T ẑ
        ∇·v = 0
        
        Returns:
            ((vx, vz), pressure)
        """
        # 简化的求解：使用流函数-涡度法或惩罚法
        vx = np.zeros((self.nx, self.nz))
        vz = np.zeros((self.nx, self.nz))
        pressure = np.zeros((self.nx, self.nz))
        
        # 浮力驱动速度估计
        # v ~ Ra * ΔT * η / (ρ g L)
        buoyancy = (self.phys.density * self.phys.gravity * 
                   self.phys.thermal_expansion * 
                   (temperature - self.phys.top_temperature))
        
        # 简化的速度场 (垂直速度)
        vz = -buoyancy * self.phys.domain_depth**2 / \
             (np.mean(viscosity) * 100) * 1e-15  # 缩放
        
        # 水平速度 (简化)
        for i in range(1, self.nx-1):
            for j in range(1, self.nz-1):
                vx[i, j] = -(vz[i+1, j] - vz[i-1, j]) / (2 * self.dx) * self.dz
        
        # 边界条件
        vx[:, 0] = 0  # 无穿透
        vx[:, -1] = 0
        vz[:, 0] = 0
        vz[:, -1] = 0
        vx[0, :] = vx[-1, :]  # 周期性
        vx[-1, :] = vx[1, :]
        
        return (vx, vz), pressure
    
    def solve_energy_equation(self,
                             temperature: np.ndarray,
                             velocity: Tuple[np.ndarray, np.ndarray],
                             dt: float) -> np.ndarray:
        """
        求解能量方程
        
        ∂T/∂t + v·∇T = ∇²T + H
        
        Args:
            temperature: 当前温度场
            velocity: (vx, vz) 速度场
            dt: 时间步长
        
        Returns:
            新温度场
        """
        vx, vz = velocity
        T_new = temperature.copy()
        
        # 无量纲参数
        dx_nondim = self.dx / self.phys.domain_depth
        dz_nondim = self.dz / self.phys.domain_depth
        
        for i in range(1, self.nx-1):
            for j in range(1, self.nz-1):
                # 扩散项
                d2T_dx2 = (temperature[i+1, j] - 2*temperature[i, j] + temperature[i-1, j]) / dx_nondim**2
                d2T_dz2 = (temperature[i, j+1] - 2*temperature[i, j] + temperature[i, j-1]) / dz_nondim**2
                
                # 对流项 (迎风格式)
                dT_dx = (temperature[i, j] - temperature[i-1, j]) / dx_nondim if vx[i, j] > 0 else \
                        (temperature[i+1, j] - temperature[i, j]) / dx_nondim
                dT_dz = (temperature[i, j] - temperature[i, j-1]) / dz_nondim if vz[i, j] > 0 else \
                        (temperature[i, j+1] - temperature[i, j]) / dz_nondim
                
                # 时间推进
                T_new[i, j] = temperature[i, j] + dt * (
                    (d2T_dx2 + d2T_dz2) / self.Ra -  # 扩散
                    (vx[i, j] * dT_dx + vz[i, j] * dT_dz) +  # 对流
                    self.H  # 内部加热
                )
        
        # 边界条件
        T_new[:, 0] = self.phys.top_temperature
        T_new[:, -1] = self.phys.bottom_temperature
        T_new[0, :] = T_new[-1, :]  # 周期性
        T_new[-1, :] = T_new[1, :]
        
        return T_new
    
    def time_stepping(self,
                     initial_state: ConvectionState,
                     total_time: float,
                     dt: float = None) -> List[ConvectionState]:
        """
        时间推进
        
        Args:
            initial_state: 初始状态
            total_time: 总模拟时间
            dt: 时间步长 (自动计算如果为None)
        
        Returns:
            状态历史
        """
        if dt is None:
            dt = 0.1 * min(self.dx, self.dz)**2 / self.phys.thermal_diffusivity
            dt = min(dt, total_time / 1000)  # 至少1000步
        
        logger.info(f"开始时间推进: 总时间={total_time:.2e}s, 时间步长={dt:.2e}s")
        
        state = initial_state
        history = [state]
        n_steps = int(total_time / dt)
        
        for step in range(n_steps):
            # 求解Stokes方程
            velocity, pressure = self.solve_stokes_equation(
                state.viscosity, state.temperature
            )
            
            # 求解能量方程
            new_temperature = self.solve_energy_equation(
                state.temperature, velocity, dt
            )
            
            # 更新粘度
            new_viscosity = self.calculate_viscosity(new_temperature)
            
            state = ConvectionState(
                temperature=new_temperature,
                velocity=velocity,
                pressure=pressure,
                viscosity=new_viscosity,
                time=state.time + dt
            )
            
            if step % max(1, n_steps // 10) == 0:
                logger.info(f"步 {step}/{n_steps}, t={state.time:.2e}s")
                history.append(state)
        
        return history
    
    def calculate_nusselt_number(self,
                                temperature: np.ndarray,
                                velocity: Tuple[np.ndarray, np.ndarray]) -> float:
        """
        计算Nusselt数 (对流热传输效率)
        
        Nu = (对流+传导热流) / 纯传导热流
        """
        # 顶部热流
        vx, vz = velocity
        
        # 传导热流
        dT_dz = (temperature[:, 1] - temperature[:, 0]) / self.dz
        q_conduction = -self.phys.thermal_conductivity * np.mean(dT_dz)
        
        # 对流热流
        q_convection = self.phys.density * self.phys.heat_capacity * np.mean(
            vz[:, 0] * temperature[:, 0]
        )
        
        # 纯传导热流
        q_pure_conduction = self.phys.thermal_conductivity * \
                          (self.phys.bottom_temperature - self.phys.top_temperature) / self.phys.domain_depth
        
        Nu = (q_conduction + q_convection) / q_pure_conduction
        
        return Nu
    
    def analyze_convection_pattern(self,
                                   velocity: Tuple[np.ndarray, np.ndarray]) -> Dict:
        """
        分析对流模式
        
        Returns:
            对流特征
        """
        vx, vz = velocity
        
        # 计算流函数
        stream_function = np.zeros((self.nx, self.nz))
        for j in range(1, self.nz):
            for i in range(self.nx):
                stream_function[i, j] = stream_function[i, j-1] + vx[i, j] * self.dz
        
        # 寻找对流单元
        max_sf = np.max(np.abs(stream_function))
        
        # 计算对流速度
        velocity_magnitude = np.sqrt(vx**2 + vz**2)
        v_rms = np.sqrt(np.mean(velocity_magnitude**2))
        
        # Peclet数
        Pe = v_rms * self.phys.domain_depth / self.phys.thermal_diffusivity
        
        return {
            'stream_function_max': max_sf,
            'rms_velocity': v_rms,
            'peclet_number': Pe,
            'convection_mode': self._classify_mode(Pe)
        }
    
    def _classify_mode(self, Pe: float) -> ConvectionMode:
        """分类对流模式"""
        if Pe < 10:
            return ConvectionMode.STEADY_STATE
        elif Pe < 100:
            return ConvectionMode.TIME_DEPENDENT
        elif Pe < 1000:
            return ConvectionMode.OSCILLATORY
        else:
            return ConvectionMode.CHAOTIC


class SubductionSimulator:
    """俯冲带模拟器"""
    
    def __init__(self, convection_solver: MantleConvectionSolver):
        self.solver = convection_solver
    
    def simulate_slab_descent(self,
                             slab_age: float,  # Myr
                             subduction_angle: float,  # degrees
                             convergence_rate: float,  # cm/yr
                             simulation_depth: float = 660e3) -> Dict:
        """
        模拟板块俯冲
        
        Args:
            slab_age: 板片年龄
            subduction_angle: 俯冲角度
            convergence_rate: 汇聚速率
            simulation_depth: 模拟深度
        
        Returns:
            俯冲演化数据
        """
        logger.info(f"模拟板块俯冲: {convergence_rate} cm/yr, {subduction_angle}°")
        
        # 板片年龄对应的热结构
        thermal_thickness = np.sqrt(
            self.solver.phys.thermal_diffusivity * slab_age * 1e6 * 365.25 * 24 * 3600
        )
        
        # 俯冲轨迹
        angle_rad = np.radians(subduction_angle)
        
        depths = np.linspace(0, simulation_depth, 100)
        horizontal_distance = depths / np.tan(angle_rad)
        
        # 板片温度 (考虑摩擦加热和绝热加热)
        ambient_mantle_temp = 1600  # K
        slab_surface_temp = 273 + 100  # K (冷表面)
        
        slab_temperature = []
        for d in depths:
            # 绝热加热
            adiabatic_heating = d * 0.3 / 1000  # K
            
            # 板片核心温度
            core_temp = ambient_mantle_temp - (ambient_mantle_temp - slab_surface_temp) * \
                       np.exp(-d / thermal_thickness)
            
            slab_temperature.append(core_temp + adiabatic_heating)
        
        # 相变延迟
        olivine_wadsleyite = 410e3
        wadsleyite_ringwoodite = 520e3
        ringwoodite_perovskite = 660e3
        
        # Clapeyron斜率引起的相界偏移
        phase_boundaries = {
            '410': olivine_wadsleyite - 10e3 * slab_age / 100,  # 冷板片下移
            '520': wadsleyite_ringwoodite - 5e3 * slab_age / 100,
            '660': ringwoodite_perovskite + 20e3 * slab_age / 100  # 负斜率，冷板片上移
        }
        
        # 脱水反应
        dehydration_depths = [100e3, 300e3, 600e3]
        water_release = [2.0, 1.0, 0.5]  # wt%
        
        return {
            'depth': depths,
            'horizontal_position': horizontal_distance,
            'slab_temperature': np.array(slab_temperature),
            'phase_boundaries': phase_boundaries,
            'dehydration_depths': dehydration_depths,
            'water_release': water_release,
            'descent_time_Myr': simulation_depth / (convergence_rate * np.sin(angle_rad)) / 1e6
        }
    
    def calculate_slab_pull_force(self,
                                 slab_length: float,
                                 slab_thickness: float,
                                 density_contrast: float = 50) -> float:
        """
        计算板片拉力
        
        F = Δρ * g * h * L * sin(θ)
        """
        force = (density_contrast * self.solver.phys.gravity * 
                slab_thickness * slab_length)
        
        return force  # N/m


class MantlePlumeSimulator:
    """地幔柱模拟器"""
    
    def __init__(self, convection_solver: MantleConvectionSolver):
        self.solver = convection_solver
    
    def simulate_thermal_plume(self,
                              plume_temperature: float,
                              plume_radius: float,
                              source_depth: float = 2800e3,
                              simulation_time: float = 100e6 * 365.25 * 24 * 3600) -> Dict:
        """
        模拟热地幔柱上升
        
        Args:
            plume_temperature: 柱头温度
            plume_radius: 柱半径
            source_depth: 源区深度
            simulation_time: 模拟时间
        
        Returns:
            地幔柱演化
        """
        logger.info(f"模拟热地幔柱: T={plume_temperature}K, R={plume_radius/1000}km")
        
        # 柱头上升速度 (Stokes定律简化)
        delta_rho = (self.solver.phys.thermal_expansion * 
                    self.solver.phys.density * 
                    (plume_temperature - self.solver.phys.top_temperature))
        
        # 粘性阻力
        eta = self.solver.phys.reference_viscosity
        
        # 上升速度 (简化)
        ascent_velocity = (2/9) * delta_rho * self.solver.phys.gravity * plume_radius**2 / eta
        
        # 到达地表时间
        time_to_surface = source_depth / ascent_velocity
        
        # 柱头扩展
        # 半径随深度减小而增大
        depths = np.linspace(source_depth, 0, 100)
        head_radius = []
        
        for d in depths:
            # 绝热膨胀 + 粘性扩散
            expansion = 1 + 0.1 * (source_depth - d) / source_depth
            diffusion = np.sqrt(1 + 4 * self.solver.phys.thermal_diffusivity * 
                              (source_depth - d) / ascent_velocity / plume_radius**2)
            head_radius.append(plume_radius * expansion * diffusion)
        
        # 温度结构
        plume_temp_profile = []
        ambient_temp = 1600  # K
        
        for d in depths:
            # 绝热冷却
            adiabatic_temp = plume_temperature - d * 0.3 / 1000
            # 与环境混合
            mixing = (ambient_temp - adiabatic_temp) * (1 - np.exp(-(source_depth - d) / (5 * plume_radius)))
            plume_temp_profile.append(adiabatic_temp + mixing)
        
        # 熔融潜力
        melting_potential = []
        solidus = 1200  # K (简化)
        
        for temp in plume_temp_profile:
            if temp > solidus:
                melting_potential.append((temp - solidus) / 200)  # 熔融分数
            else:
                melting_potential.append(0)
        
        return {
            'depth': depths,
            'ascent_velocity': ascent_velocity,
            'time_to_surface_years': time_to_surface / (365.25 * 24 * 3600),
            'head_radius': np.array(head_radius),
            'plume_temperature': np.array(plume_temp_profile),
            'melting_potential': np.array(melting_potential),
            'total_melt_production': np.trapz(melting_potential, depths) * np.pi * plume_radius**2
        }


# 应用案例演示
def mantle_convection_application():
    """地幔对流应用示例"""
    logger.info("=" * 60)
    logger.info("地幔对流应用示例")
    logger.info("=" * 60)
    
    # 1. 创建物理参数
    phys_params = PhysicalParameters(
        thermal_diffusivity=1e-6,
        thermal_expansion=3e-5,
        reference_viscosity=1e21,
        domain_depth=2890e3,
        domain_width=10000e3,
        bottom_temperature=3000,
        top_temperature=300,
        internal_heating=0  # 无内部加热
    )
    
    # 2. 创建流变学参数
    rheo_params = RheologyParameters(
        law=RheologyLaw.ARRHENIUS,
        activation_energy=300e3,
        activation_volume=10e-6
    )
    
    # 3. 初始化求解器
    solver = MantleConvectionSolver(
        phys_params, rheo_params, nx=50, nz=30
    )
    
    # 4. 初始化温度场
    T_initial = solver.initialize_temperature(mode="perturbed")
    logger.info(f"初始温度场范围: {T_initial.min():.1f} - {T_initial.max():.1f} K")
    
    # 5. 计算粘度场
    eta = solver.calculate_viscosity(T_initial)
    logger.info(f"粘度范围: {eta.min():.2e} - {eta.max():.2e} Pa·s")
    
    # 6. 求解Stokes方程
    velocity, pressure = solver.solve_stokes_equation(eta, T_initial)
    vx, vz = velocity
    logger.info(f"速度场: vx范围 [{vx.min():.2e}, {vx.max():.2e}], vz范围 [{vz.min():.2e}, {vz.max():.2e}]")
    
    # 7. 分析对流模式
    pattern = solver.analyze_convection_pattern(velocity)
    logger.info(f"\n对流模式分析:")
    logger.info(f"  RMS速度: {pattern['rms_velocity']:.2e} m/s")
    logger.info(f"  Peclet数: {pattern['peclet_number']:.2e}")
    logger.info(f"  对流模式: {pattern['convection_mode'].value}")
    
    # 8. 计算Nusselt数
    Nu = solver.calculate_nusselt_number(T_initial, velocity)
    logger.info(f"  Nusselt数: {Nu:.3f}")
    
    # 9. 俯冲带模拟
    subduction = SubductionSimulator(solver)
    
    slab_sim = subduction.simulate_slab_descent(
        slab_age=100,  # Myr
        subduction_angle=45,
        convergence_rate=8,  # cm/yr
        simulation_depth=660e3
    )
    
    logger.info(f"\n俯冲带模拟:")
    logger.info(f"  俯冲深度: {slab_sim['depth'][-1]/1000:.0f} km")
    logger.info(f"  板片表面温度@660km: {slab_sim['slab_temperature'][-1]:.1f} K")
    logger.info(f"  相界偏移(410km): {slab_sim['phase_boundaries']['410']/1000:.1f} km")
    logger.info(f"  俯冲时间: {slab_sim['descent_time_Myr']:.2f} Myr")
    
    # 板片拉力
    slab_pull = subduction.calculate_slab_pull_force(
        slab_length=1000e3, slab_thickness=100e3
    )
    logger.info(f"  板片拉力: {slab_pull/1e12:.2f} TN/m")
    
    # 10. 地幔柱模拟
    plume = MantlePlumeSimulator(solver)
    
    plume_sim = plume.simulate_thermal_plume(
        plume_temperature=3500,
        plume_radius=100e3,
        source_depth=2800e3
    )
    
    logger.info(f"\n地幔柱模拟:")
    logger.info(f"  上升速度: {plume_sim['ascent_velocity']:.2e} m/s")
    logger.info(f"  到达地表时间: {plume_sim['time_to_surface_years']/1e6:.2f} Myr")
    logger.info(f"  柱头半径@地表: {plume_sim['head_radius'][-1]/1000:.0f} km")
    logger.info(f"  熔融潜力: {np.max(plume_sim['melting_potential']):.3f}")
    logger.info(f"  总熔体产量: {plume_sim['total_melt_production']:.2e} m³")
    
    return {
        'convection_pattern': pattern,
        'nusselt_number': Nu,
        'slab_simulation': slab_sim,
        'plume_simulation': plume_sim
    }


if __name__ == "__main__":
    mantle_convection_application()
