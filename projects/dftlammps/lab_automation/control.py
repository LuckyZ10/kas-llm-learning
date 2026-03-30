"""
Closed-Loop Feedback Control System for Laboratory Automation

Provides various control strategies:
- PID Controller (Proportional-Integral-Derivative)
- MPC (Model Predictive Control)
- Optimization-based control
- Adaptive control loops
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import deque
import numpy as np
from scipy.optimize import minimize


logger = logging.getLogger(__name__)


@dataclass
class ControlSignal:
    """Control signal output"""
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessVariable:
    """Process variable measurement"""
    name: str
    value: float
    setpoint: float
    timestamp: datetime = field(default_factory=datetime.now)
    unit: str = ""


@dataclass
class ControlConfig:
    """Controller configuration"""
    name: str
    sample_time_s: float = 1.0
    output_limits: Tuple[float, float] = (0.0, 100.0)
    auto_mode: bool = True
    enabled: bool = True


class FeedbackController(ABC):
    """Abstract base class for feedback controllers"""
    
    def __init__(self, config: ControlConfig):
        self.config = config
        self._output = 0.0
        self._history: deque = deque(maxlen=1000)
        self._error_history: deque = deque(maxlen=1000)
        self._setpoint = 0.0
        self._process_value = 0.0
        self._running = False
    
    @abstractmethod
    def compute(self, 
                setpoint: float,
                process_value: float,
                dt: float = 1.0) -> float:
        """Compute control output"""
        pass
    
    def update(self, 
               setpoint: float,
               process_value: float,
               dt: Optional[float] = None) -> ControlSignal:
        """Update controller with new measurement"""
        dt = dt or self.config.sample_time_s
        
        self._setpoint = setpoint
        self._process_value = process_value
        
        error = setpoint - process_value
        self._error_history.append({
            'timestamp': datetime.now(),
            'error': error,
            'setpoint': setpoint,
            'process_value': process_value
        })
        
        if not self.config.enabled or not self.config.auto_mode:
            return ControlSignal(value=self._output)
        
        output = self.compute(setpoint, process_value, dt)
        
        # Apply output limits
        output = np.clip(output, 
                        self.config.output_limits[0],
                        self.config.output_limits[1])
        
        self._output = output
        
        self._history.append({
            'timestamp': datetime.now(),
            'output': output,
            'error': error,
            'setpoint': setpoint,
            'process_value': process_value
        })
        
        return ControlSignal(
            value=output,
            metadata={
                'error': error,
                'setpoint': setpoint,
                'process_value': process_value
            }
        )
    
    def reset(self):
        """Reset controller state"""
        self._output = 0.0
        self._history.clear()
        self._error_history.clear()
    
    def get_history(self) -> List[Dict]:
        """Get control history"""
        return list(self._history)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate control performance metrics"""
        if len(self._error_history) < 10:
            return {}
        
        errors = [e['error'] for e in self._error_history]
        
        # Integral Absolute Error (IAE)
        iae = sum(abs(e) for e in errors) * self.config.sample_time_s
        
        # Integral Squared Error (ISE)
        ise = sum(e**2 for e in errors) * self.config.sample_time_s
        
        # Mean Absolute Error
        mae = np.mean(np.abs(errors))
        
        # Standard deviation
        std = np.std(errors)
        
        return {
            'IAE': iae,
            'ISE': ise,
            'MAE': mae,
            'std_error': std,
            'max_error': max(abs(e) for e in errors)
        }


class PIDController(FeedbackController):
    """
    PID (Proportional-Integral-Derivative) Controller
    
    Standard PID control with anti-windup and derivative filtering
    """
    
    def __init__(self, 
                 config: ControlConfig,
                 kp: float = 1.0,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 derivative_filter: float = 0.1):
        super().__init__(config)
        
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.derivative_filter = derivative_filter
        
        # Internal state
        self._integral = 0.0
        self._last_error = 0.0
        self._derivative = 0.0
        
        # Anti-windup
        self._integral_limits = (-100.0, 100.0)
    
    def compute(self, 
                setpoint: float,
                process_value: float,
                dt: float = 1.0) -> float:
        """Compute PID output"""
        error = setpoint - process_value
        
        # Proportional term
        proportional = self.kp * error
        
        # Integral term with anti-windup
        self._integral += error * dt
        self._integral = np.clip(self._integral, 
                                self._integral_limits[0],
                                self._integral_limits[1])
        integral = self.ki * self._integral
        
        # Derivative term with filtering
        raw_derivative = (error - self._last_error) / dt
        self._derivative = (self.derivative_filter * self._derivative + 
                           (1 - self.derivative_filter) * raw_derivative)
        derivative = self.kd * self._derivative
        
        self._last_error = error
        
        return proportional + integral + derivative
    
    def reset(self):
        """Reset PID controller"""
        super().reset()
        self._integral = 0.0
        self._last_error = 0.0
        self._derivative = 0.0
    
    def tune_ziegler_nichols(self, 
                            ku: float,
                            tu: float) -> Dict[str, float]:
        """
        Auto-tune using Ziegler-Nichols method
        ku: Ultimate gain
        tu: Ultimate period
        """
        self.kp = 0.6 * ku
        self.ki = 2 * self.kp / tu
        self.kd = self.kp * tu / 8
        
        return {
            'Kp': self.kp,
            'Ki': self.ki,
            'Kd': self.kd
        }
    
    def auto_tune(self, 
                  process_model: Callable,
                  setpoint: float) -> Dict[str, float]:
        """
        Auto-tune PID parameters using optimization
        """
        def objective(params):
            kp, ki, kd = params
            self.kp = kp
            self.ki = ki
            self.kd = kd
            
            # Simulate response
            error_sum = 0
            pv = 0
            for t in range(100):
                control = self.compute(setpoint, pv, 0.1)
                pv = process_model(pv, control, 0.1)
                error_sum += abs(setpoint - pv)
            
            return error_sum
        
        result = minimize(
            objective,
            x0=[self.kp, self.ki, self.kd],
            method='L-BFGS-B',
            bounds=[(0, 100), (0, 100), (0, 100)]
        )
        
        if result.success:
            self.kp, self.ki, self.kd = result.x
        
        return {
            'Kp': self.kp,
            'Ki': self.ki,
            'Kd': self.kd,
            'success': result.success
        }


class MPCController(FeedbackController):
    """
    Model Predictive Controller
    
    Uses a process model to predict future behavior and optimize control actions
    """
    
    def __init__(self,
                 config: ControlConfig,
                 horizon: int = 10,
                 process_model: Optional[Callable] = None):
        super().__init__(config)
        
        self.horizon = horizon
        self.process_model = process_model or self._default_model
        
        # MPC weights
        self.q = 1.0  # Output weight
        self.r = 0.1  # Control weight
        self.s = 0.0  # Control change weight
        
        # Prediction and control horizons
        self.prediction_horizon = horizon
        self.control_horizon = horizon // 2
    
    def _default_model(self, 
                      state: float, 
                      control: float, 
                      dt: float) -> float:
        """Default first-order process model"""
        tau = 10.0  # Time constant
        k = 1.0     # Gain
        
        dstate = (k * control - state) / tau
        return state + dstate * dt
    
    def compute(self,
                setpoint: float,
                process_value: float,
                dt: float = 1.0) -> float:
        """Compute MPC output using optimization"""
        
        def objective(controls):
            # Simulate over prediction horizon
            total_cost = 0
            state = process_value
            prev_control = self._output
            
            for i in range(self.prediction_horizon):
                control = controls[min(i, len(controls) - 1)]
                
                # Prediction step
                state = self.process_model(state, control, dt)
                
                # Cost function
                error = setpoint - state
                total_cost += self.q * error**2
                total_cost += self.r * control**2
                total_cost += self.s * (control - prev_control)**2
                
                prev_control = control
            
            return total_cost
        
        # Optimize control sequence
        result = minimize(
            objective,
            x0=np.full(self.control_horizon, self._output),
            method='SLSQP',
            bounds=[self.config.output_limits] * self.control_horizon
        )
        
        if result.success:
            return result.x[0]  # Return first control action
        else:
            logger.warning("MPC optimization failed, using previous output")
            return self._output


class ControlLoop:
    """
    Main control loop that manages multiple controllers
    and interfaces with equipment
    """
    
    def __init__(self, name: str):
        self.name = name
        self.controllers: Dict[str, FeedbackController] = {}
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Callbacks
        self._on_measurement: Optional[Callable] = None
        self._on_control: Optional[Callable] = None
    
    def add_controller(self, 
                      name: str,
                      controller: FeedbackController) -> bool:
        """Add controller to loop"""
        self.controllers[name] = controller
        return True
    
    def remove_controller(self, name: str) -> bool:
        """Remove controller from loop"""
        if name in self.controllers:
            del self.controllers[name]
            return True
        return False
    
    def set_callbacks(self,
                     on_measurement: Optional[Callable] = None,
                     on_control: Optional[Callable] = None):
        """Set measurement and control callbacks"""
        self._on_measurement = on_measurement
        self._on_control = on_control
    
    async def run(self,
                 get_measurement: Callable[[], float],
                 set_output: Callable[[float], bool],
                 setpoint: float,
                 controller_name: str = "main"):
        """
        Run control loop
        
        Args:
            get_measurement: Function to get process measurement
            set_output: Function to apply control output
            setpoint: Target setpoint
            controller_name: Name of controller to use
        """
        if controller_name not in self.controllers:
            raise ValueError(f"Controller {controller_name} not found")
        
        controller = self.controllers[controller_name]
        self._running = True
        
        logger.info(f"Starting control loop '{self.name}' with controller '{controller_name}'")
        
        while self._running:
            try:
                # Get measurement
                measurement = get_measurement()
                
                if self._on_measurement:
                    self._on_measurement(measurement)
                
                # Compute control
                signal = controller.update(setpoint, measurement)
                
                # Apply control
                set_output(signal.value)
                
                if self._on_control:
                    self._on_control(signal.value)
                
                # Wait for next cycle
                await asyncio.sleep(controller.config.sample_time_s)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(1.0)
    
    def stop(self):
        """Stop control loop"""
        self._running = False
        logger.info(f"Control loop '{self.name}' stopped")
    
    def get_loop_status(self) -> Dict[str, Any]:
        """Get status of all controllers"""
        return {
            'name': self.name,
            'running': self._running,
            'controllers': {
                name: {
                    'enabled': ctrl.config.enabled,
                    'auto_mode': ctrl.config.auto_mode,
                    'output': ctrl._output,
                    'setpoint': ctrl._setpoint,
                    'process_value': ctrl._process_value,
                    'metrics': ctrl.get_performance_metrics()
                }
                for name, ctrl in self.controllers.items()
            }
        }


class OptimizationEngine:
    """
    Optimization engine for process optimization
    Uses various optimization algorithms to find optimal process parameters
    """
    
    def __init__(self):
        self.objective_history: List[Dict] = []
        self.parameter_history: List[Dict] = []
    
    def optimize(self,
                objective_function: Callable[[np.ndarray], float],
                initial_params: np.ndarray,
                bounds: List[Tuple[float, float]],
                method: str = 'L-BFGS-B',
                max_iter: int = 100) -> Dict[str, Any]:
        """
        Optimize process parameters
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            bounds: Parameter bounds
            method: Optimization method
            max_iter: Maximum iterations
        
        Returns:
            Optimization results
        """
        iteration = [0]
        
        def callback(xk):
            iteration[0] += 1
            value = objective_function(xk)
            self.parameter_history.append({
                'iteration': iteration[0],
                'params': xk.tolist(),
                'objective': value
            })
        
        result = minimize(
            objective_function,
            x0=initial_params,
            method=method,
            bounds=bounds,
            callback=callback,
            options={'maxiter': max_iter}
        )
        
        return {
            'success': result.success,
            'optimal_params': result.x.tolist(),
            'optimal_value': result.fun,
            'iterations': result.nit,
            'message': result.message
        }
    
    def bayesian_optimize(self,
                         objective_function: Callable[[np.ndarray], float],
                         param_space: List[Tuple[float, float]],
                         n_iterations: int = 50,
                         n_initial_points: int = 5) -> Dict[str, Any]:
        """
        Bayesian optimization for expensive objective functions
        """
        try:
            from skopt import gp_minimize
            
            result = gp_minimize(
                objective_function,
                param_space,
                n_calls=n_iterations,
                n_initial_points=n_initial_points,
                random_state=42
            )
            
            return {
                'success': True,
                'optimal_params': result.x,
                'optimal_value': result.fun,
                'iterations': len(result.func_vals),
                'all_values': result.func_vals.tolist()
            }
            
        except ImportError:
            logger.error("scikit-optimize required for Bayesian optimization")
            return {'success': False, 'error': 'scikit-optimize not installed'}
    
    def get_optimization_history(self) -> List[Dict]:
        """Get history of optimization runs"""
        return self.parameter_history


class AdaptiveController(FeedbackController):
    """
    Adaptive controller that adjusts its parameters online
    based on process performance
    """
    
    def __init__(self, 
                 config: ControlConfig,
                 base_controller: FeedbackController):
        super().__init__(config)
        
        self.base_controller = base_controller
        self.performance_window = deque(maxlen=50)
        self.adaptation_rate = 0.01
        self.target_performance = 0.0
    
    def compute(self,
                setpoint: float,
                process_value: float,
                dt: float = 1.0) -> float:
        """Compute adaptive control output"""
        # Get base controller output
        output = self.base_controller.compute(setpoint, process_value, dt)
        
        # Evaluate performance
        error = abs(setpoint - process_value)
        self.performance_window.append(error)
        
        # Adapt parameters if enough data
        if len(self.performance_window) >= 20:
            self._adapt_parameters()
        
        return output
    
    def _adapt_parameters(self):
        """Adapt controller parameters based on performance"""
        recent_performance = np.mean(list(self.performance_window)[-10:])
        older_performance = np.mean(list(self.performance_window)[:10])
        
        # If performance is degrading, adjust parameters
        if recent_performance > older_performance * 1.1:
            if isinstance(self.base_controller, PIDController):
                # Increase integral gain for steady-state error
                self.base_controller.ki *= (1 + self.adaptation_rate)
                # Decrease derivative gain if oscillating
                self.base_controller.kd *= (1 - self.adaptation_rate)


class MultiVariableController:
    """
    Controller for multi-variable processes (MIMO)
    Uses decoupling or model-based control
    """
    
    def __init__(self, 
                 config: ControlConfig,
                 n_inputs: int,
                 n_outputs: int):
        self.config = config
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Individual controllers for each output
        self.controllers: List[PIDController] = []
        for i in range(n_outputs):
            ctrl_config = ControlConfig(
                name=f"{config.name}_output_{i}",
                sample_time_s=config.sample_time_s,
                output_limits=config.output_limits
            )
            self.controllers.append(PIDController(ctrl_config))
        
        # Decoupling matrix (simplified identity for now)
        self.decoupling_matrix = np.eye(n_outputs)
    
    def compute(self,
                setpoints: np.ndarray,
                process_values: np.ndarray,
                dt: float = 1.0) -> np.ndarray:
        """Compute MIMO control outputs"""
        outputs = np.zeros(self.n_outputs)
        
        # Individual control calculations
        for i, ctrl in enumerate(self.controllers):
            outputs[i] = ctrl.compute(setpoints[i], process_values[i], dt)
        
        # Apply decoupling
        decoupled_outputs = self.decoupling_matrix @ outputs
        
        # Apply limits
        decoupled_outputs = np.clip(
            decoupled_outputs,
            self.config.output_limits[0],
            self.config.output_limits[1]
        )
        
        return decoupled_outputs
