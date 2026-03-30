"""
实验室自动化模块 - Lab Automation Module

自驱动实验室的核心控制模块，实现：
- 机器人合成指令生成
- 实验结果自动读取
- 闭环优化循环

作者: DFT-LAMMPS Team
"""

import os
import json
import time
import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from datetime import datetime
from enum import Enum
from pathlib import Path
import numpy as np
from collections import deque

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """实验状态枚举"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class RobotCommand(Enum):
    """机器人命令类型"""
    MOVE_TO = "move_to"
    PICK_UP = "pick_up"
    PLACE = "place"
    DISPENSE = "dispense"
    MIX = "mix"
    HEAT = "heat"
    COOL = "cool"
    STIR = "stir"
    MEASURE = "measure"
    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    CAMERA_CAPTURE = "camera_capture"
    WAIT = "wait"


@dataclass
class MaterialSpec:
    """材料规格"""
    name: str
    formula: str
    quantity: float  # mg
    purity: float = 0.99
    supplier: Optional[str] = None
    lot_number: Optional[str] = None
    storage_conditions: Optional[str] = None
    hazards: List[str] = field(default_factory=list)


@dataclass
class SynthesisParameter:
    """合成参数"""
    temperature: float  # Celsius
    pressure: float = 1.0  # atm
    time: float = 3600  # seconds
    stirring_rate: float = 0.0  # rpm
    atmosphere: str = "air"
    ph: Optional[float] = None
    solvent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobotInstruction:
    """机器人指令"""
    command: RobotCommand
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_duration: float = 0.0  # seconds
    safety_checks: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "command": self.command.value,
            "parameters": self.parameters,
            "expected_duration": self.expected_duration,
            "safety_checks": self.safety_checks,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }


@dataclass
class ExperimentResult:
    """实验结果"""
    experiment_id: str
    status: ExperimentStatus
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    measurements: Dict[str, List[float]] = field(default_factory=dict)
    observations: List[str] = field(default_factory=list)
    characterization_data: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "measurements": self.measurements,
            "observations": self.observations,
            "characterization_data": self.characterization_data,
            "success": self.success,
            "error_message": self.error_message
        }


class RobotInterface(ABC):
    """机器人接口抽象基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接机器人"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    async def execute_instruction(self, instruction: RobotInstruction) -> bool:
        """执行单个指令"""
        pass
    
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取机器人状态"""
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """紧急停止"""
        pass


class SimulatedRobot(RobotInterface):
    """模拟机器人接口，用于测试和开发"""
    
    def __init__(self, name: str = "SimulatedRobot"):
        self.name = name
        self.connected = False
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.gripper_state = "open"
        self.temperature = 25.0
        self.log: List[Dict[str, Any]] = []
        
    async def connect(self) -> bool:
        logger.info(f"[{self.name}] Connecting...")
        await asyncio.sleep(0.1)
        self.connected = True
        logger.info(f"[{self.name}] Connected")
        return True
    
    async def disconnect(self) -> bool:
        logger.info(f"[{self.name}] Disconnecting...")
        await asyncio.sleep(0.1)
        self.connected = False
        logger.info(f"[{self.name}] Disconnected")
        return True
    
    async def execute_instruction(self, instruction: RobotInstruction) -> bool:
        if not self.connected:
            raise RuntimeError("Robot not connected")
        
        logger.info(f"[{self.name}] Executing: {instruction.command.value}")
        
        # 模拟执行时间
        await asyncio.sleep(instruction.expected_duration * 0.01)
        
        # 更新状态
        if instruction.command == RobotCommand.MOVE_TO:
            self.position.update(instruction.parameters.get("position", {}))
        elif instruction.command == RobotCommand.GRIPPER_CLOSE:
            self.gripper_state = "closed"
        elif instruction.command == RobotCommand.GRIPPER_OPEN:
            self.gripper_state = "open"
        elif instruction.command == RobotCommand.HEAT:
            self.temperature = instruction.parameters.get("temperature", 25.0)
        
        self.log.append({
            "timestamp": datetime.now().isoformat(),
            "command": instruction.command.value,
            "parameters": instruction.parameters
        })
        
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "connected": self.connected,
            "position": self.position,
            "gripper_state": self.gripper_state,
            "temperature": self.temperature,
            "queue_length": len(self.log)
        }
    
    async def emergency_stop(self) -> bool:
        logger.warning(f"[{self.name}] EMERGENCY STOP!")
        self.connected = False
        return True


class UR5Robot(RobotInterface):
    """Universal Robots UR5 接口"""
    
    def __init__(self, host: str = "192.168.1.100", port: int = 30002):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    async def connect(self) -> bool:
        """连接到UR5机器人"""
        import socket
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5)
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.info(f"Connected to UR5 at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to UR5: {e}")
            return False
    
    async def disconnect(self) -> bool:
        if self.socket:
            self.socket.close()
        self.connected = False
        return True
    
    async def execute_instruction(self, instruction: RobotInstruction) -> bool:
        if not self.connected:
            return False
        
        # 生成URScript命令
        script = self._generate_urscript(instruction)
        
        try:
            self.socket.send(script.encode())
            return True
        except Exception as e:
            logger.error(f"Failed to execute instruction: {e}")
            return False
    
    def _generate_urscript(self, instruction: RobotInstruction) -> str:
        """生成URScript代码"""
        if instruction.command == RobotCommand.MOVE_TO:
            pos = instruction.parameters.get("position", {})
            return f"""
            movel(p[{pos.get('x', 0)}, {pos.get('y', 0)}, {pos.get('z', 0)}, 0, 0, 0], 
                  a=1.2, v=0.25)
            """
        elif instruction.command == RobotCommand.GRIPPER_CLOSE:
            return "set_digital_out(0, True)"
        elif instruction.command == RobotCommand.GRIPPER_OPEN:
            return "set_digital_out(0, False)"
        return ""
    
    async def get_status(self) -> Dict[str, Any]:
        # 实现状态查询
        return {"connected": self.connected, "host": self.host}
    
    async def emergency_stop(self) -> bool:
        if self.socket:
            self.socket.send("stopl(1.0)\n".encode())
        return True


class OT2Robot(RobotInterface):
    """Opentrons OT-2 液体处理机器人接口"""
    
    def __init__(self, api_host: str = "http://localhost:31950"):
        self.api_host = api_host
        self.connected = False
        self.protocol = None
        
    async def connect(self) -> bool:
        """连接到OT-2 API"""
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_host}/health") as resp:
                    if resp.status == 200:
                        self.connected = True
                        logger.info("Connected to OT-2")
                        return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to OT-2: {e}")
            return False
    
    async def disconnect(self) -> bool:
        self.connected = False
        return True
    
    async def execute_instruction(self, instruction: RobotInstruction) -> bool:
        """执行OT-2指令"""
        if not self.connected:
            return False
        
        # 转换为OT-2协议命令
        command_map = {
            RobotCommand.DISPENSE: self._dispense,
            RobotCommand.PICK_UP: self._pick_up_tip,
            RobotCommand.PLACE: self._drop_tip,
            RobotCommand.MIX: self._mix,
            RobotCommand.MOVE_TO: self._move_to
        }
        
        handler = command_map.get(instruction.command)
        if handler:
            return await handler(instruction.parameters)
        return False
    
    async def _dispense(self, params: Dict[str, Any]) -> bool:
        # 实现分液逻辑
        return True
    
    async def _pick_up_tip(self, params: Dict[str, Any]) -> bool:
        return True
    
    async def _drop_tip(self, params: Dict[str, Any]) -> bool:
        return True
    
    async def _mix(self, params: Dict[str, Any]) -> bool:
        return True
    
    async def _move_to(self, params: Dict[str, Any]) -> bool:
        return True
    
    async def get_status(self) -> Dict[str, Any]:
        return {"connected": self.connected, "api_host": self.api_host}
    
    async def emergency_stop(self) -> bool:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_host}/robot/move", 
                                       json={"target": "mount", "mount": "pipette", "point": [0, 0, 0]}):
                    return True
        except:
            return False


class SynthesisProtocol:
    """合成协议生成器"""
    
    def __init__(self, name: str):
        self.name = name
        self.materials: List[MaterialSpec] = []
        self.instructions: List[RobotInstruction] = []
        self.safety_notes: List[str] = []
        
    def add_material(self, material: MaterialSpec):
        """添加材料"""
        self.materials.append(material)
        
    def add_instruction(self, instruction: RobotInstruction):
        """添加指令"""
        self.instructions.append(instruction)
        
    def add_safety_note(self, note: str):
        """添加安全注意事项"""
        self.safety_notes.append(note)
        
    def generate_protocol(self) -> Dict[str, Any]:
        """生成完整协议"""
        return {
            "name": self.name,
            "materials": [asdict(m) for m in self.materials],
            "instructions": [i.to_dict() for i in self.instructions],
            "safety_notes": self.safety_notes,
            "estimated_duration": sum(i.expected_duration for i in self.instructions),
            "version": "1.0"
        }


class ProtocolGenerator:
    """合成协议生成器"""
    
    def __init__(self):
        self.templates: Dict[str, Callable] = {}
        self._register_default_templates()
    
    def _register_default_templates(self):
        """注册默认模板"""
        self.register_template("solid_state", self._solid_state_template)
        self.register_template("sol_gel", self._sol_gel_template)
        self.register_template("hydrothermal", self._hydrothermal_template)
        self.register_template("co_precipitation", self._co_precipitation_template)
        self.register_template("solid_electrolyte", self._solid_electrolyte_template)
    
    def register_template(self, name: str, template_func: Callable):
        """注册新模板"""
        self.templates[name] = template_func
    
    def generate_protocol(self, template_name: str, 
                         composition: Dict[str, float],
                         parameters: SynthesisParameter) -> SynthesisProtocol:
        """根据模板生成协议"""
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        return self.templates[template_name](composition, parameters)
    
    def _solid_state_template(self, composition: Dict[str, float], 
                             params: SynthesisParameter) -> SynthesisProtocol:
        """固相合成模板"""
        protocol = SynthesisProtocol("Solid State Synthesis")
        
        # 添加材料
        for element, amount in composition.items():
            protocol.add_material(MaterialSpec(
                name=f"{element}_powder",
                formula=element,
                quantity=amount,
                purity=0.999
            ))
        
        # 研磨
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.PICK_UP,
            parameters={"tool": "mortar"},
            expected_duration=5.0
        ))
        
        # 混合
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.MIX,
            parameters={"duration": 600, "method": "grinding"},
            expected_duration=600.0
        ))
        
        # 压片
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.PLACE,
            parameters={"tool": "pellet_press", "pressure": 100},
            expected_duration=60.0
        ))
        
        # 加热
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.HEAT,
            parameters={"temperature": params.temperature, 
                       "time": params.time,
                       "atmosphere": params.atmosphere},
            expected_duration=params.time + 300
        ))
        
        # 冷却
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.COOL,
            parameters={"rate": 5.0},  # 5°C/min
            expected_duration=params.temperature / 5 * 60
        ))
        
        return protocol
    
    def _sol_gel_template(self, composition: Dict[str, float], 
                         params: SynthesisParameter) -> SynthesisProtocol:
        """溶胶-凝胶合成模板"""
        protocol = SynthesisProtocol("Sol-Gel Synthesis")
        
        # 添加前驱体
        for element, amount in composition.items():
            protocol.add_material(MaterialSpec(
                name=f"{element}_alkoxide",
                formula=f"{element}(OR)4",
                quantity=amount,
                hazards=["moisture_sensitive", "flammable"]
            ))
        
        # 溶剂
        protocol.add_material(MaterialSpec(
            name="ethanol",
            formula="C2H5OH",
            quantity=100.0,
            hazards=["flammable"]
        ))
        
        # 分步指令
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.DISPENSE,
            parameters={"volume": 50, "liquid": "ethanol"},
            expected_duration=30.0
        ))
        
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.STIR,
            parameters={"rate": 500, "time": 1800},
            expected_duration=1800.0
        ))
        
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.HEAT,
            parameters={"temperature": 80, "time": 86400},  # 24h gelation
            expected_duration=87000.0
        ))
        
        return protocol
    
    def _hydrothermal_template(self, composition: Dict[str, float], 
                              params: SynthesisParameter) -> SynthesisProtocol:
        """水热合成模板"""
        protocol = SynthesisProtocol("Hydrothermal Synthesis")
        
        for element, amount in composition.items():
            protocol.add_material(MaterialSpec(
                name=f"{element}_precursor",
                formula=element,
                quantity=amount
            ))
        
        protocol.add_material(MaterialSpec(
            name="deionized_water",
            formula="H2O",
            quantity=50.0
        ))
        
        # 装入高压釜
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.DISPENSE,
            parameters={"volume": 30, "container": "autoclave"},
            expected_duration=20.0
        ))
        
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.HEAT,
            parameters={"temperature": params.temperature,
                       "pressure": params.pressure,
                       "time": params.time},
            expected_duration=params.time + 600
        ))
        
        return protocol
    
    def _co_precipitation_template(self, composition: Dict[str, float], 
                                  params: SynthesisParameter) -> SynthesisProtocol:
        """共沉淀合成模板"""
        protocol = SynthesisProtocol("Co-precipitation Synthesis")
        
        for element, amount in composition.items():
            protocol.add_material(MaterialSpec(
                name=f"{element}_nitrate",
                formula=f"{element}(NO3)n",
                quantity=amount
            ))
        
        protocol.add_material(MaterialSpec(
            name="precipitating_agent",
            formula="NH4OH",
            quantity=50.0,
            hazards=["corrosive"]
        ))
        
        # 混合盐溶液
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.MIX,
            parameters={"method": "ultrasonic"},
            expected_duration=300.0
        ))
        
        # 滴加沉淀剂
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.DISPENSE,
            parameters={"volume": 50, "rate": "dropwise", "ph_target": params.ph},
            expected_duration=1800.0
        ))
        
        # 陈化
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.WAIT,
            parameters={"time": 7200},
            expected_duration=7200.0
        ))
        
        return protocol
    
    def _solid_electrolyte_template(self, composition: Dict[str, float], 
                                   params: SynthesisParameter) -> SynthesisProtocol:
        """固态电解质专用合成模板"""
        protocol = SynthesisProtocol("Solid Electrolyte Synthesis")
        
        # 典型的硫化物固态电解质
        if "Li" in composition and "P" in composition and "S" in composition:
            protocol.add_material(MaterialSpec(
                name="lithium_sulfide",
                formula="Li2S",
                quantity=composition.get("Li", 0) * 0.5,
                hazards=["air_sensitive", "moisture_sensitive"],
                storage_conditions="argon_atmosphere"
            ))
            
            protocol.add_material(MaterialSpec(
                name="phosphorus_pentasulfide",
                formula="P2S5",
                quantity=composition.get("P", 0),
                hazards=["air_sensitive", "toxic"]
            ))
            
            protocol.add_material(MaterialSpec(
                name="lithium_halide",
                formula="LiCl",
                quantity=composition.get("Cl", 0),
                optional=True
            ))
        
        # 高能球磨
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.MIX,
            parameters={"method": "ball_milling", "energy": "high"},
            expected_duration=36000.0  # 10h
        ))
        
        # 热处理
        protocol.add_instruction(RobotInstruction(
            command=RobotCommand.HEAT,
            parameters={"temperature": params.temperature,
                       "atmosphere": "argon",
                       "time": params.time},
            expected_duration=params.time + 600
        ))
        
        return protocol


class ExperimentRunner:
    """实验运行器 - 管理实验执行"""
    
    def __init__(self, robot: RobotInterface, data_dir: str = "./experiment_data"):
        self.robot = robot
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.current_experiment: Optional[str] = None
        self.experiment_history: deque = deque(maxlen=1000)
        self.callbacks: List[Callable] = []
        self._running = False
        
    def register_callback(self, callback: Callable):
        """注册实验状态回调"""
        self.callbacks.append(callback)
    
    async def run_protocol(self, protocol: SynthesisProtocol, 
                          experiment_id: Optional[str] = None) -> ExperimentResult:
        """运行合成协议"""
        if experiment_id is None:
            experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_experiment = experiment_id
        result = ExperimentResult(
            experiment_id=experiment_id,
            status=ExperimentStatus.RUNNING,
            timestamp=datetime.now(),
            parameters=protocol.generate_protocol()
        )
        
        logger.info(f"Starting experiment: {experiment_id}")
        
        try:
            # 连接机器人
            if not await self.robot.connect():
                raise RuntimeError("Failed to connect to robot")
            
            # 执行指令序列
            for i, instruction in enumerate(protocol.instructions):
                logger.info(f"Executing instruction {i+1}/{len(protocol.instructions)}: {instruction.command.value}")
                
                # 执行并检查安全
                success = await self._execute_with_retry(instruction)
                
                if not success:
                    raise RuntimeError(f"Instruction {i+1} failed after {instruction.max_retries} retries")
                
                # 记录进度
                self._notify_callbacks({
                    "experiment_id": experiment_id,
                    "progress": (i + 1) / len(protocol.instructions),
                    "current_step": instruction.command.value
                })
            
            # 获取最终状态
            result.status = ExperimentStatus.COMPLETED
            result.success = True
            result.measurements = await self._collect_measurements()
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            
            # 尝试紧急停止
            await self.robot.emergency_stop()
            
        finally:
            await self.robot.disconnect()
            self._save_result(result)
            self.experiment_history.append(result)
            self.current_experiment = None
        
        return result
    
    async def _execute_with_retry(self, instruction: RobotInstruction) -> bool:
        """带重试的指令执行"""
        for attempt in range(instruction.max_retries):
            try:
                success = await self.robot.execute_instruction(instruction)
                if success:
                    return True
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                instruction.retry_count += 1
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return False
    
    async def _collect_measurements(self) -> Dict[str, List[float]]:
        """收集测量数据"""
        # 从各种传感器收集数据
        measurements = {
            "temperature": [],
            "pressure": [],
            "time": []
        }
        
        # 模拟数据收集
        robot_status = await self.robot.get_status()
        measurements["temperature"].append(robot_status.get("temperature", 25.0))
        
        return measurements
    
    def _notify_callbacks(self, data: Dict[str, Any]):
        """通知所有回调"""
        for callback in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def _save_result(self, result: ExperimentResult):
        """保存实验结果"""
        filepath = self.data_dir / f"{result.experiment_id}.json"
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Result saved to {filepath}")


class AutonomousLab:
    """自驱动实验室主控制器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.robots: Dict[str, RobotInterface] = {}
        self.protocol_generator = ProtocolGenerator()
        self.runners: Dict[str, ExperimentRunner] = {}
        self.optimization_engine = None
        self.active_loops: Dict[str, asyncio.Task] = {}
        
    def add_robot(self, name: str, robot: RobotInterface):
        """添加机器人"""
        self.robots[name] = robot
        self.runners[name] = ExperimentRunner(robot)
        
    def remove_robot(self, name: str):
        """移除机器人"""
        if name in self.robots:
            del self.robots[name]
            del self.runners[name]
    
    async def run_experiment(self, robot_name: str, 
                            protocol: SynthesisProtocol) -> ExperimentResult:
        """运行单个实验"""
        if robot_name not in self.runners:
            raise ValueError(f"Unknown robot: {robot_name}")
        
        runner = self.runners[robot_name]
        return await runner.run_protocol(protocol)
    
    async def start_closed_loop(self, 
                               target_property: str,
                               optimization_objective: Dict[str, Any],
                               max_iterations: int = 100) -> str:
        """启动闭环优化循环"""
        loop_id = f"loop_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        task = asyncio.create_task(
            self._optimization_loop(loop_id, target_property, 
                                   optimization_objective, max_iterations)
        )
        
        self.active_loops[loop_id] = task
        return loop_id
    
    async def _optimization_loop(self, loop_id: str,
                                 target_property: str,
                                 objective: Dict[str, Any],
                                 max_iterations: int):
        """优化循环核心"""
        logger.info(f"Starting closed-loop optimization: {loop_id}")
        
        for iteration in range(max_iterations):
            try:
                # 1. 生成候选配方
                candidate = await self._generate_candidate(target_property, objective)
                
                # 2. 生成合成协议
                protocol = self.protocol_generator.generate_protocol(
                    candidate["template"],
                    candidate["composition"],
                    candidate["parameters"]
                )
                
                # 3. 选择可用机器人
                robot_name = self._select_robot()
                
                # 4. 执行实验
                result = await self.run_experiment(robot_name, protocol)
                
                # 5. 分析结果
                feedback = await self._analyze_result(result, target_property)
                
                # 6. 更新优化器
                await self._update_optimizer(candidate, feedback)
                
                # 检查收敛
                if self._check_convergence(feedback, objective):
                    logger.info(f"Optimization converged at iteration {iteration + 1}")
                    break
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Optimization loop {loop_id} completed")
    
    async def _generate_candidate(self, target_property: str, 
                                  objective: Dict[str, Any]) -> Dict[str, Any]:
        """生成候选配方"""
        # 这里可以集成贝叶斯优化、遗传算法等
        # 简化实现：返回随机候选
        import random
        
        templates = ["solid_state", "sol_gel", "hydrothermal", "co_precipitation"]
        
        return {
            "template": random.choice(templates),
            "composition": {
                "Li": random.uniform(0.3, 0.7),
                "P": random.uniform(0.1, 0.3),
                "S": random.uniform(0.2, 0.5)
            },
            "parameters": SynthesisParameter(
                temperature=random.uniform(200, 800),
                time=random.uniform(1800, 86400)
            )
        }
    
    def _select_robot(self) -> str:
        """选择可用机器人"""
        return list(self.robots.keys())[0]
    
    async def _analyze_result(self, result: ExperimentResult, 
                             target_property: str) -> Dict[str, float]:
        """分析实验结果"""
        # 这里集成表征数据分析
        # 返回目标属性值
        return {
            target_property: np.random.uniform(0, 1),
            "yield": np.random.uniform(0.5, 1.0),
            "purity": np.random.uniform(0.9, 1.0)
        }
    
    async def _update_optimizer(self, candidate: Dict[str, Any], 
                               feedback: Dict[str, float]):
        """更新优化器"""
        # 这里集成具体的优化算法
        pass
    
    def _check_convergence(self, feedback: Dict[str, float], 
                          objective: Dict[str, Any]) -> bool:
        """检查收敛条件"""
        # 检查是否达到目标
        for key, target in objective.items():
            if key in feedback:
                if abs(feedback[key] - target) > 0.05:  # 5% tolerance
                    return False
        return True
    
    async def stop_closed_loop(self, loop_id: str):
        """停止优化循环"""
        if loop_id in self.active_loops:
            self.active_loops[loop_id].cancel()
            del self.active_loops[loop_id]


class ExperimentQueue:
    """实验队列管理"""
    
    def __init__(self, max_concurrent: int = 1):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_concurrent = max_concurrent
        self.active_experiments: Dict[str, asyncio.Task] = {}
        self.completed: List[str] = []
        self.failed: List[str] = []
        
    async def add_experiment(self, experiment_id: str, 
                            protocol: SynthesisProtocol,
                            runner: ExperimentRunner,
                            priority: int = 0):
        """添加实验到队列"""
        await self.queue.put((priority, experiment_id, protocol, runner))
        
    async def start_processing(self):
        """开始处理队列"""
        workers = [
            asyncio.create_task(self._worker())
            for _ in range(self.max_concurrent)
        ]
        await asyncio.gather(*workers)
    
    async def _worker(self):
        """队列工作器"""
        while True:
            try:
                priority, exp_id, protocol, runner = await self.queue.get()
                
                task = asyncio.create_task(
                    runner.run_protocol(protocol, exp_id)
                )
                
                self.active_experiments[exp_id] = task
                
                try:
                    result = await task
                    if result.success:
                        self.completed.append(exp_id)
                    else:
                        self.failed.append(exp_id)
                except Exception as e:
                    logger.error(f"Experiment {exp_id} failed: {e}")
                    self.failed.append(exp_id)
                finally:
                    del self.active_experiments[exp_id]
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取队列状态"""
        return {
            "queue_size": self.queue.qsize(),
            "active": list(self.active_experiments.keys()),
            "completed": self.completed,
            "failed": self.failed
        }


# ==================== 实验结果自动读取模块 ====================

class DataReader(ABC):
    """数据读取器抽象基类"""
    
    @abstractmethod
    async def read(self, source: str) -> Dict[str, Any]:
        """读取数据"""
        pass
    
    @abstractmethod
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证数据"""
        pass


class FileDataReader(DataReader):
    """文件数据读取器"""
    
    def __init__(self, file_type: str = "auto"):
        self.file_type = file_type
        self.parsers: Dict[str, Callable] = {
            "json": self._parse_json,
            "csv": self._parse_csv,
            "txt": self._parse_txt,
            "xrd": self._parse_xrd,
            "xy": self._parse_xy
        }
    
    async def read(self, source: str) -> Dict[str, Any]:
        """读取文件数据"""
        path = Path(source)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        # 自动检测文件类型
        file_type = self.file_type
        if file_type == "auto":
            file_type = path.suffix.lstrip('.') or 'txt'
        
        parser = self.parsers.get(file_type, self._parse_txt)
        
        with open(path, 'r') as f:
            content = f.read()
        
        return await parser(content)
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """验证数据完整性"""
        return "data" in data or "measurements" in data
    
    async def _parse_json(self, content: str) -> Dict[str, Any]:
        return json.loads(content)
    
    async def _parse_csv(self, content: str) -> Dict[str, Any]:
        import csv
        lines = content.strip().split('\n')
        reader = csv.DictReader(lines)
        data = list(reader)
        return {"data": data, "columns": reader.fieldnames}
    
    async def _parse_txt(self, content: str) -> Dict[str, Any]:
        lines = content.strip().split('\n')
        return {"raw_text": content, "lines": lines}
    
    async def _parse_xrd(self, content: str) -> Dict[str, Any]:
        """解析XRD数据文件"""
        lines = content.strip().split('\n')
        
        # 跳过注释行
        data_lines = [l for l in lines if not l.startswith('#') and not l.startswith('%')]
        
        two_theta = []
        intensity = []
        
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue
        
        return {
            "type": "xrd",
            "two_theta": two_theta,
            "intensity": intensity,
            "data_points": len(two_theta)
        }
    
    async def _parse_xy(self, content: str) -> Dict[str, Any]:
        """解析XY数据文件"""
        lines = content.strip().split('\n')
        x = []
        y = []
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
                except ValueError:
                    continue
        
        return {
            "type": "xy",
            "x": x,
            "y": y,
            "data_points": len(x)
        }


class InstrumentDataReader(DataReader):
    """仪器数据读取器"""
    
    def __init__(self, instrument_type: str, connection_params: Dict[str, Any]):
        self.instrument_type = instrument_type
        self.connection_params = connection_params
        
    async def read(self, source: str) -> Dict[str, Any]:
        """从仪器读取数据"""
        # 根据仪器类型选择读取方式
        readers = {
            "xrd": self._read_xrd_instrument,
            "sem": self._read_sem_instrument,
            "tem": self._read_tem_instrument,
            "raman": self._read_raman_instrument,
            "ftir": self._read_ftir_instrument
        }
        
        reader = readers.get(self.instrument_type)
        if reader:
            return await reader(source)
        
        raise ValueError(f"Unknown instrument type: {self.instrument_type}")
    
    def validate(self, data: Dict[str, Any]) -> bool:
        return "instrument_type" in data and "measurements" in data
    
    async def _read_xrd_instrument(self, measurement_id: str) -> Dict[str, Any]:
        """读取XRD仪器数据"""
        # 模拟从XRD仪器读取
        return {
            "instrument_type": "xrd",
            "instrument_model": "Bruker D8",
            "measurement_id": measurement_id,
            "two_theta_range": [5, 90],
            "step_size": 0.02,
            "data": {"two_theta": [], "intensity": []}
        }
    
    async def _read_sem_instrument(self, measurement_id: str) -> Dict[str, Any]:
        """读取SEM仪器数据"""
        return {
            "instrument_type": "sem",
            "instrument_model": "FEI Quanta",
            "measurement_id": measurement_id,
            "magnification": 10000,
            "accelerating_voltage": 10.0,
            "images": []
        }
    
    async def _read_tem_instrument(self, measurement_id: str) -> Dict[str, Any]:
        """读取TEM仪器数据"""
        return {
            "instrument_type": "tem",
            "instrument_model": "JEOL JEM",
            "measurement_id": measurement_id,
            "magnification": 500000,
            "accelerating_voltage": 200.0,
            "images": [],
            "diffraction_patterns": []
        }
    
    async def _read_raman_instrument(self, measurement_id: str) -> Dict[str, Any]:
        """读取Raman仪器数据"""
        return {
            "instrument_type": "raman",
            "instrument_model": "Horiba LabRAM",
            "measurement_id": measurement_id,
            "laser_wavelength": 532,
            "data": {"shift": [], "intensity": []}
        }
    
    async def _read_ftir_instrument(self, measurement_id: str) -> Dict[str, Any]:
        """读取FTIR仪器数据"""
        return {
            "instrument_type": "ftir",
            "instrument_model": "Thermo Nicolet",
            "measurement_id": measurement_id,
            "data": {"wavenumber": [], "transmittance": []}
        }


class DatabaseConnector:
    """实验数据库连接器"""
    
    def __init__(self, db_path: str = "./lab_database.db"):
        self.db_path = db_path
        self.connection = None
        
    async def connect(self):
        """连接数据库"""
        import aiosqlite
        self.connection = await aiosqlite.connect(self.db_path)
        await self._create_tables()
    
    async def disconnect(self):
        """断开连接"""
        if self.connection:
            await self.connection.close()
    
    async def _create_tables(self):
        """创建数据表"""
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                status TEXT,
                protocol TEXT,
                result TEXT
            )
        """)
        
        await self.connection.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                instrument_type TEXT,
                data TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """)
        
        await self.connection.commit()
    
    async def save_experiment(self, result: ExperimentResult):
        """保存实验结果"""
        await self.connection.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?, ?)",
            (result.experiment_id, 
             result.timestamp.isoformat(),
             result.status.value,
             json.dumps(result.parameters),
             json.dumps(result.to_dict()))
        )
        await self.connection.commit()
    
    async def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """获取实验数据"""
        async with self.connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row:
                return {
                    "id": row[0],
                    "timestamp": row[1],
                    "status": row[2],
                    "protocol": json.loads(row[3]),
                    "result": json.loads(row[4])
                }
            return None
    
    async def query_experiments(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查询实验"""
        # 构建查询条件
        conditions = []
        values = []
        
        if "status" in filters:
            conditions.append("status = ?")
            values.append(filters["status"])
        
        if "start_date" in filters:
            conditions.append("timestamp >= ?")
            values.append(filters["start_date"])
        
        query = "SELECT * FROM experiments"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        async with self.connection.execute(query, values) as cursor:
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "timestamp": row[1],
                    "status": row[2],
                    "protocol": json.loads(row[3]),
                    "result": json.loads(row[4])
                }
                for row in rows
            ]


# ==================== 主入口函数 ====================

def create_lab(config: Optional[Dict[str, Any]] = None) -> AutonomousLab:
    """创建自驱动实验室实例"""
    return AutonomousLab(config)


def create_simulated_robot(name: str = "SimRobot") -> SimulatedRobot:
    """创建模拟机器人"""
    return SimulatedRobot(name)


def create_protocol_generator() -> ProtocolGenerator:
    """创建协议生成器"""
    return ProtocolGenerator()


# 示例用法
if __name__ == "__main__":
    async def main():
        # 创建实验室
        lab = create_lab()
        
        # 添加模拟机器人
        robot = create_simulated_robot("LabRobot1")
        lab.add_robot("robot1", robot)
        
        # 生成合成协议
        generator = create_protocol_generator()
        protocol = generator.generate_protocol(
            "solid_state",
            {"Li": 3.0, "P": 1.0, "S": 4.0},
            SynthesisParameter(temperature=550, time=3600)
        )
        
        # 运行实验
        result = await lab.run_experiment("robot1", protocol)
        
        print(f"Experiment completed: {result.success}")
        print(f"Status: {result.status.value}")
        
    asyncio.run(main())
