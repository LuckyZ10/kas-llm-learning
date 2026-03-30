"""
Equipment Abstraction Layer for Laboratory Automation

Provides unified interfaces for various laboratory equipment including:
- Robotic arms (various brands)
- Synthesis equipment (furnaces, mixers, deposition systems)
- Characterization instruments (XRD, SEM, electrochemical workstations)
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


logger = logging.getLogger(__name__)


class EquipmentState(Enum):
    """Equipment operational states"""
    OFFLINE = auto()
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    MAINTENANCE = auto()


class EquipmentType(Enum):
    """Types of laboratory equipment"""
    ROBOT_ARM = "robot_arm"
    SYNTHESIS_FURNACE = "synthesis_furnace"
    MIXER = "mixer"
    DEPOSITION_SYSTEM = "deposition_system"
    XRD_INSTRUMENT = "xrd_instrument"
    SEM_INSTRUMENT = "sem_instrument"
    ELECTROCHEMICAL_WORKSTATION = "electrochemical_workstation"
    LIQUID_HANDLER = "liquid_handler"
    BALANCE = "balance"
    CENTRIFUGE = "centrifuge"


@dataclass
class EquipmentStatus:
    """Current status of equipment"""
    state: EquipmentState
    timestamp: datetime = field(default_factory=datetime.now)
    message: str = ""
    progress: float = 0.0  # 0-100%
    parameters: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state.name,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'progress': self.progress,
            'parameters': self.parameters,
            'errors': self.errors
        }


@dataclass  
class Position3D:
    """3D position coordinates"""
    x: float
    y: float
    z: float
    
    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]
    
    @classmethod
    def from_list(cls, coords: List[float]) -> 'Position3D':
        return cls(x=coords[0], y=coords[1], z=coords[2])


@dataclass
class GripperState:
    """Gripper/actuator state"""
    open: bool = True
    force: float = 0.0  # Newtons
    position: float = 0.0  # mm


class BaseEquipment(ABC):
    """Abstract base class for all laboratory equipment"""
    
    def __init__(self, 
                 equipment_id: str,
                 equipment_type: EquipmentType,
                 name: str,
                 model: str = "",
                 manufacturer: str = "",
                 connection_params: Optional[Dict[str, Any]] = None):
        self.equipment_id = equipment_id
        self.equipment_type = equipment_type
        self.name = name
        self.model = model
        self.manufacturer = manufacturer
        self.connection_params = connection_params or {}
        
        self._status = EquipmentStatus(state=EquipmentState.OFFLINE)
        self._connected = False
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable[[EquipmentStatus], None]] = []
        
        logger.info(f"Initialized {self.equipment_type.value}: {self.name} ({self.equipment_id})")
    
    @property
    def status(self) -> EquipmentStatus:
        return self._status
    
    def on_status_change(self, callback: Callable[[EquipmentStatus], None]):
        """Register status change callback"""
        self._callbacks.append(callback)
    
    def _update_status(self, status: EquipmentStatus):
        """Update equipment status and notify listeners"""
        self._status = status
        for callback in self._callbacks:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to equipment"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from equipment"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize equipment (homing, calibration, etc.)"""
        pass
    
    @abstractmethod
    async def emergency_stop(self) -> bool:
        """Emergency stop"""
        pass
    
    @abstractmethod
    async def get_status(self) -> EquipmentStatus:
        """Get current equipment status"""
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on equipment"""
        return {
            'equipment_id': self.equipment_id,
            'healthy': self._connected and self._status.state != EquipmentState.ERROR,
            'status': self._status.to_dict(),
            'timestamp': datetime.now().isoformat()
        }


class RobotArm(BaseEquipment):
    """
    Robotic arm interface
    Supports various robot brands (Universal Robots, FANUC, KUKA, etc.)
    """
    
    def __init__(self, 
                 equipment_id: str,
                 name: str,
                 model: str = "UR5e",
                 manufacturer: str = "Universal Robots",
                 dof: int = 6,
                 max_payload: float = 5.0,  # kg
                 max_reach: float = 850.0,  # mm
                 connection_params: Optional[Dict[str, Any]] = None):
        super().__init__(
            equipment_id=equipment_id,
            equipment_type=EquipmentType.ROBOT_ARM,
            name=name,
            model=model,
            manufacturer=manufacturer,
            connection_params=connection_params
        )
        self.dof = dof
        self.max_payload = max_payload
        self.max_reach = max_reach
        
        # Current state
        self._current_position: Optional[Position3D] = None
        self._current_joints: Optional[List[float]] = None
        self._gripper = GripperState()
        self._speed = 50.0  # %
        
    async def connect(self) -> bool:
        """Connect to robot controller"""
        try:
            # Simulate connection (replace with actual implementation)
            host = self.connection_params.get('host', '192.168.1.100')
            port = self.connection_params.get('port', 30002)
            
            logger.info(f"Connecting to robot at {host}:{port}")
            # await asyncio.sleep(0.5)  # Connection delay
            
            self._connected = True
            self._update_status(EquipmentStatus(
                state=EquipmentState.IDLE,
                message="Robot connected and ready"
            ))
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self._update_status(EquipmentStatus(
                state=EquipmentState.ERROR,
                message=f"Connection failed: {e}"
            ))
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from robot"""
        self._connected = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.OFFLINE,
            message="Robot disconnected"
        ))
        return True
    
    async def initialize(self) -> bool:
        """Initialize robot (homing sequence)"""
        if not self._connected:
            await self.connect()
        
        self._update_status(EquipmentStatus(
            state=EquipmentState.BUSY,
            message="Initializing robot (homing)...",
            progress=0.0
        ))
        
        # Simulate homing sequence
        for progress in [25, 50, 75, 100]:
            await asyncio.sleep(0.2)
            self._update_status(EquipmentStatus(
                state=EquipmentState.BUSY,
                message=f"Homing... {progress}%",
                progress=progress
            ))
        
        self._current_position = Position3D(0, 0, 0)
        self._current_joints = [0.0] * self.dof
        
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message="Robot initialized and ready",
            progress=100.0
        ))
        return True
    
    async def move_to(self, 
                      position: Position3D,
                      orientation: Optional[List[float]] = None,
                      speed: Optional[float] = None,
                      acceleration: float = 1.0) -> bool:
        """Move robot to Cartesian position"""
        async with self._lock:
            if not self._connected:
                raise RuntimeError("Robot not connected")
            
            speed = speed or self._speed
            
            self._update_status(EquipmentStatus(
                state=EquipmentState.BUSY,
                message=f"Moving to position ({position.x}, {position.y}, {position.z})",
                parameters={'target': position.to_list(), 'speed': speed}
            ))
            
            # Simulate movement
            await asyncio.sleep(1.0)
            
            self._current_position = position
            self._update_status(EquipmentStatus(
                state=EquipmentState.IDLE,
                message="Movement completed",
                parameters={'current': position.to_list()}
            ))
            return True
    
    async def move_joints(self, 
                          joint_positions: List[float],
                          speed: Optional[float] = None) -> bool:
        """Move robot using joint angles"""
        async with self._lock:
            if len(joint_positions) != self.dof:
                raise ValueError(f"Expected {self.dof} joint values, got {len(joint_positions)}")
            
            self._update_status(EquipmentStatus(
                state=EquipmentState.BUSY,
                message="Moving joints...",
                parameters={'target_joints': joint_positions}
            ))
            
            await asyncio.sleep(0.5)
            
            self._current_joints = joint_positions
            self._update_status(EquipmentStatus(state=EquipmentState.IDLE))
            return True
    
    async def pick(self, 
                   position: Position3D,
                   grip_force: float = 10.0) -> bool:
        """Pick object at position"""
        # Move above object
        above = Position3D(position.x, position.y, position.z + 50)
        await self.move_to(above)
        
        # Open gripper
        await self.set_gripper(open=True)
        
        # Move down
        await self.move_to(position)
        
        # Close gripper
        await self.set_gripper(open=False, force=grip_force)
        
        # Move up
        await self.move_to(above)
        
        return True
    
    async def place(self, position: Position3D) -> bool:
        """Place object at position"""
        # Move above position
        above = Position3D(position.x, position.y, position.z + 50)
        await self.move_to(above)
        
        # Move down
        await self.move_to(position)
        
        # Open gripper
        await self.set_gripper(open=True)
        
        # Move up
        await self.move_to(above)
        
        return True
    
    async def set_gripper(self, 
                          open: bool,
                          force: float = 10.0,
                          position: float = 0.0) -> bool:
        """Control gripper state"""
        self._gripper = GripperState(open=open, force=force, position=position)
        await asyncio.sleep(0.2)
        return True
    
    async def get_position(self) -> Optional[Position3D]:
        """Get current Cartesian position"""
        return self._current_position
    
    async def get_joints(self) -> Optional[List[float]]:
        """Get current joint angles"""
        return self._current_joints
    
    async def emergency_stop(self) -> bool:
        """Emergency stop"""
        self._update_status(EquipmentStatus(
            state=EquipmentState.ERROR,
            message="EMERGENCY STOP ACTIVATED"
        ))
        return True
    
    async def get_status(self) -> EquipmentStatus:
        return self._status


class SynthesisEquipment(BaseEquipment):
    """
    Synthesis equipment interface
    Supports furnaces, mixers, deposition systems, etc.
    """
    
    def __init__(self,
                 equipment_id: str,
                 name: str,
                 equipment_subtype: str,  # "furnace", "mixer", "deposition"
                 model: str = "",
                 manufacturer: str = "",
                 capacity: Optional[float] = None,
                 max_temperature: Optional[float] = None,
                 connection_params: Optional[Dict[str, Any]] = None):
        
        type_map = {
            'furnace': EquipmentType.SYNTHESIS_FURNACE,
            'mixer': EquipmentType.MIXER,
            'deposition': EquipmentType.DEPOSITION_SYSTEM
        }
        
        super().__init__(
            equipment_id=equipment_id,
            equipment_type=type_map.get(equipment_subtype, EquipmentType.SYNTHESIS_FURNACE),
            name=name,
            model=model,
            manufacturer=manufacturer,
            connection_params=connection_params
        )
        
        self.subtype = equipment_subtype
        self.capacity = capacity
        self.max_temperature = max_temperature
        
        # Current state
        self._current_temperature = 25.0
        self._target_temperature = 25.0
        self._heating = False
        self._mixing_speed = 0.0
        self._process_running = False
    
    async def connect(self) -> bool:
        """Connect to synthesis equipment"""
        self._connected = True
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message=f"{self.subtype.capitalize()} connected"
        ))
        return True
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def initialize(self) -> bool:
        """Initialize equipment"""
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message="Equipment initialized"
        ))
        return True
    
    async def set_temperature(self, 
                              temperature: float,
                              ramp_rate: float = 5.0) -> bool:
        """Set target temperature with ramp rate (°C/min)"""
        if self.max_temperature and temperature > self.max_temperature:
            raise ValueError(f"Temperature {temperature} exceeds max {self.max_temperature}")
        
        self._target_temperature = temperature
        self._heating = True
        
        self._update_status(EquipmentStatus(
            state=EquipmentState.BUSY,
            message=f"Heating to {temperature}°C",
            parameters={'target_temp': temperature, 'ramp_rate': ramp_rate}
        ))
        
        # Simulate heating
        while abs(self._current_temperature - temperature) > 1.0:
            diff = temperature - self._current_temperature
            step = min(abs(diff), ramp_rate / 60.0 * 0.5)  # per 0.5s
            self._current_temperature += step if diff > 0 else -step
            await asyncio.sleep(0.5)
        
        self._heating = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message=f"Temperature reached: {temperature}°C",
            parameters={'current_temp': self._current_temperature}
        ))
        return True
    
    async def get_temperature(self) -> float:
        """Get current temperature"""
        return self._current_temperature
    
    async def set_mixing_speed(self, rpm: float) -> bool:
        """Set mixing speed (RPM)"""
        self._mixing_speed = rpm
        return True
    
    async def start_process(self, 
                           duration: float,
                           parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Start synthesis process"""
        self._process_running = True
        
        self._update_status(EquipmentStatus(
            state=EquipmentState.BUSY,
            message=f"Running synthesis process ({duration}s)",
            progress=0.0,
            parameters=parameters or {}
        ))
        
        # Simulate process
        start_time = time.time()
        while time.time() - start_time < duration:
            progress = (time.time() - start_time) / duration * 100
            self._update_status(EquipmentStatus(
                state=EquipmentState.BUSY,
                message=f"Processing... {progress:.1f}%",
                progress=progress
            ))
            await asyncio.sleep(1.0)
        
        self._process_running = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message="Process completed",
            progress=100.0
        ))
        return True
    
    async def stop_process(self) -> bool:
        """Stop current process"""
        self._process_running = False
        return True
    
    async def emergency_stop(self) -> bool:
        self._heating = False
        self._process_running = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.ERROR,
            message="EMERGENCY STOP"
        ))
        return True
    
    async def get_status(self) -> EquipmentStatus:
        return self._status


class CharacterizationInstrument(BaseEquipment):
    """
    Characterization instrument interface
    Supports XRD, SEM, electrochemical workstations, etc.
    """
    
    def __init__(self,
                 equipment_id: str,
                 name: str,
                 instrument_type: str,  # "xrd", "sem", "electrochemical"
                 model: str = "",
                 manufacturer: str = "",
                 connection_params: Optional[Dict[str, Any]] = None):
        
        type_map = {
            'xrd': EquipmentType.XRD_INSTRUMENT,
            'sem': EquipmentType.SEM_INSTRUMENT,
            'electrochemical': EquipmentType.ELECTROCHEMICAL_WORKSTATION
        }
        
        super().__init__(
            equipment_id=equipment_id,
            equipment_type=type_map.get(instrument_type, EquipmentType.XRD_INSTRUMENT),
            name=name,
            model=model,
            manufacturer=manufacturer,
            connection_params=connection_params
        )
        
        self.instrument_type = instrument_type
        self._measurement_running = False
        self._last_data: Optional[Dict[str, Any]] = None
    
    async def connect(self) -> bool:
        """Connect to instrument"""
        self._connected = True
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message=f"{self.instrument_type.upper()} instrument connected"
        ))
        return True
    
    async def disconnect(self) -> bool:
        self._connected = False
        return True
    
    async def initialize(self) -> bool:
        """Initialize instrument"""
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message="Instrument initialized"
        ))
        return True
    
    async def load_sample(self, sample_id: str) -> bool:
        """Load sample for measurement"""
        self._update_status(EquipmentStatus(
            state=EquipmentState.BUSY,
            message=f"Loading sample {sample_id}"
        ))
        await asyncio.sleep(1.0)
        return True
    
    async def unload_sample(self) -> bool:
        """Unload sample"""
        await asyncio.sleep(0.5)
        return True
    
    async def run_measurement(self,
                              parameters: Dict[str, Any],
                              duration: Optional[float] = None) -> Dict[str, Any]:
        """Run measurement with given parameters"""
        self._measurement_running = True
        
        self._update_status(EquipmentStatus(
            state=EquipmentState.BUSY,
            message="Running measurement...",
            progress=0.0,
            parameters=parameters
        ))
        
        # Simulate measurement
        duration = duration or 30.0
        start_time = time.time()
        
        while time.time() - start_time < duration:
            progress = (time.time() - start_time) / duration * 100
            self._update_status(EquipmentStatus(
                state=EquipmentState.BUSY,
                message=f"Measuring... {progress:.1f}%",
                progress=progress
            ))
            await asyncio.sleep(0.5)
        
        # Generate mock data
        data = self._generate_mock_data(parameters)
        self._last_data = data
        
        self._measurement_running = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.IDLE,
            message="Measurement completed",
            progress=100.0
        ))
        
        return data
    
    def _generate_mock_data(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate mock measurement data"""
        if self.instrument_type == 'xrd':
            # Generate mock XRD pattern
            two_theta = np.linspace(10, 80, 1401)
            intensity = np.random.normal(100, 20, len(two_theta))
            # Add some peaks
            for peak_pos in [25, 38, 44, 65]:
                intensity += 500 * np.exp(-((two_theta - peak_pos) / 2)**2)
            
            return {
                'two_theta': two_theta.tolist(),
                'intensity': intensity.tolist(),
                'sample_id': parameters.get('sample_id'),
                'wavelength': parameters.get('wavelength', 1.5406),
                'scan_rate': parameters.get('scan_rate', 2.0)
            }
        
        elif self.instrument_type == 'sem':
            # Generate mock SEM metadata
            return {
                'image_data': 'mock_image_base64',
                'magnification': parameters.get('magnification', 10000),
                'accelerating_voltage': parameters.get('voltage', 15.0),
                'working_distance': parameters.get('working_distance', 10.0),
                'sample_id': parameters.get('sample_id')
            }
        
        elif self.instrument_type == 'electrochemical':
            # Generate mock electrochemical data
            if parameters.get('technique') == 'cv':
                potential = np.linspace(-0.5, 1.0, 500)
                current = np.sin(potential * 5) * 1e-4 + np.random.normal(0, 1e-5, 500)
                return {
                    'potential': potential.tolist(),
                    'current': current.tolist(),
                    'scan_rate': parameters.get('scan_rate', 0.1),
                    'sample_id': parameters.get('sample_id')
                }
            elif parameters.get('technique') == 'eis':
                freq = np.logspace(6, -2, 100)
                z_real = np.random.normal(10, 2, 100)
                z_imag = -np.random.normal(5, 1, 100)
                return {
                    'frequency': freq.tolist(),
                    'z_real': z_real.tolist(),
                    'z_imag': z_imag.tolist(),
                    'sample_id': parameters.get('sample_id')
                }
        
        return {'data': 'mock', 'sample_id': parameters.get('sample_id')}
    
    async def get_last_data(self) -> Optional[Dict[str, Any]]:
        """Get last measurement data"""
        return self._last_data
    
    async def emergency_stop(self) -> bool:
        self._measurement_running = False
        self._update_status(EquipmentStatus(
            state=EquipmentState.ERROR,
            message="EMERGENCY STOP"
        ))
        return True
    
    async def get_status(self) -> EquipmentStatus:
        return self._status


class EquipmentManager:
    """Manages all laboratory equipment"""
    
    def __init__(self):
        self._equipment: Dict[str, BaseEquipment] = {}
        self._equipment_by_type: Dict[EquipmentType, List[str]] = {}
    
    def register(self, equipment: BaseEquipment) -> bool:
        """Register equipment with manager"""
        if equipment.equipment_id in self._equipment:
            logger.warning(f"Equipment {equipment.equipment_id} already registered")
            return False
        
        self._equipment[equipment.equipment_id] = equipment
        
        # Index by type
        eq_type = equipment.equipment_type
        if eq_type not in self._equipment_by_type:
            self._equipment_by_type[eq_type] = []
        self._equipment_by_type[eq_type].append(equipment.equipment_id)
        
        logger.info(f"Registered equipment: {equipment.equipment_id}")
        return True
    
    def unregister(self, equipment_id: str) -> bool:
        """Unregister equipment"""
        if equipment_id not in self._equipment:
            return False
        
        equipment = self._equipment[equipment_id]
        eq_type = equipment.equipment_type
        
        del self._equipment[equipment_id]
        self._equipment_by_type[eq_type].remove(equipment_id)
        
        return True
    
    def get(self, equipment_id: str) -> Optional[BaseEquipment]:
        """Get equipment by ID"""
        return self._equipment.get(equipment_id)
    
    def get_by_type(self, equipment_type: EquipmentType) -> List[BaseEquipment]:
        """Get all equipment of a specific type"""
        ids = self._equipment_by_type.get(equipment_type, [])
        return [self._equipment[eid] for eid in ids]
    
    def list_all(self) -> List[str]:
        """List all registered equipment IDs"""
        return list(self._equipment.keys())
    
    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all equipment"""
        results = {}
        for eq_id, equipment in self._equipment.items():
            try:
                results[eq_id] = await equipment.connect()
            except Exception as e:
                logger.error(f"Failed to connect {eq_id}: {e}")
                results[eq_id] = False
        return results
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health check on all equipment"""
        results = {}
        for eq_id, equipment in self._equipment.items():
            try:
                results[eq_id] = await equipment.health_check()
            except Exception as e:
                results[eq_id] = {
                    'equipment_id': eq_id,
                    'healthy': False,
                    'error': str(e)
                }
        return results
    
    async def emergency_stop_all(self) -> Dict[str, bool]:
        """Emergency stop all equipment"""
        results = {}
        for eq_id, equipment in self._equipment.items():
            try:
                results[eq_id] = await equipment.emergency_stop()
            except Exception as e:
                logger.error(f"Emergency stop failed for {eq_id}: {e}")
                results[eq_id] = False
        return results
