"""
ROS2 Interface Module for Laboratory Robot Control

Provides integration with ROS2 (Robot Operating System 2) for:
- Robot arm control
- Sensor data acquisition
- Equipment state publishing
- Action client/server interfaces
- Navigation and path planning
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


logger = logging.getLogger(__name__)


# Check if ROS2 is available
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient, ActionServer
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logger.warning("ROS2 not available. Running in mock mode.")


class ROS2NodeStatus(Enum):
    """ROS2 node status"""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    ACTIVE = auto()
    ERROR = auto()
    SHUTDOWN = auto()


@dataclass
class RobotPose:
    """Robot pose in 3D space"""
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Tuple[float, float, float, float]  # quaternion x, y, z, w
    
    @classmethod
    def from_euler(cls,
                   x: float,
                   y: float,
                   z: float,
                   roll: float,
                   pitch: float,
                   yaw: float) -> 'RobotPose':
        """Create pose from Euler angles"""
        # Convert Euler to quaternion
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        
        return cls(
            position=(x, y, z),
            orientation=(qx, qy, qz, qw)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'orientation': self.orientation
        }


@dataclass
class JointState:
    """Robot joint state"""
    name: List[str]
    position: List[float]
    velocity: List[float] = field(default_factory=list)
    effort: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'position': self.position,
            'velocity': self.velocity,
            'effort': self.effort,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class TrajectoryPoint:
    """Trajectory waypoint"""
    pose: RobotPose
    time_from_start: float  # seconds
    velocities: Optional[List[float]] = None
    accelerations: Optional[List[float]] = None


class ROS2Interface(ABC):
    """Abstract base class for ROS2 interfaces"""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self._status = ROS2NodeStatus.UNINITIALIZED
        self._node = None
        self._connected = False
    
    @property
    def status(self) -> ROS2NodeStatus:
        return self._status
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize ROS2 node"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown ROS2 node"""
        pass
    
    @abstractmethod
    def spin(self):
        """Spin the node (blocking)"""
        pass


class MockROS2Interface(ROS2Interface):
    """
    Mock ROS2 interface for testing without ROS2 installation
    Simulates ROS2 behavior for development and testing
    """
    
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._subscribers: Dict[str, List[Callable]] = {}
        self._publishers: Dict[str, Any] = {}
        self._services: Dict[str, Callable] = {}
        self._actions: Dict[str, Any] = {}
        self._running = False
        
        # Simulated robot state
        self._current_pose = RobotPose(
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0)
        )
        self._joint_state = JointState(
            name=['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'],
            position=[0.0] * 6,
            velocity=[0.0] * 6,
            effort=[0.0] * 6
        )
    
    def initialize(self) -> bool:
        """Initialize mock ROS2"""
        logger.info(f"Initializing mock ROS2 node: {self.node_name}")
        self._status = ROS2NodeStatus.ACTIVE
        self._connected = True
        return True
    
    def shutdown(self) -> bool:
        """Shutdown mock ROS2"""
        self._status = ROS2NodeStatus.SHUTDOWN
        self._running = False
        self._connected = False
        return True
    
    def spin(self):
        """Mock spin - does nothing in mock mode"""
        self._running = True
        logger.info(f"Mock ROS2 node '{self.node_name}' spinning")
    
    def create_publisher(self, topic: str, msg_type: str) -> bool:
        """Create mock publisher"""
        self._publishers[topic] = {'type': msg_type}
        logger.info(f"Created mock publisher: {topic}")
        return True
    
    def create_subscriber(self, topic: str, callback: Callable) -> bool:
        """Create mock subscriber"""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        logger.info(f"Created mock subscriber: {topic}")
        return True
    
    def publish(self, topic: str, message: Any) -> bool:
        """Publish message to topic"""
        if topic in self._publishers:
            logger.debug(f"Published to {topic}: {message}")
            return True
        return False
    
    def create_service(self, service_name: str, handler: Callable) -> bool:
        """Create mock service"""
        self._services[service_name] = handler
        logger.info(f"Created mock service: {service_name}")
        return True
    
    def call_service(self, service_name: str, request: Any) -> Any:
        """Call mock service"""
        if service_name in self._services:
            return self._services[service_name](request)
        return None
    
    # Robot-specific mock methods
    def get_current_pose(self) -> RobotPose:
        """Get current robot pose"""
        return self._current_pose
    
    def get_joint_state(self) -> JointState:
        """Get current joint state"""
        return self._joint_state
    
    def set_target_pose(self, pose: RobotPose) -> bool:
        """Set target pose (simulated movement)"""
        self._current_pose = pose
        return True
    
    def set_joint_positions(self, positions: List[float]) -> bool:
        """Set joint positions"""
        self._joint_state.position = positions
        return True


class RealROS2Interface(ROS2Interface):
    """
    Real ROS2 interface using rclpy
    Requires ROS2 installation
    """
    
    def __init__(self, node_name: str, namespace: str = ""):
        super().__init__(node_name)
        self.namespace = namespace
        self._publishers = {}
        self._subscribers = {}
        self._services = {}
        self._action_clients = {}
        self._callback_group = None
    
    def initialize(self) -> bool:
        """Initialize ROS2 node"""
        if not ROS2_AVAILABLE:
            logger.error("ROS2 not available")
            return False
        
        try:
            rclpy.init()
            
            self._callback_group = ReentrantCallbackGroup()
            
            self._node = rclpy.create_node(
                self.node_name,
                namespace=self.namespace
            )
            
            self._status = ROS2NodeStatus.ACTIVE
            self._connected = True
            
            logger.info(f"ROS2 node '{self.node_name}' initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ROS2: {e}")
            self._status = ROS2NodeStatus.ERROR
            return False
    
    def shutdown(self) -> bool:
        """Shutdown ROS2 node"""
        if self._node:
            self._node.destroy_node()
            rclpy.shutdown()
        
        self._status = ROS2NodeStatus.SHUTDOWN
        self._connected = False
        return True
    
    def spin(self):
        """Spin the node"""
        if self._node:
            rclpy.spin(self._node)
    
    def create_publisher(self, topic: str, msg_type: Any, qos: int = 10) -> bool:
        """Create ROS2 publisher"""
        if not self._node:
            return False
        
        try:
            publisher = self._node.create_publisher(msg_type, topic, qos)
            self._publishers[topic] = publisher
            return True
        except Exception as e:
            logger.error(f"Failed to create publisher: {e}")
            return False
    
    def create_subscriber(self, 
                         topic: str,
                         msg_type: Any,
                         callback: Callable,
                         qos: int = 10) -> bool:
        """Create ROS2 subscriber"""
        if not self._node:
            return False
        
        try:
            subscription = self._node.create_subscription(
                msg_type,
                topic,
                callback,
                qos,
                callback_group=self._callback_group
            )
            self._subscribers[topic] = subscription
            return True
        except Exception as e:
            logger.error(f"Failed to create subscriber: {e}")
            return False
    
    def publish(self, topic: str, message: Any) -> bool:
        """Publish message"""
        if topic in self._publishers:
            self._publishers[topic].publish(message)
            return True
        return False
    
    def create_service(self,
                      service_name: str,
                      srv_type: Any,
                      callback: Callable) -> bool:
        """Create ROS2 service"""
        if not self._node:
            return False
        
        try:
            service = self._node.create_service(
                srv_type,
                service_name,
                callback
            )
            self._services[service_name] = service
            return True
        except Exception as e:
            logger.error(f"Failed to create service: {e}")
            return False
    
    def create_action_client(self,
                            action_name: str,
                            action_type: Any) -> bool:
        """Create ROS2 action client"""
        if not self._node:
            return False
        
        try:
            client = ActionClient(
                self._node,
                action_type,
                action_name
            )
            self._action_clients[action_name] = client
            return True
        except Exception as e:
            logger.error(f"Failed to create action client: {e}")
            return False


class RobotController:
    """
    High-level robot controller using ROS2
    Provides abstraction for robot arm control
    """
    
    def __init__(self,
                 ros_interface: ROS2Interface,
                 robot_name: str = "robot_arm"):
        self.ros = ros_interface
        self.robot_name = robot_name
        self._current_pose: Optional[RobotPose] = None
        self._joint_state: Optional[JointState] = None
        self._is_moving = False
    
    def initialize(self) -> bool:
        """Initialize robot controller"""
        # Setup subscribers
        self.ros.create_subscriber(
            f"/{self.robot_name}/joint_states",
            "sensor_msgs/JointState",
            self._joint_state_callback
        )
        
        self.ros.create_subscriber(
            f"/{self.robot_name}/current_pose",
            "geometry_msgs/Pose",
            self._pose_callback
        )
        
        # Setup publishers
        self.ros.create_publisher(
            f"/{self.robot_name}/target_pose",
            "geometry_msgs/Pose"
        )
        
        self.ros.create_publisher(
            f"/{self.robot_name}/joint_command",
            "trajectory_msgs/JointTrajectory"
        )
        
        return True
    
    def _joint_state_callback(self, msg: Any):
        """Handle joint state updates"""
        self._joint_state = JointState(
            name=msg.name,
            position=list(msg.position),
            velocity=list(msg.velocity) if msg.velocity else [],
            effort=list(msg.effort) if msg.effort else []
        )
    
    def _pose_callback(self, msg: Any):
        """Handle pose updates"""
        self._current_pose = RobotPose(
            position=(msg.position.x, msg.position.y, msg.position.z),
            orientation=(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            )
        )
    
    def get_current_pose(self) -> Optional[RobotPose]:
        """Get current robot pose"""
        return self._current_pose
    
    def get_joint_state(self) -> Optional[JointState]:
        """Get current joint state"""
        return self._joint_state
    
    async def move_to_pose(self,
                          target_pose: RobotPose,
                          velocity_scaling: float = 0.5,
                          acceleration_scaling: float = 0.5) -> bool:
        """Move robot to target pose"""
        if self._is_moving:
            logger.warning("Robot is already moving")
            return False
        
        self._is_moving = True
        
        try:
            # Create trajectory message
            trajectory = {
                'joint_names': self._joint_state.name if self._joint_state else [],
                'points': [{
                    'positions': [0.0] * 6,  # Would be calculated from IK
                    'time_from_start': {'sec': 2, 'nanosec': 0}
                }]
            }
            
            # Publish trajectory
            self.ros.publish(
                f"/{self.robot_name}/joint_command",
                trajectory
            )
            
            # Wait for movement to complete (mock)
            await asyncio.sleep(2.0)
            
            self._is_moving = False
            return True
            
        except Exception as e:
            logger.error(f"Move failed: {e}")
            self._is_moving = False
            return False
    
    async def move_joints(self,
                         target_positions: List[float],
                         duration: float = 2.0) -> bool:
        """Move joints to target positions"""
        if len(target_positions) != 6:
            logger.error("Expected 6 joint positions")
            return False
        
        trajectory = {
            'joint_names': ['joint1', 'joint2', 'joint3', 
                           'joint4', 'joint5', 'joint6'],
            'points': [{
                'positions': target_positions,
                'time_from_start': {'sec': int(duration), 
                                   'nanosec': int((duration % 1) * 1e9)}
            }]
        }
        
        self.ros.publish(
            f"/{self.robot_name}/joint_command",
            trajectory
        )
        
        await asyncio.sleep(duration)
        return True
    
    async def linear_move(self,
                         target_pose: RobotPose,
                         velocity: float = 0.1) -> bool:
        """Linear movement in Cartesian space"""
        # This would use Cartesian path planning
        return await self.move_to_pose(target_pose)
    
    def is_moving(self) -> bool:
        """Check if robot is currently moving"""
        return self._is_moving
    
    def stop(self) -> bool:
        """Emergency stop"""
        # Publish stop command
        self.ros.publish(
            f"/{self.robot_name}/emergency_stop",
            {'data': True}
        )
        self._is_moving = False
        return True


class EquipmentROSBridge:
    """
    Bridge between laboratory equipment and ROS2
    Publishes equipment state and accepts commands
    """
    
    def __init__(self,
                 ros_interface: ROS2Interface,
                 equipment_manager: Any):
        self.ros = ros_interface
        self.equipment_manager = equipment_manager
        self._publishers = {}
        self._services = {}
    
    def initialize(self) -> bool:
        """Initialize ROS bridge"""
        # Create publishers for equipment states
        self.ros.create_publisher(
            '/lab/equipment_states',
            'std_msgs/String'
        )
        
        # Create service for equipment commands
        self.ros.create_service(
            '/lab/equipment_command',
            'lab_automation/EquipmentCommand',
            self._handle_equipment_command
        )
        
        return True
    
    def _handle_equipment_command(self, request: Any, response: Any) -> Any:
        """Handle equipment command requests"""
        equipment_id = request.equipment_id
        command = request.command
        params = json.loads(request.parameters)
        
        equipment = self.equipment_manager.get(equipment_id)
        
        if not equipment:
            response.success = False
            response.message = f"Equipment {equipment_id} not found"
            return response
        
        try:
            # Execute command
            if command == 'initialize':
                asyncio.create_task(equipment.initialize())
            elif command == 'start':
                asyncio.create_task(
                    equipment.start_process(params.get('duration', 60))
                )
            elif command == 'stop':
                asyncio.create_task(equipment.stop_process())
            elif command == 'set_temperature':
                asyncio.create_task(
                    equipment.set_temperature(params.get('temperature', 25))
                )
            
            response.success = True
            response.message = f"Command {command} sent to {equipment_id}"
            
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response
    
    async def publish_equipment_states(self):
        """Publish equipment states periodically"""
        while True:
            try:
                states = await self.equipment_manager.health_check_all()
                
                message = {
                    'timestamp': datetime.now().isoformat(),
                    'equipment': states
                }
                
                self.ros.publish(
                    '/lab/equipment_states',
                    {'data': json.dumps(message)}
                )
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Failed to publish equipment states: {e}")
                await asyncio.sleep(5.0)


class NavigationController:
    """
    Navigation controller for mobile robots
    Integrates with ROS2 navigation stack
    """
    
    def __init__(self, ros_interface: ROS2Interface):
        self.ros = ros_interface
        self._current_position: Optional[Tuple[float, float]] = None
        self._current_orientation: float = 0.0
        self._path: List[Tuple[float, float]] = []
    
    def initialize(self) -> bool:
        """Initialize navigation"""
        # Subscribe to odometry
        self.ros.create_subscriber(
            '/odom',
            'nav_msgs/Odometry',
            self._odom_callback
        )
        
        # Subscribe to path
        self.ros.create_subscriber(
            '/plan',
            'nav_msgs/Path',
            self._path_callback
        )
        
        # Publisher for goals
        self.ros.create_publisher(
            '/move_base_simple/goal',
            'geometry_msgs/PoseStamped'
        )
        
        return True
    
    def _odom_callback(self, msg: Any):
        """Handle odometry updates"""
        self._current_position = (
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        )
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self._current_orientation = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
    
    def _path_callback(self, msg: Any):
        """Handle path updates"""
        self._path = [
            (pose.pose.position.x, pose.pose.position.y)
            for pose in msg.poses
        ]
    
    def get_current_position(self) -> Optional[Tuple[float, float]]:
        """Get current robot position"""
        return self._current_position
    
    def send_goal(self, 
                 x: float,
                 y: float,
                 theta: float = 0.0) -> bool:
        """Send navigation goal"""
        goal = {
            'header': {
                'frame_id': 'map',
                'stamp': {'sec': 0, 'nanosec': 0}
            },
            'pose': {
                'position': {'x': x, 'y': y, 'z': 0.0},
                'orientation': {
                    'x': 0.0,
                    'y': 0.0,
                    'z': np.sin(theta / 2),
                    'w': np.cos(theta / 2)
                }
            }
        }
        
        return self.ros.publish('/move_base_simple/goal', goal)
    
    def cancel_goal(self) -> bool:
        """Cancel current navigation goal"""
        # Publish cancel command
        return self.ros.publish('/move_base/cancel', {})
    
    def get_path(self) -> List[Tuple[float, float]]:
        """Get current planned path"""
        return self._path
    
    def is_at_goal(self, tolerance: float = 0.1) -> bool:
        """Check if robot is at goal position"""
        if not self._current_position or not self._path:
            return False
        
        goal = self._path[-1]
        distance = np.sqrt(
            (self._current_position[0] - goal[0]) ** 2 +
            (self._current_position[1] - goal[1]) ** 2
        )
        
        return distance < tolerance


def create_ros_interface(node_name: str,
                        use_mock: bool = False) -> ROS2Interface:
    """
    Factory function to create appropriate ROS2 interface
    
    Args:
        node_name: Name for the ROS2 node
        use_mock: Force use of mock interface even if ROS2 is available
    
    Returns:
        ROS2Interface instance
    """
    if use_mock or not ROS2_AVAILABLE:
        return MockROS2Interface(node_name)
    else:
        return RealROS2Interface(node_name)
