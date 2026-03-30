"""
DFT-LAMMPS Metaverse Module
元宇宙材料实验室模块

Provides VR/AR interfaces, collaborative spaces, and immersive visualization
for materials science research and education.

Modules:
    - vr_interface: VR visualization and gesture interaction
    - ar_overlay: AR data overlay and remote guidance
    - collaborative_space: Virtual meeting rooms and shared whiteboards

Author: XR Expert Team
Version: 1.0.0
"""

from .vr_interface import (
    VRInterface,
    VRStructureVisualizer,
    GestureRecognizer,
    VRCollaborativeSpace,
    VRVector3,
    VRQuaternion,
    VRTransform,
    AtomVRData,
    VRCrystalStructure,
    GestureEvent,
    VRUser,
    VRRenderMode,
    GestureType
)

from .ar_overlay import (
    AROverlayManager,
    ARDataOverlay,
    ARSceneUnderstanding,
    ARExperimentAssistant,
    ARRemoteExpert,
    ARVector2,
    ARVector3,
    ARBoundingBox,
    AROverlayType,
    ARAnchorType
)

from .collaborative_space import (
    CollaborativeSpace,
    VirtualMeetingRoom,
    SharedWhiteboard,
    User,
    AvatarState,
    AvatarAppearance,
    ChatMessage,
    UserRole,
    RoomType,
    PresenceStatus
)

__version__ = "1.0.0"
__all__ = [
    # VR Interface
    'VRInterface',
    'VRStructureVisualizer',
    'GestureRecognizer',
    'VRCollaborativeSpace',
    'VRVector3',
    'VRQuaternion',
    'VRTransform',
    'AtomVRData',
    'VRCrystalStructure',
    'GestureEvent',
    'VRUser',
    'VRRenderMode',
    'GestureType',
    
    # AR Overlay
    'AROverlayManager',
    'ARDataOverlay',
    'ARSceneUnderstanding',
    'ARExperimentAssistant',
    'ARRemoteExpert',
    'ARVector2',
    'ARVector3',
    'ARBoundingBox',
    'AROverlayType',
    'ARAnchorType',
    
    # Collaborative Space
    'CollaborativeSpace',
    'VirtualMeetingRoom',
    'SharedWhiteboard',
    'User',
    'AvatarState',
    'AvatarAppearance',
    'ChatMessage',
    'UserRole',
    'RoomType',
    'PresenceStatus'
]
