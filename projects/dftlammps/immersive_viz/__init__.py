"""
DFT-LAMMPS Immersive Visualization Module
沉浸式可视化模块

Provides 3D volume rendering, spatiotemporal animation, and haptic feedback
for materials science visualization.

Modules:
    - volume_renderer: 3D volume rendering and isosurface extraction
    - animation: Spatiotemporal data animation
    - haptics: Haptic feedback for VR interaction

Author: XR Expert Team
Version: 1.0.0
"""

from .volume_renderer import (
    VolumeRenderer,
    VolumeData,
    TransferFunction,
    RenderSettings,
    CameraSettings,
    VolumeRenderMode,
    ColorMap,
    AnimationFrame,
    SpatiotemporalAnimator,
    AnimationMode,
    HapticEvent,
    HapticFeedbackSystem,
    ImmersiveVizManager
)

from .animation import (
    SpatiotemporalAnimator as AdvancedAnimator,
    AnimationTrack,
    AnimationLayer,
    Keyframe,
    InterpolationMethod,
    TimeWarpMode
)

from .haptics import (
    AdvancedHapticSystem,
    HapticChannel,
    MaterialHapticProfile,
    ForceFeedbackConfig,
    HapticDeviceType,
    HapticPattern
)

__version__ = "1.0.0"
__all__ = [
    # Volume Rendering
    'VolumeRenderer',
    'VolumeData',
    'TransferFunction',
    'RenderSettings',
    'CameraSettings',
    'VolumeRenderMode',
    'ColorMap',
    
    # Animation
    'SpatiotemporalAnimator',
    'AdvancedAnimator',
    'AnimationFrame',
    'AnimationTrack',
    'AnimationLayer',
    'Keyframe',
    'AnimationMode',
    'InterpolationMethod',
    'TimeWarpMode',
    
    # Haptics
    'HapticEvent',
    'HapticFeedbackSystem',
    'AdvancedHapticSystem',
    'HapticChannel',
    'MaterialHapticProfile',
    'ForceFeedbackConfig',
    'HapticDeviceType',
    'HapticPattern',
    
    # Manager
    'ImmersiveVizManager'
]
