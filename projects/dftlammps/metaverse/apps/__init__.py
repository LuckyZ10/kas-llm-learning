"""
Metaverse Applications
元宇宙应用

Application modules for immersive materials science research and education.

Apps:
    - virtual_crystallography_lab: VR-based crystal structure analysis
    - remote_collaboration: Distributed research collaboration platform
    - immersive_teaching: VR/AR educational platform

Author: XR Expert Team
Version: 1.0.0
"""

from .virtual_crystallography_lab import VirtualCrystallographyLab
from .remote_collaboration import ResearchCollaborationSession
from .immersive_teaching import ImmersiveTeachingPlatform, CourseModule, Lesson, StudentProgress

__all__ = [
    'VirtualCrystallographyLab',
    'ResearchCollaborationSession',
    'ImmersiveTeachingPlatform',
    'CourseModule',
    'Lesson',
    'StudentProgress'
]
