#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Immersive Materials Teaching Application
沉浸式材料教学应用

Educational platform for interactive materials science learning in VR/AR.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metaverse.vr_interface import (
    VRInterface, VRRenderMode, VRVector3, VRQuaternion,
    GestureType, GestureEvent
)
from metaverse.ar_overlay import (
    AROverlayManager, ARExperimentAssistant, ARGuidanceElement
)
from metaverse.collaborative_space import (
    CollaborativeSpace, VirtualMeetingRoom, RoomType,
    User, UserRole, PresenceStatus
)
from immersive_viz.volume_renderer import ImmersiveVizManager
from immersive_viz.haptics import AdvancedHapticSystem, HapticDeviceType

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CourseModule:
    """课程模块"""
    
    def __init__(self, module_id: str, title: str, description: str = ""):
        self.module_id = module_id
        self.title = title
        self.description = description
        self.lessons: List['Lesson'] = []
        self.prerequisites: List[str] = []
        self.learning_objectives: List[str] = []
        self.duration_minutes: int = 60
        self.difficulty: str = "beginner"  # beginner, intermediate, advanced
        
    def add_lesson(self, lesson: 'Lesson') -> None:
        """添加课程"""
        self.lessons.append(lesson)
        
    def get_progress(self, completed_lessons: List[str]) -> float:
        """获取进度"""
        if not self.lessons:
            return 0.0
        completed = sum(1 for l in self.lessons if l.lesson_id in completed_lessons)
        return completed / len(self.lessons)


class Lesson:
    """单节课"""
    
    def __init__(self, lesson_id: str, title: str, content_type: str = "vr_experience"):
        self.lesson_id = lesson_id
        self.title = title
        self.content_type = content_type  # vr_experience, ar_lab, simulation, quiz
        self.content: Dict[str, any] = {}
        self.interactions: List[Dict[str, any]] = []
        self.quiz_questions: List[Dict[str, any]] = []
        self.estimated_time: int = 15  # 分钟
        
    def add_interaction(self, interaction_type: str, trigger: str, action: str) -> None:
        """添加交互"""
        self.interactions.append({
            "type": interaction_type,
            "trigger": trigger,
            "action": action
        })


class StudentProgress:
    """学生学习进度"""
    
    def __init__(self, student_id: str):
        self.student_id = student_id
        self.completed_lessons: List[str] = []
        self.completed_modules: List[str] = []
        self.quiz_scores: Dict[str, float] = {}
        self.time_spent: Dict[str, int] = {}  # 分钟
        self.achievements: List[str] = []
        self.interaction_history: List[Dict[str, any]] = []
        
    def complete_lesson(self, lesson_id: str, score: float = 0) -> None:
        """完成课程"""
        if lesson_id not in self.completed_lessons:
            self.completed_lessons.append(lesson_id)
            self.quiz_scores[lesson_id] = score
            
    def add_time_spent(self, lesson_id: str, minutes: int) -> None:
        """添加学习时间"""
        self.time_spent[lesson_id] = self.time_spent.get(lesson_id, 0) + minutes


class ImmersiveTeachingPlatform:
    """沉浸式教学平台"""
    
    def __init__(self, platform_name: str = "Materials Learning VR"):
        self.name = platform_name
        self.space = CollaborativeSpace()
        
        # 子系统
        self.vr = VRInterface()
        self.ar = AROverlayManager()
        self.viz = ImmersiveVizManager()
        self.haptics = AdvancedHapticSystem()
        self.experiment_assistant = ARExperimentAssistant()
        
        # 课程内容
        self.modules: Dict[str, CourseModule] = {}
        self.lessons: Dict[str, Lesson] = {}
        
        # 学生数据
        self.students: Dict[str, User] = {}
        self.progress: Dict[str, StudentProgress] = {}
        
        # 教室
        self.classrooms: Dict[str, VirtualMeetingRoom] = {}
        
        # 活动会话
        self.active_sessions: Dict[str, Dict[str, any]] = {}
        
    async def initialize(self) -> bool:
        """初始化平台"""
        logger.info(f"Initializing {self.name}...")
        
        await self.vr.initialize()
        self.ar.initialize()
        self.haptics.connect_device(HapticDeviceType.VR_CONTROLLER)
        
        # 加载课程
        self._load_curriculum()
        
        logger.info(f"{self.name} ready")
        return True
    
    def _load_curriculum(self) -> None:
        """加载课程"""
        # 模块1: 晶体结构基础
        module1 = CourseModule(
            "mod_001",
            "Introduction to Crystal Structures",
            "Learn the basics of crystal structures and Bravais lattices"
        )
        module1.learning_objectives = [
            "Understand the concept of unit cells",
            "Identify the 14 Bravais lattices",
            "Calculate atomic packing factors",
            "Visualize crystal structures in 3D"
        ]
        
        # 课程1.1: 简单立方结构
        lesson1 = Lesson("les_001", "Simple Cubic Structure", "vr_experience")
        lesson1.content = {
            "structure_type": "simple_cubic",
            "atoms": 8,
            "coordination": 6,
            "packing_factor": 0.52
        }
        lesson1.add_interaction("gesture", "grab", "rotate_structure")
        lesson1.add_interaction("voice", "show unit cell", "highlight_unitcell")
        
        # 课程1.2: 体心立方结构
        lesson2 = Lesson("les_002", "Body-Centered Cubic", "vr_experience")
        lesson2.content = {
            "structure_type": "bcc",
            "examples": ["Iron", "Chromium", "Tungsten"],
            "coordination": 8,
            "packing_factor": 0.68
        }
        
        # 课程1.3: 面心立方结构
        lesson3 = Lesson("les_003", "Face-Centered Cubic", "vr_experience")
        lesson3.content = {
            "structure_type": "fcc",
            "examples": ["Aluminum", "Copper", "Gold"],
            "coordination": 12,
            "packing_factor": 0.74
        }
        
        module1.add_lesson(lesson1)
        module1.add_lesson(lesson2)
        module1.add_lesson(lesson3)
        
        # 模块2: 材料力学性能
        module2 = CourseModule(
            "mod_002",
            "Mechanical Properties of Materials",
            "Explore stress, strain, and deformation mechanisms"
        )
        module2.difficulty = "intermediate"
        module2.prerequisites = ["mod_001"]
        
        # 课程2.1: 弹性变形
        lesson4 = Lesson("les_004", "Elastic Deformation", "simulation")
        lesson4.content = {
            "simulation_type": "tensile_test",
            "modulus_range": [50, 400],  # GPa
            "interactive": True
        }
        
        # 课程2.2: 塑性变形
        lesson5 = Lesson("les_005", "Plastic Deformation", "simulation")
        lesson5.content = {
            "simulation_type": "dislocation_motion",
            "show_dislocations": True,
            "haptic_feedback": True
        }
        
        module2.add_lesson(lesson4)
        module2.add_lesson(lesson5)
        
        # 模块3: 纳米材料
        module3 = CourseModule(
            "mod_003",
            "Nanomaterials and Nanotechnology",
            "Discover the unique properties of nanoscale materials"
        )
        module3.difficulty = "advanced"
        module3.prerequisites = ["mod_001", "mod_002"]
        
        # 课程3.1: 碳纳米管
        lesson6 = Lesson("les_006", "Carbon Nanotubes", "vr_experience")
        lesson6.content = {
            "structure_type": "nanotube",
            "chiralities": [(5, 5), (10, 10), (10, 0)],
            "properties": ["mechanical", "electrical", "thermal"]
        }
        
        module3.add_lesson(lesson6)
        
        # 保存课程
        self.modules[module1.module_id] = module1
        self.modules[module2.module_id] = module2
        self.modules[module3.module_id] = module3
        
        for module in [module1, module2, module3]:
            for lesson in module.lessons:
                self.lessons[lesson.lesson_id] = lesson
        
        logger.info(f"Loaded {len(self.modules)} modules with {len(self.lessons)} lessons")
    
    def register_student(self, student_id: str, name: str, email: str) -> User:
        """注册学生"""
        user = self.space.create_user(student_id, name, email, UserRole.VISITOR)
        self.students[student_id] = user
        self.progress[student_id] = StudentProgress(student_id)
        
        logger.info(f"Student registered: {name} ({student_id})")
        return user
    
    def create_classroom(self, course_id: str, instructor_id: str,
                        max_students: int = 30) -> str:
        """创建虚拟教室"""
        room_id = self.space.create_room(
            name=f"Class: {self.modules.get(course_id, CourseModule(course_id, '')).title}",
            room_type=RoomType.CLASSROOM,
            owner_id=instructor_id,
            max_participants=max_students + 1  # +1 for instructor
        )
        
        self.classrooms[room_id] = self.space.get_room(room_id)
        
        # 设置教室白板
        room = self.space.get_room(room_id)
        board_id = room.create_whiteboard("Lesson Notes")
        
        logger.info(f"Classroom created: {room_id}")
        return room_id
    
    def start_lesson(self, room_id: str, lesson_id: str,
                    instructor_id: str) -> bool:
        """开始上课"""
        if lesson_id not in self.lessons:
            return False
        
        lesson = self.lessons[lesson_id]
        room = self.space.get_room(room_id)
        
        if not room:
            return False
        
        # 发送开始通知
        room.send_system_message(f"Starting lesson: {lesson.title}")
        
        # 加载VR内容
        if lesson.content_type == "vr_experience":
            self._load_vr_lesson_content(lesson)
        elif lesson.content_type == "simulation":
            self._load_simulation_content(lesson)
        
        # 启动AR引导
        self.ar.experiment_assistant.load_protocol(
            f"lesson_{lesson_id}",
            self._generate_lesson_steps(lesson)
        )
        self.ar.start_experiment_protocol(f"lesson_{lesson_id}")
        
        # 记录会话
        self.active_sessions[lesson_id] = {
            "room_id": room_id,
            "instructor": instructor_id,
            "start_time": datetime.now(),
            "students_joined": [],
            "interactions": []
        }
        
        logger.info(f"Lesson started: {lesson.title} in room {room_id}")
        return True
    
    def _load_vr_lesson_content(self, lesson: Lesson) -> None:
        """加载VR课程内容"""
        content = lesson.content
        
        if content.get("structure_type") == "simple_cubic":
            # 创建简单立方结构
            structure_id = f"lesson_{lesson.lesson_id}_sc"
            # 使用可视化器创建结构
            logger.info(f"Loaded VR content: Simple Cubic Structure")
            
        elif content.get("structure_type") == "bcc":
            logger.info(f"Loaded VR content: BCC Structure")
            
        elif content.get("structure_type") == "fcc":
            logger.info(f"Loaded VR content: FCC Structure")
            
        elif content.get("structure_type") == "nanotube":
            logger.info(f"Loaded VR content: Carbon Nanotube")
    
    def _load_simulation_content(self, lesson: Lesson) -> None:
        """加载模拟内容"""
        content = lesson.content
        
        if content.get("simulation_type") == "tensile_test":
            logger.info("Loaded simulation: Tensile Test")
            # 启用触觉反馈
            if content.get("haptic_feedback"):
                self.haptics.connect_device(HapticDeviceType.FORCE_FEEDBACK_ARM)
                
        elif content.get("simulation_type") == "dislocation_motion":
            logger.info("Loaded simulation: Dislocation Motion")
    
    def _generate_lesson_steps(self, lesson: Lesson) -> List[Dict[str, str]]:
        """生成课程步骤"""
        steps = [
            {"instruction": f"Welcome to {lesson.title}. Follow the guide to explore."},
            {"instruction": "Use hand gestures to rotate and zoom the 3D model."},
            {"instruction": "Click on atoms to see their properties."},
            {"instruction": "Complete the quiz at the end of the lesson."}
        ]
        
        # 添加课程特定步骤
        if lesson.content.get("structure_type"):
            steps.insert(1, {
                "instruction": f"Examine the {lesson.content['structure_type'].upper()} structure carefully."
            })
        
        return steps
    
    def handle_student_interaction(self, student_id: str, lesson_id: str,
                                   interaction_type: str, data: Dict[str, any]) -> Dict[str, any]:
        """处理学生交互"""
        response = {"status": "ok"}
        
        if interaction_type == "gesture":
            # 处理手势
            gesture = data.get("gesture")
            if gesture == "rotate":
                response["action"] = "rotate_model"
            elif gesture == "zoom":
                response["action"] = "zoom_model"
            elif gesture == "select":
                response["action"] = "show_atom_info"
                
        elif interaction_type == "quiz_answer":
            # 处理测验答案
            question_id = data.get("question_id")
            answer = data.get("answer")
            is_correct = self._check_answer(question_id, answer)
            
            # 更新进度
            progress = self.progress[student_id]
            if lesson_id not in progress.quiz_scores:
                progress.quiz_scores[lesson_id] = 0
            
            if is_correct:
                progress.quiz_scores[lesson_id] += 1
                response["correct"] = True
                response["feedback"] = "Correct! Well done."
            else:
                response["correct"] = False
                response["feedback"] = "Try again. Hint: Consider the coordination number."
                
        elif interaction_type == "voice_command":
            # 处理语音命令
            command = data.get("command", "").lower()
            if "explain" in command:
                response["action"] = "play_explanation"
            elif "quiz" in command:
                response["action"] = "start_quiz"
            elif "help" in command:
                response["action"] = "show_help"
        
        # 记录交互
        if lesson_id in self.active_sessions:
            self.active_sessions[lesson_id]["interactions"].append({
                "student": student_id,
                "type": interaction_type,
                "timestamp": datetime.now()
            })
        
        return response
    
    def _check_answer(self, question_id: str, answer: any) -> bool:
        """检查答案"""
        # 简化的答案检查
        # 实际实现会从数据库加载正确答案
        correct_answers = {
            "q_001": "6",
            "q_002": "0.74",
            "q_003": "12"
        }
        return str(answer) == correct_answers.get(question_id, "")
    
    def complete_lesson(self, student_id: str, lesson_id: str) -> Dict[str, any]:
        """完成课程"""
        progress = self.progress[student_id]
        
        # 计算分数
        quiz_score = progress.quiz_scores.get(lesson_id, 0)
        max_score = len(self.lessons[lesson_id].quiz_questions) or 1
        score_percentage = (quiz_score / max_score) * 100
        
        # 记录完成
        progress.complete_lesson(lesson_id, score_percentage)
        
        # 检查成就
        achievements = self._check_achievements(student_id)
        
        result = {
            "lesson_id": lesson_id,
            "score": score_percentage,
            "completed_at": datetime.now().isoformat(),
            "achievements_earned": achievements,
            "time_spent": progress.time_spent.get(lesson_id, 0)
        }
        
        logger.info(f"Student {student_id} completed lesson {lesson_id} with score {score_percentage:.1f}%")
        return result
    
    def _check_achievements(self, student_id: str) -> List[str]:
        """检查成就"""
        progress = self.progress[student_id]
        new_achievements = []
        
        # 完成第一个课程
        if len(progress.completed_lessons) == 1 and "first_lesson" not in progress.achievements:
            new_achievements.append("first_lesson")
            progress.achievements.append("first_lesson")
        
        # 完成所有基础课程
        basic_lessons = ["les_001", "les_002", "les_003"]
        if all(l in progress.completed_lessons for l in basic_lessons):
            if "crystal_master" not in progress.achievements:
                new_achievements.append("crystal_master")
                progress.achievements.append("crystal_master")
        
        # 高分
        high_scores = sum(1 for s in progress.quiz_scores.values() if s >= 90)
        if high_scores >= 3 and "excellent_student" not in progress.achievements:
            new_achievements.append("excellent_student")
            progress.achievements.append("excellent_student")
        
        return new_achievements
    
    def get_student_dashboard(self, student_id: str) -> Dict[str, any]:
        """获取学生仪表板"""
        progress = self.progress.get(student_id)
        if not progress:
            return {}
        
        student = self.students.get(student_id)
        
        return {
            "student_name": student.display_name if student else "Unknown",
            "completed_lessons": len(progress.completed_lessons),
            "total_lessons": len(self.lessons),
            "overall_progress": len(progress.completed_lessons) / len(self.lessons) * 100 if self.lessons else 0,
            "average_score": np.mean(list(progress.quiz_scores.values())) if progress.quiz_scores else 0,
            "total_time_spent": sum(progress.time_spent.values()),
            "achievements": progress.achievements,
            "next_recommended": self._recommend_next_lesson(student_id)
        }
    
    def _recommend_next_lesson(self, student_id: str) -> Optional[str]:
        """推荐下一节课"""
        progress = self.progress[student_id]
        
        # 找到第一个未完成的课程
        for lesson_id, lesson in self.lessons.items():
            if lesson_id not in progress.completed_lessons:
                return lesson_id
        
        return None
    
    def get_teacher_analytics(self, room_id: str) -> Dict[str, any]:
        """获取教师分析数据"""
        room = self.space.get_room(room_id)
        if not room:
            return {}
        
        return {
            "active_students": room.participant_count - 1,  # Exclude instructor
            "avg_engagement": 85,  # Placeholder
            "questions_asked": len([i for i in self.active_sessions.get(room_id, {}).get("interactions", [])
                                   if i["type"] == "question"]),
            "completion_rate": 75,  # Placeholder
            "common_difficulties": ["Coordination numbers", "Packing factors"]
        }


def run_teaching_demo():
    """运行教学演示"""
    print("=" * 70)
    print("  Immersive Materials Teaching Platform")
    print("  沉浸式材料教学平台")
    print("=" * 70)
    
    # 创建教学平台
    platform = ImmersiveTeachingPlatform("Materials Science VR Classroom")
    
    # 初始化
    import asyncio
    asyncio.run(platform.initialize())
    print(f"\n✓ Platform initialized: {platform.name}")
    
    # 显示课程
    print("\n[1] Curriculum Overview")
    print(f"  Total Modules: {len(platform.modules)}")
    print(f"  Total Lessons: {len(platform.lessons)}")
    
    for mod_id, module in platform.modules.items():
        print(f"\n  Module: {module.title}")
        print(f"    Difficulty: {module.difficulty}")
        print(f"    Lessons: {len(module.lessons)}")
        for lesson in module.lessons:
            print(f"      - {lesson.title} ({lesson.content_type})")
    
    # 注册学生
    print("\n[2] Student Registration")
    students = [
        ("stu_001", "Zhang Ming", "zhang@student.edu"),
        ("stu_002", "Li Hua", "li@student.edu"),
        ("stu_003", "Wang Fang", "wang@student.edu"),
    ]
    
    for sid, name, email in students:
        user = platform.register_student(sid, name, email)
        print(f"  ✓ Registered: {name} ({sid})")
    
    # 创建教室
    print("\n[3] Creating Virtual Classroom")
    room_id = platform.create_classroom("mod_001", "instructor_001", max_students=20)
    print(f"  ✓ Classroom created: {room_id}")
    
    # 开始课程
    print("\n[4] Starting VR Lesson")
    lesson_id = "les_001"  # 简单立方结构
    success = platform.start_lesson(room_id, lesson_id, "instructor_001")
    print(f"  {'✓' if success else '✗'} Started lesson: {platform.lessons[lesson_id].title}")
    
    # 模拟学生交互
    print("\n[5] Student Interactions")
    interactions = [
        ("stu_001", "gesture", {"gesture": "rotate"}),
        ("stu_002", "gesture", {"gesture": "zoom"}),
        ("stu_001", "voice_command", {"command": "explain coordination number"}),
        ("stu_002", "quiz_answer", {"question_id": "q_001", "answer": "6"}),
        ("stu_003", "quiz_answer", {"question_id": "q_001", "answer": "8"}),
    ]
    
    for student_id, int_type, data in interactions:
        response = platform.handle_student_interaction(student_id, lesson_id, int_type, data)
        print(f"  {student_id} - {int_type}: {response.get('action', response.get('feedback', 'ok'))}")
    
    # 完成课程
    print("\n[6] Completing Lessons")
    for student_id, _, _ in students:
        result = platform.complete_lesson(student_id, lesson_id)
        print(f"  ✓ {student_id}: Score {result['score']:.1f}%, "
              f"Achievements: {result['achievements_earned']}")
    
    # 学生仪表板
    print("\n[7] Student Dashboards")
    for student_id, _, _ in students:
        dashboard = platform.get_student_dashboard(student_id)
        print(f"\n  {dashboard['student_name']}:")
        print(f"    Progress: {dashboard['overall_progress']:.1f}%")
        print(f"    Average Score: {dashboard['average_score']:.1f}%")
        print(f"    Achievements: {dashboard['achievements']}")
        if dashboard['next_recommended']:
            print(f"    Next Lesson: {platform.lessons[dashboard['next_recommended']].title}")
    
    # 教师分析
    print("\n[8] Teacher Analytics")
    analytics = platform.get_teacher_analytics(room_id)
    print(f"  Active Students: {analytics.get('active_students', 0)}")
    print(f"  Common Difficulties: {', '.join(analytics.get('common_difficulties', []))}")
    
    print("\n" + "=" * 70)
    print("  Demo Complete - Immersive Materials Teaching Platform")
    print("=" * 70)


if __name__ == "__main__":
    run_teaching_demo()
