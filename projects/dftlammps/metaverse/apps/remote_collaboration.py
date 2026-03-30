#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remote Collaborative Research Application
远程协作研究应用

Enables distributed materials research teams to collaborate in virtual space.

Author: XR Expert Team
Version: 1.0.0
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from metaverse.collaborative_space import (
    CollaborativeSpace, VirtualMeetingRoom, SharedWhiteboard,
    User, UserRole, RoomType, PresenceStatus, AvatarState, AvatarAppearance
)
from metaverse.vr_interface import VRInterface, VRVector3, VRQuaternion
from metaverse.ar_overlay import AROverlayManager
from immersive_viz.volume_renderer import ImmersiveVizManager

import logging
import asyncio
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchCollaborationSession:
    """研究协作会话"""
    
    def __init__(self, session_name: str):
        self.name = session_name
        self.space = CollaborativeSpace()
        self.room_id: str = ""
        self.participants: dict = {}
        self.shared_data: dict = {}
        self.session_log: list = []
        
        # 子系统
        self.vr = VRInterface()
        self.ar = AROverlayManager()
        self.viz = ImmersiveVizManager()
        
        # 研究数据
        self.structures: dict = {}
        self.simulations: dict = {}
        self.documents: dict = {}
        
    async def initialize(self) -> bool:
        """初始化协作会话"""
        logger.info(f"Initializing research collaboration session: {self.name}")
        
        # 创建虚拟会议室
        self.room_id = self.space.create_room(
            name=self.name,
            room_type=RoomType.LABORATORY,
            max_participants=20
        )
        
        # 初始化子系统
        await self.vr.initialize()
        self.ar.initialize()
        
        self._log_event("session_initialized", {"room_id": self.room_id})
        
        logger.info(f"Collaboration session ready: {self.room_id}")
        return True
    
    def invite_researcher(self, user_id: str, display_name: str,
                         email: str, role: UserRole = UserRole.RESEARCHER) -> User:
        """邀请研究人员"""
        # 创建用户
        user = self.space.create_user(user_id, display_name, email, role)
        
        # 设置化身
        user.avatar = AvatarState(
            user_id=user_id,
            appearance=AvatarAppearance(
                lab_coat=True,
                shirt_color=(
                    0.2 if role == UserRole.ADMIN else 0.4,
                    0.5 if role == UserRole.RESEARCHER else 0.3,
                    0.8
                )
            )
        )
        
        self.participants[user_id] = user
        
        self._log_event("researcher_invited", {
            "user_id": user_id,
            "name": display_name,
            "role": role.value
        })
        
        logger.info(f"Invited researcher: {display_name} ({role.value})")
        return user
    
    def join_session(self, user_id: str) -> bool:
        """加入会话"""
        if user_id not in self.participants:
            logger.error(f"User {user_id} not invited")
            return False
        
        user = self.participants[user_id]
        success = self.space.join_room(self.room_id, user)
        
        if success:
            room = self.space.get_room(self.room_id)
            room.send_system_message(f"{user.display_name} joined the research session")
            
            # 同步共享数据
            self._sync_shared_data(user_id)
            
            self._log_event("researcher_joined", {"user_id": user_id})
        
        return success
    
    def share_structure(self, user_id: str, structure_id: str,
                       structure_data: dict) -> bool:
        """共享晶体结构"""
        if user_id not in self.participants:
            return False
        
        self.structures[structure_id] = {
            "owner": user_id,
            "data": structure_data,
            "shared_at": datetime.now().isoformat(),
            "access": [user_id]  # 访问权限列表
        }
        
        # 广播给所有参与者
        room = self.space.get_room(self.room_id)
        if room:
            user = self.participants[user_id]
            room.send_system_message(
                f"{user.display_name} shared structure: {structure_id}"
            )
        
        self._log_event("structure_shared", {
            "user_id": user_id,
            "structure_id": structure_id
        })
        
        logger.info(f"Structure shared: {structure_id} by {user_id}")
        return True
    
    def annotate_structure(self, user_id: str, structure_id: str,
                          annotation: dict) -> str:
        """标注共享结构"""
        if structure_id not in self.structures:
            return ""
        
        anno_id = f"anno_{datetime.now().timestamp()}"
        annotation.update({
            "id": anno_id,
            "author": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        if "annotations" not in self.structures[structure_id]:
            self.structures[structure_id]["annotations"] = []
        
        self.structures[structure_id]["annotations"].append(annotation)
        
        # 实时同步
        self._broadcast_annotation(structure_id, annotation)
        
        return anno_id
    
    def start_whiteboard_collaboration(self, user_id: str,
                                      board_name: str = "Research Notes") -> str:
        """开始白板协作"""
        room = self.space.get_room(self.room_id)
        if not room:
            return ""
        
        # 创建新白板
        board_id = room.create_whiteboard(board_name)
        
        # 添加示例研究内容
        board = room.whiteboards[board_id]
        
        # 绘制晶体结构草图
        board.add_stroke(
            user_id,
            [(100, 100), (200, 100), (200, 200), (100, 200), (100, 100)],
            color=(0, 0, 200, 255),
            stroke_width=2
        )
        
        # 添加公式
        board.add_latex(
            user_id,
            r"E_{total} = \\sum_{i} \\frac{p_i^2}{2m_i} + \\sum_{i<j} V(r_{ij})",
            (250, 150),
            font_size=20
        )
        
        # 添加笔记
        board.add_text(
            user_id,
            "Research Notes:\n- Band gap calculation\n- Phonon dispersion\n- Defect analysis",
            (100, 300),
            font_size=16
        )
        
        self._log_event("whiteboard_created", {
            "user_id": user_id,
            "board_id": board_id
        })
        
        logger.info(f"Whiteboard created: {board_name} ({board_id})")
        return board_id
    
    def conduct_virtual_experiment(self, experiment_type: str,
                                   parameters: dict) -> dict:
        """进行虚拟实验"""
        results = {
            "experiment_type": experiment_type,
            "parameters": parameters,
            "start_time": datetime.now().isoformat(),
            "status": "running"
        }
        
        # 模拟实验过程
        if experiment_type == "md_simulation":
            # 分子动力学模拟
            results["progress"] = 0
            results["estimated_time"] = 3600  # 秒
            
        elif experiment_type == "dft_calculation":
            # DFT计算
            results["progress"] = 0
            results["estimated_time"] = 7200
            
        elif experiment_type == "xrd_analysis":
            # XRD分析
            results["peaks"] = []
            results["progress"] = 100
            results["status"] = "completed"
        
        exp_id = f"exp_{datetime.now().timestamp()}"
        self.simulations[exp_id] = results
        
        # 广播实验开始
        room = self.space.get_room(self.room_id)
        if room:
            room.send_system_message(f"Virtual experiment started: {experiment_type}")
        
        self._log_event("experiment_started", {
            "experiment_id": exp_id,
            "type": experiment_type
        })
        
        return {"experiment_id": exp_id, **results}
    
    def share_experiment_results(self, experiment_id: str,
                                 user_id: str) -> bool:
        """共享实验结果"""
        if experiment_id not in self.simulations:
            return False
        
        exp_data = self.simulations[experiment_id]
        exp_data["shared_by"] = user_id
        exp_data["shared_at"] = datetime.now().isoformat()
        
        # 在AR中显示结果
        if "results" in exp_data:
            self.ar.overlay.create_data_viz(
                exp_data["results"],
                type('ARV2', (), {'x': 100, 'y': 400})(),
                "line"
            )
        
        # 添加到白板
        room = self.space.get_room(self.room_id)
        if room and room.active_whiteboard_id:
            board = room.whiteboards[room.active_whiteboard_id]
            board.add_text(
                user_id,
                f"Experiment: {exp_data['experiment_type']}\n"
                f"Status: {exp_data['status']}\n"
                f"Shared by: {self.participants.get(user_id, User(user_id, '')).display_name}",
                (100, 500),
                font_size=14
            )
        
        return True
    
    def initiate_voice_discussion(self, topic: str, 
                                 moderator_id: str) -> dict:
        """发起语音讨论"""
        discussion = {
            "topic": topic,
            "moderator": moderator_id,
            "start_time": datetime.now().isoformat(),
            "participants": [],
            "transcript": []
        }
        
        room = self.space.get_room(self.room_id)
        if room:
            room.send_system_message(f"Voice discussion started: {topic}")
            
            # 通知所有参与者
            for user_id in room.participants:
                if user_id != moderator_id:
                    discussion["participants"].append(user_id)
        
        self._log_event("discussion_started", {
            "topic": topic,
            "moderator": moderator_id
        })
        
        return discussion
    
    def record_collaboration_session(self) -> bool:
        """录制协作会话"""
        room = self.space.get_room(self.room_id)
        if not room:
            return False
        
        # 启用录制
        room.settings['recording'] = True
        
        self._log_event("recording_started", {})
        logger.info("Session recording started")
        return True
    
    def export_collaboration_report(self, filepath: str) -> bool:
        """导出协作报告"""
        report = {
            "session_name": self.name,
            "room_id": self.room_id,
            "duration": self._calculate_duration(),
            "participants": [
                {
                    "id": u.user_id,
                    "name": u.display_name,
                    "role": u.role.value,
                    "joined_at": u.joined_at
                }
                for u in self.participants.values()
            ],
            "shared_structures": list(self.structures.keys()),
            "experiments": list(self.simulations.keys()),
            "event_log": self.session_log,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            import json
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Collaboration report exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False
    
    def _sync_shared_data(self, user_id: str) -> None:
        """同步共享数据给新用户"""
        # 发送所有共享结构
        for struct_id, struct_data in self.structures.items():
            logger.debug(f"Syncing structure {struct_id} to {user_id}")
    
    def _broadcast_annotation(self, structure_id: str,
                             annotation: dict) -> None:
        """广播标注"""
        room = self.space.get_room(self.room_id)
        if room:
            # 发送给所有参与者
            pass
    
    def _log_event(self, event_type: str, data: dict) -> None:
        """记录事件"""
        self.session_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "data": data
        })
    
    def _calculate_duration(self) -> float:
        """计算会话持续时间"""
        if not self.session_log:
            return 0
        start = datetime.fromisoformat(self.session_log[0]["timestamp"])
        end = datetime.now()
        return (end - start).total_seconds()
    
    def get_session_summary(self) -> dict:
        """获取会话摘要"""
        room = self.space.get_room(self.room_id)
        
        return {
            "session_name": self.name,
            "room_id": self.room_id,
            "active": room.is_active if room else False,
            "participant_count": len(self.participants),
            "online_count": sum(1 for u in self.participants.values()
                               if u.presence == PresenceStatus.ONLINE),
            "shared_structures": len(self.structures),
            "active_experiments": sum(1 for e in self.simulations.values()
                                     if e.get("status") == "running"),
            "whiteboards": len(room.whiteboards) if room else 0,
            "session_duration": self._calculate_duration()
        }


def run_collaboration_demo():
    """运行协作演示"""
    print("=" * 70)
    print("  Remote Collaborative Research Platform")
    print("  远程协作研究平台 - 分布式材料研究团队")
    print("=" * 70)
    
    # 创建协作会话
    session = ResearchCollaborationSession(
        "Novel Battery Materials Research"
    )
    
    # 初始化
    import asyncio
    asyncio.run(session.initialize())
    print(f"\n✓ Research collaboration session initialized")
    print(f"  Session: {session.name}")
    print(f"  Room ID: {session.room_id}")
    
    # 邀请研究团队成员
    print("\n[1] Inviting Research Team...")
    researchers = [
        ("res_001", "Dr. Chen Li", "chen@university.edu", UserRole.ADMIN),
        ("res_002", "Dr. Sarah Johnson", "sarah@lab.org", UserRole.RESEARCHER),
        ("res_003", "Prof. Wang Wei", "wang@institute.cn", UserRole.RESEARCHER),
        ("res_004", "Alice Brown", "alice@student.edu", UserRole.VISITOR),
    ]
    
    for uid, name, email, role in researchers:
        user = session.invite_researcher(uid, name, email, role)
        print(f"  ✓ Invited {name} ({role.value})")
    
    # 团队成员加入
    print("\n[2] Team Members Joining...")
    for uid, _, _, _ in researchers:
        success = session.join_session(uid)
        status = "✓" if success else "✗"
        print(f"  {status} {session.participants[uid].display_name} joined")
    
    # 共享研究数据
    print("\n[3] Sharing Research Data...")
    structures = [
        ("res_001", "LiCoO2_structure", {"formula": "LiCoO2", "space_group": "R-3m"}),
        ("res_002", "graphite_anode", {"formula": "C", "layers": 5}),
        ("res_003", "solid_electrolyte", {"formula": "Li7La3Zr2O12", "conductivity": 1e-4}),
    ]
    
    for user_id, struct_id, data in structures:
        success = session.share_structure(user_id, struct_id, data)
        if success:
            print(f"  ✓ {struct_id} shared by {session.participants[user_id].display_name}")
    
    # 启动白板协作
    print("\n[4] Starting Whiteboard Collaboration...")
    board_id = session.start_whiteboard_collaboration("res_001", "Research Plan")
    print(f"  ✓ Whiteboard created: {board_id}")
    
    room = session.space.get_room(session.room_id)
    board = room.whiteboards[board_id]
    print(f"  - Strokes: {len(board.current_page.strokes)}")
    print(f"  - Elements: {len(board.current_page.elements)}")
    
    # 虚拟实验
    print("\n[5] Conducting Virtual Experiments...")
    experiments = [
        ("md_simulation", {"temperature": 300, "pressure": 1, "time": 1000}),
        ("dft_calculation", {"functional": "PBE", "kpoints": [4, 4, 4]}),
        ("xrd_analysis", {"wavelength": 1.54, "two_theta_range": [10, 90]}),
    ]
    
    for exp_type, params in experiments:
        result = session.conduct_virtual_experiment(exp_type, params)
        print(f"  ✓ {exp_type} started (ID: {result['experiment_id']})")
        print(f"    Status: {result['status']}, Est. time: {result.get('estimated_time', 0)}s")
    
    # 语音讨论
    print("\n[6] Voice Discussion...")
    discussion = session.initiate_voice_discussion(
        "Battery Performance Optimization",
        "res_001"
    )
    print(f"  ✓ Discussion started: {discussion['topic']}")
    print(f"    Moderator: {session.participants[discussion['moderator']].display_name}")
    print(f"    Participants: {len(discussion['participants'])}")
    
    # 添加标注
    print("\n[7] Collaborative Annotation...")
    annotation = {
        "type": "defect_site",
        "position": [1.5, 2.0, 0.5],
        "description": "Possible Li vacancy site with low formation energy",
        "suggested_action": "Perform NEB calculation"
    }
    anno_id = session.annotate_structure("res_002", "LiCoO2_structure", annotation)
    print(f"  ✓ Annotation added: {anno_id}")
    
    # 录制会话
    print("\n[8] Recording Session...")
    session.record_collaboration_session()
    print(f"  ✓ Session recording enabled")
    
    # 会话摘要
    print("\n[9] Session Summary...")
    summary = session.get_session_summary()
    print(f"  Session: {summary['session_name']}")
    print(f"  Participants: {summary['participant_count']} "
          f"({summary['online_count']} online)")
    print(f"  Shared Structures: {summary['shared_structures']}")
    print(f"  Active Experiments: {summary['active_experiments']}")
    print(f"  Whiteboards: {summary['whiteboards']}")
    print(f"  Duration: {summary['session_duration']:.1f}s")
    
    # 导出报告
    print("\n[10] Exporting Collaboration Report...")
    report_path = "/tmp/collaboration_report.json"
    success = session.export_collaboration_report(report_path)
    print(f"  {'✓' if success else '✗'} Report exported to {report_path}")
    
    print("\n" + "=" * 70)
    print("  Demo Complete - Remote Collaborative Research Platform")
    print("=" * 70)


if __name__ == "__main__":
    run_collaboration_demo()
