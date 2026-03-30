"""
KAS A/B 测试系统
Agent 版本对比测试
"""
import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import random

from kas.core.config import get_config


class ABTestStatus(Enum):
    """A/B 测试状态"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class ABTestSession:
    """A/B 测试会话"""
    id: str
    test_id: str
    user_id: Optional[str] = None
    assigned_version: Optional[str] = None  # 'A' 或 'B'
    prompt: str = ""
    response_a: Optional[str] = None
    response_b: Optional[str] = None
    user_choice: Optional[str] = None  # 'A', 'B', 或 'tie'
    rating_a: Optional[int] = None  # 1-5
    rating_b: Optional[int] = None  # 1-5
    feedback: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


@dataclass
class ABTest:
    """A/B 测试定义"""
    id: str
    name: str
    agent_name: str
    version_a: str
    version_b: str
    description: Optional[str] = None
    status: ABTestStatus = ABTestStatus.RUNNING
    sample_size: int = 100  # 目标样本数
    min_confidence: float = 0.95  # 最小置信度
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'agent_name': self.agent_name,
            'version_a': self.version_a,
            'version_b': self.version_b,
            'description': self.description,
            'status': self.status.value,
            'sample_size': self.sample_size,
            'min_confidence': self.min_confidence,
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ABTest':
        """从字典创建"""
        test = cls(
            id=data['id'],
            name=data['name'],
            agent_name=data['agent_name'],
            version_a=data['version_a'],
            version_b=data['version_b'],
            description=data.get('description'),
            status=ABTestStatus(data.get('status', 'running')),
            sample_size=data.get('sample_size', 100),
            min_confidence=data.get('min_confidence', 0.95),
            created_at=data.get('created_at', datetime.now().isoformat()),
            completed_at=data.get('completed_at')
        )
        return test


class ABTestEngine:
    """A/B 测试引擎"""
    
    def __init__(self):
        self.tests_dir = Path(get_config().config_dir) / "ab_tests"
        self.tests_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir = self.tests_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
    
    def _get_test_path(self, test_id: str) -> Path:
        """获取测试文件路径"""
        return self.tests_dir / f"{test_id}.json"
    
    def _get_sessions_path(self, test_id: str) -> Path:
        """获取会话文件路径"""
        return self.sessions_dir / f"{test_id}.json"
    
    def create(
        self,
        name: str,
        agent_name: str,
        version_a: str,
        version_b: str,
        description: str = None,
        sample_size: int = 100
    ) -> ABTest:
        """创建 A/B 测试"""
        test_id = str(uuid.uuid4())[:8]
        
        test = ABTest(
            id=test_id,
            name=name,
            agent_name=agent_name,
            version_a=version_a,
            version_b=version_b,
            description=description,
            sample_size=sample_size
        )
        
        self._save_test(test)
        
        # 初始化空会话列表
        self._save_sessions(test_id, [])
        
        return test
    
    def _save_test(self, test: ABTest):
        """保存测试"""
        path = self._get_test_path(test.id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(test.to_dict(), f, indent=2, ensure_ascii=False)
    
    def _load_test(self, test_id: str) -> Optional[ABTest]:
        """加载测试"""
        path = self._get_test_path(test_id)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ABTest.from_dict(data)
    
    def _save_sessions(self, test_id: str, sessions: List[Dict]):
        """保存会话"""
        path = self._get_sessions_path(test_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sessions, f, indent=2, ensure_ascii=False)
    
    def _load_sessions(self, test_id: str) -> List[Dict]:
        """加载会话"""
        path = self._get_sessions_path(test_id)
        if not path.exists():
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_tests(self) -> List[ABTest]:
        """列出所有测试"""
        tests = []
        for path in self.tests_dir.glob("*.json"):
            test_id = path.stem
            test = self._load_test(test_id)
            if test:
                tests.append(test)
        return sorted(tests, key=lambda t: t.created_at, reverse=True)
    
    def get_test(self, test_id: str) -> Optional[ABTest]:
        """获取测试"""
        return self._load_test(test_id)
    
    def assign_version(self, test_id: str, user_id: str = None) -> Optional[str]:
        """
        为用户分配版本（A 或 B）
        使用随机分配，确保均衡
        """
        test = self._load_test(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        sessions = self._load_sessions(test_id)
        
        # 统计当前分配
        count_a = sum(1 for s in sessions if s.get('assigned_version') == 'A')
        count_b = sum(1 for s in sessions if s.get('assigned_version') == 'B')
        
        # 均衡分配
        if count_a <= count_b:
            return 'A'
        else:
            return 'B'
    
    def create_session(
        self,
        test_id: str,
        prompt: str,
        user_id: str = None
    ) -> Optional[ABTestSession]:
        """创建测试会话"""
        test = self._load_test(test_id)
        if not test or test.status != ABTestStatus.RUNNING:
            return None
        
        assigned = self.assign_version(test_id, user_id)
        if not assigned:
            return None
        
        session = ABTestSession(
            id=str(uuid.uuid4())[:12],
            test_id=test_id,
            user_id=user_id,
            assigned_version=assigned,
            prompt=prompt
        )
        
        # 保存会话
        sessions = self._load_sessions(test_id)
        sessions.append(asdict(session))
        self._save_sessions(test_id, sessions)
        
        return session
    
    def run_comparison(
        self,
        session_id: str,
        test_id: str,
        use_mock: bool = False
    ) -> Dict[str, str]:
        """
        运行 A/B 对比测试
        返回两个版本的响应
        """
        from kas.core.chat import ChatEngine
        from kas.core.versioning import get_version_manager
        
        test = self._load_test(test_id)
        if not test:
            raise ValueError(f"Test not found: {test_id}")
        
        sessions = self._load_sessions(test_id)
        session_data = next((s for s in sessions if s['id'] == session_id), None)
        if not session_data:
            raise ValueError(f"Session not found: {session_id}")
        
        prompt = session_data['prompt']
        
        # 加载版本 A
        vm = get_version_manager(test.agent_name)
        agent_a = vm.load_version(test.version_a)
        
        # 加载版本 B（如果是当前版本，直接使用）
        if test.version_b == 'current':
            from kas.core.config import get_config
            agent_path = Path(get_config().agents_dir) / test.agent_name
            from kas.core.models import Agent
            agent_b = Agent.load(agent_path)
        else:
            agent_b = vm.load_version(test.version_b)
        
        # 运行 A
        chat_a = ChatEngine(test.agent_name)
        chat_a.agent = agent_a
        response_a = chat_a.run(prompt, use_mock=use_mock)
        
        # 运行 B
        chat_b = ChatEngine(test.agent_name)
        chat_b.agent = agent_b
        response_b = chat_b.run(prompt, use_mock=use_mock)
        
        # 更新会话
        for s in sessions:
            if s['id'] == session_id:
                s['response_a'] = response_a
                s['response_b'] = response_b
                break
        
        self._save_sessions(test_id, sessions)
        
        return {
            'A': response_a,
            'B': response_b
        }
    
    def record_result(
        self,
        session_id: str,
        test_id: str,
        user_choice: str,  # 'A', 'B', 'tie'
        rating_a: int = None,
        rating_b: int = None,
        feedback: str = None
    ):
        """记录用户选择"""
        sessions = self._load_sessions(test_id)
        
        for s in sessions:
            if s['id'] == session_id:
                s['user_choice'] = user_choice
                s['rating_a'] = rating_a
                s['rating_b'] = rating_b
                s['feedback'] = feedback
                s['completed_at'] = datetime.now().isoformat()
                break
        
        self._save_sessions(test_id, sessions)
        
        # 检查是否达到样本量
        self._check_completion(test_id)
    
    def _check_completion(self, test_id: str):
        """检查是否完成"""
        test = self._load_test(test_id)
        if not test:
            return
        
        sessions = self._load_sessions(test_id)
        completed = [s for s in sessions if s.get('user_choice')]
        
        if len(completed) >= test.sample_size:
            test.status = ABTestStatus.COMPLETED
            test.completed_at = datetime.now().isoformat()
            self._save_test(test)
    
    def get_stats(self, test_id: str) -> Dict[str, Any]:
        """获取测试统计"""
        test = self._load_test(test_id)
        if not test:
            return None
        
        sessions = self._load_sessions(test_id)
        completed = [s for s in sessions if s.get('user_choice')]
        
        if not completed:
            return {
                'test': test.to_dict(),
                'total_sessions': len(sessions),
                'completed_sessions': 0,
                'a_wins': 0,
                'b_wins': 0,
                'ties': 0,
                'winner': None,
                'confidence': 0
            }
        
        # 统计结果
        a_wins = sum(1 for s in completed if s['user_choice'] == 'A')
        b_wins = sum(1 for s in completed if s['user_choice'] == 'B')
        ties = sum(1 for s in completed if s['user_choice'] == 'tie')
        
        total = a_wins + b_wins + ties
        
        # 计算置信度（简化版，实际应使用统计检验）
        if total > 0:
            a_rate = a_wins / total
            b_rate = b_wins / total
            
            # 简单判断胜者
            if a_rate > b_rate and a_wins > total * 0.6:
                winner = 'A'
                confidence = a_rate
            elif b_rate > a_rate and b_wins > total * 0.6:
                winner = 'B'
                confidence = b_rate
            else:
                winner = None
                confidence = max(a_rate, b_rate)
        else:
            winner = None
            confidence = 0
        
        # 评分统计
        ratings_a = [s.get('rating_a') for s in completed if s.get('rating_a')]
        ratings_b = [s.get('rating_b') for s in completed if s.get('rating_b')]
        
        return {
            'test': test.to_dict(),
            'total_sessions': len(sessions),
            'completed_sessions': len(completed),
            'a_wins': a_wins,
            'b_wins': b_wins,
            'ties': ties,
            'winner': winner,
            'confidence': round(confidence, 3),
            'avg_rating_a': round(sum(ratings_a) / len(ratings_a), 2) if ratings_a else None,
            'avg_rating_b': round(sum(ratings_b) / len(ratings_b), 2) if ratings_b else None
        }
    
    def declare_winner(self, test_id: str, version: str) -> bool:
        """手动宣布胜者"""
        test = self._load_test(test_id)
        if not test:
            return False
        
        test.status = ABTestStatus.COMPLETED
        test.completed_at = datetime.now().isoformat()
        self._save_test(test)
        
        return True


# 便捷函数
def get_abtest_engine() -> ABTestEngine:
    """获取 A/B 测试引擎"""
    return ABTestEngine()
