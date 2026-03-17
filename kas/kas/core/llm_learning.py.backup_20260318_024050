"""
LLM Enhanced Learning System
强化学习驱动的 Agent 元学习与收敛系统

核心组件:
- LLMMetaAnalyzer: LLM深度分析遥测数据
- LLMQualityEvaluator: LLM语义质量评估
- RLFeedbackSystem: 强化学习反馈系统
- ConvergenceEngine: 收敛检测与迭代优化
- LLMEvolutionAdvisor: LLM生成进化计划
"""

import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import deque
import random

from openai import OpenAI


@dataclass
class InteractionRecord:
    """单次交互记录"""
    timestamp: float
    user_message: str
    agent_response: str
    task_type: str
    quality_score: float  # 0-100
    user_feedback: Optional[str] = None  # 👍/👎/💬
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CapabilityMetrics:
    """能力指标跟踪"""
    capability_name: str
    total_uses: int = 0
    success_count: int = 0
    avg_quality: float = 0.0
    user_satisfaction: float = 0.0
    trend: str = "stable"  # improving/declining/stable
    last_updated: float = 0.0


@dataclass
class EvolutionState:
    """进化状态"""
    generation: int = 0
    convergence_score: float = 0.0
    is_converged: bool = False
    last_evolution: float = 0.0
    improvements_applied: List[str] = None
    
    def __post_init__(self):
        if self.improvements_applied is None:
            self.improvements_applied = []


class LLMClient:
    """统一的 LLM 客户端"""
    
    def __init__(self):
        self.client = None
        self.model = None
        self._init_client()
    
    def _init_client(self):
        """自动检测并初始化 LLM 客户端"""
        if os.environ.get('DEEPSEEK_API_KEY'):
            self.client = OpenAI(
                api_key=os.environ['DEEPSEEK_API_KEY'],
                base_url="https://api.deepseek.com/v1"
            )
            self.model = "deepseek-chat"
        elif os.environ.get('KIMI_API_KEY'):
            self.client = OpenAI(
                api_key=os.environ['KIMI_API_KEY'],
                base_url="https://api.moonshot.cn/v1"
            )
            self.model = "moonshot-v1-8k"
        elif os.environ.get('OPENAI_API_KEY'):
            self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
            self.model = "gpt-3.5-turbo"
    
    def chat(self, system_prompt: str, user_message: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """调用 LLM"""
        if not self.client:
            raise RuntimeError("No LLM API key configured")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


class LLMQualityEvaluator:
    """
    LLM 语义质量评估器
    
    对比规则评估，LLM 可以:
    - 理解语义相关性（不只是关键词匹配）
    - 评估代码的准确性和实用性
    - 判断回答是否真正解决了用户问题
    """
    
    EVALUATION_PROMPT = """你是一个专业的 AI 助手回答质量评估专家。

请对以下 Agent 回答进行深度质量评估（0-100分）：

【用户问题】
{user_message}

【Agent回答】
{agent_response}

【Agent能力】
{capabilities}

请从以下维度评估（每个维度0-25分）：
1. **语义相关性**: 回答是否与用户问题高度相关？
2. **准确性**: 技术内容是否正确？代码能否运行？
3. **完整性**: 是否全面回答了问题？有没有遗漏关键点？
4. **实用性**: 用户能否直接使用这个回答？有没有可操作的指导？

请以 JSON 格式输出:
{{
    "semantic_relevance": <0-25>,
    "accuracy": <0-25>,
    "completeness": <0-25>,
    "practicality": <0-25>,
    "total_score": <0-100>,
    "reasoning": "简要评价",
    "suggestions": "改进建议"
}}
"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def evaluate(self, user_message: str, agent_response: str, capabilities: List[str]) -> Dict:
        """评估单次交互质量"""
        prompt = self.EVALUATION_PROMPT.format(
            user_message=user_message,
            agent_response=agent_response,
            capabilities=", ".join(capabilities)
        )
        
        result = self.llm.chat(
            system_prompt="你是一个严格但公正的质量评估专家。只输出 JSON，不要其他内容。",
            user_message=prompt,
            temperature=0.3,  # 低温度确保一致性
            max_tokens=800
        )
        
        # 解析 JSON 结果
        try:
            # 提取 JSON 部分
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0].strip()
            else:
                json_str = result.strip()
            
            evaluation = json.loads(json_str)
            return {
                "score": evaluation.get("total_score", 50),
                "dimensions": {
                    "semantic_relevance": evaluation.get("semantic_relevance", 0),
                    "accuracy": evaluation.get("accuracy", 0),
                    "completeness": evaluation.get("completeness", 0),
                    "practicality": evaluation.get("practicality", 0)
                },
                "reasoning": evaluation.get("reasoning", ""),
                "suggestions": evaluation.get("suggestions", ""),
                "raw_response": result
            }
        except json.JSONDecodeError:
            # 回退到简单评分
            return {
                "score": 50,
                "dimensions": {},
                "reasoning": "Parse error",
                "suggestions": "Failed to parse LLM evaluation",
                "raw_response": result
            }


class RLFeedbackSystem:
    """
    强化学习反馈系统
    
    将用户交互建模为 MDP:
    - State: 当前 Agent 状态（能力配置、历史表现）
    - Action: 调整某个能力参数（如增加/减少置信度）
    - Reward: 用户满意度 + 质量评分
    - Policy: 学习何时以及如何调整 Agent
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = {}  # Q-value table
        self.exploration_rate = 0.2
        self.action_history: List[Tuple[str, str, float]] = []  # (state, action, reward)
    
    def get_state(self, metrics: Dict[str, CapabilityMetrics]) -> str:
        """将当前指标编码为状态"""
        # 简化状态：根据平均质量分桶
        avg_quality = sum(m.avg_quality for m in metrics.values()) / len(metrics) if metrics else 50
        if avg_quality >= 80:
            return "high_quality"
        elif avg_quality >= 60:
            return "medium_quality"
        else:
            return "low_quality"
    
    def get_available_actions(self) -> List[str]:
        """可用动作"""
        return [
            "increase_confidence",
            "decrease_confidence",
            "add_capability",
            "refine_prompt",
            "no_action"
        ]
    
    def select_action(self, state: str) -> str:
        """ε-贪婪策略选择动作"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.get_available_actions()}
        
        if random.random() < self.exploration_rate:
            return random.choice(self.get_available_actions())
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update(self, state: str, action: str, reward: float, next_state: str):
        """Q-learning 更新"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.get_available_actions()}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.get_available_actions()}
        
        # Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        
        self.action_history.append((state, action, reward))
    
    def calculate_reward(self, quality_score: float, user_feedback: Optional[str]) -> float:
        """计算奖励信号"""
        reward = quality_score / 100.0  # 基础质量分
        
        # 用户反馈加成
        if user_feedback:
            if "👍" in user_feedback or "good" in user_feedback.lower():
                reward += 0.5
            elif "👎" in user_feedback or "bad" in user_feedback.lower():
                reward -= 0.5
        
        return max(-1.0, min(1.0, reward))  # 限制在 [-1, 1]
    
    def get_policy_insights(self) -> str:
        """获取策略洞察"""
        insights = []
        for state, actions in self.q_table.items():
            best_action = max(actions, key=actions.get)
            best_q = actions[best_action]
            insights.append(f"状态 '{state}': 最优动作 '{best_action}' (Q={best_q:.2f})")
        return "\n".join(insights) if insights else "策略还在学习中..."


class ConvergenceEngine:
    """
    收敛检测与迭代优化引擎
    
    核心思想:
    - 持续监控 Agent 性能指标
    - 当性能趋于稳定时判定为"收敛"
    - 触发自动化改进或提示人工介入
    """
    
    def __init__(self, window_size: int = 10, convergence_threshold: float = 0.85):
        self.window_size = window_size
        self.threshold = convergence_threshold
        self.quality_history: deque = deque(maxlen=window_size)
        self.metrics_history: List[Dict] = []
        self.generation = 0
    
    def add_sample(self, quality_score: float, metrics: Dict[str, Any]):
        """添加新样本"""
        self.quality_history.append(quality_score)
        self.metrics_history.append({
            "timestamp": time.time(),
            "quality": quality_score,
            "metrics": metrics
        })
    
    def check_convergence(self) -> Tuple[bool, float]:
        """
        检测是否收敛
        
        收敛标准:
        1. 样本数 >= window_size
        2. 最近 window 的平均质量 >= threshold * 100
        3. 方差较小（性能稳定）
        """
        if len(self.quality_history) < self.window_size:
            return False, 0.0
        
        recent_scores = list(self.quality_history)
        avg_quality = sum(recent_scores) / len(recent_scores)
        variance = sum((x - avg_quality) ** 2 for x in recent_scores) / len(recent_scores)
        
        # 收敛分数 = 质量分数 * 稳定性系数
        stability = max(0, 1 - variance / 1000)  # 方差越小越稳定
        convergence_score = (avg_quality / 100) * stability
        
        is_converged = (
            avg_quality >= self.threshold * 100 and
            variance <= 100 and
            convergence_score >= 0.7
        )
        
        return is_converged, convergence_score
    
    def get_trend_analysis(self) -> Dict:
        """趋势分析"""
        if len(self.quality_history) < 5:
            return {"trend": "insufficient_data", "slope": 0, "current_avg": 0, "variance": 0}
        
        # 简单线性回归计算趋势
        n = len(self.quality_history)
        x_mean = (n - 1) / 2
        y_mean = sum(self.quality_history) / n
        
        numerator = sum((i - x_mean) * (score - y_mean) for i, score in enumerate(self.quality_history))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator > 0 else 0
        
        if slope > 1:
            trend = "improving"
        elif slope < -1:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": slope,
            "current_avg": y_mean,
            "variance": sum((x - y_mean) ** 2 for x in self.quality_history) / n
        }
    
    def should_trigger_evolution(self) -> Tuple[bool, str]:
        """是否应该触发进化"""
        is_converged, score = self.check_convergence()
        trend = self.get_trend_analysis()
        
        if is_converged:
            return True, f"已收敛 (score={score:.2f})，准备进化到下一代"
        
        if trend["trend"] == "declining" and trend["current_avg"] < 60:
            return True, f"性能下降 (avg={trend['current_avg']:.1f})，需要紧急优化"
        
        return False, f"学习中... (trend={trend['trend']}, avg={trend['current_avg']:.1f})"


class LLMMetaAnalyzer:
    """
    LLM 深度元分析器
    
    不只是统计保留率，而是真正理解:
    - 为什么某些能力被保留/丢弃
    - 能力之间的依赖关系
    - 用户使用模式的深层原因
    """
    
    ANALYSIS_PROMPT = """你是一个 AI Agent 行为分析师。请分析以下 Agent 的使用遥测数据。

【Agent信息】
名称: {agent_name}
能力: {capabilities}

【最近交互记录】
{interaction_records}

【当前指标】
{metrics}

请进行深度分析，回答:
1. 用户最看重 Agent 的哪些能力？为什么？
2. 哪些能力表现不佳？根本原因是什么？
3. 用户通常在什么场景下使用这个 Agent？
4. 有哪些隐藏的使用模式或需求没有被满足？
5. 如果只能改进一件事，应该改什么？为什么？

请以结构化 JSON 输出:
{{
    "key_strengths": ["优势1", "优势2"],
    "weaknesses": [{{"capability": "能力名", "root_cause": "根本原因"}}],
    "usage_patterns": ["模式1", "模式2"],
    "hidden_needs": ["未满足需求1"],
    "top_recommendation": "最重要的改进建议",
    "confidence": 0.85
}}
"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def analyze(self, agent_name: str, capabilities: List[str], 
                records: List[InteractionRecord], metrics: Dict) -> Dict:
        """深度分析 Agent 表现"""
        
        # 格式化交互记录
        records_text = "\n".join([
            f"- [{datetime.fromtimestamp(r.timestamp).strftime('%m-%d %H:%M')}] "
            f"Q: {r.user_message[:50]}... | "
            f"Quality: {r.quality_score} | Feedback: {r.user_feedback or 'none'}"
            for r in records[-20:]  # 最近20条
        ])
        
        # 格式化指标
        metrics_text = "\n".join([
            f"- {name}: avg_quality={m.avg_quality:.1f}, satisfaction={m.user_satisfaction:.1f}, trend={m.trend}"
            for name, m in metrics.items()
        ])
        
        prompt = self.ANALYSIS_PROMPT.format(
            agent_name=agent_name,
            capabilities=", ".join(capabilities),
            interaction_records=records_text,
            metrics=metrics_text
        )
        
        result = self.llm.chat(
            system_prompt="你是一个资深的产品分析师，擅长从数据中发现深层洞察。只输出 JSON。",
            user_message=prompt,
            temperature=0.4,
            max_tokens=1500
        )
        
        # 解析结果
        try:
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0].strip()
            else:
                json_str = result.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "key_strengths": ["Analysis failed"],
                "weaknesses": [],
                "usage_patterns": [],
                "hidden_needs": [],
                "top_recommendation": "Please check raw response",
                "confidence": 0.0,
                "raw_response": result
            }


class LLMEvolutionAdvisor:
    """
    LLM 进化顾问
    
    基于分析结果，生成具体的进化计划，包括:
    - 重写 system prompt
    - 调整能力配置
    - 生成新的能力建议
    """
    
    EVOLUTION_PROMPT = """你是一个 AI Agent 架构师。请基于以下分析结果，生成具体的 Agent 进化方案。

【当前 Agent】
名称: {agent_name}
当前 System Prompt:
{current_prompt}

【分析洞察】
{analysis_result}

【当前问题】
{issues}

请生成进化方案，包括:

1. **改进后的 System Prompt** (保持核心能力，但针对性优化)
2. **能力调整** (添加/修改/删除哪些能力)
3. **配置优化** (temperature, max_tokens 等)
4. **实施步骤** (优先级排序)

请以 JSON 格式输出:
{{
    "new_system_prompt": "优化后的完整 prompt",
    "capability_changes": [
        {{"action": "add|modify|remove", "name": "能力名", "reason": "原因"}}
    ],
    "config_changes": {{"temperature": 0.7, "max_tokens": 2000}},
    "implementation_steps": ["步骤1", "步骤2"],
    "expected_improvement": "预期改进效果"
}}
"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def generate_evolution_plan(self, agent_name: str, current_prompt: str,
                                analysis: Dict, issues: List[str]) -> Dict:
        """生成进化计划"""
        
        prompt = self.EVOLUTION_PROMPT.format(
            agent_name=agent_name,
            current_prompt=current_prompt[:1000],  # 限制长度
            analysis_result=json.dumps(analysis, ensure_ascii=False, indent=2),
            issues="\n".join(f"- {i}" for i in issues)
        )
        
        result = self.llm.chat(
            system_prompt="你是一个专家级的 AI 系统架构师，擅长优化 Agent 性能。",
            user_message=prompt,
            temperature=0.5,
            max_tokens=2000
        )
        
        try:
            if "```json" in result:
                json_str = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                json_str = result.split("```")[1].split("```")[0].strip()
            else:
                json_str = result.strip()
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {
                "new_system_prompt": current_prompt,
                "capability_changes": [],
                "config_changes": {},
                "implementation_steps": ["Parse error - manual review needed"],
                "expected_improvement": "Unknown",
                "raw_response": result
            }


class LLMEnhancedLearningEngine:
    """
    统一的 LLM 增强学习引擎
    
    整合所有组件，提供完整的元学习能力:
    - 记录交互
    - 评估质量
    - 强化学习
    - 检测收敛
    - 生成进化计划
    """
    
    def __init__(self, agent_name: str, storage_dir: Optional[str] = None):
        self.agent_name = agent_name
        self.storage_dir = Path(storage_dir or Path.home() / '.kas' / 'learning' / agent_name)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.llm = LLMClient()
        self.quality_evaluator = LLMQualityEvaluator(self.llm)
        self.rl_system = RLFeedbackSystem()
        self.convergence_engine = ConvergenceEngine()
        self.meta_analyzer = LLMMetaAnalyzer(self.llm)
        self.evolution_advisor = LLMEvolutionAdvisor(self.llm)
        
        # 数据存储
        self.interactions: List[InteractionRecord] = []
        self.capability_metrics: Dict[str, CapabilityMetrics] = {}
        self.evolution_state = EvolutionState()
        
        # 加载历史数据
        self._load_data()
    
    def _load_data(self):
        """加载持久化数据"""
        data_file = self.storage_dir / 'learning_data.json'
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
                self.interactions = [InteractionRecord(**r) for r in data.get('interactions', [])]
                self.capability_metrics = {
                    k: CapabilityMetrics(**v) for k, v in data.get('metrics', {}).items()
                }
                self.evolution_state = EvolutionState(**data.get('evolution', {}))
    
    def _save_data(self):
        """保存数据"""
        data = {
            'interactions': [asdict(r) for r in self.interactions],
            'metrics': {k: asdict(v) for k, v in self.capability_metrics.items()},
            'evolution': asdict(self.evolution_state),
            'q_table': self.rl_system.q_table,
            'last_saved': time.time()
        }
        with open(self.storage_dir / 'learning_data.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_interaction(self, user_message: str, agent_response: str,
                          task_type: str, capabilities: List[str],
                          user_feedback: Optional[str] = None) -> Dict:
        """
        记录一次交互并进行完整的学习流程
        
        流程:
        1. LLM 评估质量
        2. 更新能力指标
        3. 强化学习更新
        4. 检测收敛
        5. 如果收敛，触发进化建议
        """
        
        # 1. LLM 质量评估
        quality_result = self.quality_evaluator.evaluate(
            user_message, agent_response, capabilities
        )
        quality_score = quality_result['score']
        
        # 2. 创建记录
        record = InteractionRecord(
            timestamp=time.time(),
            user_message=user_message,
            agent_response=agent_response,
            task_type=task_type,
            quality_score=quality_score,
            user_feedback=user_feedback,
            metadata={'llm_evaluation': quality_result}
        )
        self.interactions.append(record)
        
        # 3. 更新能力指标
        for cap in capabilities:
            if cap not in self.capability_metrics:
                self.capability_metrics[cap] = CapabilityMetrics(capability_name=cap)
            
            metric = self.capability_metrics[cap]
            metric.total_uses += 1
            metric.avg_quality = (metric.avg_quality * (metric.total_uses - 1) + quality_score) / metric.total_uses
            if user_feedback and ("👍" in user_feedback or "good" in user_feedback.lower()):
                metric.success_count += 1
            metric.user_satisfaction = metric.success_count / metric.total_uses if metric.total_uses > 0 else 0
            metric.last_updated = time.time()
        
        # 4. 强化学习更新
        current_state = self.rl_system.get_state(self.capability_metrics)
        action = self.rl_system.select_action(current_state)
        reward = self.rl_system.calculate_reward(quality_score, user_feedback)
        
        # 执行动作（简化版）
        # 实际实现中，这里会根据 action 修改 Agent 配置
        
        next_state = self.rl_system.get_state(self.capability_metrics)
        self.rl_system.update(current_state, action, reward, next_state)
        
        # 5. 收敛检测
        self.convergence_engine.add_sample(quality_score, self.capability_metrics)
        should_evolve, reason = self.convergence_engine.should_trigger_evolution()
        
        evolution_plan = None
        if should_evolve and not self.evolution_state.is_converged:
            # 触发进化
            self.evolution_state.is_converged = True
            self.evolution_state.convergence_score = self.convergence_engine.check_convergence()[1]
            
            # 6. 生成进化计划
            analysis = self.meta_analyzer.analyze(
                self.agent_name, capabilities, self.interactions, self.capability_metrics
            )
            
            issues = [w['root_cause'] for w in analysis.get('weaknesses', [])]
            current_prompt = "【需要加载实际 system prompt】"  # 占位符
            
            evolution_plan = self.evolution_advisor.generate_evolution_plan(
                self.agent_name, current_prompt, analysis, issues
            )
            
            self.evolution_state.generation += 1
            self.evolution_state.last_evolution = time.time()
            self.evolution_state.improvements_applied.append(f"Gen{self.evolution_state.generation}: {analysis.get('top_recommendation', 'unknown')}")
        
        # 保存数据
        self._save_data()
        
        return {
            'quality_score': quality_score,
            'quality_details': quality_result,
            'rl_action': action,
            'rl_reward': reward,
            'convergence': {
                'should_evolve': should_evolve,
                'reason': reason,
                'trend': self.convergence_engine.get_trend_analysis()
            },
            'evolution_plan': evolution_plan,
            'generation': self.evolution_state.generation
        }
    
    def get_learning_report(self) -> str:
        """生成学习报告"""
        trend = self.convergence_engine.get_trend_analysis()
        
        report = f"""
# 🧠 {self.agent_name} 学习报告

## 📊 总体状态
- **代数**: {self.evolution_state.generation}
- **交互数**: {len(self.interactions)}
- **收敛状态**: {'✅ 已收敛' if self.evolution_state.is_converged else '🔄 学习中'}
- **趋势**: {trend['trend']} (斜率: {trend['slope']:.2f})

## 📈 能力表现
"""
        for name, metric in self.capability_metrics.items():
            report += f"- **{name}**: 使用{metric.total_uses}次, 质量{metric.avg_quality:.1f}, 满意度{metric.user_satisfaction:.0%}\n"
        
        report += f"\n## 🎯 强化学习策略\n"
        report += self.rl_system.get_policy_insights()
        
        if self.evolution_state.improvements_applied:
            report += f"\n\n## 🧬 进化历史\n"
            for imp in self.evolution_state.improvements_applied:
                report += f"- {imp}\n"
        
        return report
    
    def force_evolution(self, current_prompt: str, capabilities: List[str]) -> Dict:
        """强制触发一次进化"""
        analysis = self.meta_analyzer.analyze(
            self.agent_name, capabilities, self.interactions, self.capability_metrics
        )
        
        issues = [w['root_cause'] for w in analysis.get('weaknesses', [])]
        
        plan = self.evolution_advisor.generate_evolution_plan(
            self.agent_name, current_prompt, analysis, issues
        )
        
        self.evolution_state.generation += 1
        self.evolution_state.last_evolution = time.time()
        self.evolution_state.improvements_applied.append(f"Manual Gen{self.evolution_state.generation}")
        self._save_data()
        
        return {
            'analysis': analysis,
            'evolution_plan': plan,
            'generation': self.evolution_state.generation
        }


# CLI 命令支持
def create_evolve_command():
    """创建 kas evolve 命令"""
    import click
    
    @click.command()
    @click.argument('agent')
    @click.option('--force', is_flag=True, help='强制进化，忽略收敛状态')
    @click.option('--output', '-o', help='输出报告到文件')
    def evolve(agent, force, output):
        """触发 Agent 进化"""
        from rich.console import Console
        from rich.panel import Panel
        
        console = Console()
        
        # 初始化学习引擎
        engine = LLMEnhancedLearningEngine(agent)
        
        # 加载 agent 信息
        agent_dir = Path.home() / '.kas' / 'agents' / agent
        if not agent_dir.exists():
            console.print(f"❌ Agent not found: {agent}")
            return
        
        import yaml
        with open(agent_dir / 'agent.yaml', 'r') as f:
            agent_data = yaml.safe_load(f)
        
        current_prompt = agent_data.get('system_prompt', '')
        capabilities = [c['name'] for c in agent_data.get('capabilities', [])]
        
        if force or len(engine.interactions) == 0:
            # 强制进化
            console.print(f"🧬 [bold blue]Force evolving {agent}...[/bold blue]")
            result = engine.force_evolution(current_prompt, capabilities)
        else:
            # 显示当前状态
            console.print(engine.get_learning_report())
            result = None
        
        if result:
            console.print(Panel.fit(
                f"[bold cyan]Evolution Plan for {agent}[/bold cyan]\n\n"
                f"Generation: {result['generation']}\n"
                f"Top Recommendation: {result['analysis'].get('top_recommendation', 'N/A')}\n\n"
                f"[bold]New Prompt Preview:[/bold]\n"
                f"{result['evolution_plan'].get('new_system_prompt', 'N/A')[:200]}...",
                title="🧬 Evolution Result"
            ))
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                console.print(f"\n💾 Report saved to: {output}")
    
    return evolve


if __name__ == "__main__":
    # 测试代码
    print("🧠 LLM Enhanced Learning System")
    print("=" * 50)
    
    # 创建测试引擎
    engine = LLMEnhancedLearningEngine("TestAgent", storage_dir="/tmp/kas_test")
    
    # 模拟一些交互
    test_interactions = [
        ("如何写一个好的 Python 函数？", "一个好的 Python 函数应该遵循单一职责原则...", "👍"),
        ("这段代码有 bug 吗？", "让我看看... 这里确实有个问题，你应该...", None),
        ("帮我优化这个算法", "这个算法的时间复杂度是 O(n²)，可以优化到 O(n log n)...", "👍"),
    ]
    
    for user_msg, response, feedback in test_interactions:
        print(f"\n📝 Recording: {user_msg[:30]}...")
        result = engine.record_interaction(
            user_message=user_msg,
            agent_response=response,
            task_type="coding",
            capabilities=["Code Review", "Optimization"],
            user_feedback=feedback
        )
        print(f"   Quality: {result['quality_score']:.1f}/100")
        print(f"   RL Reward: {result['rl_reward']:+.2f}")
        print(f"   Converged: {result['convergence']['should_evolve']}")
    
    print("\n" + "=" * 50)
    print(engine.get_learning_report())
