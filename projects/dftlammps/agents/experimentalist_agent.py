"""
Experimentalist Agent - 实验员Agent
负责实验设计、执行、数据分析
"""
from __future__ import annotations
import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import uuid

from ..multi_agent.agent_core import (
    DeliberativeAgent, Message, MessageType,
    Observation, Action, AgentCapability
)


@dataclass
class ExperimentDesign:
    """实验设计"""
    id: str
    name: str
    objective: str
    hypothesis_id: Optional[str] = None
    
    # 实验变量
    independent_variables: List[Dict[str, Any]] = field(default_factory=list)
    dependent_variables: List[Dict[str, Any]] = field(default_factory=list)
    control_variables: List[Dict[str, Any]] = field(default_factory=list)
    
    # 实验条件
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    sample_size: int = 0
    
    # 测量
    measurements: List[Dict[str, Any]] = field(default_factory=list)
    measurement_frequency: str = "once"  # once, continuous, periodic
    
    # 成功标准
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "objective": self.objective,
            "hypothesis_id": self.hypothesis_id,
            "independent_variables": self.independent_variables,
            "dependent_variables": self.dependent_variables,
            "control_variables": self.control_variables,
            "conditions": self.conditions,
            "sample_size": self.sample_size,
            "measurements": self.measurements,
            "success_criteria": self.success_criteria
        }


@dataclass
class Experiment:
    """实验实例"""
    id: str
    design_id: str
    status: str = "planned"  # planned, running, completed, failed
    
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # 实际参数
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 数据采集
    data_points: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    
    # 结果
    results: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "design_id": self.design_id,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "parameters": self.parameters,
            "data_points_count": len(self.data_points),
            "observations": self.observations,
            "results": self.results
        }


@dataclass
class DataAnalysis:
    """数据分析结果"""
    id: str
    experiment_id: str
    analysis_type: str
    
    # 统计结果
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # 可视化
    visualizations: List[str] = field(default_factory=list)
    
    # 发现
    findings: List[Dict[str, Any]] = field(default_factory=list)
    
    # 假设检验
    hypothesis_tests: List[Dict[str, Any]] = field(default_factory=list)
    
    # 置信度
    confidence_level: float = 0.95
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "experiment_id": self.experiment_id,
            "analysis_type": self.analysis_type,
            "statistics": self.statistics,
            "findings": self.findings,
            "hypothesis_tests": self.hypothesis_tests,
            "confidence_level": self.confidence_level
        }


class ExperimentalistAgent(DeliberativeAgent):
    """
    实验员Agent
    负责科学研究中的实验工作
    """
    
    def __init__(self, **kwargs):
        kwargs.setdefault("name", "ExperimentalistAgent")
        kwargs.setdefault("description", "Designs and executes experiments, analyzes data")
        super().__init__(**kwargs)
        
        # 知识库
        self.experiment_designs: Dict[str, ExperimentDesign] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.data_analyses: Dict[str, DataAnalysis] = {}
        
        # 仪器和方法库
        self.available_methods: Dict[str, Dict[str, Any]] = {}
        self.measurement_tools: Dict[str, Dict[str, Any]] = {}
        
        # 注册能力
        self._register_capabilities()
        
        # 注册消息处理器
        self.register_message_handler(
            MessageType.COMMUNICATION,
            self._handle_collaboration_message
        )
    
    def _register_capabilities(self) -> None:
        """注册专业能力"""
        self.register_capability(AgentCapability(
            name="design_experiment",
            description="Design experiment to test hypothesis",
            handler=self._design_experiment_handler
        ))
        
        self.register_capability(AgentCapability(
            name="execute_experiment",
            description="Execute designed experiment",
            handler=self._execute_experiment_handler
        ))
        
        self.register_capability(AgentCapability(
            name="collect_data",
            description="Collect experimental data",
            handler=self._collect_data_handler
        ))
        
        self.register_capability(AgentCapability(
            name="analyze_data",
            description="Analyze experimental data",
            handler=self._analyze_data_handler
        ))
        
        self.register_capability(AgentCapability(
            name="validate_results",
            description="Validate experimental results",
            handler=self._validate_results_handler
        ))
        
        self.register_capability(AgentCapability(
            name="replicate_experiment",
            description="Replicate previous experiment",
            handler=self._replicate_experiment_handler
        ))
    
    async def perceive(self) -> List[Observation]:
        """感知"""
        observations = []
        
        # 检查消息队列
        while not self.inbox.empty():
            try:
                message = self.inbox.get_nowait()
                observations.append(Observation(
                    source="message",
                    data=message.content,
                    confidence=0.95
                ))
            except asyncio.QueueEmpty:
                break
        
        # 检查正在运行的实验状态
        for exp in self.experiments.values():
            if exp.status == "running":
                observations.append(Observation(
                    source=f"experiment_{exp.id}",
                    data={"status": "in_progress", "experiment_id": exp.id},
                    confidence=0.9
                ))
        
        return observations
    
    async def reason(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """推理"""
        decision = {
            "actions": [],
            "reasoning_log": []
        }
        
        trigger = context.get("trigger")
        
        if trigger:
            trigger_type = trigger.get("type")
            
            if trigger_type == "design_request":
                # 设计实验请求
                hypothesis = trigger.get("hypothesis")
                design = await self.design_experiment(hypothesis)
                if design:
                    decision["actions"].append({
                        "type": "store_design",
                        "design": design.to_dict()
                    })
            
            elif trigger_type == "execute_request":
                design_id = trigger.get("design_id")
                if design_id in self.experiment_designs:
                    decision["actions"].append({
                        "type": "execute",
                        "design_id": design_id
                    })
            
            elif trigger_type == "analyze_request":
                experiment_id = trigger.get("experiment_id")
                if experiment_id in self.experiments:
                    decision["actions"].append({
                        "type": "analyze",
                        "experiment_id": experiment_id
                    })
        
        # 检查是否有完成的实验需要分析
        completed_experiments = [
            exp for exp in self.experiments.values()
            if exp.status == "completed" and not exp.results
        ]
        
        for exp in completed_experiments:
            decision["actions"].append({
                "type": "analyze",
                "experiment_id": exp.id
            })
        
        return decision
    
    async def act(self, decision: Dict[str, Any]) -> List[Action]:
        """行动"""
        actions = []
        
        for action_data in decision.get("actions", []):
            action_type = action_data.get("type")
            
            if action_type == "store_design":
                actions.append(Action(
                    action_type="broadcast_design",
                    params={"design": action_data.get("design")},
                    priority=2
                ))
            
            elif action_type == "execute":
                actions.append(Action(
                    action_type="execute_experiment",
                    params={"design_id": action_data.get("design_id")},
                    priority=3
                ))
            
            elif action_type == "analyze":
                actions.append(Action(
                    action_type="analyze_data",
                    params={"experiment_id": action_data.get("experiment_id")},
                    priority=2
                ))
        
        return actions
    
    # ===== 核心能力实现 =====
    
    async def design_experiment(
        self,
        hypothesis: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[ExperimentDesign]:
        """
        设计实验来验证假设
        """
        design_id = f"design_{uuid.uuid4().hex[:8]}"
        
        # 从假设中提取关键信息
        hypothesis_statement = hypothesis.get("statement", "")
        hypothesis_id = hypothesis.get("id", "")
        
        # 设计实验变量
        independent_vars = self._identify_independent_variables(hypothesis)
        dependent_vars = self._identify_dependent_variables(hypothesis)
        control_vars = self._identify_control_variables(hypothesis)
        
        # 设计实验条件
        conditions = self._design_conditions(independent_vars)
        
        # 确定样本量
        sample_size = self._calculate_sample_size(
            len(conditions),
            effect_size=hypothesis.get("expected_effect_size", 0.5),
            power=0.8
        )
        
        # 设计测量方案
        measurements = self._design_measurements(dependent_vars)
        
        # 制定成功标准
        success_criteria = {
            "statistical_significance": "p < 0.05",
            "effect_size": "medium or larger",
            "reproducibility": "confirmed in at least 2 replicates"
        }
        
        design = ExperimentDesign(
            id=design_id,
            name=f"Experiment for {hypothesis_id}",
            objective=f"Test hypothesis: {hypothesis_statement[:50]}...",
            hypothesis_id=hypothesis_id,
            independent_variables=independent_vars,
            dependent_variables=dependent_vars,
            control_variables=control_vars,
            conditions=conditions,
            sample_size=sample_size,
            measurements=measurements,
            success_criteria=success_criteria
        )
        
        self.experiment_designs[design_id] = design
        
        # 存储到长期记忆
        self.long_term_memory.add_experience({
            "type": "experiment_design",
            "design_id": design_id,
            "hypothesis_id": hypothesis_id
        })
        
        return design
    
    async def execute_experiment(self, design_id: str) -> Optional[Experiment]:
        """
        执行实验
        """
        if design_id not in self.experiment_designs:
            return None
        
        design = self.experiment_designs[design_id]
        
        experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
        
        experiment = Experiment(
            id=experiment_id,
            design_id=design_id,
            status="running",
            start_time=datetime.now(),
            parameters={
                "conditions": design.conditions,
                "sample_size": design.sample_size
            }
        )
        
        self.experiments[experiment_id] = experiment
        
        # 模拟实验执行
        # 实际应用中这里应该控制真实的实验设备或调用仿真
        await self._run_experiment_simulation(experiment, design)
        
        return experiment
    
    async def _run_experiment_simulation(
        self,
        experiment: Experiment,
        design: ExperimentDesign
    ) -> None:
        """
        模拟实验执行
        实际应用中应该替换为真实的实验控制代码
        """
        # 模拟数据采集
        for condition in design.conditions:
            for sample in range(design.sample_size // len(design.conditions)):
                # 生成模拟数据
                data_point = self._generate_data_point(condition, design)
                experiment.data_points.append(data_point)
                
                # 模拟执行时间
                await asyncio.sleep(0.01)
        
        # 记录观察
        experiment.observations = [
            f"Executed {len(design.conditions)} conditions",
            f"Collected {len(experiment.data_points)} data points",
            "No anomalies detected during execution"
        ]
        
        # 完成实验
        experiment.status = "completed"
        experiment.end_time = datetime.now()
    
    async def analyze_experiment(
        self,
        experiment_id: str,
        analysis_types: Optional[List[str]] = None
    ) -> Optional[DataAnalysis]:
        """
        分析实验数据
        """
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != "completed":
            return None
        
        analysis_id = f"analysis_{uuid.uuid4().hex[:8]}"
        
        analysis_types = analysis_types or ["descriptive", "inferential"]
        
        statistics = {}
        findings = []
        hypothesis_tests = []
        
        # 描述性统计
        if "descriptive" in analysis_types:
            statistics["descriptive"] = self._calculate_descriptive_stats(
                experiment.data_points
            )
        
        # 推论统计
        if "inferential" in analysis_types:
            stats, tests = self._perform_inferential_analysis(
                experiment.data_points,
                experiment.design_id
            )
            statistics["inferential"] = stats
            hypothesis_tests = tests
        
        # 相关性分析
        if "correlation" in analysis_types:
            statistics["correlation"] = self._calculate_correlations(
                experiment.data_points
            )
        
        # 提取发现
        findings = self._extract_findings(statistics, hypothesis_tests)
        
        data_analysis = DataAnalysis(
            id=analysis_id,
            experiment_id=experiment_id,
            analysis_type=",".join(analysis_types),
            statistics=statistics,
            findings=findings,
            hypothesis_tests=hypothesis_tests
        )
        
        self.data_analyses[analysis_id] = data_analysis
        
        # 更新实验结果
        experiment.results = {
            "analysis_id": analysis_id,
            "summary": findings,
            "statistics": statistics
        }
        
        return data_analysis
    
    async def validate_results(
        self,
        experiment_id: str,
        validation_method: str = "replication"
    ) -> Dict[str, Any]:
        """
        验证实验结果
        """
        if experiment_id not in self.experiments:
            return {"error": "Experiment not found"}
        
        experiment = self.experiments[experiment_id]
        
        validation_result = {
            "experiment_id": experiment_id,
            "validation_method": validation_method,
            "passed": False,
            "details": {}
        }
        
        if validation_method == "replication":
            # 重复实验
            replicate = await self.execute_experiment(experiment.design_id)
            if replicate:
                analysis = await self.analyze_experiment(replicate.id)
                
                # 比较结果
                original_results = experiment.results or {}
                replicate_results = replicate.results or {}
                
                # 简化：检查主要发现是否一致
                consistency = self._compare_results(
                    original_results,
                    replicate_results
                )
                
                validation_result["passed"] = consistency > 0.7
                validation_result["details"]["consistency"] = consistency
                validation_result["details"]["replicate_id"] = replicate.id
        
        elif validation_method == "cross_validation":
            # 交叉验证
            validation_result["details"]["k_fold_results"] = self._cross_validate(
                experiment.data_points
            )
            validation_result["passed"] = True
        
        return validation_result
    
    async def replicate_experiment(
        self,
        original_experiment_id: str,
        variations: Optional[Dict[str, Any]] = None
    ) -> Optional[Experiment]:
        """
        重复实验（可能包含变化）
        """
        if original_experiment_id not in self.experiments:
            return None
        
        original = self.experiments[original_experiment_id]
        
        # 使用相同的设计
        design_id = original.design_id
        
        # 创建新的实验实例
        replicate = await self.execute_experiment(design_id)
        
        if replicate and variations:
            # 应用变化
            replicate.parameters.update(variations)
        
        return replicate
    
    # ===== 辅助方法 =====
    
    def _identify_independent_variables(
        self,
        hypothesis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别自变量"""
        # 从假设中提取
        statement = hypothesis.get("statement", "")
        
        # 简化的变量识别
        variables = []
        
        # 常见科学变量模式
        patterns = [
            ("temperature", "continuous", ["low", "medium", "high"]),
            ("pressure", "continuous", ["1 atm", "2 atm", "5 atm"]),
            ("concentration", "continuous", ["0.1 M", "0.5 M", "1.0 M"]),
            ("time", "continuous", ["short", "medium", "long"])
        ]
        
        for var_name, var_type, levels in patterns:
            if var_name.lower() in statement.lower():
                variables.append({
                    "name": var_name,
                    "type": var_type,
                    "levels": levels
                })
        
        if not variables:
            # 默认变量
            variables.append({
                "name": "condition",
                "type": "categorical",
                "levels": ["control", "treatment"]
            })
        
        return variables
    
    def _identify_dependent_variables(
        self,
        hypothesis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别因变量"""
        statement = hypothesis.get("statement", "")
        
        # 查找预测结果
        predictions = hypothesis.get("predictions", [])
        
        variables = []
        for pred in predictions:
            # 简化提取
            variables.append({
                "name": "outcome",
                "type": "continuous",
                "unit": "arbitrary_units"
            })
        
        if not variables:
            variables.append({
                "name": "response",
                "type": "continuous",
                "unit": "measured_value"
            })
        
        return variables
    
    def _identify_control_variables(
        self,
        hypothesis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """识别控制变量"""
        # 从假设的假设中识别
        assumptions = hypothesis.get("assumptions", [])
        
        controls = []
        for assumption in assumptions:
            controls.append({
                "name": f"controlled_{len(controls)}",
                "value": "constant",
                "rationale": assumption
            })
        
        # 默认控制
        controls.extend([
            {"name": "environment", "value": "laboratory", "rationale": "Standard conditions"},
            {"name": "equipment", "value": "calibrated", "rationale": "Consistency"}
        ])
        
        return controls
    
    def _design_conditions(
        self,
        independent_vars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """设计实验条件"""
        conditions = []
        
        # 全因子设计（简化）
        if len(independent_vars) == 1:
            var = independent_vars[0]
            for level in var.get("levels", []):
                conditions.append({
                    var["name"]: level,
                    "description": f"{var['name']} = {level}"
                })
        elif len(independent_vars) == 2:
            var1, var2 = independent_vars
            for level1 in var1.get("levels", []):
                for level2 in var2.get("levels", []):
                    conditions.append({
                        var1["name"]: level1,
                        var2["name"]: level2,
                        "description": f"{var1['name']}={level1}, {var2['name']}={level2}"
                    })
        else:
            # 单因素变化
            conditions.append({"type": "control", "description": "Control condition"})
            for var in independent_vars:
                for level in var.get("levels", [])[1:]:  # 跳过第一个（控制）
                    conditions.append({
                        var["name"]: level,
                        "type": "treatment",
                        "description": f"Treatment: {var['name']} = {level}"
                    })
        
        return conditions
    
    def _calculate_sample_size(
        self,
        num_conditions: int,
        effect_size: float = 0.5,
        power: float = 0.8,
        alpha: float = 0.05
    ) -> int:
        """计算样本量"""
        # 简化的样本量计算
        # 实际应该使用功效分析
        base_sample = 30
        
        # 考虑条件数量
        samples_per_condition = max(10, int(20 * (1 / effect_size)))
        
        total_sample = samples_per_condition * num_conditions
        
        return min(total_sample, 1000)  # 上限
    
    def _design_measurements(
        self,
        dependent_vars: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """设计测量方案"""
        measurements = []
        
        for var in dependent_vars:
            measurements.append({
                "variable": var["name"],
                "method": "direct_measurement",
                "instrument": "appropriate_sensor",
                "precision": "0.01",
                "frequency": "per_sample"
            })
        
        return measurements
    
    def _generate_data_point(
        self,
        condition: Dict[str, Any],
        design: ExperimentDesign
    ) -> Dict[str, Any]:
        """生成模拟数据点"""
        data = {
            "condition": condition,
            "timestamp": datetime.now().isoformat()
        }
        
        # 根据条件生成因变量值
        base_value = 10.0
        
        if "treatment" in str(condition).lower():
            base_value += 2.5  # 处理效应
        
        # 添加噪声
        for var in design.dependent_variables:
            noise = random.gauss(0, 1)
            data[var["name"]] = base_value + noise
        
        return data
    
    def _calculate_descriptive_stats(
        self,
        data_points: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """计算描述性统计"""
        if not data_points:
            return {}
        
        # 收集数值
        values = []
        for dp in data_points:
            for key, val in dp.items():
                if isinstance(val, (int, float)) and key != "timestamp":
                    values.append(val)
        
        if not values:
            return {}
        
        n = len(values)
        mean = sum(values) / n
        variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0
        std_dev = variance ** 0.5
        
        sorted_values = sorted(values)
        median = sorted_values[n // 2] if n % 2 else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        
        return {
            "n": n,
            "mean": mean,
            "median": median,
            "std_dev": std_dev,
            "min": min(values),
            "max": max(values)
        }
    
    def _perform_inferential_analysis(
        self,
        data_points: List[Dict[str, Any]],
        design_id: str
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """执行推论分析"""
        # 简化：模拟t检验
        
        # 分组数据
        control = [dp for dp in data_points if "control" in str(dp.get("condition", ""))]
        treatment = [dp for dp in data_points if "treatment" in str(dp.get("condition", ""))]
        
        stats = {
            "control_n": len(control),
            "treatment_n": len(treatment)
        }
        
        # 模拟假设检验
        hypothesis_tests = [{
            "test": "t_test",
            "null_hypothesis": "No difference between groups",
            "p_value": random.uniform(0.01, 0.1),
            "significant": random.random() > 0.3,
            "effect_size": random.uniform(0.2, 0.8)
        }]
        
        return stats, hypothesis_tests
    
    def _calculate_correlations(
        self,
        data_points: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算相关性"""
        # 简化实现
        return {"variable_1_vs_2": random.uniform(-0.5, 0.8)}
    
    def _extract_findings(
        self,
        statistics: Dict[str, Any],
        hypothesis_tests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """提取研究发现"""
        findings = []
        
        for test in hypothesis_tests:
            if test.get("significant"):
                findings.append({
                    "type": "significant_effect",
                    "description": f"Significant effect detected (p={test['p_value']:.4f})",
                    "confidence": "high" if test["p_value"] < 0.01 else "medium"
                })
        
        if not findings:
            findings.append({
                "type": "no_significant_effect",
                "description": "No statistically significant effects detected",
                "confidence": "medium"
            })
        
        return findings
    
    def _compare_results(
        self,
        result1: Dict[str, Any],
        result2: Dict[str, Any]
    ) -> float:
        """比较两个结果的一致性"""
        # 简化：返回随机一致性分数
        return random.uniform(0.6, 1.0)
    
    def _cross_validate(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行交叉验证"""
        return {
            "k": 5,
            "average_score": random.uniform(0.7, 0.95)
        }
    
    # ===== 消息处理器 =====
    
    async def _handle_collaboration_message(self, message: Message) -> None:
        """处理协作消息"""
        content = message.content
        
        if content.get("request_type") == "design_experiment":
            design = await self.design_experiment(
                content.get("hypothesis", {}),
                content.get("constraints")
            )
            
            await self.send_message(
                {"design": design.to_dict() if design else None},
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT
            )
        
        elif content.get("request_type") == "execute":
            design_id = content.get("design_id")
            experiment = await self.execute_experiment(design_id)
            
            await self.send_message(
                {"experiment": experiment.to_dict() if experiment else None},
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT
            )
        
        elif content.get("request_type") == "analyze":
            analysis = await self.analyze_experiment(
                content.get("experiment_id"),
                content.get("analysis_types")
            )
            
            await self.send_message(
                {"analysis": analysis.to_dict() if analysis else None},
                receiver_id=message.sender_id,
                message_type=MessageType.RESULT
            )
    
    # ===== 能力处理器 =====
    
    async def _design_experiment_handler(self, **kwargs) -> Dict[str, Any]:
        design = await self.design_experiment(
            kwargs.get("hypothesis", {}),
            kwargs.get("constraints")
        )
        return {"design": design.to_dict() if design else None}
    
    async def _execute_experiment_handler(self, **kwargs) -> Dict[str, Any]:
        experiment = await self.execute_experiment(kwargs.get("design_id"))
        return {"experiment": experiment.to_dict() if experiment else None}
    
    async def _collect_data_handler(self, **kwargs) -> Dict[str, Any]:
        experiment_id = kwargs.get("experiment_id")
        if experiment_id in self.experiments:
            return {"data_points": self.experiments[experiment_id].data_points}
        return {"error": "Experiment not found"}
    
    async def _analyze_data_handler(self, **kwargs) -> Dict[str, Any]:
        analysis = await self.analyze_experiment(
            kwargs.get("experiment_id"),
            kwargs.get("analysis_types")
        )
        return {"analysis": analysis.to_dict() if analysis else None}
    
    async def _validate_results_handler(self, **kwargs) -> Dict[str, Any]:
        result = await self.validate_results(
            kwargs.get("experiment_id"),
            kwargs.get("validation_method", "replication")
        )
        return result
    
    async def _replicate_experiment_handler(self, **kwargs) -> Dict[str, Any]:
        replicate = await self.replicate_experiment(
            kwargs.get("original_experiment_id"),
            kwargs.get("variations")
        )
        return {"replicate": replicate.to_dict() if replicate else None}
    
    # ===== 公共API =====
    
    def get_experiment_designs(self) -> List[Dict[str, Any]]:
        """获取所有实验设计"""
        return [d.to_dict() for d in self.experiment_designs.values()]
    
    def get_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取实验列表"""
        experiments = self.experiments.values()
        if status:
            experiments = [e for e in experiments if e.status == status]
        return [e.to_dict() for e in experiments]
    
    def get_analyses(self) -> List[Dict[str, Any]]:
        """获取所有分析"""
        return [a.to_dict() for a in self.data_analyses.values()]
