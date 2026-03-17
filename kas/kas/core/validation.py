"""
KAS Core - Capability Validation
能力验证系统 - 给 Agent 出考题，确保不是"吹牛"

功能:
- 自动化测试套件
- 基准测试题目库
- 能力评分系统
- 测试报告生成
- 历史追踪
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from .models import Agent, Capability, CapabilityType
from .chat import ChatEngine

# 设置日志
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class TestResult:
    """单个测试结果"""
    test_id: str
    test_name: str
    capability_type: CapabilityType
    status: TestStatus
    score: float  # 0-100
    max_score: float
    feedback: str  # Agent 的回答/输出
    expected: str  # 期望的输出
    execution_time: float  # 执行时间(秒)
    error_message: str = ""
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED
    
    @property
    def percentage(self) -> float:
        if self.max_score > 0:
            return (self.score / self.max_score) * 100
        return 0.0


@dataclass
class ValidationReport:
    """验证报告"""
    agent_name: str
    agent_version: str
    timestamp: str
    overall_score: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    results: List[TestResult]
    capability_scores: Dict[str, float]  # 每种能力的平均分
    summary: str = ""
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class TestCase:
    """测试用例基类"""
    
    def __init__(self, test_id: str, name: str, capability: CapabilityType,
                 description: str = "", difficulty: int = 1):
        self.test_id = test_id
        self.name = name
        self.capability = capability
        self.description = description
        self.difficulty = difficulty  # 1-5
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        """运行测试，子类实现"""
        raise NotImplementedError


class CodeReviewTest(TestCase):
    """代码审查测试"""
    
    # 测试用例：有 bug 的代码
    BUGGY_CODE_SAMPLES = [
        {
            "id": "cr_1",
            "name": "空指针检查",
            "code": '''def process_user(user):
    print(user.name)  # 没有检查 user 是否为 None
    return user.id''',
            "bugs": ["未检查 user 是否为 None", "潜在的 NullPointerException"],
            "expected_keywords": ["None", "null", "检查", "判空"]
        },
        {
            "id": "cr_2", 
            "name": "资源泄露",
            "code": '''def read_file(path):
    f = open(path, 'r')
    return f.read()  # 没有关闭文件''',
            "bugs": ["文件未关闭", "资源泄露"],
            "expected_keywords": ["close", "with", "finally", "资源"]
        },
        {
            "id": "cr_3",
            "name": "SQL 注入",
            "code": '''def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)''',
            "bugs": ["SQL 注入风险", "字符串拼接 SQL"],
            "expected_keywords": ["注入", "参数化", "prepared", "安全"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.BUGGY_CODE_SAMPLES[sample_index % len(self.BUGGY_CODE_SAMPLES)]
        super().__init__(
            test_id=sample["id"],
            name=f"代码审查 - {sample['name']}",
            capability=CapabilityType.CODE_REVIEW,
            description=f"找出代码中的 {sample['name']} 问题",
            difficulty=2
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        # 构建提示
        prompt = f"""请审查以下代码，找出潜在问题:

```python
{self.sample['code']}
```

请列出发现的所有问题。"""
        
        # 调用 Agent
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            # 评分
            score = 0
            max_score = len(self.sample["expected_keywords"]) * 20
            
            found_keywords = []
            for keyword in self.sample["expected_keywords"]:
                if keyword.lower() in response.lower():
                    score += 20
                    found_keywords.append(keyword)
            
            # 检查是否提到了 bug
            for bug in self.sample["bugs"]:
                if any(kw in response.lower() for kw in bug.lower().split()):
                    score += 10
            
            score = min(score, 100)  # 满分 100
            
            status = TestStatus.PASSED if score >= 70 else TestStatus.FAILED  # 阈值从60提高到70
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected=f"应发现: {', '.join(self.sample['bugs'])}",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"代码审查测试失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected=f"应发现: {', '.join(self.sample['bugs'])}",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class DocumentationTest(TestCase):
    """文档编写测试"""
    
    CODE_SAMPLES = [
        {
            "id": "doc_1",
            "name": "函数文档",
            "code": '''def calculate_discount(price, customer_type):
    if customer_type == "vip":
        return price * 0.8
    elif customer_type == "member":
        return price * 0.9
    return price''',
            "expected_elements": ["参数", "返回", "customer_type", "vip", "member", "折扣"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.CODE_SAMPLES[sample_index % len(self.CODE_SAMPLES)]
        super().__init__(
            test_id=sample["id"],
            name=f"文档编写 - {sample['name']}",
            capability=CapabilityType.DOCUMENTATION,
            description=f"为代码编写文档",
            difficulty=2
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        prompt = f"""请为以下代码编写文档注释:

```python
{self.sample['code']}
```

请包含: 功能描述、参数说明、返回值说明。"""
        
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            # 评分
            score = 0
            for element in self.sample["expected_elements"]:
                if element.lower() in response.lower():
                    score += 15
            
            score = min(score, 100)
            status = TestStatus.PASSED if score >= 60 else TestStatus.FAILED  # 阈值从50提高到60
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected=f"应包含: {', '.join(self.sample['expected_elements'][:3])}...",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"文档编写测试失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected="文档生成",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class DebugTest(TestCase):
    """调试能力测试"""
    
    DEBUG_SCENARIOS = [
        {
            "id": "debug_1",
            "name": "异常处理",
            "error": '''Traceback (most recent call last):
  File "app.py", line 15, in <module>
    result = divide(10, 0)
  File "app.py", line 3, in divide
    return a / b
ZeroDivisionError: division by zero''',
            "expected_solution": ["ZeroDivisionError", "除零", "try", "except", "判断"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.DEBUG_SCENARIOS[sample_index % len(self.DEBUG_SCENARIOS)]
        super().__init__(
            test_id=sample["id"],
            name=f"调试 - {sample['name']}",
            capability=CapabilityType.DEBUGGING,
            description="分析错误并提供解决方案",
            difficulty=2
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        prompt = f"""请分析以下错误信息，说明原因和解决方案:

```
{self.sample['error']}
```"""
        
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            score = 0
            for keyword in self.sample["expected_solution"]:
                if keyword.lower() in response.lower():
                    score += 20
            
            score = min(score, 100)
            status = TestStatus.PASSED if score >= 60 else TestStatus.FAILED
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected="应解释错误原因并提供修复方案",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"调试测试失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected="调试分析",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class TestGenerationTest(TestCase):
    """测试生成能力测试"""
    
    CODE_SAMPLES = [
        {
            "id": "test_1",
            "name": "函数测试",
            "code": '''def calculate_discount(price, customer_type):
    """Calculate discount based on customer type"""
    if customer_type == "vip":
        return price * 0.8
    elif customer_type == "member":
        return price * 0.9
    return price''',
            "expected_tests": ["test", "assert", "vip", "member", "price", "边界"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.CODE_SAMPLES[sample_index % len(self.CODE_SAMPLES)]
        super().__init__(
            test_id=sample["id"],
            name=f"测试生成 - {sample['name']}",
            capability=CapabilityType.TEST_GENERATION,
            description=f"为代码生成测试用例",
            difficulty=3
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        prompt = f"""请为以下函数生成单元测试:

```python
{self.sample['code']}
```

请包含: 正常情况、边界情况、异常情况。"""
        
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            score = 0
            for keyword in self.sample["expected_tests"]:
                if keyword.lower() in response.lower():
                    score += 15
            
            score = min(score, 100)
            status = TestStatus.PASSED if score >= 60 else TestStatus.FAILED
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected=f"应包含: {', '.join(self.sample['expected_tests'][:4])}...",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"测试生成失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected="测试用例生成",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class RefactoringTest(TestCase):
    """重构能力测试"""
    
    REFACTOR_SCENARIOS = [
        {
            "id": "refactor_1",
            "name": "简化条件",
            "code": '''def get_price(quantity):
    if quantity == 1:
        return 100
    elif quantity == 2:
        return 180
    elif quantity == 3:
        return 250
    else:
        return quantity * 80''',
            "expected_improvements": ["字典", "dict", "映射", "表驱动", "简化"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.REFACTOR_SCENARIOS[sample_index % len(self.REFACTOR_SCENARIOS)]
        super().__init__(
            test_id=sample["id"],
            name=f"重构 - {sample['name']}",
            capability=CapabilityType.REFACTORING,
            description="重构代码以提高质量",
            difficulty=3
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        prompt = f"""请重构以下代码，使其更清晰:

```python
{self.sample['code']}
```

请说明: 1) 问题所在 2) 重构后的代码 3) 改进点。"""
        
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            score = 0
            for keyword in self.sample["expected_improvements"]:
                if keyword.lower() in response.lower():
                    score += 20
            
            score = min(score, 100)
            status = TestStatus.PASSED if score >= 60 else TestStatus.FAILED
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected="应使用字典/映射表等简化条件",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"重构测试失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected="代码重构",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class ArchitectureTest(TestCase):
    """架构能力测试"""
    
    ARCH_SCENARIOS = [
        {
            "id": "arch_1",
            "name": "设计评估",
            "description": '''现有系统是一个单体应用，包含:
- 用户管理模块 (用户注册/登录/权限)
- 订单处理模块 (下单/支付/退款)
- 库存管理模块 (库存查询/更新)
- 报表模块 (数据统计/导出)

目前所有模块共享同一个数据库，部署在一起。''',
            "expected_points": ["微服务", "拆分", "独立", "数据库", "解耦", "扩展"]
        }
    ]
    
    def __init__(self, sample_index: int = 0):
        sample = self.ARCH_SCENARIOS[sample_index % len(self.ARCH_SCENARIOS)]
        super().__init__(
            test_id=sample["id"],
            name=f"架构 - {sample['name']}",
            capability=CapabilityType.ARCHITECTURE,
            description="评估架构设计并提供改进建议",
            difficulty=4
        )
        self.sample = sample
    
    def run(self, agent: Agent, chat_engine: ChatEngine) -> TestResult:
        import time
        start_time = time.time()
        
        prompt = f"""请评估以下架构设计:

{self.sample['description']}

请分析: 1) 当前架构的问题 2) 改进建议 3) 迁移方案。"""
        
        try:
            response = chat_engine.chat(agent, prompt)
            execution_time = time.time() - start_time
            
            score = 0
            for keyword in self.sample["expected_points"]:
                if keyword.lower() in response.lower():
                    score += 15
            
            score = min(score, 100)
            status = TestStatus.PASSED if score >= 60 else TestStatus.FAILED
            
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=status,
                score=score,
                max_score=100,
                feedback=response[:500],
                expected="应考虑微服务拆分、数据库解耦等",
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"架构评估失败 {self.test_id}: {e}", exc_info=True)
            return TestResult(
                test_id=self.test_id,
                test_name=self.name,
                capability_type=self.capability,
                status=TestStatus.ERROR,
                score=0,
                max_score=100,
                feedback="",
                expected="架构设计评估",
                execution_time=time.time() - start_time,
                error_message=str(e)
            )


class CapabilityValidator:
    """
    能力验证器 - 给 Agent 出考题
    
    用法:
        validator = CapabilityValidator()
        report = validator.validate(agent)
        
        # 特定能力测试
        report = validator.validate_capability(agent, CapabilityType.CODE_REVIEW)
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.chat_engine = ChatEngine(llm_client) if llm_client else None
        self.test_cases: List[TestCase] = []
        self._init_test_cases()
    
    def _init_test_cases(self):
        """初始化测试用例"""
        # 代码审查测试
        for i in range(len(CodeReviewTest.BUGGY_CODE_SAMPLES)):
            self.test_cases.append(CodeReviewTest(i))
        
        # 文档测试
        for i in range(len(DocumentationTest.CODE_SAMPLES)):
            self.test_cases.append(DocumentationTest(i))
        
        # 调试测试
        for i in range(len(DebugTest.DEBUG_SCENARIOS)):
            self.test_cases.append(DebugTest(i))
        
        # 测试生成
        for i in range(len(TestGenerationTest.CODE_SAMPLES)):
            self.test_cases.append(TestGenerationTest(i))
        
        # 重构测试
        for i in range(len(RefactoringTest.REFACTOR_SCENARIOS)):
            self.test_cases.append(RefactoringTest(i))
        
        # 架构测试
        for i in range(len(ArchitectureTest.ARCH_SCENARIOS)):
            self.test_cases.append(ArchitectureTest(i))
    
    def validate(self, agent: Agent, capability_filter: List[CapabilityType] = None) -> ValidationReport:
        """
        全面验证 Agent
        
        Args:
            agent: 要验证的 Agent
            capability_filter: 只测试特定能力（可选）
        
        Returns:
            验证报告
        """
        if not self.chat_engine:
            return self._create_error_report(agent, "未配置 LLM 客户端")
        
        results = []
        
        # 筛选测试用例
        test_cases = self.test_cases
        if capability_filter:
            test_cases = [t for t in test_cases if t.capability in capability_filter]
        
        print(f"🧪 运行 {len(test_cases)} 个测试...")
        
        for i, test in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] {test.name}...", end=" ")
            result = test.run(agent, self.chat_engine)
            results.append(result)
            
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
            print(f"{status_icon} {result.score:.0f}分")
        
        # 生成报告
        return self._generate_report(agent, results)
    
    def validate_capability(self, agent: Agent, capability: CapabilityType) -> ValidationReport:
        """验证特定能力"""
        return self.validate(agent, [capability])
    
    def _generate_report(self, agent: Agent, results: List[TestResult]) -> ValidationReport:
        """生成验证报告"""
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in results if r.status == TestStatus.FAILED)
        
        # 计算总体分数
        total_score = sum(r.score for r in results)
        max_total = sum(r.max_score for r in results)
        overall_score = (total_score / max_total * 100) if max_total > 0 else 0
        
        # 按能力分类统计
        capability_scores = {}
        capability_counts = {}
        
        for r in results:
            cap_name = r.capability_type.value
            if cap_name not in capability_scores:
                capability_scores[cap_name] = 0
                capability_counts[cap_name] = 0
            capability_scores[cap_name] += r.score
            capability_counts[cap_name] += 1
        
        # 计算平均分
        for cap in capability_scores:
            capability_scores[cap] /= capability_counts[cap]
        
        # 生成建议
        recommendations = []
        for cap, score in capability_scores.items():
            if score < 50:
                recommendations.append(f"{cap} 能力较弱，建议针对性训练")
            elif score < 70:
                recommendations.append(f"{cap} 能力一般，有提升空间")
        
        if overall_score >= 80:
            summary = f"🌟 优秀！{agent.name} 综合得分 {overall_score:.1f}，所有能力都很强"
        elif overall_score >= 60:
            summary = f"✅ 良好。{agent.name} 综合得分 {overall_score:.1f}，大部分能力合格"
        else:
            summary = f"⚠️  需改进。{agent.name} 综合得分 {overall_score:.1f}，建议继续进化"
        
        return ValidationReport(
            agent_name=agent.name,
            agent_version=agent.version,
            timestamp=datetime.now().isoformat(),
            overall_score=overall_score,
            total_tests=len(results),
            passed_tests=passed,
            failed_tests=failed,
            results=results,
            capability_scores=capability_scores,
            summary=summary,
            recommendations=recommendations
        )
    
    def _create_error_report(self, agent: Agent, error: str) -> ValidationReport:
        """创建错误报告"""
        return ValidationReport(
            agent_name=agent.name,
            agent_version=agent.version,
            timestamp=datetime.now().isoformat(),
            overall_score=0,
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            results=[],
            capability_scores={},
            summary=f"验证失败: {error}",
            recommendations=["请配置 LLM API 以进行验证"]
        )
    
    def format_report(self, report: ValidationReport) -> str:
        """格式化报告（用于 CLI 显示）"""
        lines = [
            f"\n📋 {report.agent_name} v{report.agent_version} 能力验证报告",
            "=" * 60,
            f"\n🎯 综合得分: {report.overall_score:.1f}/100",
            f"📊 测试通过: {report.passed_tests}/{report.total_tests}",
            f"\n📌 {report.summary}",
            "\n💪 能力细分:",
        ]
        
        for cap, score in report.capability_scores.items():
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"  {cap:20s} {bar} {score:.1f}")
        
        if report.recommendations:
            lines.append("\n💡 建议:")
            for rec in report.recommendations:
                lines.append(f"  • {rec}")
        
        lines.append("\n📑 详细结果:")
        for r in report.results:
            icon = "✅" if r.status == TestStatus.PASSED else "❌" if r.status == TestStatus.FAILED else "⚠️"
            lines.append(f"  {icon} {r.test_name}: {r.score:.0f}/{r.max_score:.0f}分 ({r.execution_time:.1f}s)")
        
        return "\n".join(lines)


# 便捷函数
def validate_agent(agent: Agent, llm_client=None) -> ValidationReport:
    """验证 Agent"""
    validator = CapabilityValidator(llm_client)
    return validator.validate(agent)


if __name__ == "__main__":
    # 测试
    print("🧪 测试验证系统")
    
    from kas.core.models import Agent, Capability, CapabilityType
    
    # 创建测试 Agent
    agent = Agent(
        name="TestValidator",
        capabilities=[
            Capability("Code Review", CapabilityType.CODE_REVIEW, "审查代码", 0.9),
            Capability("Docs", CapabilityType.DOCUMENTATION, "写文档", 0.8)
        ],
        system_prompt="你是一个代码专家..."
    )
    
    # 测试（需要 LLM）
    print("\n✅ 验证系统已就绪")
    print(f"   可用测试用例: {len(CapabilityValidator().test_cases)} 个")
