"""
交互式助手模块 - Interactive Chat Assistant
=========================================
提供问答系统、实时计算指导和故障诊断建议。

Author: DFT-LAMMPS Team
Date: 2025
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
from pathlib import Path
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntentType(Enum):
    """用户意图类型"""
    GENERAL_QUESTION = auto()      # 一般问题
    CODE_REQUEST = auto()          # 代码请求
    ERROR_HELP = auto()            # 错误求助
    PARAMETER_ADVICE = auto()      # 参数建议
    INTERPRETATION = auto()        # 结果解释
    WORKFLOW_HELP = auto()         # 工作流帮助
    TUTORIAL_REQUEST = auto()      # 教程请求
    TOOL_RECOMMENDATION = auto()   # 工具推荐
    GREETING = auto()              # 问候
    UNKNOWN = auto()               # 未知


class ExpertiseLevel(Enum):
    """用户专业水平"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    user_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    current_topic: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    active_calculation: Optional[str] = None
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """添加消息到历史"""
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
        # 限制历史长度
        if len(self.history) > 20:
            self.history = self.history[-20:]


@dataclass
class QAResponse:
    """问答响应"""
    answer: str
    confidence: float
    sources: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """诊断报告"""
    problem_type: str
    severity: str  # low, medium, high, critical
    description: str
    possible_causes: List[str] = field(default_factory=list)
    recommended_solutions: List[Dict[str, Any]] = field(default_factory=list)
    prevention_tips: List[str] = field(default_factory=list)


class KnowledgeBase:
    """知识库 - 存储常见问题和答案"""
    
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or Path(__file__).parent / "data"
        self.qa_pairs: Dict[str, Dict[str, Any]] = {}
        self.tutorials: Dict[str, Dict[str, Any]] = {}
        self.troubleshooting: Dict[str, List[Dict[str, Any]]] = {}
        
        self._load_knowledge()
    
    def _load_knowledge(self):
        """加载知识库"""
        # FAQ数据
        self.qa_pairs = {
            "encut_convergence": {
                "question": "How do I determine the ENCUT value for VASP calculations?",
                "answer": """The ENCUT (energy cutoff) should be chosen based on the maximum ENMAX of all POTCARs used:

1. Check ENMAX values: `grep ENMAX POTCAR`
2. Set ENCUT to 1.3 × max(ENMAX) for good accuracy
3. For high-precision calculations, use 1.5 × max(ENMAX)

Example workflow:
- Run test calculations with ENCUT = 300, 400, 500, 600 eV
- Plot total energy vs ENCUT
- Choose ENCUT where energy converges to < 1 meV/atom

Note: Higher ENCUT increases computational cost significantly.""",
                "category": "vasp",
                "difficulty": "beginner",
                "tags": ["encut", "convergence", "accuracy"]
            },
            "kpoint_sampling": {
                "question": "What k-point grid should I use?",
                "answer": """K-point sampling depends on system size and type:

**Metals:**
- Dense k-point grid needed (e.g., 12×12×12 for bulk)
- Use Methfessel-Paxton or Gaussian smearing
- Check convergence carefully

**Semiconductors/Insulators:**
- Coarser grid acceptable (e.g., 6×6×6 for bulk)
- Use tetrahedron method (ISMEAR = -5) for accurate DOS

**Molecules/Clusters:**
- Gamma-point only may be sufficient
- Test with 1×1×1 vs 2×2×2

**General rule:**
- k-point density should be inversely proportional to real-space cell size
- Aim for similar k-density in all directions
- Always check convergence with respect to k-points""",
                "category": "dft_theory",
                "difficulty": "intermediate",
                "tags": ["kpoints", "sampling", "brillouin_zone"]
            },
            "magnetic_calculations": {
                "question": "How do I set up magnetic calculations in VASP?",
                "answer": """For magnetic calculations in VASP:

1. **Enable spin polarization:**
   ISPIN = 2

2. **Set initial magnetic moments:**
   MAGMOM = N×m1 N×m2 ...  (for each atom type)
   Example for Fe3O4: MAGMOM = 4*5.0 8*-5.0 (Fe↑ Fe↓ O)

3. **For ferromagnets:**
   - Set all spins parallel
   - MAGMOM = 5 5 5 5 (for Fe)

4. **For antiferromagnets:**
   - Set alternating spins
   - MAGMOM = 5 -5 5 -5

5. **Convergence tips:**
   - Start with larger SIGMA (~0.2) for initial convergence
   - Reduce SIGMA to 0.05 for final calculation
   - Use ALGO = Normal or All for better stability

6. **Output:**
   - Check OSZICAR for magnetic moments
   - OUTCAR contains detailed magnetization info""",
                "category": "vasp",
                "difficulty": "advanced",
                "tags": ["magnetism", "spin", "ispin", "magmom"]
            },
            "lammps_timestep": {
                "question": "What timestep should I use in LAMMPS MD?",
                "answer": """Timestep selection in LAMMPS:

**General guidelines:**
- Metals (EAM potentials): 0.001 ps = 1 fs
- Soft matter (beads): 0.005-0.01 ps = 5-10 fs
- Coarse-grained: up to 0.05 ps = 50 fs
- Rigid molecules: 0.002-0.004 ps

**Factors affecting timestep:**
1. Fastest motion in system (bond vibrations)
2. Temperature (higher T → smaller dt needed)
3. Potential stiffness

**Stability check:**
- Monitor total energy conservation
- Energy drift < 0.1% over simulation
- If unstable, reduce timestep by factor of 2

**Practical values:**
- C-C bonds: dt ≤ 0.5 fs
- Metals: dt = 1-2 fs
- Water (rigid): dt = 2 fs
- Coarse-grained: dt = 10-50 fs""",
                "category": "lammps",
                "difficulty": "beginner",
                "tags": ["timestep", "molecular_dynamics", "stability"]
            },
            "hubbard_u": {
                "question": "When and how should I use Hubbard U corrections?",
                "answer": """Hubbard U corrections (DFT+U):

**When to use:**
- Strongly correlated systems (d/f electrons)
- Transition metal oxides (FeO, NiO, etc.)
- Systems with localized electrons
- When standard DFT gives wrong metallic behavior for insulators

**Implementation in VASP:**
```
LDAU = .TRUE.
LDAUTYPE = 2          (Dudarev scheme, simpler)
LDAUL = -1 2 -1       (l quantum number per species)
LDAUU = 0 3.5 0       (U parameter per species)
LDAUJ = 0 0 0         (J parameter, often set to 0)
LMAXMIX = 4           (for d electrons)
LDAUPRINT = 2         (verbose output)
```

**Choosing U values:**
1. Literature values for similar systems
2. Linear response method
3. Self-consistent procedure (cDFT)

**Typical U values (eV):**
- Fe: 3.5-5.3
- Co: 3.3-5.0
- Ni: 3.5-6.5
- Cu: 2.5-6.0
- Mn: 3.5-5.0

**Warning:** U values are system-dependent and not transferable!""",
                "category": "advanced_dft",
                "difficulty": "advanced",
                "tags": ["hubbard_u", "dft+u", "correlation", "transition_metals"]
            },
            "vasp_mpi_parallel": {
                "question": "How do I parallelize VASP calculations efficiently?",
                "answer": """VASP Parallelization Strategies:

**Three levels of parallelism:**
1. **KPAR** - Parallelization over k-points
   - KPAR ≤ number of k-points
   - Best scaling: KPAR = nkpts
   - Set KPAR to divide nkpts evenly

2. **NCORE** - Parallelization over bands
   - NCORE × NPAR = total cores
   - Typical: NCORE = 4-16
   - Larger systems → larger NCORE

3. **NPAR** - Alternative band parallelization
   - NPAR = total_cores / NCORE
   - Choose so bands divide evenly

**Recommended settings:**

Small system (~10 atoms), many k-points:
- KPAR = 4-8, NCORE = 4

Large system (>100 atoms), few k-points:
- KPAR = 1, NCORE = 8-16

**Memory considerations:**
- Each k-point group needs full memory
- More KPAR → more memory per node
- Reduce KPAR if memory limited

**Performance tuning:**
```
# Test different combinations
KPAR = 4
NCORE = 8

# Alternative
NPAR = 4
NSIM = 4
```

Monitor with: `grep "Elapsed time" OUTCAR`""",
                "category": "vasp",
                "difficulty": "advanced",
                "tags": ["parallelization", "mpi", "kpar", "ncore", "performance"]
            }
        }
        
        # 教程数据
        self.tutorials = {
            "vasp_basics": {
                "title": "VASP Basics Tutorial",
                "steps": [
                    {
                        "title": "Introduction",
                        "content": "VASP (Vienna Ab initio Simulation Package) performs DFT calculations."
                    },
                    {
                        "title": "Input Files",
                        "content": "VASP requires four input files: INCAR, POSCAR, POTCAR, KPOINTS"
                    },
                    {
                        "title": "Running Calculations",
                        "content": "Execute: mpirun -np 16 vasp_std > vasp.out 2>&1"
                    },
                    {
                        "title": "Analyzing Output",
                        "content": "Check OSZICAR for convergence, OUTCAR for detailed results"
                    }
                ],
                "difficulty": "beginner"
            },
            "lammps_workflow": {
                "title": "LAMMPS MD Workflow",
                "steps": [
                    {"title": "Prepare Structure", "content": "Create data file with atom positions"},
                    {"title": "Choose Potential", "content": "Select appropriate interatomic potential"},
                    {"title": "Write Input Script", "content": "Define simulation parameters and commands"},
                    {"title": "Equilibration", "content": "Run NVT then NPT to equilibrate system"},
                    {"title": "Production Run", "content": "Run production simulation for statistics"},
                    {"title": "Analysis", "content": "Use post-processing tools to analyze trajectory"}
                ],
                "difficulty": "intermediate"
            }
        }
        
        # 故障排除数据
        self.troubleshooting = {
            "vasp": [
                {
                    "symptom": "BRMIX: very serious problems",
                    "cause": "Charge density mixing instability",
                    "solutions": [
                        "Add BMIX = 0.001 to INCAR",
                        "Use AMIN = 0.01",
                        "Try IALGO = 38 (Davidson)"
                    ],
                    "severity": "high"
                },
                {
                    "symptom": "ZHEGV failed",
                    "cause": "Numerical instability in diagonalization",
                    "solutions": [
                        "Use ALGO = Fast or All",
                        "Increase ENCUT by 20%",
                        "Set LREAL = Auto"
                    ],
                    "severity": "high"
                },
                {
                    "symptom": "EDDDAV: Call to ZHEGV failed",
                    "cause": "Diagonalization failure",
                    "solutions": [
                        "Increase NBANDS by 10-20%",
                        "Use ALGO = Damped",
                        "Check for very short bonds in structure"
                    ],
                    "severity": "critical"
                },
                {
                    "symptom": "VERY BAD NEWS! internal error in subroutine INVGRP",
                    "cause": "Symmetry determination problem",
                    "solutions": [
                        "Add ISYM = 0 to disable symmetry",
                        "Check POSCAR for near-degenerate lattice vectors"
                    ],
                    "severity": "medium"
                }
            ],
            "lammps": [
                {
                    "symptom": "Neighbor list overflow",
                    "cause": "Too many neighbors per atom",
                    "solutions": [
                        "Increase neighbor skin distance",
                        "Reduce cutoff in pair_style",
                        "Increase page size in neigh_modify"
                    ],
                    "severity": "medium"
                },
                {
                    "symptom": "Bond atoms # # # missing",
                    "cause": "Bonds stretched beyond cutoff",
                    "solutions": [
                        "Reduce timestep",
                        "Check initial structure for overlapping atoms",
                        "Add energy minimization before MD"
                    ],
                    "severity": "high"
                },
                {
                    "symptom": "Lost atoms",
                    "cause": "Atoms moved outside simulation box",
                    "solutions": [
                        "Check if boundary conditions are correct",
                        "Increase simulation box size",
                        "Add walls or restraints"
                    ],
                    "severity": "high"
                }
            ],
            "general": [
                {
                    "symptom": "Job killed / Out of memory",
                    "cause": "Insufficient memory allocated",
                    "solutions": [
                        "Request more memory in job script",
                        "Reduce parallelization (fewer cores)",
                        "Reduce system size or ENCUT"
                    ],
                    "severity": "critical"
                },
                {
                    "symptom": "Job times out",
                    "cause": "Calculation takes longer than walltime limit",
                    "solutions": [
                        "Request longer walltime",
                        "Improve convergence settings",
                        "Use WAVECAR/CHGCAR restart",
                        "Reduce system size"
                    ],
                    "severity": "medium"
                }
            ]
        }
    
    def search_faq(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """搜索FAQ"""
        results = []
        query_lower = query.lower()
        
        for key, qa in self.qa_pairs.items():
            score = 0
            
            # 问题匹配
            if query_lower in qa["question"].lower():
                score += 3
            
            # 答案匹配
            if query_lower in qa["answer"].lower():
                score += 1
            
            # 标签匹配
            for tag in qa.get("tags", []):
                if tag in query_lower:
                    score += 2
            
            if score > 0:
                results.append({"key": key, **qa, "score": score})
        
        # 按分数排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    def search_tutorial(self, topic: str) -> Optional[Dict[str, Any]]:
        """搜索教程"""
        for key, tutorial in self.tutorials.items():
            if topic.lower() in tutorial["title"].lower():
                return {"key": key, **tutorial}
        return None
    
    def search_troubleshooting(
        self,
        software: str,
        error_message: str
    ) -> List[Dict[str, Any]]:
        """搜索故障排除方案"""
        results = []
        error_lower = error_message.lower()
        
        for category in [software, "general"]:
            if category in self.troubleshooting:
                for entry in self.troubleshooting[category]:
                    if entry["symptom"].lower() in error_lower or \
                       error_lower in entry["symptom"].lower():
                        results.append(entry)
        
        return results


class IntentClassifier:
    """意图分类器"""
    
    def __init__(self):
        self.patterns = {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|greetings)\b',
                r'\b(good morning|good afternoon|good evening)\b'
            ],
            IntentType.CODE_REQUEST: [
                r'\b(generate|create|write).*(input|script|file)\b',
                r'\bhow.*(write|make|create).*(input|incar|script)\b',
                r'\b(input|script).*for\b'
            ],
            IntentType.ERROR_HELP: [
                r'\b(error|fail|crash|problem|issue).*\b',
                r'\b(not working|doesn\'t work|failed)\b',
                r'\b(help|fix|solve).*error\b'
            ],
            IntentType.PARAMETER_ADVICE: [
                r'\b(what|which|how).*(parameter|setting|value)\b',
                r'\b(encut|kpoint|timestep|cutoff)\b',
                r'\b(recommend|suggest).*(value|parameter)\b'
            ],
            IntentType.INTERPRETATION: [
                r'\b(interpret|explain|analyze).*result\b',
                r'\bwhat.*(mean|indicate|show)\b',
                r'\b(band gap|dos|energy|pressure).*(mean|interpret)\b'
            ],
            IntentType.WORKFLOW_HELP: [
                r'\b(workflow|pipeline|automation)\b',
                r'\b(how.*run|sequence|step).*\b'
            ],
            IntentType.TUTORIAL_REQUEST: [
                r'\b(tutorial|guide|learn|how to|documentation)\b',
                r'\b(basics|introduction|getting started)\b'
            ],
            IntentType.TOOL_RECOMMENDATION: [
                r'\b(recommend|suggest|which|best).*(tool|software|code)\b',
                r'\b(compare|vasp vs|qe vs|difference between)\b'
            ]
        }
    
    def classify(self, text: str) -> IntentType:
        """分类用户意图"""
        text_lower = text.lower()
        
        scores = {intent: 0 for intent in IntentType}
        
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    scores[intent] += 1
        
        # 返回得分最高的意图
        max_intent = max(scores, key=scores.get)
        
        if scores[max_intent] == 0:
            return IntentType.GENERAL_QUESTION
        
        return max_intent


class ChatAssistant:
    """
    交互式助手主类
    
    提供问答、实时指导和故障诊断功能。
    """
    
    def __init__(self, llm_provider: Optional[Any] = None):
        """
        初始化聊天助手
        
        Args:
            llm_provider: LLM提供商实例
        """
        self.llm = llm_provider
        self.knowledge_base = KnowledgeBase()
        self.intent_classifier = IntentClassifier()
        self.sessions: Dict[str, ConversationContext] = {}
        
        logger.info("ChatAssistant initialized")
    
    def get_or_create_session(
        self,
        session_id: str,
        user_level: ExpertiseLevel = ExpertiseLevel.INTERMEDIATE
    ) -> ConversationContext:
        """获取或创建会话"""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationContext(
                session_id=session_id,
                user_level=user_level
            )
        return self.sessions[session_id]
    
    def chat(
        self,
        message: str,
        session_id: str = "default",
        context: Optional[Dict[str, Any]] = None
    ) -> QAResponse:
        """
        处理用户消息并返回响应
        
        Args:
            message: 用户消息
            session_id: 会话ID
            context: 额外上下文
            
        Returns:
            问答响应
        """
        # 获取会话
        session = self.get_or_create_session(session_id)
        
        # 记录用户消息
        session.add_message("user", message, context)
        
        # 分类意图
        intent = self.intent_classifier.classify(message)
        
        # 根据意图处理
        if intent == IntentType.GREETING:
            response = self._handle_greeting(session)
        elif intent == IntentType.CODE_REQUEST:
            response = self._handle_code_request(message, session)
        elif intent == IntentType.ERROR_HELP:
            response = self._handle_error_help(message, session)
        elif intent == IntentType.PARAMETER_ADVICE:
            response = self._handle_parameter_advice(message, session)
        elif intent == IntentType.INTERPRETATION:
            response = self._handle_interpretation(message, session)
        elif intent == IntentType.TUTORIAL_REQUEST:
            response = self._handle_tutorial_request(message, session)
        else:
            response = self._handle_general_question(message, session)
        
        # 记录助手响应
        session.add_message("assistant", response.answer)
        
        return response
    
    def _handle_greeting(self, session: ConversationContext) -> QAResponse:
        """处理问候"""
        level_msg = {
            ExpertiseLevel.BEGINNER: " I'll guide you through the basics.",
            ExpertiseLevel.INTERMEDIATE: " Ready for some advanced topics?",
            ExpertiseLevel.ADVANCED: " Let's tackle some complex problems.",
            ExpertiseLevel.EXPERT: " Ready for in-depth discussions."
        }
        
        greeting = f"Hello! I'm your materials science assistant.{level_msg.get(session.user_level, '')}\n\n"
        greeting += "I can help you with:\n"
        greeting += "• 📝 Generating DFT/MD input files\n"
        greeting += "• 🔧 Troubleshooting errors\n"
        greeting += "• 📊 Interpreting results\n"
        greeting += "• 📚 Recommending parameters and best practices\n"
        greeting += "• 🎓 Tutorials and explanations\n\n"
        greeting += "What would you like to work on?"
        
        return QAResponse(
            answer=greeting,
            confidence=1.0,
            suggestions=[
                "Generate VASP input for band structure",
                "Help with SCF convergence issues",
                "Explain how to choose ENCUT"
            ]
        )
    
    def _handle_code_request(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理代码请求"""
        # 尝试提取软件类型
        software = self._detect_software(message)
        calc_type = self._detect_calculation_type(message)
        
        # 生成代码（这里简化处理，实际应调用CodeGenerator）
        code_example = f"""# {software.upper()} {calc_type.upper()} Input
# Based on your request: {message[:50]}...

# Key parameters to set:
# - Check convergence with respect to key parameters
# - Monitor output files for errors
# - Adjust settings based on your specific system

# Would you like me to generate a complete input file?"""
        
        return QAResponse(
            answer=f"I can help you generate {software} input for {calc_type} calculation. "
                   f"Here's a template:\n\n```\n{code_example}\n```",
            confidence=0.8,
            code_examples=[code_example],
            follow_up_questions=[
                f"What's your target material?",
                f"What property are you calculating?",
                f"Any specific accuracy requirements?"
            ]
        )
    
    def _handle_error_help(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理错误求助"""
        # 检测软件类型
        software = self._detect_software(message)
        
        # 搜索知识库
        solutions = self.knowledge_base.search_troubleshooting(software, message)
        
        if solutions:
            answer = f"I found potential solutions for your {software} error:\n\n"
            for i, sol in enumerate(solutions[:3], 1):
                answer += f"**{i}. {sol['symptom']}** (Severity: {sol['severity']})\n"
                answer += f"   Cause: {sol['cause']}\n"
                answer += f"   Solutions:\n"
                for s in sol['solutions']:
                    answer += f"   - {s}\n"
                answer += "\n"
        else:
            answer = f"I don't have a specific solution for this {software} error in my database.\n\n"
            answer += "Here are general debugging steps:\n"
            answer += "1. Check the error message carefully for clues\n"
            answer += "2. Verify input file syntax\n"
            answer += "3. Check that all required files are present\n"
            answer += "4. Try simplifying the calculation\n"
            answer += "5. Consult the software manual or forums\n"
        
        return QAResponse(
            answer=answer,
            confidence=0.9 if solutions else 0.5,
            suggestions=[
                "Show me how to check input syntax",
                "What are common mistakes in {}".format(software),
                "How do I get more detailed error output?"
            ]
        )
    
    def _handle_parameter_advice(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理参数建议请求"""
        # 搜索FAQ
        results = self.knowledge_base.search_faq(message, max_results=2)
        
        if results:
            answer = "Based on best practices:\n\n"
            for result in results:
                answer += f"**{result['question']}**\n\n"
                # 根据用户级别调整详细程度
                if session.user_level == ExpertiseLevel.BEGINNER:
                    # 只取前几行
                    answer += '\n'.join(result['answer'].split('\n')[:10]) + "\n...\n"
                else:
                    answer += result['answer']
                answer += "\n\n"
        else:
            answer = "I need more information to recommend parameters.\n\n"
            answer += "Please tell me:\n"
            answer += "- What software are you using? (VASP, LAMMPS, QE, etc.)\n"
            answer += "- What type of calculation?\n"
            answer += "- What material/system are you studying?\n"
            answer += "- What property are you interested in?\n"
        
        return QAResponse(
            answer=answer,
            confidence=0.85 if results else 0.4,
            related_topics=["convergence testing", "accuracy vs cost trade-off"]
        )
    
    def _handle_interpretation(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理结果解释请求"""
        answer = "To help interpret your results, I'll need some information:\n\n"
        answer += "1. What software produced the results?\n"
        answer += "2. What type of calculation did you run?\n"
        answer += "3. What specific values or output are you looking at?\n\n"
        answer += "Common interpretation topics:\n"
        answer += "- **Total energy**: Check convergence and magnitude\n"
        answer += "- **Band structure**: Look for band gap, dispersion\n"
        answer += "- **Forces**: Should be small for relaxed structures\n"
        answer += "- **Pressure**: Near zero for optimized cells\n"
        
        return QAResponse(
            answer=answer,
            confidence=0.6,
            suggestions=[
                "How do I know if my calculation converged?",
                "What is a good total energy value?",
                "How do I calculate the band gap from output?"
            ]
        )
    
    def _handle_tutorial_request(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理教程请求"""
        # 检测主题
        topic = None
        if "vasp" in message.lower():
            topic = "vasp_basics"
        elif "lammps" in message.lower():
            topic = "lammps_workflow"
        
        if topic and topic in self.knowledge_base.tutorials:
            tutorial = self.knowledge_base.tutorials[topic]
            answer = f"# {tutorial['title']}\n\n"
            for i, step in enumerate(tutorial['steps'], 1):
                answer += f"**Step {i}: {step['title']}**\n"
                answer += f"{step['content']}\n\n"
        else:
            answer = "Available tutorials:\n\n"
            answer += "1. **VASP Basics** - Introduction to VASP calculations\n"
            answer += "2. **LAMMPS Workflow** - Setting up MD simulations\n\n"
            answer += "Which one would you like to explore?"
        
        return QAResponse(
            answer=answer,
            confidence=0.9,
            follow_up_questions=[
                "Show me the next step",
                "Give me a practical example",
                "What are common mistakes to avoid?"
            ]
        )
    
    def _handle_general_question(
        self,
        message: str,
        session: ConversationContext
    ) -> QAResponse:
        """处理一般问题"""
        # 搜索FAQ
        results = self.knowledge_base.search_faq(message, max_results=3)
        
        if results:
            answer = "I found some relevant information:\n\n"
            for result in results:
                answer += f"**Q: {result['question']}**\n\n"
                answer += f"A: {result['answer'][:500]}...\n\n"
        else:
            answer = "I'm not sure I understand your question completely.\n\n"
            answer += "I can help you with:\n"
            answer += "• Generating input files for DFT/MD calculations\n"
            answer += "• Debugging errors and convergence issues\n"
            answer += "• Explaining methods and parameters\n"
            answer += "• Interpreting calculation results\n\n"
            answer += "Could you rephrase your question or provide more context?"
        
        return QAResponse(
            answer=answer,
            confidence=0.5 if not results else 0.7,
            suggestions=[
                "How do I start a VASP calculation?",
                "What software should I use for molecular dynamics?",
                "How do I choose between DFT functionals?"
            ]
        )
    
    def diagnose(
        self,
        error_log: str,
        software: str,
        input_files: Optional[Dict[str, str]] = None
    ) -> DiagnosisReport:
        """
        诊断计算问题
        
        Args:
            error_log: 错误日志
            software: 使用的软件
            input_files: 输入文件内容
            
        Returns:
            诊断报告
        """
        # 搜索匹配的故障
        matches = self.knowledge_base.search_troubleshooting(software, error_log)
        
        if matches:
            match = matches[0]
            
            solutions = [
                {
                    "solution": sol,
                    "difficulty": "easy" if i == 0 else "medium",
                    "expected_success_rate": 0.8 if i == 0 else 0.5
                }
                for i, sol in enumerate(match['solutions'])
            ]
            
            return DiagnosisReport(
                problem_type=match['cause'],
                severity=match['severity'],
                description=match['symptom'],
                possible_causes=[match['cause']],
                recommended_solutions=solutions,
                prevention_tips=[
                    "Always test with small systems first",
                    "Check input file syntax before running",
                    "Monitor convergence during calculation"
                ]
            )
        
        # 通用诊断
        return DiagnosisReport(
            problem_type="unknown",
            severity="medium",
            description="Unable to identify specific error pattern",
            possible_causes=[
                "Unknown error type",
                "May require manual investigation"
            ],
            recommended_solutions=[
                {
                    "solution": "Check software documentation",
                    "difficulty": "medium",
                    "expected_success_rate": 0.5
                }
            ],
            prevention_tips=[
                "Keep backups of working configurations",
                "Document changes to input files"
            ]
        )
    
    def get_guidance(
        self,
        current_step: str,
        calculation_type: str,
        session_id: str = "default"
    ) -> str:
        """
        获取实时计算指导
        
        Args:
            current_step: 当前步骤
            calculation_type: 计算类型
            session_id: 会话ID
            
        Returns:
            指导信息
        """
        guidance_map = {
            "scf_initialization": {
                "vasp": "Setting up SCF calculation:\n"
                        "1. Ensure ISPIN is set correctly for magnetic systems\n"
                        "2. Start with ICHARG=2 for initial run\n"
                        "3. Check that ENCUT covers all elements\n"
                        "4. Monitor convergence in OSZICAR",
                "qe": "Setting up PWscf calculation:\n"
                      "1. Verify pseudopotential files exist\n"
                      "2. Check ecutwfc and ecutrho are appropriate\n"
                      "3. Ensure k-point sampling is adequate\n"
                      "4. Monitor total energy convergence"
            },
            "relaxation": {
                "vasp": "Geometry optimization:\n"
                        "1. Use IBRION=2 (CG) or 1 (RMM-DIIS)\n"
                        "2. Set EDIFFG negative for force convergence\n"
                        "3. ISIF=2 for ions only, ISIF=3 for cell+ions\n"
                        "4. Watch for negative pressures indicating expansion",
                "qe": "Cell/ion relaxation:\n"
                      "1. Use calculation='relax' or 'vc-relax'\n"
                      "2. Set forc_conv_thr for force convergence\n"
                      "3. Monitor cell parameters in output\n"
                      "4. Check final forces are below threshold"
            },
            "md_equilibration": {
                "lammps": "NVT equilibration:\n"
                          "1. Start with energy minimization\n"
                          "2. Use NVT fix with appropriate Tdamp\n"
                          "3. Monitor temperature fluctuations\n"
                          "4. Check energy drift is minimal\n"
                          "5. Equilibration typically 10-100 ps"
            }
        }
        
        calc_guidance = guidance_map.get(current_step, {})
        return calc_guidance.get(calculation_type, 
                                 f"No specific guidance for {current_step} in {calculation_type}")
    
    def _detect_software(self, text: str) -> str:
        """检测提到的软件"""
        text_lower = text.lower()
        if "vasp" in text_lower:
            return "vasp"
        elif "lammps" in text_lower:
            return "lammps"
        elif "quantum" in text_lower or "espresso" in text_lower or "qe" in text_lower:
            return "qe"
        elif "cp2k" in text_lower:
            return "cp2k"
        elif "gaussian" in text_lower:
            return "gaussian"
        else:
            return "vasp"  # 默认
    
    def _detect_calculation_type(self, text: str) -> str:
        """检测计算类型"""
        text_lower = text.lower()
        if any(x in text_lower for x in ["scf", "self-consistent", "static"]):
            return "scf"
        elif any(x in text_lower for x in ["relax", "optimization", "optimize"]):
            return "relax"
        elif any(x in text_lower for x in ["band", "bands", "bandstructure"]):
            return "bands"
        elif any(x in text_lower for x in ["md", "molecular dynamics", "nvt", "npt"]):
            return "md"
        elif any(x in text_lower for x in ["dos", "density of states"]):
            return "dos"
        else:
            return "scf"
    
    def export_conversation(self, session_id: str, filepath: str):
        """导出会话记录"""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.sessions[session_id]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                "session_id": session.session_id,
                "user_level": session.user_level.value,
                "history": session.history
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversation exported to {filepath}")


# 便捷函数
def quick_chat(message: str, session_id: str = "default") -> str:
    """快速聊天"""
    assistant = ChatAssistant()
    response = assistant.chat(message, session_id)
    return response.answer


def quick_diagnose(error_log: str, software: str = "vasp") -> DiagnosisReport:
    """快速诊断"""
    assistant = ChatAssistant()
    return assistant.diagnose(error_log, software)


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("ChatAssistant Test")
    print("=" * 60)
    
    assistant = ChatAssistant()
    
    # 测试1: 问候
    print("\n1. Testing greeting...")
    response = assistant.chat("Hello!", session_id="test1")
    print(f"Response preview: {response.answer[:100]}...")
    
    # 测试2: 代码请求
    print("\n2. Testing code request...")
    response = assistant.chat(
        "Generate VASP input for Si band structure",
        session_id="test2"
    )
    print(f"Response contains code: {len(response.code_examples) > 0}")
    
    # 测试3: 错误帮助
    print("\n3. Testing error help...")
    response = assistant.chat(
        "I'm getting 'ZHEGV failed' error in VASP",
        session_id="test3"
    )
    print(f"Response length: {len(response.answer)}")
    
    # 测试4: 参数建议
    print("\n4. Testing parameter advice...")
    response = assistant.chat(
        "What ENCUT should I use?",
        session_id="test4"
    )
    print(f"Confidence: {response.confidence}")
    
    # 测试5: 诊断
    print("\n5. Testing diagnose...")
    report = assistant.diagnose(
        "Error EDDDAV: Call to ZHEGV failed",
        software="vasp"
    )
    print(f"Problem type: {report.problem_type}")
    print(f"Severity: {report.severity}")
    
    # 测试6: 实时指导
    print("\n6. Testing guidance...")
    guidance = assistant.get_guidance("scf_initialization", "vasp")
    print(f"Guidance length: {len(guidance)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
