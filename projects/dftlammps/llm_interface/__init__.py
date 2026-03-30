"""
LLM接口模块初始化文件
====================

此模块提供材料科学相关的LLM集成功能，包括：
- 材料科学GPT (materials_gpt.py)
- 智能代码生成 (code_generator.py)
- 交互式助手 (chat_assistant.py)
- 应用案例 (application_examples.py)

Author: DFT-LAMMPS Team
Date: 2025
"""

from .materials_gpt import (
    MaterialsGPT,
    LiteratureMiner,
    TaskType,
    MaterialEntity,
    ComputationParameters,
    ExperimentDesign,
    OpenAIProvider,
    LocalLLMProvider,
    create_default_gpt,
    quick_extract,
    quick_design_experiment,
    quick_interpret_results
)

from .code_generator import (
    CodeGenerator,
    CodeLanguage,
    CalculationType,
    CodeTemplate,
    GeneratedCode,
    ErrorFix,
    PromptEngineer,
    quick_generate,
    quick_fix
)

from .chat_assistant import (
    ChatAssistant,
    IntentType,
    ExpertiseLevel,
    ConversationContext,
    QAResponse,
    DiagnosisReport,
    KnowledgeBase,
    IntentClassifier,
    quick_chat,
    quick_diagnose
)

from .application_examples import (
    LiteratureMiningExample,
    NLWorkflowDesignExample,
    SmartLabNotebookExample,
    IntegrationExample
)

__all__ = [
    # Materials GPT
    'MaterialsGPT',
    'LiteratureMiner',
    'TaskType',
    'MaterialEntity',
    'ComputationParameters',
    'ExperimentDesign',
    'OpenAIProvider',
    'LocalLLMProvider',
    'create_default_gpt',
    'quick_extract',
    'quick_design_experiment',
    'quick_interpret_results',
    
    # Code Generator
    'CodeGenerator',
    'CodeLanguage',
    'CalculationType',
    'CodeTemplate',
    'GeneratedCode',
    'ErrorFix',
    'PromptEngineer',
    'quick_generate',
    'quick_fix',
    
    # Chat Assistant
    'ChatAssistant',
    'IntentType',
    'ExpertiseLevel',
    'ConversationContext',
    'QAResponse',
    'DiagnosisReport',
    'KnowledgeBase',
    'IntentClassifier',
    'quick_chat',
    'quick_diagnose',
    
    # Application Examples
    'LiteratureMiningExample',
    'NLWorkflowDesignExample',
    'SmartLabNotebookExample',
    'IntegrationExample'
]

__version__ = "1.0.0"
