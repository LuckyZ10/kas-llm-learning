# LLM Integration for DFT-LAMMPS Scientific Platform

A comprehensive framework for integrating Large Language Models (LLMs) into computational materials science research workflows. This module provides intelligent hypothesis explanation, scientific reasoning enhancement, paper writing assistance, and interactive scientific dialogue capabilities.

## Overview

The LLM Integration module bridges the gap between computational chemistry data and scientific understanding by leveraging state-of-the-art language models to:

- **Interpret numerical results** in natural language with domain expertise
- **Generate and test hypotheses** from computational data
- **Perform causal inference** and counterfactual analysis
- **Assist in academic writing** from structure to publication
- **Provide interactive scientific Q&A** with context awareness

## Features

### 🧠 Unified LLM Interface
- Multi-provider support: OpenAI (GPT-4), Anthropic (Claude), DeepSeek, Local Models (vLLM, llama.cpp)
- Async operations with streaming support
- Automatic retry with exponential backoff
- Prompt engineering utilities (Few-shot, Chain-of-Thought, ReAct)

### 🔬 Hypothesis Explainer
- Transform DFT/MD results into scientific narratives
- Multiple explanation styles (technical, pedagogical, executive)
- Hypothesis generation from numerical data
- Domain-specific interpretation (catalysis, batteries, semiconductors)

### 🎯 Scientific Reasoning
- Causal inference from computational data
- Counterfactual analysis (what-if scenarios)
- Evidence-based reasoning chains
- Abductive and analogical reasoning

### 📝 Paper Assistant
- Automated paper structure planning
- Section-by-section writing assistance
- Citation management and bibliography generation
- Language polishing and style adaptation
- Reviewer response generation

### 💬 Scientific Chat
- Multi-turn scientific conversations
- Domain-aware context tracking
- Code generation for simulations
- Literature recommendations

## Installation

### Prerequisites

```bash
# Python 3.8+
pip install asyncio dataclasses typing

# LLM provider packages (choose based on your needs)
pip install openai          # For OpenAI GPT-4/3.5
pip install anthropic       # For Claude
# DeepSeek uses OpenAI-compatible API
# Local models need vLLM or llama-cpp-python
```

### Setup

Clone or navigate to the dftlammps directory and ensure the `llm_integration` module is in your Python path.

## Configuration

### API Keys

Set environment variables for your chosen LLM providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4"  # or gpt-3.5-turbo

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export ANTHROPIC_MODEL="claude-3-sonnet-20240229"

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_MODEL="deepseek-chat"

# Local vLLM (optional)
export LOCAL_VLLM_BASE_URL="http://localhost:8000/v1"
```

### Basic Configuration

```python
from dftlammps.llm_integration import (
    UnifiedLLMInterface,
    LLMConfig,
    LLMProvider,
)

# Method 1: From environment variables
config = LLMConfig.from_env(LLMProvider.OPENAI)
llm = UnifiedLLMInterface(config)

# Method 2: Manual configuration
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_retries=3,
)
llm = UnifiedLLMInterface(config)
```

## Quick Start

### 1. Basic Completion

```python
import asyncio
from dftlammps.llm_integration import UnifiedLLMInterface, LLMConfig, LLMProvider

async def main():
    config = LLMConfig.from_env(LLMProvider.OPENAI)
    llm = UnifiedLLMInterface(config)
    
    response = await llm.complete(
        prompt="Explain the band gap in semiconductors",
        temperature=0.7,
    )
    print(response.content)

asyncio.run(main())
```

### 2. Streaming Response

```python
async for chunk in llm.stream(prompt="Explain DFT in detail"):
    print(chunk.content, end="", flush=True)
```

### 3. Hypothesis Explanation

```python
from dftlammps.llm_integration import (
    HypothesisExplainer,
    NumericalContext,
    NumericalValue,
    ExplanationStyle,
)

async def analyze_results():
    explainer = HypothesisExplainer(llm)
    
    context = NumericalContext(
        system_description="TiO2 anatase with oxygen vacancy",
        calculation_method="DFT-PBE+U",
        numerical_results={
            "band_gap": NumericalValue(3.21, "eV", uncertainty=0.05),
            "formation_energy": NumericalValue(4.85, "eV"),
        }
    )
    
    explanation = await explainer.explain_results(
        context=context,
        style=ExplanationStyle.TECHNICAL,
        domain="semiconductor"
    )
    
    print(explanation.explanation)
    print(f"Confidence: {explanation.confidence_score:.1%}")

asyncio.run(analyze_results())
```

### 4. Scientific Chat

```python
from dftlammps.llm_integration import ScientificChatInterface

async def chat_demo():
    chat = ScientificChatInterface(llm)
    session = chat.create_session(domain="catalysis")
    
    response = await chat.chat(
        message="What makes a good CO2 reduction catalyst?",
        session=session
    )
    print(response)

asyncio.run(chat_demo())
```

## Module Documentation

### LLM Interface (`llm_interface.py`)

#### Classes

**`UnifiedLLMInterface`** - Main interface for LLM interactions

```python
interface = UnifiedLLMInterface(config)

# Basic completion
response = await interface.complete("Your prompt here")

# Streaming
async for chunk in interface.stream("Your prompt"):
    print(chunk.content)

# Few-shot prompting
response = await interface.generate_with_few_shot(
    task_description="Classify material type",
    examples=[("Si", "semiconductor"), ("Fe", "metal")],
    query="GaAs"
)

# Chain-of-thought
response = await interface.generate_with_chain_of_thought(
    question="Why does band gap increase with quantum confinement?",
    reasoning_steps=["Consider particle in a box", "Apply to electrons and holes"]
)

# Structured output
result = await interface.generate_structured(
    prompt="Analyze this material",
    output_schema={
        "crystal_structure": "string",
        "band_gap": "number",
        "applications": ["string"]
    }
)
```

**`LLMConfig`** - Configuration for LLM connections

```python
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7,
    max_tokens=2000,
    max_retries=3,
    retry_delay=1.0,
    system_prompt="You are a materials science expert."
)
```

**`Conversation`** - Manage conversation history

```python
conv = Conversation(system_prompt="You are a helpful assistant.")
conv.add_user_message("Explain DFT")
conv.add_assistant_message("DFT is...")

# Get conversation as dicts
messages = conv.to_dicts()
```

### Hypothesis Explainer (`hypothesis_explainer.py`)

#### Classes

**`HypothesisExplainer`** - Explain numerical results

```python
explainer = HypothesisExplainer(llm)

# Basic explanation
explanation = await explainer.explain_results(
    context=numerical_context,
    style=ExplanationStyle.TECHNICAL,
    domain="catalysis"
)

# Generate hypotheses from results
hypotheses = await explainer.generate_hypothesis_from_results(
    context=context,
    domain="battery"
)

# Explain with hypothesis testing
explanation = await explainer.explain_results(
    context=context,
    hypothesis=hypothesis,
    style=ExplanationStyle.HYPOTHESIS_DRIVEN
)

# Compare multiple calculations
comparison = await explainer.compare_calculations(
    contexts=[dft_context, md_context],
    comparison_focus="methodology"
)
```

**`NumericalContext`** - Container for numerical results

```python
context = NumericalContext(
    system_description="CO on Pt(111)",
    calculation_method="DFT-PBE with D3 dispersion",
    numerical_results={
        "binding_energy": NumericalValue(-1.85, "eV", uncertainty=0.05),
        "distance": NumericalValue(1.92, "Å"),
    },
    comparison_values={
        "literature": NumericalValue(-1.92, "eV", source="Experiment")
    },
    validation_metrics={
        "lattice_constant_error": 0.8
    }
)
```

### Scientific Reasoning (`scientific_reasoning.py`)

#### Classes

**`ScientificReasoningEngine`** - Advanced reasoning capabilities

```python
reasoning = ScientificReasoningEngine(llm)

# Causal inference
relationships = await reasoning.causal_inference.infer_relationships(
    variables=["temperature", "pressure", "conductivity"],
    data={"temperature": [300, 400, 500], "conductivity": [0.1, 0.5, 1.2]}
)

# Build reasoning chain
chain = await reasoning.construct_reasoning_chain(
    goal="Determine if material is a good thermoelectric",
    evidence=[evidence1, evidence2],
    constraints=["ZT > 1 required"]
)

# Test hypothesis
test_result = await reasoning.test_hypothesis(
    hypothesis="Doping increases conductivity",
    evidence=evidence_list
)

# Counterfactual analysis
scenarios = await reasoning.counterfactual.generate_scenarios(
    base_conditions={"temperature": 300, "doping": 0.01},
    intervention_space={"doping": [0.01, 0.05, 0.1]}
)
```

### Paper Assistant (`paper_assistant.py`)

#### Classes

**`PaperAssistant`** - Complete paper writing support

```python
assistant = PaperAssistant(llm, writing_style=WritingStyle.FORMAL_ACADEMIC)

# Plan structure
structure = await assistant.plan_structure(
    title="Machine Learning for Catalysis",
    research_type="computational_study",
    key_findings=["GNN achieves 0.15 eV MAE"],
    target_journal="Nature Catalysis"
)

# Write section
draft = await assistant.write_section(
    section=PaperSection.ABSTRACT,
    key_points=["CO2 reduction importance", "ML approach", "Key results"],
    target_word_count=250
)

# Polish text
polished = await assistant.polish_text(
    text=draft.content,
    polish_type="conciseness"
)

# Review paper
review = await assistant.review_paper(sections=paper_sections)

# Manage citations
citation = Citation(id="ref1", title="...", authors=["Smith, J."], year=2023)
assistant.citation_manager.add_citation(citation)
marker = assistant.citation_manager.cite("ref1")
bibliography = assistant.citation_manager.generate_bibliography()
```

### Chat Interface (`chat_interface.py`)

#### Classes

**`ScientificChatInterface`** - Interactive scientific dialogue

```python
chat = ScientificChatInterface(llm)

# Create session
session = chat.create_session(domain="battery_materials")

# Basic chat
response = await chat.chat(
    message="What causes capacity fade?",
    session=session
)

# Mode-specific chat
response = await chat.chat(
    message="Generate ideas for new cathodes",
    session=session,
    mode=ChatMode.BRAINSTORM
)

# Explain concept
explanation = await chat.explain_with_context(
    concept="phonon dispersion",
    session=session,
    detail_level="intermediate"
)

# Get literature suggestions
papers = await chat.get_literature_suggestions(
    topic="solid electrolytes",
    session=session
)

# Export session
markdown = chat.export_session(session, format="markdown")
json_export = chat.export_session(session, format="json")
```

## Examples

### Example 1: Hypothesis Generation from DFT Results

```python
import asyncio
from dftlammps.llm_integration import (
    HypothesisExplainer, NumericalContext, NumericalValue
)

async def main():
    explainer = HypothesisExplainer(llm)
    
    # Your DFT calculation results
    context = NumericalContext(
        system_description="LiFePO4 with Mg doping",
        calculation_method="DFT-HSE06",
        numerical_results={
            "voltage": NumericalValue(3.45, "V"),
            "diffusion_barrier": NumericalValue(0.27, "eV"),
        }
    )
    
    # Generate hypotheses
    hypotheses = await explainer.generate_hypothesis_from_results(
        context, domain="battery"
    )
    
    for h in hypotheses:
        print(f"Hypothesis: {h.statement}")
        print(f"Expected: {h.expected_outcome}\n")

asyncio.run(main())
```

### Example 2: Paper Section Writing

```python
from dftlammps.llm_integration import PaperAssistant, PaperSection

async def write_paper():
    assistant = PaperAssistant(llm)
    
    # Write abstract
    draft = await assistant.write_section(
        section=PaperSection.ABSTRACT,
        key_points=[
            "CO2 reduction is critical for carbon neutrality",
            "ML can accelerate catalyst discovery",
            "Our model achieves 0.15 eV MAE",
            "Identified 3 promising candidates"
        ],
        target_word_count=200
    )
    
    print(draft.content)

asyncio.run(write_paper())
```

### Example 3: Interactive Research Assistant

```python
from dftlammps.llm_integration import ScientificChatInterface, ChatMode

async def research_assistant():
    chat = ScientificChatInterface(llm)
    session = chat.create_session(domain="catalysis")
    
    # Ask questions
    questions = [
        "What makes a good CO2 reduction catalyst?",
        "How do scaling relations affect screening?",
        "Suggest recent papers on this topic",
    ]
    
    for q in questions:
        response = await chat.chat(q, session=session)
        print(f"Q: {q}\nA: {response}\n")

asyncio.run(research_assistant())
```

### Example 4: Causal Analysis of Simulation Data

```python
from dftlammps.llm_integration.scientific_reasoning import (
    ScientificReasoningEngine, Evidence, EvidenceType, ConfidenceLevel
)

async def causal_analysis():
    reasoning = ScientificReasoningEngine(llm)
    
    # Add evidence from calculations
    evidence = [
        Evidence(
            source="DFT Calculation",
            type=EvidenceType.COMPUTATIONAL,
            description="Binding energy correlates with d-band center (R²=0.85)",
            confidence=ConfidenceLevel.HIGH
        ),
        Evidence(
            source="Experiment",
            type=EvidenceType.EXPERIMENTAL,
            description="Measured activity increases with compressive strain",
            confidence=ConfidenceLevel.HIGH
        ),
    ]
    
    # Build reasoning chain
    chain = await reasoning.construct_reasoning_chain(
        goal="Determine optimal strain for CO2RR catalyst",
        evidence=evidence
    )
    
    print(chain.to_markdown())

asyncio.run(causal_analysis())
```

## Advanced Usage

### Custom Prompt Engineering

```python
from dftlammps.llm_integration.llm_interface import PromptEngineer

# Few-shot prompting
prompt = PromptEngineer.create_few_shot_prompt(
    task_description="Predict crystal structure from composition",
    examples=[
        ("SiO2", "tetrahedral network"),
        ("NaCl", "rock salt"),
    ],
    query="ZnS",
    include_explanation=True
)

# Chain-of-thought
prompt = PromptEngineer.create_chain_of_thought_prompt(
    question="Why does band bending occur at semiconductor interfaces?",
    reasoning_steps=[
        "Consider Fermi level alignment",
        "Account for charge redistribution",
        "Calculate potential variation"
    ]
)
```

### Batch Processing

```python
from dftlammps.llm_integration import AsyncLLMInterface

async_interface = AsyncLLMInterface(config)

# Process multiple prompts concurrently
prompts = ["Explain DFT", "Explain MD", "Explain Monte Carlo"]
responses = await async_interface.batch_complete(prompts, max_concurrent=3)

# Ensemble generation for self-consistency
responses = await async_interface.ensemble_generate(
    prompt="Predict the band gap of GaN",
    temperatures=[0.3, 0.5, 0.7]
)
```

### Local Model Support

```python
from dftlammps.llm_integration import (
    create_local_vllm_interface,
    LLMConfig,
    LLMProvider
)

# Connect to local vLLM server
llm = create_local_vllm_interface(
    model="meta-llama/Llama-2-70b",
    base_url="http://localhost:8000/v1"
)

# Or use generic interface
config = LLMConfig(
    provider=LLMProvider.LOCAL_VLLM,
    model="local-model",
    base_url="http://localhost:8000/v1"
)
llm = UnifiedLLMInterface(config)
```

## Error Handling

All methods include automatic retry logic with exponential backoff:

```python
config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    max_retries=3,
    retry_delay=1.0,
    retry_exponential=True,  # Delays: 1s, 2s, 4s
)
```

Custom error handling:

```python
from dftlammps.llm_integration import LLMConfig, LLMProvider

try:
    config = LLMConfig.from_env(LLMProvider.OPENAI)
except ValueError as e:
    print(f"API key not found: {e}")
    # Fall back to alternative provider
    config = LLMConfig.from_env(LLMProvider.DEEPSEEK)
```

## Performance Considerations

- **Async operations**: Use `async/await` for concurrent requests
- **Streaming**: Use streaming for long responses to improve perceived latency
- **Batching**: Use `AsyncLLMInterface.batch_complete()` for multiple prompts
- **Caching**: Consider implementing response caching for repeated queries
- **Rate limiting**: Built-in retry logic handles rate limits automatically

## Testing

Run the example demos:

```bash
# Set your API key
export OPENAI_API_KEY="your-key"

# Run demos
cd dftlammps/llm_examples
python llm_hypothesis_demo.py
python paper_writing_assistant.py
python scientific_qa_bot.py
```

## API Reference

See the docstrings in each module for detailed API documentation:

- `llm_interface.py`: Core LLM interface classes
- `hypothesis_explainer.py`: Result explanation and hypothesis generation
- `scientific_reasoning.py`: Advanced reasoning capabilities
- `paper_assistant.py`: Academic writing support
- `chat_interface.py`: Interactive dialogue system

## Contributing

When contributing to this module:

1. Follow type hints and docstring conventions
2. Add tests for new functionality
3. Update this README with new features
4. Ensure async functions are properly awaited

## License

This module is part of the DFT-LAMMPS project. See the main project license for details.

## Citation

If you use this module in your research, please cite:

```bibtex
@software{dftlammps_llm,
  title={DFT-LAMMPS LLM Integration Module},
  year={2024},
  note={Scientific reasoning and hypothesis explanation with LLMs}
}
```

## Support

For issues, questions, or feature requests related to the LLM integration module,
please refer to the main DFT-LAMMPS project documentation or open an issue.
