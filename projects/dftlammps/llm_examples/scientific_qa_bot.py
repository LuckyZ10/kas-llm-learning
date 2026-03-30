"""
Scientific Q&A Bot - Materials Science Knowledge Assistant

This example demonstrates a conversational AI assistant for materials science
research, capable of answering questions, explaining concepts, and providing
context-aware responses.

Features:
- Multi-turn scientific conversations
- Domain-specific knowledge retrieval
- Code generation for simulations
- Literature recommendations
- Interactive hypothesis testing
"""

import asyncio
from typing import Optional

try:
    from dftlammps.llm_integration import (
        UnifiedLLMInterface,
        LLMConfig,
        LLMProvider,
    )
    from dftlammps.llm_integration.chat_interface import (
        ScientificChatInterface,
        ChatMode,
        ScientificContext,
        ResponseFormat,
    )
except ImportError:
    # Allow standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from llm_integration import (
        UnifiedLLMInterface,
        LLMConfig,
        LLMProvider,
    )
    from llm_integration.chat_interface import (
        ScientificChatInterface,
        ChatMode,
        ScientificContext,
        ResponseFormat,
    )


class MaterialsScienceQABot:
    """Interactive Q&A bot for materials science."""
    
    def __init__(self):
        """Initialize the Q&A bot."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
            self.llm = UnifiedLLMInterface(config)
            self.chat = ScientificChatInterface(self.llm)
            self.session = None
        except ValueError:
            self.llm = None
            self.chat = None
            self.session = None
    
    async def start_session(self, domain: str = "materials"):
        """Start a new chat session."""
        if self.chat:
            context = ScientificContext(
                domain=domain,
                key_concepts=["density functional theory", "molecular dynamics", 
                             "electronic structure", "thermodynamics"],
            )
            self.session = self.chat.create_session(
                domain=domain,
                initial_context=context
            )
            print(f"✓ Started new session: {self.session.session_id}")
        else:
            print("Note: LLM not configured. Running in demo mode.")
            print(f"Would create session for domain: {domain}")
    
    async def answer_question(self, question: str, mode: Optional[ChatMode] = None):
        """Answer a scientific question."""
        print(f"\n👤 User: {question}")
        print("\n🤖 Assistant: ", end="", flush=True)
        
        if self.chat and self.session:
            response = await self.chat.chat(
                message=question,
                session=self.session,
                mode=mode,
                stream=False
            )
            print(response)
            return response
        else:
            # Demo mode - show expected response structure
            demo_responses = {
                "band gap": "The band gap is the energy difference between the valence band maximum and conduction band minimum...",
                "dft": "Density Functional Theory (DFT) is a computational quantum mechanical modeling method...",
                "catalyst": "Catalysts accelerate chemical reactions by providing alternative reaction pathways...",
            }
            
            for key, resp in demo_responses.items():
                if key in question.lower():
                    print(f"{resp[:100]}...")
                    return resp
            
            print("(Would provide detailed scientific answer based on question)")
            return ""
    
    async def explain_concept(self, concept: str, detail_level: str = "intermediate"):
        """Explain a scientific concept."""
        print(f"\n📚 Explaining: {concept} ({detail_level} level)")
        
        if self.chat and self.session:
            explanation = await self.chat.explain_with_context(
                concept=concept,
                session=self.session,
                detail_level=detail_level
            )
            print(f"\n{explanation[:500]}...")
            return explanation
        else:
            print(f"\n(Would provide {detail_level}-level explanation of {concept})")
            return ""
    
    async def generate_code(self, task: str, language: str = "python"):
        """Generate code for a simulation task."""
        print(f"\n💻 Generating {language} code for: {task}")
        
        code_request = f"""Generate {language} code for the following task:
{task}

Requirements:
- Use best practices for scientific computing
- Include comments explaining key steps
- Handle common edge cases
- Make it reproducible"""
        
        if self.chat and self.session:
            response = await self.chat.chat(
                message=code_request,
                session=self.session,
                mode=ChatMode.CODING,
                response_format=ResponseFormat.CODE
            )
            print(f"\n```python\n{response[:800]}\n```")
            return response
        else:
            print("\n(Would generate code with comments and error handling)")
            return ""
    
    async def suggest_literature(self, topic: str):
        """Suggest relevant literature."""
        print(f"\n📖 Literature suggestions for: {topic}")
        
        if self.chat and self.session:
            papers = await self.chat.get_literature_suggestions(
                topic=topic,
                session=self.session,
                num_papers=5
            )
            
            print(f"\nFound {len(papers)} relevant papers:")
            for i, paper in enumerate(papers, 1):
                print(f"\n{i}. {paper.get('title', 'N/A')}")
                print(f"   {paper.get('authors', 'N/A')} ({paper.get('year', 'N/A')})")
                print(f"   {paper.get('journal', 'N/A')}")
            
            return papers
        else:
            print("\n(Would suggest 5 relevant papers with summaries)")
            return []
    
    async def brainstorm(self, topic: str, constraints: Optional[list] = None):
        """Brainstorm research ideas."""
        print(f"\n💡 Brainstorming ideas for: {topic}")
        
        if self.chat and self.session:
            ideas = await self.chat.brainstorm_ideas(
                topic=topic,
                session=self.session,
                num_ideas=5,
                constraints=constraints
            )
            
            print(f"\nGenerated {len(ideas)} ideas:")
            for i, idea in enumerate(ideas, 1):
                print(f"\n{i}. {idea.get('title', 'N/A')}")
                print(f"   {idea.get('description', 'N/A')[:150]}...")
            
            return ideas
        else:
            print("\n(Would generate 5 creative research ideas)")
            return []
    
    def get_session_info(self):
        """Display session information."""
        if self.chat and self.session:
            summary = self.chat.get_session_summary(self.session)
            print(f"\n{summary}")
        else:
            print("\nNo active session")


async def demo_basic_qa():
    """Demonstrate basic Q&A functionality."""
    print("=" * 70)
    print("DEMO 1: Basic Scientific Q&A")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("materials")
    
    questions = [
        "What is the band gap and why is it important in semiconductors?",
        "How does density functional theory work?",
        "What makes a good catalyst for CO2 reduction?",
        "Explain the difference between GGA and hybrid functionals in DFT",
    ]
    
    for q in questions:
        await bot.answer_question(q)
        await asyncio.sleep(0.5)


async def demo_concept_explanation():
    """Demonstrate concept explanation at different levels."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Level Concept Explanation")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("materials")
    
    concept = "phonon dispersion"
    levels = ["basic", "intermediate", "advanced"]
    
    for level in levels:
        await bot.explain_concept(concept, level)
        print()


async def demo_coding_assistance():
    """Demonstrate code generation for simulations."""
    print("\n" + "=" * 70)
    print("DEMO 3: Code Generation for Materials Simulations")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("computational_chemistry")
    
    tasks = [
        "Parse VASP OUTCAR and extract total energies",
        "Calculate radial distribution function from MD trajectory",
        "Fit a Birch-Murnaghan equation of state to energy-volume data",
    ]
    
    for task in tasks:
        await bot.generate_code(task)
        print()


async def demo_literature_suggestions():
    """Demonstrate literature recommendation."""
    print("\n" + "=" * 70)
    print("DEMO 4: Literature Recommendations")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("materials")
    
    topics = [
        "machine learning for catalyst discovery",
        "high-entropy alloys for energy applications",
    ]
    
    for topic in topics:
        await bot.suggest_literature(topic)
        print()


async def demo_brainstorming():
    """Demonstrate research idea generation."""
    print("\n" + "=" * 70)
    print("DEMO 5: Research Idea Brainstorming")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("catalysis")
    
    await bot.brainstorm(
        topic="novel single-atom catalysts for nitrogen reduction",
        constraints=[
            "Must use earth-abundant elements only",
            "Should be stable under electrochemical conditions",
        ]
    )


async def demo_multiturn_conversation():
    """Demonstrate multi-turn conversation with context."""
    print("\n" + "=" * 70)
    print("DEMO 6: Multi-Turn Conversation with Context")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("battery_materials")
    
    # Simulate a conversation flow
    conversation = [
        ("What are the main challenges in solid-state batteries?", None),
        ("Tell me more about the interface issues", ChatMode.EXPLAIN),
        ("How can we characterize these interfaces experimentally?", ChatMode.METHODOLOGY),
        ("What computational methods are used to model them?", ChatMode.CODING),
        ("Can you suggest some recent papers on this topic?", ChatMode.LITERATURE),
    ]
    
    for message, mode in conversation:
        await bot.answer_question(message, mode)
        print()
    
    # Show session summary
    bot.get_session_info()


async def demo_hypothesis_testing():
    """Demonstrate hypothesis formulation assistance."""
    print("\n" + "=" * 70)
    print("DEMO 7: Hypothesis Formulation Assistance")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("catalysis")
    
    observation = """
    We observed that adding 5% dopant concentration of Nb to TiO2 
    increased the photocatalytic activity by 40% under visible light, 
    but further increasing to 10% decreased activity back to near 
    pristine levels.
    """
    
    print(f"\nObservation:")
    print(observation)
    
    if bot.chat and bot.session:
        result = await bot.chat.help_formulate_hypothesis(
            observation=observation,
            session=bot.session
        )
        
        print("\nFormulated Hypotheses:")
        for i, h in enumerate(result.get("hypotheses", []), 1):
            print(f"\n{i}. {h.get('statement', 'N/A')}")
            print(f"   Tests: {', '.join(h.get('suggested_tests', [])[:2])}")
    else:
        print("\n(Would generate testable hypotheses with predictions)")


async def demo_domain_specific():
    """Demonstrate domain-specific expertise."""
    print("\n" + "=" * 70)
    print("DEMO 8: Domain-Specific Expertise")
    print("=" * 70)
    
    domains = [
        ("semiconductors", "What determines carrier mobility in 2D materials?"),
        ("catalysis", "How do scaling relations affect catalyst screening?"),
        ("battery", "What causes capacity fade in silicon anodes?"),
        ("quantum", "What is the significance of Berry curvature in topology?"),
    ]
    
    for domain, question in domains:
        print(f"\n--- Domain: {domain.upper()} ---")
        bot = MaterialsScienceQABot()
        await bot.start_session(domain)
        await bot.answer_question(question)


async def demo_analysis_discussion():
    """Demonstrate data analysis discussion."""
    print("\n" + "=" * 70)
    print("DEMO 9: Data Analysis Discussion")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("materials")
    
    data_description = """
    XRD analysis of a synthesized perovskite material shows:
    - Expected peaks at 2θ = 23.2°, 32.9°, 40.5°
    - Additional small peaks at 2θ = 28.5°, 41.2°
    - Peak broadening FWHM = 0.15° vs expected 0.08°
    - Main peaks match reference pattern well
    """
    
    print(f"\nData Description:")
    print(data_description)
    
    if bot.chat and bot.session:
        analysis = await bot.chat.analyze_data_discussion(
            data_description=data_description,
            session=bot.session,
            specific_question="What can we conclude about the sample quality?"
        )
        print(f"\nAnalysis:\n{analysis[:500]}...")
    else:
        print("\n(Would analyze XRD data and suggest interpretations)")


async def demo_export_conversation():
    """Demonstrate conversation export."""
    print("\n" + "=" * 70)
    print("DEMO 10: Conversation Export")
    print("=" * 70)
    
    bot = MaterialsScienceQABot()
    await bot.start_session("materials")
    
    # Add some messages
    if bot.session:
        from dftlammps.llm_integration.chat_interface import ChatMessage
        bot.session.add_message(ChatMessage(role="user", content="What is DFT?"))
        bot.session.add_message(ChatMessage(
            role="assistant", 
            content="DFT (Density Functional Theory) is a computational method..."
        ))
        bot.session.add_message(ChatMessage(
            role="user", 
            content="How accurate is it?"
        ))
    
    if bot.chat and bot.session:
        # Export as markdown
        md_export = bot.chat.export_session(bot.session, format="markdown")
        print("\nMarkdown Export (sample):")
        print("-" * 50)
        print(md_export[:600])
        
        # Export as JSON
        json_export = bot.chat.export_session(bot.session, format="json")
        print("\n\nJSON Export (sample):")
        print("-" * 50)
        print(json_export[:500])


def print_summary():
    """Print summary of the demo."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demo showcased the Scientific Q&A Bot capabilities:

1. BASIC Q&A
   - Answer scientific questions with domain expertise
   - Context-aware responses
   - Multi-mode operation (explain/analyze/brainstorm)

2. CONCEPT EXPLANATION
   - Multi-level explanations (basic/intermediate/advanced)
   - Progressive complexity
   - Related concept linking

3. CODE GENERATION
   - Simulation code for common tasks
   - Best practices included
   - Language-specific idioms

4. LITERATURE RECOMMENDATIONS
   - Relevant paper suggestions
   - Citation information
   - Relevance scoring

5. RESEARCH BRAINSTORMING
   - Idea generation with constraints
   - Impact and feasibility assessment
   - Related work identification

6. MULTI-TURN CONVERSATIONS
   - Context retention across messages
   - Mode switching during conversation
   - Topic tracking

7. HYPOTHESIS FORMULATION
   - Generate testable hypotheses
   - Prediction derivation
   - Experimental design suggestions

8. DOMAIN EXPERTISE
   - Specialized knowledge per domain
   - Appropriate terminology
   - Field-specific insights

9. DATA ANALYSIS
   - Interpret experimental data
   - Suggest follow-up experiments
   - Identify anomalies

10. CONVERSATION EXPORT
    - Markdown format for readability
    - JSON format for processing
    - Full context preservation

Key Features:
✓ Scientific context tracking
✓ Domain-specific system prompts
✓ Multi-modal responses (text/code/equations)
✓ Conversation memory and summarization
✓ Session export for documentation

To run with actual LLM responses:
  export OPENAI_API_KEY=your_key
  python scientific_qa_bot.py
""")


async def main():
    """Main demo function."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║            Scientific Q&A Bot - Materials Science Assistant         ║
║                                                                      ║
║  This demo demonstrates an interactive AI assistant for materials   ║
║  science research, providing answers, explanations, and assistance. ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    await demo_basic_qa()
    await demo_concept_explanation()
    await demo_coding_assistance()
    await demo_literature_suggestions()
    await demo_brainstorming()
    await demo_multiturn_conversation()
    await demo_hypothesis_testing()
    await demo_domain_specific()
    await demo_analysis_discussion()
    await demo_export_conversation()
    
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
