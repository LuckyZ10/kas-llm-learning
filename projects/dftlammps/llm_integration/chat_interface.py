"""
Scientific Chat Interface Module

Provides an intelligent conversational interface for scientific research,
supporting multi-turn dialogues with context awareness, scientific knowledge
retrieval, and specialized reasoning modes.

Features:
- Multi-turn conversation management
- Scientific context tracking
- Domain-specific response modes
- Tool integration (calculations, literature search)
- Conversation memory and summarization
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncIterator
from datetime import datetime
from collections import deque
import asyncio

from .llm_interface import (
    UnifiedLLMInterface,
    LLMConfig,
    LLMProvider,
    StreamResponse,
    Message,
    Conversation,
)
from .hypothesis_explainer import HypothesisExplainer, ExplanationStyle, NumericalContext
from .scientific_reasoning import ScientificReasoningEngine, Evidence, EvidenceType


class ChatMode(Enum):
    """Modes for scientific chat."""
    GENERAL = auto()           # General scientific discussion
    EXPLAIN = auto()           # Explanation mode
    BRAINSTORM = auto()        # Idea generation
    ANALYZE = auto()           # Data analysis discussion
    CRITIQUE = auto()          # Critical analysis
    LITERATURE = auto()        # Literature discussion
    HYPOTHESIS = auto()        # Hypothesis testing
    METHODOLOGY = auto()       # Methods discussion
    CODING = auto()            # Code assistance
    WRITING = auto()           # Writing assistance


class ResponseFormat(Enum):
    """Response format preferences."""
    CONVERSATIONAL = auto()    # Natural conversation
    STRUCTURED = auto()        # Structured/bulleted
    STEP_BY_STEP = auto()      # Sequential steps
    COMPARISON = auto()        # Side-by-side comparison
    CODE = auto()              # Code-focused
    EQUATION = auto()          # Math/equation focused


@dataclass
class ScientificContext:
    """Scientific context for chat sessions."""
    domain: str = "general"                    # Research domain
    topic: Optional[str] = None                # Current topic
    relevant_papers: List[Dict[str, Any]] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    active_hypotheses: List[str] = field(default_factory=list)
    numerical_data: Optional[NumericalContext] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    
    def to_prompt_context(self) -> str:
        """Convert to string for prompting."""
        lines = [f"Domain: {self.domain}"]
        
        if self.topic:
            lines.append(f"Current Topic: {self.topic}")
        
        if self.key_concepts:
            lines.append(f"Key Concepts: {', '.join(self.key_concepts)}")
        
        if self.active_hypotheses:
            lines.append("Active Hypotheses:")
            for h in self.active_hypotheses:
                lines.append(f"  - {h}")
        
        return "\n".join(lines)


@dataclass
class ChatMessage:
    """A message in the chat."""
    role: str  # user, assistant, system, tool
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    mode: Optional[ChatMode] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "mode": self.mode.name if self.mode else None,
            "metadata": self.metadata,
        }


@dataclass
class ChatSession:
    """A chat session with full context."""
    session_id: str
    created_at: datetime = field(default_factory=datetime.now)
    messages: List[ChatMessage] = field(default_factory=list)
    scientific_context: ScientificContext = field(default_factory=ScientificContext)
    current_mode: ChatMode = ChatMode.GENERAL
    summary: Optional[str] = None
    
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session."""
        self.messages.append(message)
    
    def get_recent_context(self, n_messages: int = 5) -> List[ChatMessage]:
        """Get recent message context."""
        return self.messages[-n_messages:] if len(self.messages) > n_messages else self.messages
    
    def to_conversation(self) -> Conversation:
        """Convert to Conversation for LLM."""
        conv = Conversation()
        for msg in self.messages:
            conv.add_message(Message(role=msg.role, content=msg.content))
        return conv


@dataclass
class MultiTurnConversation:
    """Manages multi-turn conversation state."""
    session: ChatSession
    turn_count: int = 0
    topic_history: deque = field(default_factory=lambda: deque(maxlen=10))
    clarification_needed: bool = False
    pending_questions: List[str] = field(default_factory=list)
    
    def advance_turn(self) -> None:
        """Advance to next turn."""
        self.turn_count += 1


class ScientificChatInterface:
    """Main scientific chat interface."""
    
    # Mode-specific system prompts
    MODE_PROMPTS = {
        ChatMode.GENERAL: """You are a helpful scientific research assistant. 
Provide accurate, well-reasoned responses to scientific questions. Acknowledge 
uncertainty when appropriate and cite relevant principles or literature.""",
        
        ChatMode.EXPLAIN: """You are an expert science communicator. Explain 
complex concepts clearly and accessibly. Use analogies where helpful, define 
terms, and build from fundamentals to advanced concepts.""",
        
        ChatMode.BRAINSTORM: """You are a creative research collaborator. 
Generate diverse ideas, challenge assumptions constructively, and help explore 
unconventional approaches. Encourage divergent thinking.""",
        
        ChatMode.ANALYZE: """You are a data analysis expert. Help interpret 
results, identify patterns, suggest statistical approaches, and discuss 
limitations. Be precise about what can and cannot be concluded.""",
        
        ChatMode.CRITIQUE: """You are a critical reviewer. Provide constructive 
criticism, identify weaknesses, question assumptions, and suggest improvements. 
Be thorough but fair.""",
        
        ChatMode.LITERATURE: """You are a literature expert. Discuss papers, 
identify key works, summarize findings, and place research in historical 
context. Note conflicting results and emerging trends.""",
        
        ChatMode.HYPOTHESIS: """You are a hypothesis testing advisor. Help 
formulate testable hypotheses, design experiments, and evaluate evidence. 
Distinguish between correlation and causation.""",
        
        ChatMode.METHODOLOGY: """You are a methodology expert. Discuss 
computational and experimental methods, compare approaches, suggest best 
practices, and troubleshoot issues.""",
        
        ChatMode.CODING: """You are a scientific computing expert. Provide 
code, debug issues, explain algorithms, and suggest optimizations. Focus on 
clarity and correctness.""",
        
        ChatMode.WRITING: """You are a scientific writing coach. Help structure 
arguments, improve clarity, suggest phrasing, and ensure appropriate tone for 
the target audience.""",
    }
    
    def __init__(
        self,
        llm_interface: Optional[UnifiedLLMInterface] = None,
        reasoning_engine: Optional[ScientificReasoningEngine] = None,
        hypothesis_explainer: Optional[HypothesisExplainer] = None,
        default_mode: ChatMode = ChatMode.GENERAL,
    ):
        """Initialize chat interface.
        
        Args:
            llm_interface: LLM interface
            reasoning_engine: Scientific reasoning engine
            hypothesis_explainer: Hypothesis explainer
            default_mode: Default chat mode
        """
        self.llm = llm_interface or self._create_default_llm()
        self.reasoning = reasoning_engine or ScientificReasoningEngine(self.llm)
        self.explainer = hypothesis_explainer or HypothesisExplainer(self.llm)
        self.default_mode = default_mode
        self.sessions: Dict[str, ChatSession] = {}
    
    def _create_default_llm(self) -> UnifiedLLMInterface:
        """Create default LLM interface."""
        try:
            config = LLMConfig.from_env(LLMProvider.OPENAI)
        except ValueError:
            for provider in [LLMProvider.ANTHROPIC, LLMProvider.DEEPSEEK]:
                try:
                    config = LLMConfig.from_env(provider)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("No LLM API keys found")
        
        return UnifiedLLMInterface(config)
    
    def create_session(
        self,
        session_id: Optional[str] = None,
        domain: str = "general",
        initial_context: Optional[ScientificContext] = None,
    ) -> ChatSession:
        """Create a new chat session.
        
        Args:
            session_id: Session identifier
            domain: Scientific domain
            initial_context: Initial scientific context
            
        Returns:
            New chat session
        """
        import uuid
        session_id = session_id or str(uuid.uuid4())[:8]
        
        context = initial_context or ScientificContext(domain=domain)
        
        session = ChatSession(
            session_id=session_id,
            scientific_context=context,
        )
        
        self.sessions[session_id] = session
        return session
    
    async def chat(
        self,
        message: str,
        session: Optional[ChatSession] = None,
        mode: Optional[ChatMode] = None,
        stream: bool = False,
        response_format: ResponseFormat = ResponseFormat.CONVERSATIONAL,
    ) -> Union[str, AsyncIterator[str]]:
        """Send a message and get response.
        
        Args:
            message: User message
            session: Chat session (creates new if None)
            mode: Chat mode
            stream: Whether to stream response
            response_format: Response format preference
            
        Returns:
            Response string or async iterator
        """
        # Get or create session
        if session is None:
            session = self.create_session()
        
        # Determine mode
        detected_mode = mode or self._detect_mode(message) or session.current_mode
        session.current_mode = detected_mode
        
        # Add user message
        user_msg = ChatMessage(
            role="user",
            content=message,
            mode=detected_mode,
        )
        session.add_message(user_msg)
        
        # Build system prompt
        system_prompt = self._build_system_prompt(
            detected_mode,
            session.scientific_context,
            response_format,
        )
        
        # Build conversation context
        recent_messages = session.get_recent_context(10)
        conversation = Conversation()
        conversation.add_message(Message.system(system_prompt))
        
        for msg in recent_messages[:-1] if len(recent_messages) > 1 else []:
            conversation.add_message(Message(role=msg.role, content=msg.content))
        
        # Generate response
        if stream:
            return self._stream_response(message, conversation, session)
        else:
            response_text = await self._generate_response(message, conversation, session)
            
            # Add assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=response_text,
                mode=detected_mode,
            )
            session.add_message(assistant_msg)
            
            return response_text
    
    async def _generate_response(
        self,
        message: str,
        conversation: Conversation,
        session: ChatSession,
    ) -> str:
        """Generate non-streaming response."""
        response = await self.llm.complete(
            prompt=message,
            conversation=conversation,
            temperature=0.7,
            max_tokens=2000,
        )
        return response.content
    
    async def _stream_response(
        self,
        message: str,
        conversation: Conversation,
        session: ChatSession,
    ) -> AsyncIterator[str]:
        """Generate streaming response."""
        full_response = []
        
        async for chunk in self.llm.stream(
            prompt=message,
            conversation=conversation,
            temperature=0.7,
            max_tokens=2000,
        ):
            if chunk.content:
                full_response.append(chunk.content)
                yield chunk.content
        
        # Add complete message to session
        assistant_msg = ChatMessage(
            role="assistant",
            content="".join(full_response),
            mode=session.current_mode,
        )
        session.add_message(assistant_msg)
    
    def _detect_mode(self, message: str) -> Optional[ChatMode]:
        """Detect chat mode from message."""
        message_lower = message.lower()
        
        mode_indicators = {
            ChatMode.EXPLAIN: ["explain", "what is", "how does", "describe", "define", "clarify"],
            ChatMode.BRAINSTORM: ["brainstorm", "ideas", "suggest", "what if", "possibilities", "alternatives"],
            ChatMode.ANALYZE: ["analyze", "interpret", "what do these results mean", "pattern", "trend"],
            ChatMode.CRITIQUE: ["critique", "review", "criticism", "weakness", "problem", "limitation"],
            ChatMode.LITERATURE: ["paper", "publication", "cite", "reference", "who found", "study"],
            ChatMode.HYPOTHESIS: ["hypothesis", "test", "predict", "expect", "if then"],
            ChatMode.METHODOLOGY: ["method", "protocol", "procedure", "how to", "setup", "calculate"],
            ChatMode.CODING: ["code", "program", "script", "function", "algorithm", "debug", "python"],
            ChatMode.WRITING: ["write", "draft", "phrase", "wording", "sentence", "paragraph", "section"],
        }
        
        for mode, indicators in mode_indicators.items():
            if any(ind in message_lower for ind in indicators):
                return mode
        
        return None
    
    def _build_system_prompt(
        self,
        mode: ChatMode,
        context: ScientificContext,
        response_format: ResponseFormat,
    ) -> str:
        """Build system prompt for the session."""
        parts = [self.MODE_PROMPTS.get(mode, self.MODE_PROMPTS[ChatMode.GENERAL])]
        
        # Add context
        context_str = context.to_prompt_context()
        if context_str:
            parts.extend([
                "",
                "Current Context:",
                context_str,
            ])
        
        # Add format instructions
        format_instructions = {
            ResponseFormat.STRUCTURED: "Structure your response with clear headings and bullet points.",
            ResponseFormat.STEP_BY_STEP: "Provide your response as a numbered sequence of steps.",
            ResponseFormat.COMPARISON: "Present information in a comparative format, highlighting similarities and differences.",
            ResponseFormat.CODE: "Focus on code examples and technical implementation details.",
            ResponseFormat.EQUATION: "Include relevant equations and mathematical formulations where appropriate.",
        }
        
        if response_format in format_instructions:
            parts.extend(["", format_instructions[response_format]])
        
        return "\n".join(parts)
    
    async def explain_with_context(
        self,
        concept: str,
        session: ChatSession,
        detail_level: str = "intermediate",
    ) -> str:
        """Explain a concept with full context awareness.
        
        Args:
            concept: Concept to explain
            session: Chat session
            detail_level: Detail level (basic/intermediate/advanced)
            
        Returns:
            Explanation
        """
        # Get relevant context from session
        recent_topics = [m.content for m in session.messages[-5:]]
        
        prompt = f"""Explain "{concept}" at a {detail_level} level.

Context from conversation:
{chr(10).join(recent_topics[-3:])}

Domain: {session.scientific_context.domain}

Provide:
1. Clear definition
2. Key principles
3. Relevant examples
4. Connection to previous discussion (if applicable)
5. When this concept is important"""
        
        response = await self.llm.complete(prompt, temperature=0.6, max_tokens=1500)
        
        # Add to session
        session.add_message(ChatMessage(role="user", content=f"Explain {concept}"))
        session.add_message(ChatMessage(role="assistant", content=response.content))
        
        return response.content
    
    async def analyze_data_discussion(
        self,
        data_description: str,
        session: ChatSession,
        specific_question: Optional[str] = None,
    ) -> str:
        """Discuss data analysis in context.
        
        Args:
            data_description: Description of data
            session: Chat session
            specific_question: Specific question about the data
            
        Returns:
            Analysis discussion
        """
        session.current_mode = ChatMode.ANALYZE
        
        prompt = f"""Analyze the following data:

{data_description}

{specific_question if specific_question else "What insights can we draw from this data?"}

Consider:
1. Statistical significance
2. Practical significance
3. Potential confounders
4. Alternative interpretations
5. Suggested follow-up analyses"""
        
        response = await self.llm.complete(prompt, temperature=0.4, max_tokens=1500)
        
        session.add_message(ChatMessage(role="user", content=data_description[:200]))
        session.add_message(ChatMessage(role="assistant", content=response.content, mode=ChatMode.ANALYZE))
        
        return response.content
    
    async def brainstorm_ideas(
        self,
        topic: str,
        session: ChatSession,
        num_ideas: int = 5,
        constraints: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Brainstorm research ideas.
        
        Args:
            topic: Topic for brainstorming
            session: Chat session
            num_ideas: Number of ideas to generate
            constraints: Constraints to consider
            
        Returns:
            List of ideas with details
        """
        session.current_mode = ChatMode.BRAINSTORM
        
        constraints_text = f"\nConstraints:\n{chr(10).join(f'- {c}' for c in constraints)}" if constraints else ""
        
        prompt = f"""Brainstorm {num_ideas} research ideas related to: {topic}

Domain: {session.scientific_context.domain}
{constraints_text}

For each idea, provide:
1. Title/concept
2. Brief description
3. Potential impact
4. Feasibility considerations
5. Related work or precedents

Format as JSON array."""
        
        schema = {
            "ideas": [
                {
                    "title": "string",
                    "description": "string",
                    "potential_impact": "string",
                    "feasibility": "string",
                    "related_work": "string",
                }
            ]
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.8)
        
        return result.get("ideas", [])
    
    async def get_literature_suggestions(
        self,
        topic: str,
        session: ChatSession,
        num_papers: int = 5,
    ) -> List[Dict[str, str]]:
        """Suggest relevant literature.
        
        Args:
            topic: Research topic
            session: Chat session
            num_papers: Number of papers to suggest
            
        Returns:
            List of paper suggestions
        """
        session.current_mode = ChatMode.LITERATURE
        
        prompt = f"""Suggest {num_papers} key papers related to: {topic}

Domain: {session.scientific_context.domain}

For each paper, provide:
1. Authors (key authors)
2. Title
3. Year
4. Journal
5. Key contribution
6. Why it's relevant

Format as JSON array."""
        
        schema = {
            "papers": [
                {
                    "authors": "string",
                    "title": "string",
                    "year": "integer",
                    "journal": "string",
                    "contribution": "string",
                    "relevance": "string",
                }
            ]
        }
        
        result = await self.llm.generate_structured(prompt, schema, temperature=0.5)
        return result.get("papers", [])
    
    async def help_formulate_hypothesis(
        self,
        observation: str,
        session: ChatSession,
    ) -> Dict[str, Any]:
        """Help formulate testable hypotheses.
        
        Args:
            observation: Observation to base hypothesis on
            session: Chat session
            
        Returns:
            Hypothesis formulation
        """
        session.current_mode = ChatMode.HYPOTHESIS
        
        prompt = f"""Based on the following observation, help formulate testable hypotheses:

Observation: {observation}

Domain: {session.scientific_context.domain}

Provide:
1. 2-3 alternative hypotheses
2. Predictions for each hypothesis
3. Suggested experiments/tests
4. Key variables to measure
5. Potential confounders to control

Format as JSON."""
        
        schema = {
            "hypotheses": [
                {
                    "statement": "string",
                    "predictions": ["string"],
                    "suggested_tests": ["string"],
                    "key_variables": ["string"],
                    "confounders": ["string"],
                }
            ],
            "recommendations": "string",
        }
        
        return await self.llm.generate_structured(prompt, schema, temperature=0.6)
    
    def get_session_summary(self, session: ChatSession) -> str:
        """Generate summary of chat session.
        
        Args:
            session: Chat session
            
        Returns:
            Session summary
        """
        lines = [
            f"Session: {session.session_id}",
            f"Domain: {session.scientific_context.domain}",
            f"Messages: {len(session.messages)}",
            f"Current Mode: {session.current_mode.name}",
            "",
            "Recent Topics:",
        ]
        
        # Extract topics from messages
        for msg in session.messages[-5:]:
            if msg.role == "user":
                preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                lines.append(f"- {preview}")
        
        return "\n".join(lines)
    
    async def summarize_conversation(
        self,
        session: ChatSession,
        focus: str = "key_points",
    ) -> str:
        """Generate AI summary of conversation.
        
        Args:
            session: Chat session
            focus: Summary focus
            
        Returns:
            Conversation summary
        """
        # Build conversation text
        conversation_text = "\n\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in session.messages
        ])
        
        prompt = f"""Summarize the following scientific conversation.

Focus on: {focus}

Conversation:
{conversation_text}

Provide a concise summary capturing:
1. Main topics discussed
2. Key conclusions or insights
3. Open questions or next steps
4. Overall progression of the discussion"""
        
        response = await self.llm.complete(prompt, temperature=0.4, max_tokens=1000)
        session.summary = response.content
        
        return response.content
    
    def export_session(
        self,
        session: ChatSession,
        format: str = "json",
    ) -> str:
        """Export session to file format.
        
        Args:
            session: Chat session
            format: Export format (json/markdown)
            
        Returns:
            Exported content
        """
        if format == "json":
            data = {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "domain": session.scientific_context.domain,
                "messages": [m.to_dict() for m in session.messages],
                "summary": session.summary,
            }
            return json.dumps(data, indent=2)
        
        elif format == "markdown":
            lines = [
                f"# Chat Session: {session.session_id}",
                f"**Domain:** {session.scientific_context.domain}",
                f"**Created:** {session.created_at.isoformat()}",
                "",
                "## Conversation",
                "",
            ]
            
            for msg in session.messages:
                role_emoji = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(msg.role, "💬")
                lines.append(f"### {role_emoji} {msg.role.title()}")
                lines.append(msg.content)
                lines.append("")
            
            if session.summary:
                lines.extend(["## Summary", session.summary])
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format}")
