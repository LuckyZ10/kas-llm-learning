"""
Paper Writing Assistant Demo - Full Paper Generation Workflow

This example demonstrates the complete workflow for using LLMs to assist
in academic paper writing, from structure planning to final polishing.

Features demonstrated:
1. Paper structure planning
2. Section-by-section drafting
3. Citation management
4. Abstract generation
5. Review and revision suggestions
6. Language polishing
"""

import asyncio
from typing import Dict, List, Optional

try:
    from dftlammps.llm_integration import (
        UnifiedLLMInterface,
        LLMConfig,
        LLMProvider,
    )
    from dftlammps.llm_integration.paper_assistant import (
        PaperAssistant,
        PaperSection,
        WritingStyle,
        CitationStyle,
        Citation,
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
    from llm_integration.paper_assistant import (
        PaperAssistant,
        PaperSection,
        WritingStyle,
        CitationStyle,
        Citation,
    )


async def demo_structure_planning():
    """Demonstrate paper structure planning."""
    print("=" * 70)
    print("STEP 1: Paper Structure Planning")
    print("=" * 70)
    
    try:
        config = LLMConfig.from_env(LLMProvider.OPENAI)
        llm = UnifiedLLMInterface(config)
        assistant = PaperAssistant(llm, writing_style=WritingStyle.FORMAL_ACADEMIC)
        
        structure = await assistant.plan_structure(
            title="Machine Learning Predictions of Catalytic Activity for CO2 Reduction",
            research_type="computational_study",
            key_findings=[
                "Graph neural network achieves 0.15 eV MAE on binding energies",
                "Identified 3 novel high-activity catalyst candidates",
                "Transfer learning reduces data requirements by 60%",
                "Model interprets importance of coordination environment"
            ],
            target_journal="Nature Catalysis"
        )
        
        print("\nRecommended Structure:")
        print(structure.to_outline())
        
        return assistant, structure
        
    except ValueError as e:
        print(f"\nNote: {e}")
        print("Showing example structure...")
        print("""
Recommended Structure:

## Abstract (~250 words)
- Context and motivation
- Methods summary
- Key results with numbers
- Main conclusion

## Introduction (~800 words)
- CO2 reduction importance
- Current catalyst limitations
- ML in catalysis overview
- This work's contributions

## Methods (~1000 words)
- Dataset construction
- GNN architecture
- Training procedures
- DFT validation

## Results (~1500 words)
- Model performance metrics
- Feature importance analysis
- New catalyst predictions
- Experimental validation

## Discussion (~1000 words)
- Implications for catalyst design
- Comparison with prior work
- Limitations and future work

## Conclusion (~300 words)
- Summary of achievements
- Broader impact
""")
        return None, None


async def demo_section_writing(assistant: PaperAssistant):
    """Demonstrate section writing."""
    print("\n" + "=" * 70)
    print("STEP 2: Section Writing")
    print("=" * 70)
    
    # Add some citations
    citations = [
        Citation(
            id="ref1",
            title="Machine Learning for Catalyst Design",
            authors=["Smith, J.", "Jones, A."],
            year=2023,
            journal="Nature Catalysis",
            volume="15",
            pages="123-135",
        ),
        Citation(
            id="ref2",
            title="Graph Neural Networks for Materials Discovery",
            authors=["Wang, L.", "Chen, B.", "Liu, X."],
            year=2022,
            journal="J. Phys. Chem. C",
            volume="126",
            pages="4567-4578",
        ),
        Citation(
            id="ref3",
            title="High-Throughput Screening of CO2 Reduction Catalysts",
            authors=["Brown, M."],
            year=2021,
            journal="ACS Catalysis",
            volume="11",
            pages="2345-2356",
        ),
    ]
    
    for cite in citations:
        assistant.citation_manager.add_citation(cite)
    
    sections_to_write = [
        (PaperSection.ABSTRACT, [
            "CO2 electroreduction is critical for carbon neutrality",
            "Current catalyst discovery is slow and expensive",
            "Machine learning can accelerate screening",
            "GNN model achieves 0.15 eV MAE on binding energies",
            "Identified 3 promising candidates",
        ], None),
        (PaperSection.INTRODUCTION, [
            "Climate change requires carbon capture and utilization",
            "Electrochemical CO2 reduction can produce valuable chemicals",
            "Catalyst discovery is the key bottleneck",
            "Computational screening using DFT is accurate but slow",
            "Machine learning offers acceleration potential",
            "Graph neural networks are promising for materials",
        ], 800),
    ]
    
    for section, key_points, word_count in sections_to_write:
        print(f"\n--- Writing {section.value.upper()} ---")
        
        try:
            draft = await assistant.write_section(
                section=section,
                key_points=key_points,
                target_word_count=word_count,
            )
            
            print(f"Draft word count: {draft.word_count}")
            print(f"\nPreview (first 300 chars):")
            print(f"{draft.content[:300]}...")
            
        except Exception as e:
            print(f"Could not generate section: {e}")
            print("(Mock content would go here)")


async def demo_abstract_generation(assistant: PaperAssistant):
    """Demonstrate abstract generation from sections."""
    print("\n" + "=" * 70)
    print("STEP 3: Abstract Generation from Sections")
    print("=" * 70)
    
    # Simulate having written sections
    sections = {
        PaperSection.INTRODUCTION: """
The electrochemical reduction of carbon dioxide (CO2RR) represents a promising 
pathway toward carbon neutrality and sustainable chemical production. However, 
the discovery of efficient, selective, and stable catalysts remains a significant 
challenge. Traditional experimental screening is time-consuming and expensive, 
while computational screening using density functional theory (DFT), though 
accurate, is computationally intensive and limits the searchable chemical space.

Recent advances in machine learning (ML) offer potential solutions for 
accelerating catalyst discovery. Graph neural networks (GNNs) have emerged 
as particularly promising architectures for materials prediction due to their 
ability to learn representations directly from atomic structures.
        """,
        PaperSection.METHODS: """
We developed a GNN model based on the SchNet architecture with custom modifications 
for catalysis applications. The model was trained on a dataset of 15,000 DFT 
calculations of CO and H binding energies on transition metal surfaces.

The dataset spans 40 different metal elements and 12 surface facets. We employed 
a 80/10/10 train/validation/test split. The model was trained using the Adam 
optimizer with a learning rate of 1e-4 for 500 epochs.
        """,
        PaperSection.RESULTS: """
Our GNN model achieves a mean absolute error (MAE) of 0.15 eV on the test set, 
significantly outperforming baseline models. The model successfully identifies 
key chemical trends, including the d-band center correlation with adsorption 
strength.

Using the trained model, we screened a library of 10,000 bimetallic alloys and 
identified three promising candidates with predicted CO binding energies 
(-0.4 to -0.6 eV) in the optimal range for CO2RR activity.
        """,
        PaperSection.CONCLUSION: """
We have demonstrated that graph neural networks can accurately predict catalytic 
properties with errors comparable to DFT at a fraction of the computational cost. 
The identified catalyst candidates warrant experimental validation. This work 
establishes a framework for accelerated catalyst discovery that can be extended 
to other catalytic reactions.
        """,
    }
    
    try:
        abstract = await assistant.generate_abstract_from_sections(
            sections=sections,
            max_words=200
        )
        
        print("\nGenerated Abstract:")
        print("-" * 50)
        print(abstract)
        print("-" * 50)
        
    except Exception as e:
        print(f"Could not generate abstract: {e}")
        print("\n(Mock abstract would be generated here)")


async def demo_polishing(assistant: PaperAssistant):
    """Demonstrate text polishing."""
    print("\n" + "=" * 70)
    print("STEP 4: Text Polishing")
    print("=" * 70)
    
    # Example text that needs improvement
    draft_text = """
We did a study on CO2 reduction catalysts using machine learning. 
The results show that our model works pretty well. We found some 
new catalysts that might be good. The model has low error compared 
to DFT. Future work could look at other reactions too.
"""
    
    print("\nOriginal Text:")
    print("-" * 50)
    print(draft_text)
    
    polish_types = ["general", "conciseness", "impact"]
    
    for polish_type in polish_types:
        print(f"\n--- Polishing for {polish_type.upper()} ---")
        
        try:
            polished = await assistant.polish_text(
                text=draft_text,
                polish_type=polish_type,
                target_audience="expert"
            )
            
            print(f"Polished ({polish_type}):")
            print("-" * 50)
            print(polished[:500])
            
        except Exception as e:
            print(f"Could not polish: {e}")
            print("(Polished text would appear here)")


async def demo_review(assistant: PaperAssistant):
    """Demonstrate paper review."""
    print("\n" + "=" * 70)
    print("STEP 5: Paper Review Simulation")
    print("=" * 70)
    
    # Simulate a complete paper
    paper_sections = {
        PaperSection.ABSTRACT: """
We present a machine learning approach for predicting catalytic activity in CO2 
reduction. A graph neural network trained on DFT data achieves 0.15 eV MAE.
Screening identified 3 promising catalyst candidates.
""",
        PaperSection.INTRODUCTION: """
Carbon dioxide reduction is important for sustainability. Current methods are 
slow. Machine learning can help. We use GNNs for prediction.
""",
        PaperSection.METHODS: """
We used SchNet architecture with 3 interaction blocks. Training used 15,000 
structures. Learning rate was 1e-4.
""",
        PaperSection.RESULTS: """
The model works well. MAE is 0.15 eV. Three catalysts were found.
""",
        PaperSection.DISCUSSION: """
Our results show ML can predict catalysis. Future work should extend this.
""",
    }
    
    try:
        review = await assistant.review_paper(
            sections=paper_sections,
            paper_type="research_article"
        )
        
        print(f"\nOverall Score: {review.overall_score}/10")
        print(f"Estimated Impact: {review.estimated_impact}")
        
        print("\nStrengths:")
        for s in review.strengths[:3]:
            print(f"  ✓ {s}")
        
        print("\nPriority Revisions:")
        for i, r in enumerate(review.priority_revisions[:3], 1):
            print(f"  {i}. {r}")
            
    except Exception as e:
        print(f"Could not generate review: {e}")
        print("\n(Mock review would be generated here)")


async def demo_citation_management(assistant: PaperAssistant):
    """Demonstrate citation management."""
    print("\n" + "=" * 70)
    print("STEP 6: Citation Management")
    print("=" * 70)
    
    # Generate citations for a topic
    try:
        citations = await assistant.citation_manager.find_related_papers(
            llm=assistant.llm,
            topic="machine learning for catalysis",
            num_papers=5
        )
        
        print("\nSuggested Citations:")
        for i, cite in enumerate(citations, 1):
            print(f"\n{i}. {cite.title}")
            print(f"   Authors: {', '.join(cite.authors)}")
            print(f"   Journal: {cite.journal} ({cite.year})")
        
        # Add to manager
        for cite in citations:
            assistant.citation_manager.add_citation(cite)
        
        # Simulate citations in text
        print("\n\nCitation Usage:")
        for i, cite in enumerate(citations[:3], 1):
            marker = assistant.citation_manager.cite(cite.id, "introduction")
            print(f"  Text with citation {marker}: 'Recent work shows...{marker}'")
        
        # Generate bibliography
        print("\n\nGenerated Bibliography (sample):")
        bibliography = assistant.citation_manager.generate_bibliography()
        print(bibliography[:800] + "...")
        
        # Check balance
        print("\n\nCitation Balance Analysis:")
        balance = assistant.citation_manager.check_citation_balance()
        print(f"  Total citation instances: {balance['total_citation_instances']}")
        print(f"  Unique papers cited: {balance['unique_citations_cited']}")
        print(f"  Recommendation: {balance['recommendation']}")
        
    except Exception as e:
        print(f"Could not manage citations: {e}")
        print("\n(Mock citations would be generated here)")


async def demo_reviewer_response(assistant: PaperAssistant):
    """Demonstrate reviewer response generation."""
    print("\n" + "=" * 70)
    print("STEP 7: Reviewer Response Generation")
    print("=" * 70)
    
    # Simulate reviewer comments
    review_comments = [
        "The authors should provide more details about the training procedure, including hyperparameter selection.",
        "How does the model performance compare to other recent ML models for catalysis?",
        "The computational cost savings should be quantified more precisely.",
        "Include uncertainty quantification for the predictions.",
        "Can the authors comment on the transferability to other catalytic systems?",
    ]
    
    print("\nReviewer Comments:")
    for i, comment in enumerate(review_comments, 1):
        print(f"\n{i}. {comment}")
    
    try:
        response = await assistant.generate_reviewer_response(
            review_comments=review_comments,
            paper_sections={},  # Would contain actual paper
            tone="professional"
        )
        
        print("\n\nGenerated Response Letter (excerpt):")
        print("-" * 50)
        print(response[:1000])
        print("...")
        
    except Exception as e:
        print(f"\nCould not generate response: {e}")
        print("\n(Mock response letter would be generated here)")


def print_summary():
    """Print summary of the demo."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
This demo showcased the Paper Writing Assistant capabilities:

1. STRUCTURE PLANNING
   - Automatic paper structure recommendations
   - Word count allocation per section
   - Journal-specific guidance

2. SECTION WRITING
   - Draft generation from key points
   - Context-aware content creation
   - Automatic citation insertion

3. ABSTRACT GENERATION
   - Automated summary from paper sections
   - Key result highlighting
   - Word count optimization

4. TEXT POLISHING
   - Multiple polish types (clarity, conciseness, impact)
   - Audience-appropriate language adjustment
   - Style consistency

5. PEER REVIEW
   - Automated paper review
   - Strength/weakness identification
   - Priority revision suggestions

6. CITATION MANAGEMENT
   - Automated citation finding
   - Format consistency
   - Balance analysis

7. REVIEWER RESPONSES
   - Professional response generation
   - Point-by-point addressing
   - Revision tracking

Key Features:
✓ Domain-aware writing (materials science focus)
✓ Citation style flexibility (Nature, Science, etc.)
✓ Multiple writing styles (formal, concise, review)
✓ Iterative refinement support
✓ Export to multiple formats

To run with actual LLM responses:
  export OPENAI_API_KEY=your_key
  python paper_writing_assistant.py
""")


async def main():
    """Main demo function."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Paper Writing Assistant - Full Workflow Demo              ║
║                                                                      ║
║  This demo demonstrates the complete paper writing workflow using   ║
║  LLM assistance, from structure planning to final submission.       ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    assistant, structure = await demo_structure_planning()
    
    if assistant:
        await demo_section_writing(assistant)
        await demo_abstract_generation(assistant)
        await demo_polishing(assistant)
        await demo_review(assistant)
        await demo_citation_management(assistant)
        await demo_reviewer_response(assistant)
    
    print_summary()


if __name__ == "__main__":
    asyncio.run(main())
