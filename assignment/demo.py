#!/usr/bin/env python3
"""
Demo script showcasing the Data Transformation System capabilities

This script demonstrates:
1. Basic company profile extraction
2. Complex nested schema handling
3. Large document processing with chunking
4. Confidence estimation and human review flagging
5. Performance metrics and analysis
"""

import asyncio
import json
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .main import DataTransformationSystem
from .config import config

console = Console()

# =============================================================================
# Sample Documents and Schemas
# =============================================================================

SIMPLE_COMPANY_DOCUMENT = """
TechCorp Inc is a leading technology company founded in 2018 and headquartered in San Francisco, California. 
The company specializes in artificial intelligence and machine learning solutions for enterprise clients.

Leadership Team:
- CEO: John Smith (15 years experience, former VP at Google)
- CTO: Sarah Johnson (12 years experience, AI researcher)  
- CFO: Michael Brown (10 years experience in fintech)

Company Details:
- Employee Count: 500-750 employees
- Revenue: $50M ARR
- Funding: Series B, $25M raised
- Industry: Technology/AI

Contact Information:
- Email: info@techcorp.com
- Phone: +1-555-0123
- Address: 123 Innovation Drive, San Francisco, CA 94105

Products:
1. AI Analytics Platform - Machine learning analytics for business intelligence
2. Natural Language Processing API - Text analysis and understanding
3. Computer Vision SDK - Image and video analysis tools
"""

COMPLEX_RESEARCH_DOCUMENT = """
Research Paper: "Advanced Neural Network Architectures for Multi-Modal Learning"

Authors:
- Dr. Alice Chen, Stanford University, Department of Computer Science, 10 years in ML research
- Prof. Bob Rodriguez, MIT, AI Lab, 15 years experience, specializes in deep learning
- Dr. Carol Wang, Google Research, 8 years in industry research

Abstract:
This paper presents novel architectures for multi-modal learning combining vision, text, and audio inputs.
We introduce the Fusion Transformer architecture that achieves state-of-the-art results on three benchmark datasets.

Methodology:
Our approach uses attention mechanisms to align features across modalities. The architecture consists of:
1. Vision Encoder: ResNet-50 backbone with attention pooling
2. Text Encoder: BERT-large with custom tokenization
3. Audio Encoder: Wav2Vec2 with temporal convolutions
4. Fusion Module: Cross-modal attention with learnable position embeddings

Results:
- COCO Captions: 95.2% accuracy (previous SOTA: 92.1%)
- VQA 2.0: 78.9% accuracy (previous SOTA: 76.3%)
- AudioSet Classification: 85.7% mAP (previous SOTA: 83.2%)

Funding and Affiliations:
This research was supported by NSF Grant #1234567 ($500,000 over 3 years).
Additional funding from Google Research Grant ($200,000) and Stanford HAI Initiative ($100,000).

Publication Details:
- Conference: NeurIPS 2024
- Submission Date: May 15, 2024
- Acceptance Date: September 2, 2024
- Paper ID: NeurIPS24-5678
"""

LARGE_COMPANY_DOCUMENT = """
MetaGlobal Corporation - Annual Report 2024

Executive Summary:
MetaGlobal Corporation is a multinational technology conglomerate with operations in 47 countries.
Founded in 1995 by Jennifer Thompson and Marcus Lee, the company has grown from a small software startup
to a global enterprise serving over 100 million customers worldwide.

Corporate Structure:
The company operates through five main divisions:

1. Cloud Computing Division
   - Head: Dr. Sarah Kim, 20 years experience
   - Revenue: $2.5B (Q4 2024)
   - Employees: 8,500
   - Key Products: CloudScale Platform, MetaData Analytics, ServerlessOps

2. Artificial Intelligence Division  
   - Head: Prof. Ahmed Hassan, 18 years in AI research
   - Revenue: $1.8B (Q4 2024)
   - Employees: 6,200
   - Key Products: AI Assistant Pro, Computer Vision API, NLP Toolkit

3. Cybersecurity Division
   - Head: Lt. Col. Maria Santos, 25 years in cybersecurity
   - Revenue: $1.2B (Q4 2024)  
   - Employees: 4,800
   - Key Products: SecureShield Enterprise, ThreatDetect AI, CryptoVault

4. Mobile Technology Division
   - Head: David Chen, 15 years in mobile development
   - Revenue: $3.1B (Q4 2024)
   - Employees: 12,000
   - Key Products: MetaPhone OS, AppDeveloper Studio, Mobile Analytics

5. Research and Development Division
   - Head: Dr. Elena Volkov, 22 years in R&D
   - Budget: $800M annually
   - Employees: 3,500
   - Focus Areas: Quantum Computing, Biotech Integration, Space Technology

Financial Performance:
- Total Revenue 2024: $8.6B (18% YoY growth)
- Net Income: $1.9B (22% YoY growth)
- Operating Margin: 28.5%
- Market Capitalization: $45B
- Debt-to-Equity Ratio: 0.23

Global Offices:
- Headquarters: San Francisco, California, USA
- Major Offices: London (UK), Tokyo (Japan), Bangalore (India), São Paulo (Brazil)
- Total Facilities: 127 offices worldwide
- Total Employees: 34,000+

Strategic Partnerships:
- Microsoft: 5-year cloud infrastructure partnership ($500M)
- Amazon: Joint AI research initiative ($300M investment)
- IBM: Quantum computing collaboration (3-year program)
- Tesla: Autonomous vehicle AI partnership
- Stanford University: Research collaboration ($50M endowment)

Recent Acquisitions:
- DataFlow Analytics (Q1 2024): $850M acquisition
- SecureNet Solutions (Q3 2024): $1.2B acquisition  
- AIStudio Labs (Q4 2024): $650M acquisition

Regulatory and Compliance:
The company operates under various regulatory frameworks including GDPR, CCPA, and SOC 2 Type II.
Recent compliance initiatives include:
- Privacy by Design implementation across all products
- Ethical AI governance framework
- Carbon neutrality commitment by 2030
- Diversity and inclusion targets: 50% women in leadership by 2025
"""

VERY_UNSTRUCTURED_DOCUMENT = """
A Deep Dive into QuantumLeap Dynamics: Pioneering the Next Technological Frontier. This comprehensive memorandum provides an in-depth analysis of QuantumLeap Dynamics, a trailblazing deep-tech corporation headquartered in the innovation hub of Boston, Massachusetts. This is not just another tech company; it is an architect of future possibilities, operating at the complex and promising intersection of quantum computing, artificial intelligence, and advanced materials science. To understand QuantumLeap Dynamics is to understand the trajectory of modern technology itself. The firm was founded in 2012, born from the intellectual synergy of two exceptional minds: Dr. Evelyn Reed, a celebrated physicist in the quantum domain with an impressive fifteen years of dedicated research experience, and the equally distinguished materials scientist, Dr. Kenji Tanaka. Their partnership was founded on a shared conviction: that the world's most challenging problems demand solutions rooted in fundamental scientific discovery, not just incremental engineering. This core belief has permeated every facet of the company's operations and strategy from day one. From its headquarters in Boston, QuantumLeap Dynamics has methodically expanded its global influence. It has established crucial research and development centers in Zurich, Switzerland, a city known for its scientific prowess, and a strategic commercial office in Singapore, which serves as a gateway to the burgeoning Asian markets. This global distribution allows the company to attract premier talent from around the world and maintain a 24/7 innovation cycle. The company’s human capital is perhaps its greatest asset, with a total global workforce of around 350 individuals. This is no ordinary team; it is a curated assembly of the brightest minds in their respective fields, including a high concentration of PhDs, specialized engineers, and forward-thinking business strategists, all working in concert.

A review of the company's financial health paints a picture of a vibrant, rapidly growing enterprise. In its most recent fiscal year, QuantumLeap Dynamics posted impressive total revenues amounting to $110 million. Even more telling is its net income, which stood at a robust $20 million, a clear indicator of its strong operational leverage and the premium value of its intellectual property and product offerings. The investment community has taken notice, bestowing upon the company a current market capitalization of $2 billion. This valuation is not based on hype, but on a tangible and consistent year-over-year growth rate of 30%, a performance metric that significantly outshines most players in the tech sector. The company’s internal structure is a model of modern organizational design, meticulously crafted to maximize both innovation and operational efficiency. It is composed of three primary divisions, each functioning as a semi-autonomous unit with its own leadership and P&L responsibility, while also engaging in deep, synergistic collaboration on projects that cut across disciplines. At the heart of the company is the Quantum Computing Division, personally spearheaded by co-founder Dr. Evelyn Reed. This division is the company's crown jewel and its primary revenue engine, bringing in $50 million annually. With a dedicated team of 150 quantum physicists and engineers, it has successfully commercialized its flagship products: the 'Q-Processor', a revolutionary quantum computing chip, and the 'QuantumCloud' platform, which provides quantum computing as a service. These offerings are providing game-changing computational advantages to clients in sectors like pharmaceutical discovery, complex financial modeling, and national defense. Dr. Reed's 15 years of deep expertise have been the guiding force behind these breakthroughs.

The second key pillar is the AI Solutions Division, skillfully managed by David Chen, a respected AI visionary with twelve years of hands-on experience in building and deploying large-scale AI systems. His division, composed of 120 employees, is a significant contributor to the company's top line, generating $35 million in revenue. Its core products, 'Synapse AI', a sophisticated framework for developing and training neural networks, and 'Visionary AI', a cutting-edge platform for real-time computer vision analysis, are empowering businesses to unlock new efficiencies and capabilities. Rounding out the corporate structure is the Advanced Materials Division, led by the eminent Dr. Maria Petrova, who brings an unparalleled twenty years of experience to the table. This agile team of 80 scientists and engineers is responsible for some of the company’s most tangible innovations, including 'GrapheneMatrix', a composite material that redefines the standards for strength and weight, and 'AeroGel X', a novel insulating material with applications ranging from aerospace to green building. This division contributes $25 million in revenue and is pivotal to the company's long-term vision of seamlessly integrating its digital innovations with the physical world.

QuantumLeap Dynamics understands that no company, no matter how brilliant, can invent the future in isolation. Thus, it has woven a rich tapestry of strategic partnerships. A landmark collaboration is its $10 million joint research initiative with the Massachusetts Institute of Technology (MIT), which is focused on exploring the frontiers of quantum computing architecture. In the artificial intelligence arena, a strategic alliance with NVIDIA, the leader in accelerated computing, provides QuantumLeap with early access to next-generation hardware and facilitates co-development of specialized processors for AI workloads. Demonstrating its cross-industry appeal, the company has also forged a powerful partnership with aerospace titan Rolls-Royce. This collaboration is centered on testing and integrating 'GrapheneMatrix' and other advanced materials into the design of next-generation jet engines, a project that could revolutionize aviation efficiency and safety.

Growth has also been fueled by a shrewd and targeted acquisition strategy. In the third quarter of 2023, QuantumLeap Dynamics made a significant move by acquiring 'Photonics Innovations', a boutique firm specializing in quantum photonics, for a sum of $75 million. This was strategically followed in the first quarter of 2024 with the acquisition of 'AI Weaver', an innovative startup at the forefront of generative AI models, for $50 million. These are not mere roll-up acquisitions; they are surgical additions of key technologies and invaluable talent that have been rapidly and effectively integrated into the company’s existing divisions, creating immediate synergistic value.

In conclusion, the story of QuantumLeap Dynamics is a compelling narrative of bold ambition backed by rigorous execution. Its unique strategic posture, combining quantum computing, AI, and advanced materials, creates a virtuous cycle of innovation, where a breakthrough in one domain acts as a catalyst for progress in the others. The leadership team’s profound expertise and unwavering vision provide a steady hand on the tiller. Its global footprint is lean yet strategically placed in the world’s most critical innovation ecosystems. The financial metrics are not just strong; they are indicative of a durable, high-growth business model. Its partnerships with world-renowned academic and corporate institutions serve as both a validation of its technology and a powerful accelerant for its R&D pipeline. Its acquisition strategy has been both disciplined and synergistic. As industries across the globe confront the dual challenges of digital disruption and the approaching limits of classical computation, QuantumLeap Dynamics offers a new paradigm. Its technological offerings are not incremental improvements; they are foundational building blocks for the next economy. Whether it's accelerating drug discovery with quantum simulations, building more intelligent systems with its AI, or manufacturing the high-performance components of tomorrow with its advanced materials, QuantumLeap Dynamics is not just reacting to the future; it is actively writing its script. This document is for informational purposes only. Prospective investors should conduct their own thorough due diligence. The forward-looking statements herein are subject to a number of risks and uncertainties. Yet, the fundamental strengths of QuantumLeap Dynamics' technology, its people, and its strategy present a powerful argument for its potential to become a defining enterprise of the next technological age, making it an exceptionally attractive prospect for investors seeking to back truly world-changing companies. The company’s unwavering commitment to ethical innovation and sustainable practices further cements its long-term value and social license to operate. The masterful synthesis of these diverse yet deeply interconnected fields into a single, cohesive vision is what elevates QuantumLeap Dynamics from a mere company to a force of nature in the global technology landscape.
"""

SIMPLE_SCHEMA = {
    "type": "object",
    "required": ["company_name", "location"],
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Name of the company"
        },
        "location": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "state": {"type": "string"},
                "address": {"type": "string"}
            }
        },
        "leadership": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "title": {"type": "string"},
                    "experience_years": {"type": "integer"}
                }
            }
        }
    }
}

COMPLEX_RESEARCH_SCHEMA = {
    "type": "object",
    "required": ["title", "authors", "results"],
    "properties": {
        "title": {
            "type": "string",
            "description": "Title of the research paper"
        },
        "authors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "affiliation": {"type": "string"},
                    "department": {"type": "string"},
                    "experience_years": {"type": "integer"},
                    "specialization": {"type": "string"}
                },
                "required": ["name", "affiliation"]
            }
        },
        "methodology": {
            "type": "object",
            "properties": {
                "architecture_components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "component_name": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                },
                "approach": {"type": "string"}
            }
        },
        "results": {
            "type": "object",
            "properties": {
                "benchmarks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset": {"type": "string"},
                            "accuracy": {"type": "number"},
                            "previous_sota": {"type": "number"}
                        }
                    }
                }
            }
        },
        "funding": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "amount": {"type": "number"},
                    "duration": {"type": "string"}
                }
            }
        },
        "publication_info": {
            "type": "object",
            "properties": {
                "conference": {"type": "string"},
                "submission_date": {"type": "string"},
                "acceptance_date": {"type": "string"}
            }
        }
    }
}

ENTERPRISE_SCHEMA = {
    "type": "object",
    "required": ["company_name", "divisions", "financial_performance"],
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Name of the corporation"
        },
        "founding_info": {
            "type": "object",
            "properties": {
                "year": {"type": "integer"},
                "founders": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        },
        "divisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "head": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "experience_years": {"type": "integer"}
                        }
                    },
                    "revenue": {"type": "string"},
                    "employees": {"type": "integer"},
                    "products": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name"]
            }
        },
        "financial_performance": {
            "type": "object",
            "properties": {
                "total_revenue": {"type": "string"},
                "net_income": {"type": "string"},
                "market_cap": {"type": "string"},
                "growth_rate": {"type": "string"}
            }
        },
        "global_offices": {
            "type": "object",
            "properties": {
                "headquarters": {"type": "string"},
                "major_offices": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "total_employees": {"type": "integer"}
            }
        },
        "partnerships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "partner": {"type": "string"},
                    "type": {"type": "string"},
                    "value": {"type": "string"}
                }
            }
        },
        "acquisitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "date": {"type": "string"},
                    "value": {"type": "string"}
                }
            }
        }
    }
}

# =============================================================================
# Demo Functions
# =============================================================================

async def demo_simple_extraction():
    """Demo 1: Simple company profile extraction"""
    console.print(Panel("[bold blue]Demo 1: Simple Company Profile Extraction[/bold blue]"))
    
    system = DataTransformationSystem()
    start_time = time.time()
    
    result = await system.process_document(SIMPLE_COMPANY_DOCUMENT, SIMPLE_SCHEMA)
    
    processing_time = time.time() - start_time
    
    console.print(f"[green]✓ Processing completed in {processing_time:.2f} seconds[/green]")
    console.print(f"[cyan]Extracted {len(result.confidence_scores)} fields[/cyan]")
    console.print(f"[yellow]Average confidence: {sum(result.confidence_scores.values()) / len(result.confidence_scores):.2f}[/yellow]")
    
    return result

async def demo_complex_extraction():
    """Demo 2: Complex research paper extraction"""
    console.print(Panel("[bold blue]Demo 2: Complex Research Paper Extraction[/bold blue]"))
    
    system = DataTransformationSystem()
    start_time = time.time()
    
    result = await system.process_document(COMPLEX_RESEARCH_DOCUMENT, COMPLEX_RESEARCH_SCHEMA)
    
    processing_time = time.time() - start_time
    
    console.print(f"[green]✓ Processing completed in {processing_time:.2f} seconds[/green]")
    console.print(f"[cyan]Extracted {len(result.confidence_scores)} fields[/cyan]")
    
    if result.flagged_for_review:
        console.print(f"[red]⚠ {len(result.flagged_for_review)} fields flagged for human review[/red]")
    
    return result

async def demo_large_document():
    """Demo 3: Large enterprise document processing"""
    console.print(Panel("[bold blue]Demo 3: Large Enterprise Document Processing[/bold blue]"))
    
    system = DataTransformationSystem()
    start_time = time.time()
    
    result = await system.process_document(LARGE_COMPANY_DOCUMENT, ENTERPRISE_SCHEMA)
    
    processing_time = time.time() - start_time
    
    console.print(f"[green]✓ Processing completed in {processing_time:.2f} seconds[/green]")
    console.print(f"[cyan]Document chunks processed: {result.processing_metadata.get('total_chunks', 0)}[/cyan]")
    console.print(f"[cyan]Schema paths identified: {result.processing_metadata.get('schema_paths_count', 0)}[/cyan]")
    
    return result

async def demo_unstructured_extraction():
    """Demo 4: Highly unstructured text extraction"""
    console.print(Panel("[bold blue]Demo 4: Highly Unstructured Text Extraction[/bold blue]"))
    
    system = DataTransformationSystem()
    start_time = time.time()
    
    # We use the enterprise schema as it fits the narrative content
    result = await system.process_document(VERY_UNSTRUCTURED_DOCUMENT, ENTERPRISE_SCHEMA)
    
    processing_time = time.time() - start_time
    
    console.print(f"[green]✓ Processing completed in {processing_time:.2f} seconds[/green]")
    console.print(f"[cyan]Document chunks processed: {result.processing_metadata.get('total_chunks', 0)}[/cyan]")
    console.print(f"[cyan]Extracted {len(result.confidence_scores)} fields[/cyan]")
    
    if result.flagged_for_review:
        console.print(f"[red]⚠ {len(result.flagged_for_review)} fields flagged for human review[/red]")
    
    return result

def create_comparison_table(results: list):
    """Create a comparison table of different extraction results"""
    table = Table(title="Extraction Performance Comparison")
    
    table.add_column("Demo", style="cyan")
    table.add_column("Fields Extracted", style="green")
    table.add_column("Avg Confidence", style="yellow")
    table.add_column("Flagged Fields", style="red")
    table.add_column("Validation Errors", style="magenta")
    
    demo_names = ["Simple Company", "Complex Research", "Large Enterprise", "Unstructured Text"]
    
    for i, (name, result) in enumerate(zip(demo_names, results)):
        if result:
            avg_conf = sum(result.confidence_scores.values()) / len(result.confidence_scores) if result.confidence_scores else 0
            table.add_row(
                name,
                str(len(result.confidence_scores)),
                f"{avg_conf:.2f}",
                str(len(result.flagged_for_review)),
                str(len(result.validation_errors))
            )
        else:
            table.add_row(name, "Failed", "N/A", "N/A", "N/A")
    
    return table

async def save_demo_results(results: list):
    """Save all demo results to files"""
    output_dir = Path("demo_results")
    output_dir.mkdir(exist_ok=True)
    
    demo_names = ["simple_company", "complex_research", "large_enterprise", "unstructured_text"]
    
    for name, result in zip(demo_names, results):
        if result:
            output_file = output_dir / f"{name}_extraction.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "extracted_data": result.data,
                    "confidence_scores": result.confidence_scores,
                    "flagged_for_review": result.flagged_for_review,
                    "skeleton_nodes": len(result.skeleton),
                    "processing_metadata": result.processing_metadata,
                    "validation_errors": result.validation_errors
                }, f, indent=2, default=str)
            
            console.print(f"[green]✓ {name} results saved to {output_file}[/green]")

async def main():
    """Main demo execution"""
    console.print(Panel(
        "[bold green]Data Transformation System - Comprehensive Demo[/bold green]\n\n"
        "This demo showcases the system's capabilities across different document types and complexity levels."
    ))
    
    # Check configuration
    if not config.openai_api_key:
        console.print(Panel(
            "[bold yellow]⚠ Warning: OPENAI_API_KEY not set[/bold yellow]\n\n"
            "The system will use mock responses for demonstration purposes.\n"
            "For full functionality, configure your OpenAI API key:\n\n"
            "[cyan]Option 1 (Recommended):[/cyan]\n"
            "1. cp env_template.txt .env\n"
            "2. Edit .env and add your API key\n\n"
            "[cyan]Option 2:[/cyan]\n"
            "export OPENAI_API_KEY='your-api-key-here'"
        ))
    
    results = []
    
    try:
        # Run all demos
        console.print("\n[bold cyan]Running demonstration scenarios...[/bold cyan]\n")
        
        result1 = await demo_simple_extraction()
        results.append(result1)
        
        console.print("\n" + "="*80 + "\n")
        
        result2 = await demo_complex_extraction()
        results.append(result2)
        
        console.print("\n" + "="*80 + "\n")
        
        result3 = await demo_large_document()
        results.append(result3)
        
        console.print("\n" + "="*80 + "\n")
        
        result4 = await demo_unstructured_extraction()
        results.append(result4)
        
        # Display comparison
        console.print("\n" + "="*80 + "\n")
        console.print(Panel("[bold green]Performance Summary[/bold green]"))
        
        comparison_table = create_comparison_table(results)
        console.print(comparison_table)
        
        # Save results
        await save_demo_results(results)
        
        console.print(Panel(
            "[bold green]Demo completed successfully![/bold green]\n\n"
            "The system demonstrates:\n"
            "✓ Modular DSPy-inspired architecture\n"
            "✓ Two-pass skeleton-of-thought extraction\n"
            "✓ Semantic chunking with vector retrieval\n"
            "✓ Confidence estimation and human review flagging\n"
            "✓ Schema-constrained decoding approach\n"
            "✓ Comprehensive validation and error handling"
        ))
        
    except Exception as e:
        console.print(f"[red]Demo failed with error: {e}[/red]")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 