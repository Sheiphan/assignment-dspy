# Text-to-Structured Data Transformation Pipeline

## Overview

This repository contains a comprehensive analysis and implementation guide for building modular pipelines that transform unstructured text into deeply nested structured data. It includes framework comparisons, architectural patterns, practical decision matrices, and a working prototype.

## 📁 Repository Contents

### Core Documents

- **`data_transformation_pipeline_design.md`** - Comprehensive design document with framework analysis, architecture patterns, and implementation strategies
- **`framework_decision_matrix.md`** - Practical decision framework to choose optimal approaches based on specific requirements
- **`main.py`** - Complete implementation of the modular data transformation system
- **`demo.py`** - Comprehensive demo showcasing different document types and schemas
- **`config.py`** - Configuration management for the system
- **`pyproject.toml`** - Project dependencies and build configuration

## 🚀 Quick Start

### 1. Set Up Environment

```bash
# Clone and navigate to directory
cd assignment

# Install dependencies using uv (recommended) or pip
uv sync
# or
pip install -e .
```

### 2. Configure Environment

**Option A: Using .env file (Recommended)**
```bash
# Copy the template and customize
cp env_template.txt .env
# Edit .env with your API key and preferences
```

**Option B: Environment variables**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Run the System

```bash
# Run the main system with sample data
uv run python -m assignment.main

# Or run the comprehensive demo
uv run python -m assignment.demo
```

### 3. Use the Decision Framework

1. Read `framework_decision_matrix.md`
2. Answer the requirements assessment questions
3. Use the decision matrix to select your optimal stack
4. Follow the implementation roadmap

## 📊 Key Deliverables

### Framework Comparison Matrix

| Framework | Best For | Complexity | Strengths |
|-----------|----------|------------|-----------|
| **LangChain** | Quick prototyping | Low-Medium | Mature ecosystem, easy setup |
| **DSPy** | Complex workflows | Medium | Declarative, swappable components |
| **LlamaIndex** | Document-heavy apps | Medium | RAG integration, Pydantic support |
| **Haystack** | Production systems | High | End-to-end pipelines, validation |

### Architectural Patterns Covered

1. **Retrieval-Augmented Generation (RAG)** - For large document processing
2. **Two-Pass Extraction** - For complex schemas requiring planning
3. **Schema-First Planning** - For strict output requirements
4. **Token-Aware Compute Allocation** - For efficient resource usage

### Implementation Features

- ✅ **DSPy-inspired modular architecture** with clear separation of concerns
- ✅ **Two-pass Skeleton-of-Thought extraction** (skeleton → detailed population)
- ✅ **Semantic chunking** with recursive document splitting
- ✅ **Vector-based retrieval** using FAISS and OpenAI embeddings
- ✅ **Schema-constrained decoding** approach (JSONFormer-inspired)
- ✅ **Confidence estimation** with automatic human review flagging
- ✅ **JSON Schema flattening** for token-aware compute allocation
- ✅ **Comprehensive validation** against original schemas
- ✅ **Rich terminal output** with progress tracking and visualization
- ✅ **Asynchronous processing** for improved performance

## 🎯 Assignment Implementation

This prototype implements the **exact specifications** from the assignment:

### ✅ Research Foundations Implemented
- **DSPy**: Modular pipeline design with clear retrieval, planning, generation, and validation stages
- **JSONFormer**: Schema-constrained decoding approach to reduce hallucinations
- **Skeleton-of-Thought**: Two-pass extraction (skeleton → detailed population) with parallel processing

### ✅ Core Challenges Addressed
- **Deeply nested structured fields**: JSON Schema flattening with path-based extraction
- **Large, variable schema compliance**: Token-level analysis for compute allocation
- **Uncertain extraction flagging**: Confidence estimation with automatic human review routing
- **Adaptive processing strategy**: Dynamic approach based on input and schema complexity

### ✅ System Architecture Components
1. **Input Handling**: Semantic chunking + OpenAI embeddings + FAISS vector indexing
2. **Schema Loader & Planner**: JSON Schema flattening to path-based format with complexity scoring
3. **Two-Pass Pipeline**: Skeleton extraction → Field population with constrained decoding

## 🎯 Usage Scenarios

### For Assignment/Academic Work

1. **Run the system** to demonstrate all required capabilities in action
2. **Reference** the comprehensive design documents for theoretical analysis
3. **Use** the decision matrix to justify architectural choices
4. **Analyze** the trade-offs between different implementation approaches

### For Prototype Development

1. **Start** with the basic prototype structure
2. **Customize** the schema definitions for your domain
3. **Swap** the MockLLM with a real LLM provider
4. **Extend** with additional validation or processing steps

### For Production Planning

1. **Follow** the phased implementation roadmap
2. **Use** the framework suitability scores for technology selection
3. **Implement** the recommended architecture patterns
4. **Monitor** using the suggested success metrics

## 🔧 Customization Guide

### Adapting Schemas

Define your JSON schemas in the demo or main scripts:

```python
custom_schema = {
    "type": "object",
    "required": ["essential_field"],
    "properties": {
        "essential_field": {
            "type": "string",
            "description": "Critical business data"
        },
        "nested_structure": {
            "type": "object",
            "properties": {
                "sub_field": {"type": "string"},
                "array_field": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
}
```

### Configuring the System

**Option 1: Using .env file (Recommended)**
```bash
# Copy template and edit
cp env_template.txt .env
# Edit .env file with your settings
```

**Option 2: Environment variables**
```bash
export CONFIDENCE_THRESHOLD=0.8
export MAX_CHUNK_SIZE=1500
export VECTOR_SEARCH_TOP_K=10
```

**Option 3: Modify config.py directly**
```python
config.confidence_threshold = 0.8
config.max_chunk_size = 1500
```

The system loads configuration in this order: `.env file` → `environment variables` → `defaults`

### Adding Custom Document Processing

Extend the `SemanticChunker` for domain-specific chunking:

```python
class CustomChunker(SemanticChunker):
    def chunk_document(self, text: str) -> List[DocumentChunk]:
        # Your custom chunking logic
        return super().chunk_document(text)
```

### Extending Confidence Estimation

Add domain-specific confidence metrics in `ExtractionPipeline`:

```python
def _assess_field_confidence(self, field_path: str, value: Any) -> float:
    # Custom confidence logic for your domain
    if field_path.startswith("financial"):
        return self._financial_confidence(value)
    return self._default_confidence(value)
```

## 📈 Performance Optimization

### For High Accuracy (>95%)

- Use GPT-4 or Claude-3
- Implement two-pass extraction
- Add Guardrails validation
- Include human-in-the-loop for edge cases

### For Low Latency (<1s)

- Use constrained decoding (JSONFormer)
- Implement aggressive caching
- Consider smaller, specialized models
- Optimize prompt length

### For Large Scale (>10K docs)

- Implement RAG with vector search
- Use async/parallel processing
- Set up proper infrastructure (Haystack/DSPy)
- Monitor with evaluation frameworks

## 🧪 Testing and Evaluation

The prototype includes basic validation, but for production use:

1. **Set up DeepEval** for automated quality metrics
2. **Create test datasets** with ground truth data
3. **Implement A/B testing** for different approaches
4. **Monitor drift** in extraction quality over time

## 📚 Further Reading

### Recommended Papers
- "DSPy: Programming—not prompting—Foundation Models" (2023)
- "JSONFormer: A Bulletproof Way to Generate Structured JSON from Language Models" (2023)
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)

### Framework Documentation
- [LangChain Documentation](https://docs.langchain.com/)
- [DSPy GitHub](https://github.com/stanfordnlp/dspy)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [Haystack Documentation](https://docs.haystack.deepset.ai/)

## 🤝 Contributing

This framework is designed to be extensible. Key areas for enhancement:

1. **Additional LLM providers** (Anthropic, Cohere, local models)
2. **More sophisticated confidence estimation**
3. **Advanced RAG strategies** (hybrid search, reranking)
4. **Domain-specific validators**
5. **Performance optimizations**

## 📋 Next Steps

Based on your specific requirements:

1. **For immediate prototyping**: Start with the basic pipeline + LangChain
2. **For production deployment**: Follow the phased roadmap in the decision matrix
3. **For research/experimentation**: Use DSPy with multiple model comparisons
4. **For enterprise use**: Consider Haystack with full infrastructure setup

---

*This repository provides a foundation for building robust text-to-structured-data pipelines. Adapt the components based on your specific domain, performance requirements, and constraints.* 