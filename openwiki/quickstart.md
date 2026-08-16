---
type: overview
title: Quick Start
description: Getting started with Local PDF Chat RAG - navigation, concepts, and task routing
tags: [quickstart, navigation, getting-started]
---

# Quick Start

Welcome to the Local PDF Chat RAG documentation. This guide helps you quickly find what you need.

## About This Project

**Local PDF Chat RAG** is a transparent, educational implementation of Retrieval-Augmented Generation (RAG). It demonstrates the complete RAG pipeline with modular, replaceable components.

**Version**: 2.1.0

**Key Features**:
- **Inspectable Pipeline**: Core modules follow RAG execution order
- **Hybrid Retrieval**: FAISS dense + BM25 sparse search
- **Optional Reranking**: CrossEncoder or LLM-based
- **Multiple Model Backends**: Ollama, SiliconFlow, Magick API
- **Document Support**: PDF, TXT, Markdown, DOCX, XLS/XLSX, PPTX
- **Dual Interfaces**: Gradio Web UI and FastAPI REST API

## Quick Navigation

### For First-Time Users

<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
1. **[Configuration](/configuration/environment-and-models.md)** - Set up API keys and models
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
2. **[Operations](/operations/running-the-application.md)** - Start the application
<!-- openwiki: broken internal link [/interfaces/gradio-ui.md] file "/interfaces/gradio-ui.md" does not exist. Fix the href or restore the target, then delete this comment. -->
3. **[Gradio Web UI](/interfaces/gradio-ui.md)** - Use the web interface

### For Developers

<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
1. **[Development Guide](/development/development-guide.md)** - Contribution workflow
<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
2. **[Testing](/testing/overview.md)** - Test structure and coverage
<!-- openwiki: broken internal link [/architecture/overview.md] file "/architecture/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
3. **[Architecture Overview](/architecture/overview.md)** - System design

### For Maintenance

1. **[Core Modules](/core/)** - RAG pipeline implementation
2. **[Features](/features/)** - Extended capabilities
<!-- openwiki: broken internal link [/interfaces/rest-api.md] file "/interfaces/rest-api.md" does not exist. Fix the href or restore the target, then delete this comment. -->
3. **[REST API](/interfaces/rest-api.md)** - API reference

## Major Concepts

### RAG Pipeline

The system follows this processing flow:

```
Documents → Parse → Chunk → Embed → Store → Retrieve → Rerank → Generate
```

| Stage | Module | Purpose |
|-------|--------|---------|
| Document Loading | `core/document_loader.py` | Extract text from PDF, DOCX, etc. |
| Chunking | `core/text_splitter.py` | Split text into retrieval-friendly chunks |
| Embeddings | `core/embeddings.py` | Convert text to vectors |
| Vector Store | `core/vector_store.py` | FAISS index for dense retrieval |
| BM25 Index | `core/bm25_index.py` | BM25 index for keyword retrieval |
| Retrieval | `core/retriever.py` | Hybrid search + recursive retrieval |
| Reranking | `core/reranker.py` | Refine retrieval results |
| Generation | `core/generator.py` | Build prompt and generate answer |

### Hybrid Search

Combines semantic and keyword search:

```
Final Score = α × Semantic Score + (1-α) × BM25 Score
```

- **α = 0.7** (default): 70% semantic + 30% keyword
<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- Adjust in [`config.py`](/configuration/environment-and-models.md)

### Model Backends

| Backend | Configuration | Use Case |
|---------|---------------|----------|
| SiliconFlow | `SILICONFLOW_API_KEY` | Cloud, stable |
| Magick API | `MAGICK_API_KEY` | OpenAI-compatible |
| Ollama | `OLLAMA_MODEL_NAME` | Local, private |

## Task Routing

Find the right page for your task:

| Task | Page | Entry Point | Test |
|------|------|-------------|------|
<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Configure API keys** | [Environment and Models](/configuration/environment-and-models.md) | `config.py` | `test_config.py` |
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Start Gradio UI** | [Operations](/operations/running-the-application.md) | `rag_demo.py` | - |
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Start REST API** | [Operations](/operations/running-the-application.md) | `api_router.py` | `test_api_status.py` |
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Add document format** | [Development Guide](/development/development-guide.md) | `core/document_loader.py` | `test_document_loader.py` |
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Add LLM provider** | [Development Guide](/development/development-guide.md) | `core/generator.py` | (none) |
<!-- openwiki: broken internal link [/core/retrieval-and-reranking.md] file "/core/retrieval-and-reranking.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Adjust retrieval** | [Retrieval and Reranking](/core/retrieval-and-reranking.md) | `core/retriever.py` | `test_retrieval.py` |
<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Debug generation** | [Answer Generation](/core/generation.md) | `core/generator.py` | (none) |
<!-- openwiki: broken internal link [/features/web-search-and-conflict-detection.md] file "/features/web-search-and-conflict-detection.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Enable web search** | [Web Search](/features/web-search-and-conflict-detection.md) | `features/web_search.py` | (none) |
<!-- openwiki: broken internal link [/features/web-search-and-conflict-detection.md] file "/features/web-search-and-conflict-detection.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Fix conflicts** | [Conflict Detection](/features/web-search-and-conflict-detection.md) | `features/conflict_detector.py` | (none) |
<!-- openwiki: broken internal link [/features/thinking-chain.md] file "/features/thinking-chain.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Format thinking** | [Thinking Chain](/features/thinking-chain.md) | `features/thinking_chain.py` | (none) |
<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Run tests** | [Testing](/testing/overview.md) | `tests/` | All test files |
<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| **Add tests** | [Testing](/testing/overview.md) | `tests/` | - |

## API Reference

### Gradio Web UI

**File**: `rag_demo.py`

**Tabs**:
1. **问答对话** - Upload documents and ask questions
2. **分块可视化** - Inspect processed chunks
3. **系统监控** - System resource monitoring

**Entry Points**:
- `process_multiple_files()` - Document processing
- `query_answer()` - Question answering
- `get_document_chunks()` - Chunk visualization

### REST API

**File**: `api_router.py`

**Endpoints**:
- `GET /api/status` - System status
- `POST /api/upload` - Upload document
- `POST /api/ask` - Ask question

**Request Models**:
- `QuestionRequest` - Question with options
- `AnswerResponse` - Answer with sources
- `FileProcessResult` - Processing result

## Core Modules

### Document Processing

| Module | File | Purpose |
|--------|------|---------|
| Document Loader | `core/document_loader.py` | Parse PDF, DOCX, etc. |
| Text Splitter | `core/text_splitter.py` | Chunk text |
| Embeddings | `core/embeddings.py` | Vector encoding |

<!-- openwiki: broken internal link [/core/document-processing.md] file "/core/document-processing.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Document Processing](/core/document-processing.md)

### Vector Storage

| Module | File | Purpose |
|--------|------|---------|
| Vector Store | `core/vector_store.py` | FAISS index |
| BM25 Index | `core/bm25_index.py` | Keyword index |

<!-- openwiki: broken internal link [/core/embeddings-and-vector-store.md] file "/core/embeddings-and-vector-store.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Embeddings and Vector Store](/core/embeddings-and-vector-store.md)

### Retrieval

| Module | File | Purpose |
|--------|------|---------|
| Retriever | `core/retriever.py` | Hybrid + recursive search |
| Reranker | `core/reranker.py` | Result refinement |

<!-- openwiki: broken internal link [/core/retrieval-and-reranking.md] file "/core/retrieval-and-reranking.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Retrieval and Reranking](/core/retrieval-and-reranking.md)

### Generation

| Module | File | Purpose |
|--------|------|---------|
| Generator | `core/generator.py` | Prompt + LLM generation |

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Answer Generation](/core/generation.md)

## Features

| Feature | File | Purpose |
|---------|------|---------|
| Web Search | `features/web_search.py` | SerpAPI integration |
| Conflict Detection | `features/conflict_detector.py` | Multi-source conflict detection |
| Thinking Chain | `features/thinking_chain.py` | Thinking content formatting |

**See**: [Features Directory](/features/)

## Utilities

| Utility | File | Purpose |
|---------|------|---------|
| Network | `utils/network.py` | HTTP session with retry |

## Configuration

| Configuration | File | Purpose |
|---------------|------|---------|
| Environment | `config.py` | API keys, model params |
| Version | `version.py` | Version info |

<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Environment and Models](/configuration/environment-and-models.md)

## Testing

| Test File | Coverage |
|-----------|----------|
| `test_config.py` | Configuration, model selection |
| `test_document_loader.py` | Document extraction |
| `test_retrieval.py` | BM25, hybrid merge |
| `test_api_status.py` | API status endpoint |
| `test_no_key_fallback.py` | API key fallback |

<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
**See**: [Testing Overview](/testing/overview.md)

## Operations

| Topic | Page |
|-------|------|
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Running the Application | [Operations](/operations/running-the-application.md) |
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Deployment Considerations | [Operations](/operations/running-the-application.md) |
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Troubleshooting | [Operations](/operations/running-the-application.md) |

## Development

| Topic | Page |
|-------|------|
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Contribution Workflow | [Development Guide](/development/development-guide.md) |
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Coding Conventions | [Development Guide](/development/development-guide.md) |
<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Testing | [Testing](/testing/overview.md) |

## Architecture

| Topic | Page |
|-------|------|
<!-- openwiki: broken internal link [/architecture/overview.md] file "/architecture/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| System Overview | [Architecture](/architecture/overview.md) |
<!-- openwiki: broken internal link [/architecture/overview.md] file "/architecture/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Data Flow | [Architecture](/architecture/overview.md) |
<!-- openwiki: broken internal link [/architecture/overview.md] file "/architecture/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
| Design Decisions | [Architecture](/architecture/overview.md) |

## Known Limitations

- **No OCR**: PDF text extraction only, no image recognition
- **In-Memory State**: No persistence across restarts
- **Single Instance**: No horizontal scaling support
- **No Authentication**: Open access by default

<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
See [Operations](/operations/running-the-application.md) for deployment considerations.

## Getting Help

1. **Check Documentation**: Browse this wiki
2. **Read Source Code**: Modules have detailed docstrings
3. **Run Tests**: See expected behavior
4. **Open Issue**: On GitHub (for bugs/features)

## Next Steps

<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- **New User**: Start with [Configuration](/configuration/environment-and-models.md), then [Operations](/operations/running-the-application.md)
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- **Developer**: Read [Development Guide](/development/development-guide.md)
<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- **Contributor**: Review [Testing](/testing/overview.md) and [Development Guide](/development/development-guide.md)

## Related Documentation

- [README.md](https://github.com/weiwill88/Local_Pdf_Chat_RAG) - Project overview
- [CONTRIBUTING.md](https://github.com/weiwill88/Local_Pdf_Chat_RAG) - Contribution guidelines
- [CHANGELOG.md](https://github.com/weiwill88/Local_Pdf_Chat_RAG) - Version history
