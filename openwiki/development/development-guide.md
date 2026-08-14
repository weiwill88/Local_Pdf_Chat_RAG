---
type: overview
title: Development Guide
description: Contribution workflow, coding conventions, and development setup
tags: [development, contributing, workflow]
---

# Development Guide

This document covers contribution workflow, coding conventions, and development setup.

## Getting Started

### Development Environment

```bash
# Clone repository
git clone https://github.com/weiwill88/Local_Pdf_Chat_RAG.git
cd Local_Pdf_Chat_RAG

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (including dev)
pip install -r requirements-dev.txt
```

### Project Structure

```
Local_Pdf_Chat_RAG/
├── config.py              # Configuration and environment
├── rag_demo.py            # Gradio Web UI entry point
├── api_router.py          # FastAPI REST API entry point
├── version.py             # Version information
├── core/                  # Core RAG modules
│   ├── document_loader.py # Document parsing
│   ├── text_splitter.py   # Text chunking
│   ├── embeddings.py      # Vector embeddings
│   ├── vector_store.py    # FAISS index
│   ├── bm25_index.py      # BM25 index
│   ├── retriever.py       # Hybrid retrieval
│   ├── reranker.py        # Result reranking
│   └── generator.py       # LLM generation
├── features/              # Extended features
│   ├── web_search.py      # Web search integration
│   ├── conflict_detector.py # Conflict detection
│   └── thinking_chain.py  # Thinking content formatting
├── utils/                 # Utilities
│   └── network.py         # Network utilities
├── tests/                 # Test suite
│   ├── test_config.py
│   ├── test_document_loader.py
│   ├── test_retrieval.py
│   ├── test_api_status.py
│   └── test_no_key_fallback.py
├── .github/               # GitHub configuration
│   ├── workflows/         # CI/CD pipelines
│   └── ISSUE_TEMPLATE/    # Issue templates
└── openwiki/              # Generated documentation
```

## Contribution Workflow

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/Local_Pdf_Chat_RAG.git
cd Local_Pdf_Chat_RAG
git remote add upstream https://github.com/weiwill88/Local_Pdf_Chat_RAG.git
```

### 2. Create Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Changes

Follow coding conventions (see below).

### 4. Run Tests

```bash
python -m pytest -v
```

### 5. Commit

```bash
git add .
git commit -m "feat: add new feature"
# or
git commit -m "fix: resolve bug"
```

### 6. Push and Pull Request

```bash
git push origin feature/your-feature-name
# Create PR on GitHub
```

## Coding Conventions

### Code Style

**Python Style Guide**: PEP 8

**Key Conventions**:

1. **Imports**: Group and order
```python
# Standard library
import os
import logging

# Third-party
import requests
from dotenv import load_dotenv

# Local
from config import CHUNK_SIZE
from core.document_loader import extract_text
```

2. **Naming**:
   - Functions: `snake_case`
   - Classes: `PascalCase`
   - Constants: `UPPER_CASE`
   - Private: `_leading_underscore`

3. **Docstrings**: All public functions and classes

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Detailed description if needed.
    
    Args:
        param1: Description
        param2: Description
    
    Returns:
        Description
    """
```

### Documentation

**Inline Comments**:
- Explain "why", not "what"
- Keep concise
- Update when code changes

**Module Docstrings**:
Each module should have a docstring explaining:
- Purpose
- Key functions
- Dependencies

```python
"""
Module name

Brief description.

Key functions:
- function1: Description
- function2: Description
"""
```

### Testing

**Test Naming**:
```python
def test_function_name_specific_scenario():
    # Arrange
    # Act
    # Assert
    pass
```

**Test Coverage**:
- Add tests for new functionality
- Maintain or improve coverage
- Test edge cases

### Error Handling

**Patterns**:

1. **Explicit validation**:
```python
if not api_key:
    logging.error("API key not configured")
    return "Error: API key required"
```

2. **Exception handling**:
```python
try:
    result = risky_operation()
except SpecificError as e:
    logging.error(f"Operation failed: {e}")
    return f"Error: {e}"
```

3. **Graceful degradation**:
```python
try:
    return primary_implementation()
except Exception:
    logging.warning("Primary failed, using fallback")
    return fallback_implementation()
```

## Development Patterns

### RAG Pipeline Pattern

Modules follow RAG processing order:

```
Document → Chunk → Embed → Store → Retrieve → Rerank → Generate
```

**Adding a new module**:
1. Determine position in pipeline
2. Define clear input/output interfaces
3. Follow existing module patterns
4. Add tests

### Singleton Pattern

Used for shared state (vector store, BM25 index):

```python
class SingletonClass:
    def __init__(self):
        self.state = {}

# Module-level singleton
singleton_instance = SingletonClass()
```

### Lazy Loading Pattern

Used for expensive resources (models):

```python
_model = None
_model_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = load_expensive_model()
    return _model
```

### Progress Callback Pattern

Used for long-running operations:

```python
def long_operation(progress=None):
    progress(0.1, desc="Starting...")
    # ... work ...
    progress(0.5, desc="Midpoint...")
    # ... work ...
    progress(1.0, desc="Complete")
```

## Feature Development

### Adding New Document Format

1. **Update `core/document_loader.py`**:
```python
elif file_ext == '.newext':
    try:
        from newlib import NewParser
        parser = NewParser(filepath)
        return parser.extract_text()
    except ImportError:
        logging.error("Install newlib to process .newext files")
        return ""
```

2. **Add dependency to `requirements.txt`**:
```
newlib>=1.0.0
```

3. **Add test in `tests/test_document_loader.py`**:
```python
def test_newext_extraction(tmp_path):
    file = tmp_path / "sample.newext"
    file.write_text("Test content")
    assert "Test content" in extract_text(str(file))
```

### Adding New LLM Provider

1. **Add configuration in `config.py`**:
```python
NEW_PROVIDER_API_KEY = os.getenv("NEW_PROVIDER_API_KEY")
NEW_PROVIDER_API_URL = os.getenv("NEW_PROVIDER_API_URL")
NEW_PROVIDER_MODEL_NAME = os.getenv("NEW_PROVIDER_MODEL_NAME")
```

2. **Add API caller in `core/generator.py`**:
```python
def call_new_provider_api(prompt, temperature=0.7, max_tokens=1024):
    return _call_openai_compatible_api(
        "New Provider",
        NEW_PROVIDER_API_KEY,
        NEW_PROVIDER_API_URL,
        NEW_PROVIDER_MODEL_NAME,
        prompt,
        temperature,
        max_tokens
    )
```

3. **Update `call_cloud_api()`**:
```python
if model_choice == "new_provider":
    return call_new_provider_api(prompt, temperature, max_tokens)
```

4. **Update model choices**:
```python
MODEL_CHOICES = ["ollama", "siliconflow", "magick", "new_provider"]
```

### Adding New Retrieval Method

1. **Implement in new module or extend `core/retriever.py`**

2. **Update `hybrid_merge()` or create new merge function**

3. **Add configuration parameter**:
```python
NEW_RETRIEVAL_WEIGHT = os.getenv("NEW_RETRIEVAL_WEIGHT", "0.1")
```

4. **Add tests**

## Code Review

### Checklist

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New code has tests
- [ ] Documentation updated
- [ ] No hardcoded credentials
- [ ] Error handling adequate
- [ ] Logging appropriate
- [ ] Performance acceptable
- [ ] No unnecessary dependencies

### Review Process

1. Author submits PR
2. Reviewer assigns
3. Reviewer provides feedback
4. Author addresses feedback
5. Reviewer approves
6. PR merged

## CI/CD

### GitHub Actions

**Trigger**: Push or PR to `main`

**Jobs**:
1. **Test**: Run pytest with coverage
2. **Lint**: Check code style (optional)
3. **Build**: Verify compilation

### Local Pre-commit

```bash
# Run tests before commit
python -m pytest

# Optional: Add pre-commit hook
echo "python -m pytest && git add ." > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Version Management

### Version File

**Location**: `version.py`

```python
__version__ = "2.1.0"
```

### Versioning Scheme

**Semantic Versioning**: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Process

1. Update `version.py`
2. Update `CHANGELOG.md`
3. Create git tag
4. Push tag
5. GitHub Actions creates release

## Debugging

### Logging

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
logger.critical("Critical message")
```

### Debug Mode

```bash
# Set debug environment variable
export DEBUG=1

# Or modify config
logging.basicConfig(level=logging.DEBUG)
```

### Interactive Debugging

```python
import pdb; pdb.set_trace()  # Breakpoint
```

## Related Components

<!-- openwiki: broken internal link [/testing/overview.md] file "/testing/overview.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Testing](/testing/overview.md) - Test conventions
<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Configuration](/configuration/environment-and-models.md) - Config patterns
<!-- openwiki: broken internal link [/operations/running-the-application.md] file "/operations/running-the-application.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Operations](/operations/running-the-application.md) - Deployment
