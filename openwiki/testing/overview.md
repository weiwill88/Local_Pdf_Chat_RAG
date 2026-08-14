---
type: overview
title: Testing
description: Test structure, coverage, and validation strategies
tags: [testing, validation, quality]
---

# Testing

This document covers the test structure, coverage, and validation strategies for the Local PDF Chat RAG system.

## Test Structure

### Test Files

**Location**: `tests/` directory

| File | Purpose | Modules Covered |
|------|---------|-----------------|
| `test_config.py` | Configuration and model selection | `config.py` |
| `test_document_loader.py` | Document extraction | `core/document_loader.py` |
| `test_retrieval.py` | BM25 and hybrid search | `core/bm25_index.py`, `core/retriever.py` |
| `test_api_status.py` | API endpoint health | `api_router.py` |
| `test_no_key_fallback.py` | API key fallback behavior | `config.py`, `core/generator.py` |

### Running Tests

**Location**: `pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

**Command**:
```bash
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

**CI Command** (from `.github/workflows/ci.yml`):
```bash
python -m pytest -p pytest_cov.plugin --cov=core --cov=config --cov-report=term-missing
```

## Test Coverage

### Covered Modules

| Module | Test File | Coverage |
|--------|-----------|----------|
| `config.py` | `test_config.py` | API key validation, model selection |
| `core/document_loader.py` | `test_document_loader.py` | TXT/MD extraction, unsupported formats |
| `core/bm25_index.py` | `test_retrieval.py` | BM25 search, keyword matching |
| `core/retriever.py` | `test_retrieval.py` | Hybrid merge algorithm |
| `api_router.py` | `test_api_status.py` | Status endpoint |

### Test Coverage Gaps

The following modules have **no dedicated tests**:

| Module | Gap | Risk |
|--------|-----|------|
| `core/embeddings.py` | No tests for encoding | Model loading failures undetected |
| `core/vector_store.py` | No tests for FAISS operations | Index corruption undetected |
| `core/text_splitter.py` | No tests for chunking | Chunk quality unverified |
| `core/reranker.py` | No tests for reranking | Rerank failures undetected |
| `core/generator.py` | No tests for generation | Prompt/LLM issues undetected |
| `features/web_search.py` | No tests for web search | Search failures undetected |
| `features/conflict_detector.py` | No tests for conflict detection | False positives/negatives |
| `features/thinking_chain.py` | No tests for thinking processing | HTML injection risk |
| `utils/network.py` | No tests for network utilities | Network issues undetected |
| `rag_demo.py` | No UI tests | UI regressions undetected |

## Test Details

### test_config.py

**Location**: `tests/test_config.py`

#### `test_api_key_validation_rejects_placeholders()`

**Purpose**: Verify API key validation rejects placeholder values

**Test Cases**:
```python
assert config.is_configured_api_key(None) is False
assert config.is_configured_api_key("") is False
assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
assert config.is_configured_api_key("sk-real-value") is True
```

**Coverage**: Placeholder detection logic (config.py line 61-63)

#### `test_default_model_selection_order()`

**Purpose**: Verify model selection priority order

**Test Cases**:
```python
assert config.choose_default_model("sk-silicon", "sk-magick", True) == "siliconflow"
assert config.choose_default_model(None, "sk-magick", True) == "magick"
assert config.choose_default_model(None, None, True) == "ollama"
assert config.choose_default_model(None, None, False) == "siliconflow"
```

**Coverage**: Model priority logic (config.py line 66-75)

### test_document_loader.py

**Location**: `tests/test_document_loader.py`

#### `test_extract_utf8_text_and_markdown()`

**Purpose**: Verify UTF-8 text and Markdown extraction

**Setup**:
```python
text_file = tmp_path / "sample.txt"
text_file.write_text("RAG combines retrieval and generation.\n中文内容。", encoding="utf-8")

markdown_file = tmp_path / "sample.md"
markdown_file.write_text("# Heading\n\nHybrid retrieval", encoding="utf-8")
```

**Assertions**:
```python
assert "中文内容" in extract_text(str(text_file))
assert "Hybrid retrieval" in extract_text(str(markdown_file))
```

**Coverage**: TXT/MD extraction (document_loader.py lines 35-37)

#### `test_unsupported_extension_returns_empty_string()`

**Purpose**: Verify unsupported formats return empty string

**Setup**:
```python
unsupported = tmp_path / "sample.bin"
unsupported.write_bytes(b"not a supported document")
```

**Assertion**:
```python
assert extract_text(str(unsupported)) == ""
```

**Coverage**: Unsupported format handling (document_loader.py lines 76-78)

### test_retrieval.py

**Location**: `tests/test_retrieval.py`

#### `test_bm25_returns_relevant_document_first()`

**Purpose**: Verify BM25 returns keyword-matching document first

**Setup**:
```python
manager = BM25IndexManager()
manager.build_index(
    ["FAISS supports dense vector retrieval",
     "BM25 supports exact keyword retrieval",
     "RAG combines retrieval with generation"],
    ["dense", "sparse", "rag"]
)
```

**Assertion**:
```python
results = manager.search("BM25 keyword", top_k=2)
assert results
assert results[0]["id"] == "sparse"  # Keyword match ranks first
```

**Coverage**: BM25 search (bm25_index.py lines 40-57)

#### `test_hybrid_merge_combines_semantic_and_sparse_scores()`

**Purpose**: Verify hybrid search combines scores correctly

**Setup**:
```python
semantic = {
    "ids": [["doc-a", "doc-b"]],
    "documents": [["semantic result", "shared result"]],
    "metadatas": [[{"source": "a"}, {"source": "b"}]],
}
sparse = [
    {"id": "doc-b", "score": 4.0, "content": "shared result"},
    {"id": "doc-c", "score": 2.0, "content": "keyword result"},
]
```

**Assertions**:
```python
merged = hybrid_merge(semantic, sparse, alpha=0.5)
merged_by_id = dict(merged)

assert set(merged_by_id) == {"doc-a", "doc-b", "doc-c"}
assert merged[0][0] == "doc-b"  # doc-b wins with combined score
assert merged_by_id["doc-b"]["score"] > merged_by_id["doc-a"]["score"]
```

**Coverage**: Hybrid merge algorithm (retriever.py lines 19-76)

### test_api_status.py

**Location**: `tests/test_api_status.py`

#### `test_status_endpoint()`

**Purpose**: Verify status endpoint returns valid response

**Setup**:
```python
from fastapi.testclient import TestClient
from api_router import app

client = TestClient(app)
response = client.get("/api/status")
```

**Assertions**:
```python
assert response.status_code == 200
data = response.json()
assert "status" in data
assert "version" in data
assert "model_configured" in data
```

**Coverage**: Status endpoint (api_router.py lines 103-125)

### test_no_key_fallback.py

**Location**: `tests/test_no_key_fallback.py`

#### `test_cloud_provider_fails_cleanly_without_api_key()`

**Purpose**: Verify no HTTP request is made when API key is missing

**Setup**:
```python
from core.generator import _call_openai_compatible_api

def unexpected_network_call(*args, **kwargs):
    raise AssertionError("a missing API key must not trigger an HTTP request")

monkeypatch.setattr("core.generator.requests.post", unexpected_network_call)
```

**Test**:
```python
result = _call_openai_compatible_api(
    provider_name="Test Provider",
    api_key=None,
    api_url="https://example.com/v1",
    model_name="example-model",
    prompt="hello",
)
```

**Assertion**:
```python
assert "未配置 Test Provider API Key" in result
```

**Coverage**: No-network fallback when API key missing (generator.py lines 59-64)

#### `test_no_api_key_fallback_returns_error_message()`

**Purpose**: Verify graceful handling when API key not configured

**Setup**:
```python
from config import is_configured_api_key
assert is_configured_api_key(None) is False
```

**Coverage**: API key validation fallback

## CI/CD Integration

### GitHub Actions

**Location**: `.github/workflows/ci.yml`

**Trigger**:
- Push to `main` branch
- Pull requests to `main` branch

**Job**:
```yaml
name: CI
runs-on: ubuntu-latest
timeout-minutes: 20

steps:
  - Checkout repository
  - Set up Python 3.10
  - Install dependencies
  - Compile Python sources
  - Run tests with coverage
```

**Coverage Report**:
```bash
--cov=core --cov=config --cov-report=term-missing
```

## Test Utilities

### Temporary Files

Tests use pytest's `tmp_path` fixture for temporary file creation:

```python
def test_extract_utf8_text_and_markdown(tmp_path: Path):
    text_file = tmp_path / "sample.txt"
    text_file.write_text("Content", encoding="utf-8")
    # ... test ...
```

### TestClient

FastAPI's TestClient for API testing:

```python
from fastapi.testclient import TestClient
from api_router import app

client = TestClient(app)
response = client.get("/api/status")
```

## Change Recipes

### Adding a New Test

1. Create test file in `tests/`:
```python
# tests/test_new_module.py
def test_new_functionality():
    # Arrange
    input_data = ...
    
    # Act
    result = new_function(input_data)
    
    # Assert
    assert result == expected
```

2. Run tests:
```bash
python -m pytest tests/test_new_module.py -v
```

### Adding Test Coverage for Untested Module

1. Identify module to test (e.g., `core/embeddings.py`)
2. Create test file:
```python
# tests/test_embeddings.py
from core.embeddings import encode_texts, encode_query, get_embed_model

def test_encode_texts_returns_numpy_array():
    texts = ["Hello", "World"]
    embeddings = encode_texts(texts)
    
    assert embeddings.shape == (2, 384)  # 2 texts, 384 dimensions
    assert embeddings.dtype == np.float32

def test_encode_query_returns_2d_array():
    query = "Test query"
    embedding = encode_query(query)
    
    assert embedding.shape == (1, 384)
```

3. Run and verify:
```bash
python -m pytest tests/test_embeddings.py -v
```

### Increasing Coverage Threshold

Modify CI configuration to require higher coverage:

```yaml
# .github/workflows/ci.yml
- name: Run tests
  run: >-
    python -m pytest --cov=core --cov=config
    --cov-report=term-missing --cov-fail-under=80
```

### Adding Integration Tests

Create integration tests that test multiple modules together:

```python
# tests/test_integration.py
def test_full_rag_pipeline(tmp_path):
    # 1. Create test document
    doc_file = tmp_path / "test.txt"
    doc_file.write_text("RAG combines retrieval and generation", encoding="utf-8")
    
    # 2. Process document
    from core.document_loader import extract_text
    from core.text_splitter import split_text
    from core.embeddings import encode_texts
    from core.vector_store import VectorStore
    
    text = extract_text(str(doc_file))
    chunks = split_text(text)
    embeddings = encode_texts(chunks)
    
    store = VectorStore()
    store.build_index(chunks, ["chunk_0"], [{"source": "test.txt"}], embeddings)
    
    # 3. Verify index built
    assert store.is_ready
    assert store.total_chunks == 1
    
    # 4. Search
    from core.embeddings import encode_query
    from core.retriever import hybrid_merge
    from core.bm25_index import BM25IndexManager
    
    query_emb = encode_query("retrieval")
    docs, ids, metas = store.search(query_emb, k=1)
    
    assert "retrieval" in docs[0]
```

## Quality Metrics

### Current Coverage

Based on test files:
- **config.py**: ~80% (key functions covered)
- **core/document_loader.py**: ~40% (TXT/MD only)
- **core/bm25_index.py**: ~60% (core search covered)
- **core/retriever.py**: ~50% (hybrid merge covered)
- **api_router.py**: ~20% (status only)
- **Other modules**: 0% (no tests)

### Recommended Priority

1. **High Priority**: `core/generator.py`, `core/vector_store.py`
2. **Medium Priority**: `core/embeddings.py`, `core/reranker.py`
3. **Lower Priority**: `features/`, `utils/`

## Related Components

<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Configuration](/configuration/environment-and-models.md) - Test configuration
<!-- openwiki: broken internal link [/interfaces/rest-api.md] file "/interfaces/rest-api.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [REST API](/interfaces/rest-api.md) - API testing patterns
<!-- openwiki: broken internal link [/development/development-guide.md] file "/development/development-guide.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Development Guide](/development/development-guide.md) - Testing conventions
