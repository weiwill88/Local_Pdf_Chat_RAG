---
type: configuration
title: Environment and Models
description: Environment variables, model configuration, and RAG hyperparameters
tags: [configuration, environment, models]
---

# Environment and Models

This document covers environment configuration, model selection, and RAG hyperparameters.

## Configuration File

### Environment Loading

**Location**: `config.py` lines 10-24

```python
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)
```

### Configuration Priority

1. `.env` file (user configuration) - highest priority
2. `example.env` (example configuration)
3. Environment variables from system
4. Default values in code

### Example Configuration

**File**: `example.env`

```env
SERPAPI_KEY=Your SERPAPI_KEY
SILICONFLOW_API_KEY=
MAGICK_API_KEY=
MAGICK_API_URL=https://api.magickapi.com/v1/chat/completions

# Model configuration (optional, uses defaults if not set)
# OLLAMA_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# SILICONFLOW_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# MAGICK_MODEL_NAME=gpt-4o-mini
```

## API Keys

### SiliconFlow

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SILICONFLOW_API_KEY` | Optional | Not set | SiliconFlow API authentication key |
| `SILICONFLOW_API_URL` | Optional | `https://api.siliconflow.cn/v1/chat/completions` | API endpoint URL |
| `SILICONFLOW_MODEL_NAME` | Optional | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | Model identifier |

### Magick API

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MAGICK_API_KEY` | Optional | Not set | Magick API authentication key |
| `MAGICK_API_URL` | Optional | `https://api.magickapi.com/v1/chat/completions` | API endpoint URL |
| `MAGICK_MODEL_NAME` | Optional | `gpt-4o-mini` | Model identifier |

### Ollama

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OLLAMA_MODEL_NAME` | Optional | `deepseek-r1:8b` | Local Ollama model name |

**Note**: Ollama requires a running local service at `http://localhost:11434`.

### SerpAPI (Web Search)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `SERPAPI_KEY` | Optional | Not set | SerpAPI authentication key for web search |

### API Key Validation

**Location**: `config.py` lines 61-63

```python
def is_configured_api_key(api_key):
    """Determine if API key is an actual configured value."""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))
```

**Validation Rules**:
- Key must not be `None` or empty
- Key must not start with "Your" (placeholder detection)
- Leading/trailing whitespace is stripped

## Model Selection

### Model Auto-Detection

**Location**: `config.py` lines 98-125

```python
def detect_default_model():
    """
    Auto-detect available LLM backend, return default model choice.
    
    Priority order:
    1. SiliconFlow (if API key configured)
    2. Magick API (if API key configured)
    3. Ollama (if local service available)
    4. Default to siliconflow (for UI stability)
    """
```

**Detection Logic**:
1. Checks `SILICONFLOW_API_KEY` with `is_configured_api_key()`
2. Checks `MAGICK_API_KEY` with `is_configured_api_key()`
3. Attempts Ollama connectivity check
4. Falls back to "siliconflow" if none available

### Default Model Selection

**Location**: `config.py` lines 66-75

```python
def choose_default_model(siliconflow_key, magick_key, ollama_available=False):
    """Select default model backend by stability and testability priority."""
    if is_configured_api_key(siliconflow_key):
        return "siliconflow"
    if is_configured_api_key(magick_key):
        return "magick"
    if ollama_available:
        return "ollama"
    return "siliconflow"  # Keep UI default stable
```

**Priority Order**:
1. SiliconFlow (cloud, stable)
2. Magick API (cloud, OpenAI-compatible)
3. Ollama (local, requires running service)
4. Default to SiliconFlow configuration (for UI stability)

### Model Display Names

**Location**: `config.py` lines 54-58

```python
MODEL_DISPLAY_NAMES = {
    "ollama": "本地 Ollama 模型",
    "siliconflow": "Cloud DeepSeek-R1 模型",
    "magick": "Magick API 模型"
}
```

### Model Choices

**Location**: `config.py` line 53

```python
MODEL_CHOICES = ["ollama", "siliconflow", "magick"]
```

These choices populate the Gradio UI dropdown and are validated in API requests.

## RAG Hyperparameters

### Document Processing

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `CHUNK_SIZE` | 400 | `config.py` line 80 | Maximum characters per text chunk |
| `CHUNK_OVERLAP` | 40 | `config.py` line 81 | Overlap between adjacent chunks |

**Trade-offs**:
- **Larger CHUNK_SIZE**: More context per chunk, coarser retrieval granularity
- **Smaller CHUNK_SIZE**: Finer retrieval, may lose context
- **Larger CHUNK_OVERLAP**: Better context preservation, more duplicates
- **Smaller CHUNK_OVERLAP**: Less redundancy, may cut important info at boundaries

### Retrieval

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `HYBRID_ALPHA` | 0.7 | `config.py` line 82 | Weight for semantic vs keyword search |
| `RETRIEVAL_TOP_K` | 10 | `config.py` line 83 | Initial retrieval candidates |
| `RERANK_TOP_K` | 5 | `config.py` line 84 | Final results after reranking |
| `MAX_RETRIEVAL_ITERATIONS` | 3 | `config.py` line 85 | Max recursive retrieval iterations |

**Trade-offs**:
- **HYBRID_ALPHA = 0.7**: 70% semantic + 30% keyword
- **Higher alpha**: More emphasis on semantic similarity
- **Lower alpha**: More emphasis on exact keyword matching
- **Larger TOP_K**: More candidates, slower processing
- **More iterations**: Better recall, slower response

### Reranking

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `RERANK_METHOD` | "cross_encoder" | `config.py` line 51 | Reranking algorithm |

**Options**:
- `"cross_encoder"`: Uses sentence-transformers CrossEncoder (faster, good precision)
- `"llm"`: Uses LLM for relevance scoring (slower, very high precision)

## Runtime Environment Configuration

### Environment Variables

**Location**: `config.py` lines 90-93

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # Hugging Face mirror
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable TensorFlow optimizations
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'  # Proxy bypass for local
requests.adapters.DEFAULT_RETRIES = 3  # HTTP retry count
```

### Purpose

- **HF_ENDPOINT**: Use Chinese mirror for Hugging Face model downloads
- **TF_ENABLE_ONEDNN_OPTS**: Disable TensorFlow-specific optimizations
- **NO_PROXY**: Bypass proxy for localhost connections
- **DEFAULT_RETRIES**: HTTP request retry count

## Version Information

**Location**: `version.py` line 2

```python
__version__ = "2.1.0"
```

Version is shared across UI, API, and release tooling.

## Configuration Examples

### Minimal Configuration (SiliconFlow)

```env
SILICONFLOW_API_KEY=sk-your-actual-key-here
```

### Minimal Configuration (Magick API)

```env
MAGICK_API_KEY=sk-your-actual-key-here
MAGICK_API_URL=https://api.magickapi.com/v1/chat/completions
```

### Minimal Configuration (Ollama)

```env
# No API key needed for local Ollama
OLLAMA_MODEL_NAME=deepseek-r1:8b
```

### Full Configuration

```env
# API Keys
SILICONFLOW_API_KEY=sk-your-siliconflow-key
MAGICK_API_KEY=sk-your-magick-key
SERPAPI_KEY=your-serpapi-key

# Model Names
SILICONFLOW_MODEL_NAME=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MAGICK_MODEL_NAME=gpt-4o-mini
OLLAMA_MODEL_NAME=deepseek-r1:8b

# RAG Parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=50
HYBRID_ALPHA=0.6
RERANK_METHOD=cross_encoder
```

## Change Recipes

### Adding New Environment Variable

1. Add to `config.py`:
```python
NEW_VARIABLE = os.getenv("NEW_VARIABLE", "default_value")
```

2. Add to `example.env`:
```env
NEW_VARIABLE=default_value
```

3. Document in this file

### Changing Default Model

1. Modify `choose_default_model()` priority order
2. Update `MODEL_DISPLAY_NAMES` if needed
3. Test with each backend

### Adjusting RAG Parameters

1. Modify values in `config.py`
2. Consider trade-offs (see tables above)
3. Test with representative queries
4. Document changes

### Adding Custom Model Provider

1. Add configuration variables:
```python
CUSTOM_API_KEY = os.getenv("CUSTOM_API_KEY")
CUSTOM_API_URL = os.getenv("CUSTOM_API_URL")
CUSTOM_MODEL_NAME = os.getenv("CUSTOM_MODEL_NAME")
```

2. Add to `MODEL_CHOICES`:
```python
MODEL_CHOICES = ["ollama", "siliconflow", "magick", "custom"]
```

3. Add to `choose_default_model()`:
```python
if is_configured_api_key(CUSTOM_API_KEY):
    return "custom"
```

4. Implement API caller in `core/generator.py`

## Security Considerations

### API Key Storage

- **Store**: In `.env` file (gitignored)
- **Never**: Commit real API keys to version control
- **Use**: `example.env` with placeholder values for documentation

### Environment File Protection

```env
# .env should be in .gitignore
# Never commit real credentials
```

### Production Deployment

Before production use:
- Implement authentication
- Use secure secret management (e.g., AWS Secrets Manager, HashiCorp Vault)
- Enable HTTPS
- Restrict CORS origins

## Focused Tests

**File**: `tests/test_config.py`

### Test Cases

**`test_api_key_validation_rejects_placeholders()`**:
```python
def test_api_key_validation_rejects_placeholders():
    assert config.is_configured_api_key(None) is False
    assert config.is_configured_api_key("") is False
    assert config.is_configured_api_key("Your_SILICONFLOW_API_KEY") is False
    assert config.is_configured_api_key("sk-real-value") is True
```

**`test_default_model_selection_order()`**:
```python
def test_default_model_selection_order():
    assert config.choose_default_model("sk-silicon", "sk-magick", True) == "siliconflow"
    assert config.choose_default_model(None, "sk-magick", True) == "magick"
    assert config.choose_default_model(None, None, True) == "ollama"
    assert config.choose_default_model(None, None, False) == "siliconflow"
```

## Related Components

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Answer Generation](/core/generation.md) - Model provider integration
<!-- openwiki: broken internal link [/interfaces/gradio-ui.md] file "/interfaces/gradio-ui.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Gradio Web UI](/interfaces/gradio-ui.md) - Model selection UI
<!-- openwiki: broken internal link [/interfaces/rest-api.md] file "/interfaces/rest-api.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [REST API](/interfaces/rest-api.md) - Model choice in API requests
