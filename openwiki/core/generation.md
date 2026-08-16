---
type: component
title: Answer Generation
description: Prompt engineering, LLM provider integration, and response generation strategies
tags: [generation, llm, prompt-engineering]
---

# Answer Generation

This document covers LLM integration, prompt engineering, and answer generation in the RAG system.

## Overview

The generator module (`core/generator.py`) handles:
1. Prompt construction with retrieved context
2. LLM provider integration (Ollama, SiliconFlow, Magick API)
3. Response generation and processing
4. Thinking chain formatting

## LLM Provider Integration

### Supported Providers

| Provider | Configuration | Endpoint | Model |
|----------|---------------|----------|-------|
| **SiliconFlow** | `SILICONFLOW_API_KEY` | `https://api.siliconflow.cn/v1/chat/completions` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` |
| **Magick API** | `MAGICK_API_KEY`, `MAGICK_API_URL` | Configurable | `gpt-4o-mini` (default) |
| **Ollama** | `OLLAMA_MODEL_NAME` | `http://localhost:11434` | `deepseek-r1:8b` (default) |

### Model Selection Priority

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

### API Key Validation

**Location**: `config.py` lines 61-63

```python
def is_configured_api_key(api_key):
    """Determine if API key is an actual configured value."""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))
```

## Prompt Engineering

### Prompt Template

**Location**: `core/generator.py` lines 161-188

```python
def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected):
    """Build prompt with context and instructions."""
    prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""

    return prompt_template.format(
        context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else (
            "网络搜索结果" if enable_web_search else "本地文档"),
        context=context if context else (
            "网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
        question=question,
        time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
        conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
    )
```

### Prompt Principles

1. **Grounded Generation**: Answer only from provided context
2. **Honest Uncertainty**: Admit when context is insufficient
3. **Structured Response**: Use paragraphs and clear structure
4. **Language**: Chinese output
5. **Source Attribution**: Cite information sources
6. **Time Sensitivity**: Prioritize recent info when web search enabled
7. **Conflict Handling**: Explicitly note source differences when conflicts detected

### Context Building

**Location**: `core/generator.py` lines 191-213

```python
def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """Build context string and source list."""
    context_parts = []
    sources_for_conflict = []

    for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
        source_type = metadata.get('source', '本地文档')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            url = metadata.get('url', '未知 URL')
            title = metadata.get('title', '未知标题')
            context_parts.append(f"[网络来源：{title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
        else:
            source = metadata.get('source', '未知来源')
            context_parts.append(f"[本地文档：{source}]\n{doc}")
            source_item['source'] = source

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict
```

## API Calling

### OpenAI-Compatible API

**Location**: `core/generator.py` lines 55-98

```python
def _call_openai_compatible_api(provider_name, api_key, api_url, model_name,
                                prompt, temperature=0.7, max_tokens=1024,
                                extra_payload=None):
    """Call OpenAI-compatible Chat Completions API."""
    if not api_key:
        logging.error(f"未设置 {provider_name} API Key")
        return f"错误：未配置 {provider_name} API Key。"
    if not api_url:
        logging.error(f"未设置 {provider_name} API URL")
        return f"错误：未配置 {provider_name} API URL。"

    chat_url = _normalize_chat_completions_url(api_url)
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if extra_payload:
            payload.update(extra_payload)
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(chat_url, data=json_payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
        return _extract_openai_compatible_content(result)

    except requests.exceptions.HTTPError as e:
        logging.error(f"调用{provider_name} API 时出错：{str(e)}")
        return f"调用{provider_name} API 时出错：{str(e)}。"
    except requests.exceptions.RequestException as e:
        logging.error(f"调用{provider_name} API 时出错：{str(e)}")
        return f"调用{provider_name} API 时出错：{str(e)}"
    except Exception as e:
        logging.error(f"{provider_name} API 未知错误：{str(e)}")
        return f"发生未知错误：{str(e)}"
```

### URL Normalization

**Location**: `core/generator.py` lines 25-32

```python
def _normalize_chat_completions_url(api_url):
    """Normalize base URL or full chat completions URL."""
    if not api_url:
        return ""
    url = api_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"
```

### Content Extraction

**Location**: `core/generator.py` lines 35-52

```python
def _extract_openai_compatible_content(result):
    """Extract answer text and reasoning content from OpenAI-compatible response."""
    if "choices" not in result or not result["choices"]:
        return "API 返回结果格式异常"

    message = result["choices"][0].get("message", {})
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    if reasoning:
        return f"{content}<think>{reasoning}</think>"
    return content
```

### Provider-Specific Functions

#### SiliconFlow

**Location**: `core/generator.py` lines 101-118

```python
def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """Call SiliconFlow cloud API."""
    return _call_openai_compatible_api(
        "SiliconFlow",
        SILICONFLOW_API_KEY,
        SILICONFLOW_API_URL,
        SILICONFLOW_MODEL_NAME,
        prompt,
        temperature,
        max_tokens,
        extra_payload={
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
    )
```

#### Magick API

**Location**: `core/generator.py` lines 121-131

```python
def call_magick_api(prompt, temperature=0.7, max_tokens=1024):
    """Call Magick API."""
    return _call_openai_compatible_api(
        "Magick API",
        MAGICK_API_KEY,
        MAGICK_API_URL,
        MAGICK_MODEL_NAME,
        prompt,
        temperature,
        max_tokens
    )
```

#### Ollama

**Location**: `core/generator.py` lines 151-158

```python
def call_ollama_api(prompt, model_choice="ollama"):
    """Call local Ollama API."""
    response = get_session().post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
        timeout=180
    )
    return response.json().get("response", "").strip()
```

### Unified Interface

**Location**: `core/generator.py` lines 134-158

```python
def call_cloud_api(prompt, model_choice="siliconflow", temperature=0.7, max_tokens=1024):
    """Unified cloud API call."""
    if model_choice == "siliconflow":
        return call_siliconflow_api(prompt, temperature, max_tokens)
    if model_choice == "magick":
        return call_magick_api(prompt, temperature, max_tokens)
    raise ValueError(f"未知云端模型服务：{model_choice}")


def call_llm_simple(prompt, model_choice="siliconflow"):
    """Simple LLM call (for recursive retrieval query optimization)."""
    if model_choice in ("siliconflow", "magick"):
        result = call_cloud_api(prompt, model_choice)
        # Strip thinking content for simple calls
        if "<think>" in result:
            result = result.split("</think>")[1].strip()
        return result
    elif model_choice == "ollama":
        return call_ollama_api(prompt, model_choice)
    raise ValueError(f"未知模型选择：{model_choice}")
```

## Main Generation Flow

### Primary Symbol: `query_answer()`

**Location**: `core/generator.py` line 216

**Signature**:
```python
def query_answer(
    question: str,
    enable_web_search: bool = False,
    model_choice: str = "siliconflow",
    progress = None
) -> str:
    """
    Q&A processing flow (non-streaming).
    
    Flow: recursive retrieval → build context → conflict detection → 
          build prompt → LLM generation → process thinking content
    
    Args:
        question: User query
        enable_web_search: Whether to enable web search
        model_choice: Model provider (siliconflow/magick/ollama)
        progress: Optional progress callback
    
    Returns:
        Generated answer with thinking content formatted
    """
```

### Implementation Flow

```python
# core/generator.py lines 222-265
def query_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None):
    try:
        # 1. Check knowledge base
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ 知识库为空，请先上传文档。"

        # 2. Progress update
        if progress:
            progress(0.3, desc="执行递归检索...")

        # 3. Recursive retrieval
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        # 4. Build context
        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        
        # 5. Conflict detection
        conflict_detected = detect_conflicts(sources)
        
        # 6. Time sensitivity check
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

        # 7. Build prompt
        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected)

        # 8. Progress update
        if progress:
            progress(0.8, desc="生成回答...")

        # 9. LLM generation
        if model_choice in ("siliconflow", "magick"):
            result = call_cloud_api(prompt, model_choice, temperature=0.7, max_tokens=1536)
        elif model_choice == "ollama":
            response = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=180, headers={'Connection': 'close'}
            )
            result = response.json().get("response", "").strip()
        else:
            result = f"错误：未知模型选择 {model_choice}"

        # 10. Process thinking content
        from features.thinking_chain import process_thinking_content
        result = process_thinking_content(result)

        return result

    except Exception as e:
        logging.error(f"问答处理失败：{str(e)}")
        return f"处理失败：{str(e)}"
```

## Error Handling

### Missing API Key

```python
# core/generator.py lines 59-64
if not api_key:
    logging.error(f"未设置 {provider_name} API Key")
    return f"错误：未配置 {provider_name} API Key。"
```

### HTTP Errors

```python
# core/generator.py lines 87-98
except requests.exceptions.HTTPError as e:
    logging.error(f"调用{provider_name} API 时出错：{str(e)}")
    return (
        f"调用{provider_name} API 时出错：{str(e)}。"
        f"请检查 API Key、API URL 和模型名称 {model_name} 是否可用。"
    )
except requests.exceptions.RequestException as e:
    logging.error(f"调用{provider_name} API 时出错：{str(e)}")
    return f"调用{provider_name} API 时出错：{str(e)}"
```

### General Exception

```python
# core/generator.py line 266-269
except Exception as e:
    logging.error(f"问答处理失败：{str(e)}")
    return f"处理失败：{str(e)}"
```

## Configuration

### Model Parameters

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `SILICONFLOW_MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `config.py` line 49 | SiliconFlow model ID |
| `MAGICK_MODEL_NAME` | `gpt-4o-mini` | `config.py` line 50 | Magick API model ID |
| `OLLAMA_MODEL_NAME` | `deepseek-r1:8b` | `config.py` line 48 | Ollama model name |
| Temperature | 0.7 | `generator.py` lines 56, 102, 123 | Generation randomness |
| Max Tokens | 1024-1536 | `generator.py` lines 102, 123, 245 | Response length limit |

## Thinking Chain Processing

### Overview

The thinking chain module (`features/thinking_chain.py`) processes DeepSeek-R1's `<think>` tags and converts them to collapsible HTML.

### Primary Symbol: `process_thinking_content()`

**Location**: `features/thinking_chain.py` line 12

```python
def process_thinking_content(text: str) -> str:
    """
    Process <think>...</think> content into collapsible HTML.
    
    Converts <think>推理过程</think> to <details> tags.
    """
```

### Implementation

```python
# features/thinking_chain.py lines 28-42
while "<think>" in processed_text and "</think>" in processed_text:
    start_idx = processed_text.find("<think>")
    end_idx = processed_text.find("</think>")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking_content = processed_text[start_idx + 7:end_idx]
        before = processed_text[:start_idx]
        after = processed_text[end_idx + 8:]
        processed_text = (
            before +
            "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" +
            thinking_content +
            "\n\n</details>\n\n" +
            after
        )
```

### HTML Escaping

After processing thinking content, remaining HTML tags are escaped to prevent injection:

```python
# features/thinking_chain.py lines 44-65
processed_html = []
i = 0
while i < len(processed_text):
    if (processed_text[i:i + 8] == "<details" or
            processed_text[i:i + 9] == "</details" or
            processed_text[i:i + 8] == "<summary" or
            processed_text[i:i + 9] == "</summary"):
        tag_end = processed_text.find(">", i)
        if tag_end != -1:
            processed_html.append(processed_text[i:tag_end + 1])
            i = tag_end + 1
            continue
    if processed_text[i] == "<":
        processed_html.append("&lt;")
    elif processed_text[i] == ">":
        processed_html.append("&gt;")
    else:
        processed_html.append(processed_text[i])
    i += 1
```

## Focused Tests

No dedicated tests for generator module in current test suite.

**Test Coverage Gap**: `tests/` directory does not include tests for:
- `core/generator.py`
- `features/thinking_chain.py`

## Change Recipes

### Adding a New LLM Provider

1. Add configuration variables in `config.py`:
   ```python
   NEW_PROVIDER_API_KEY = os.getenv("NEW_PROVIDER_API_KEY")
   NEW_PROVIDER_API_URL = os.getenv("NEW_PROVIDER_API_URL")
   NEW_PROVIDER_MODEL_NAME = os.getenv("NEW_PROVIDER_MODEL_NAME")
   ```

2. Add provider function in `generator.py`:
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

3. Update `call_cloud_api()`:
   ```python
   if model_choice == "new_provider":
       return call_new_provider_api(prompt, temperature, max_tokens)
   ```

4. Update model selection in `config.py`:
   ```python
   MODEL_CHOICES = ["ollama", "siliconflow", "magick", "new_provider"]
   ```

### Adjusting Temperature

1. Modify default temperature in `_call_openai_compatible_api()` (line 56)
2. Or pass different temperature per call
3. Higher values (0.8-1.0): More creative, diverse responses
4. Lower values (0.2-0.5): More focused, deterministic responses

### Customizing Prompt Template

1. Modify `_build_prompt()` in `generator.py` (lines 161-188)
2. Consider adding:
   - Domain-specific instructions
   - Output format requirements
   - Citation style guidelines
3. Test with representative queries

## Related Components

<!-- openwiki: broken internal link [/core/retrieval-and-reranking.md] file "/core/retrieval-and-reranking.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Retrieval and Reranking](/core/retrieval-and-reranking.md) - Context retrieval for generation
<!-- openwiki: broken internal link [/features/thinking-chain.md] file "/features/thinking-chain.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Thinking Chain Processing](/features/thinking-chain.md) - Thinking content formatting
<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Configuration](/configuration/environment-and-models.md) - Model configuration
