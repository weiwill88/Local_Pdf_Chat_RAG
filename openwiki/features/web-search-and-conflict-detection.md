---
type: component
title: Web Search and Conflict Detection
description: Web search integration, conflict detection, and credibility evaluation for multi-source RAG
tags: [web-search, conflict-detection, features]
---

# Web Search and Conflict Detection

This document covers web search integration, conflict detection, and source credibility evaluation features.

## Web Search

### Overview

The web search module (`features/web_search.py`) integrates SerpAPI to provide real-time web information alongside local document retrieval.

### Key Design Decisions

1. **No Indexing**: Web search results are NOT added to FAISS index
2. **Context Only**: Results are provided directly as context to LLM
3. **Optional**: Enabled per-query via `enable_web_search` flag
4. **SerpAPI**: Uses Google Search via SerpAPI (requires API key)

### Configuration

**Location**: `config.py` lines 29-30

```python
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"
```

### Primary Symbols

#### `check_serpapi_key()`

**Location**: `features/web_search.py` line 15

```python
def check_serpapi_key() -> bool:
    """Check if valid SERPAPI_KEY is configured."""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != "" and not SERPAPI_KEY.startswith("Your")
```

#### `search_web()`

**Location**: `features/web_search.py` line 55

**Signature**:
```python
def search_web(query: str, num_results: int = 5) -> List[Dict]:
    """
    Execute web search (results not added to FAISS index, only used as context).
    
    Args:
        query: Search query
        num_results: Number of results to return
    
    Returns:
        List of {'title': ..., 'url': ..., 'snippet': ..., 'timestamp': ...}
    """
```

**Implementation**:
```python
# features/web_search.py lines 55-62
def search_web(query, num_results=5):
    results = serpapi_search(query, num_results)
    if not results:
        logging.info("网络搜索没有返回结果")
    else:
        logging.info(f"网络搜索返回 {len(results)} 条结果")
    return results
```

#### `serpapi_search()`

**Location**: `features/web_search.py` line 20

```python
def serpapi_search(query: str, num_results: int = 5) -> List[Dict]:
    """Execute SerpAPI search."""
    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量")
    try:
        params = {
            "engine": SEARCH_ENGINE, "q": query, "api_key": SERPAPI_KEY,
            "num": num_results, "hl": "zh-CN", "gl": "cn"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as e:
        logging.error(f"网络搜索失败：{str(e)}")
        return []
```

#### `_parse_serpapi_results()`

**Location**: `features/web_search.py` line 37

```python
def _parse_serpapi_results(data: Dict) -> List[Dict]:
    """Parse SerpAPI response."""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "timestamp": item.get("date")
            })
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"),
            "source": "knowledge_graph"
        })
    return results
```

### Integration with RAG Flow

Web search is integrated in `recursive_retrieval()` (retriever.py lines 97-108):

```python
# Web search supplement
web_texts = []
if enable_web_search and check_serpapi_key():
    try:
        web_results = search_web(query)
        for item in web_results:
<!-- openwiki: broken internal link [{item.get('url'] file "{item.get('url'" does not exist. Fix the href or restore the target, then delete this comment. -->
            web_texts.append(f"[{item.get('title')}]({item.get('url')}): {item.get('snippet')}")
            all_metadata.append({
                'source': 'web',
                'url': item.get('url'),
                'title': item.get('title')
            })
    except Exception as e:
        logging.error(f"网络搜索失败：{str(e)}")
```

## Conflict Detection

### Overview

The conflict detector (`features/conflict_detector.py`) identifies contradictions across multiple information sources (local documents + web search).

### Key Design Decisions

1. **Rule-Based**: Uses simple regex patterns for fact extraction
2. **Key Facts Only**: Focuses on dates, percentages, technical terms
3. **Binary Detection**: Returns True/False for conflict presence
4. **Credibility Scoring**: Domain-based credibility evaluation

### Primary Symbols

#### `detect_conflicts()`

**Location**: `features/conflict_detector.py` line 12

**Signature**:
```python
def detect_conflicts(sources: List[Dict]) -> bool:
    """
    Detect conflicts in multi-source information.
    
    Args:
        sources: List of {'text': ..., 'type': ..., 'url': ...}
    
    Returns:
        True if conflicts detected, False otherwise
    """
```

**Implementation**:
```python
# features/conflict_detector.py lines 12-21
def detect_conflicts(sources):
    key_facts = {}
    for item in sources:
        facts = _extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
        for fact, value in facts.items():
            if fact in key_facts and key_facts[fact] != value:
                return True
            key_facts[fact] = value
    return False
```

#### `_extract_facts()`

**Location**: `features/conflict_detector.py` line 24

```python
def _extract_facts(text: str) -> Dict:
    """Extract key facts from text."""
    facts = {}
    # Extract years and percentages
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    # Extract technical terms
    if "产业图谱" in text:
        facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
    return facts
```

**Fact Extraction Patterns**:
- `\b\d{4}年`: Four-digit year with Chinese character (e.g., "2024 年")
- `\b\d+%`: Percentage values (e.g., "85%")
- `[A-Za-z]+模型`: Model names (e.g., "Transformer 模型")
- `[A-Z]{2,}算法`: Algorithm names (e.g., "BERT 算法")

## Credibility Evaluation

### Overview

The credibility evaluator assesses source reliability based on domain names.

### Primary Symbol: `evaluate_source_credibility()`

**Location**: `features/conflict_detector.py` line 35

**Signature**:
```python
def evaluate_source_credibility(source: Dict) -> float:
    """
    Evaluate source credibility based on domain.
    
    Args:
        source: {'url': ..., 'title': ..., ...}
    
    Returns:
        Credibility score (0.0-1.0)
    """
```

### Credibility Scores

**Location**: `features/conflict_detector.py` lines 37-38

```python
credibility_scores = {
    "gov.cn": 0.9,      # Government websites
    "edu.cn": 0.85,     # Educational institutions
    "weixin": 0.7,      # WeChat official accounts
    "zhihu": 0.6,       # Zhihu (Q&A platform)
    "baidu": 0.5        # Baidu (general search)
}
```

### Implementation

```python
# features/conflict_detector.py lines 35-50
def evaluate_source_credibility(source):
    credibility_scores = {
        "gov.cn": 0.9, "edu.cn": 0.85, "weixin": 0.7, "zhihu": 0.6, "baidu": 0.5
    }
    url = source.get('url', '')
    if not url:
        return 0.5
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    domain = domain_match.group(1)
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    return 0.5  # Default score for unknown domains
```

## Integration in Generation Flow

### Conflict Detection in Prompt

When conflicts are detected, the prompt includes special instructions:

```python
# core/generator.py line 187
conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
```

### Context Building with Source Types

```python
# core/generator.py lines 196-211
for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
    source_type = metadata.get('source', '本地文档')
    source_item = {'text': doc, 'type': source_type}

    if source_type == 'web':
        url = metadata.get('url', '未知 URL')
        title = metadata.get('title', '未知标题')
        context_parts.append(f"[网络来源：{title}] (URL: {url})\n{doc}")
    else:
        source = metadata.get('source', '未知来源')
        context_parts.append(f"[本地文档：{source}]\n{doc}")
```

## Configuration

### Web Search Configuration

| Parameter | Default | Location | Purpose |
|-----------|---------|----------|---------|
| `SERPAPI_KEY` | Not set | `config.py` line 29 | SerpAPI authentication |
| `SEARCH_ENGINE` | "google" | `config.py` line 30 | Search engine to use |

### Enable/Disable Web Search

**Per-Query**: Pass `enable_web_search=True` to `query_answer()`:

```python
result = query_answer(
    question="What is the latest AI news?",
    enable_web_search=True,
    model_choice="siliconflow"
)
```

**Gradio UI**: Checkbox in Q&A tab allows users to toggle web search.

## Focused Tests

No dedicated tests for web search or conflict detection in current test suite.

**Test Coverage Gap**: `tests/` directory does not include tests for:
- `features/web_search.py`
- `features/conflict_detector.py`

## Change Recipes

### Adding Alternative Search Engine

1. Modify `SEARCH_ENGINE` in `config.py`:
   ```python
   SEARCH_ENGINE = "bing"  # or "duckduckgo", "yahoo"
   ```

2. SerpAPI supports multiple engines - check [SerpAPI documentation](https://serpapi.com/search-engines)

### Enhancing Fact Extraction

1. Add more regex patterns in `_extract_facts()`:
   ```python
   def _extract_facts(text):
       facts = {}
       # Add phone numbers
       phones = re.findall(r'\b1[3-9]\d{9}\b', text)
       if phones:
           facts['电话号码'] = phones
       # Add email addresses
       emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', text)
       if emails:
           facts['邮箱'] = emails
       # Existing patterns...
       return facts
   ```

2. Consider using NER (Named Entity Recognition) for more robust extraction

### Improving Credibility Scoring

1. Expand domain list in `evaluate_source_credibility()`:
   ```python
   credibility_scores = {
       "gov.cn": 0.9, "edu.cn": 0.85, "weixin": 0.7,
       "zhihu": 0.6, "baidu": 0.5,
       "github.com": 0.8, "medium.com": 0.65,
       "twitter.com": 0.55, "reddit.com": 0.5
   }
   ```

2. Consider TLD-based scoring:
   ```python
   if url.endswith(".gov"):
       return 0.9
   elif url.endswith(".edu"):
       return 0.85
   ```

3. Implement multi-factor scoring (domain + author + publication date)

### Adding Web Search Caching

To avoid redundant searches:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def search_web_cached(query: str, num_results: int = 5) -> List[Dict]:
    return search_web(query, num_results)
```

## Related Components

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Answer Generation](/core/generation.md) - Web search integration in generation flow
<!-- openwiki: broken internal link [/core/retrieval-and-reranking.md] file "/core/retrieval-and-reranking.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Retrieval and Reranking](/core/retrieval-and-reranking.md) - Recursive retrieval with web search
