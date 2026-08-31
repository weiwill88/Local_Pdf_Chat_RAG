---
type: component
title: Thinking Chain Processing
description: DeepSeek-R1 thinking content formatting and HTML conversion
tags: [thinking-chain, features, formatting]
---

# Thinking Chain Processing

This document covers the thinking chain processing feature that formats DeepSeek-R1's reasoning content.

## Overview

The thinking chain module (`features/thinking_chain.py`) processes DeepSeek-R1 model outputs that contain `<think>` tags, converting them into user-friendly collapsible HTML elements.

## Background

DeepSeek-R1 and similar reasoning models output their thought process within `<think>` and `</think>` tags:

```
</think>

Based on the context, the answer is X.</think>
```

The thinking chain processor converts this to collapsible HTML:

```html
Based on the context, the answer is X.

<details>
<summary>思考过程（点击展开）</summary>

[reasoning content here]

</details>
```

## Primary Symbol

### `process_thinking_content()`

**Location**: `features/thinking_chain.py` line 12

**Signature**:
```python
def process_thinking_content(text: str) -> str:
    """
    Process <think>...</think> content into collapsible HTML format.
    
    Converts <think>推理过程</think> to <details> tags that users can click to expand.
    
    Args:
        text: Text potentially containing <think> tags
    
    Returns:
        Text with thinking content wrapped in collapsible HTML
    """
```

## Implementation Details

### Thinking Content Extraction

**Location**: `features/thinking_chain.py` lines 28-42

```python
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

**Algorithm**:
1. Find first `<think>` tag
2. Find corresponding `</think>` tag
3. Extract content between tags
4. Replace with HTML `<details>` structure
5. Repeat for multiple thinking sections

### HTML Escaping

After processing thinking content, remaining HTML special characters are escaped to prevent injection:

**Location**: `features/thinking_chain.py` lines 44-65

```python
# Preserve details/summary tags, escape others
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

processed_text = "".join(processed_html)
```

**Design Rationale**:
- Preserves `<details>` and `<summary>` tags for collapsible UI
- Escapes all other `<` and `>` to prevent HTML injection
- Handles edge cases where thinking content might contain `<` or `>`

### Error Handling

**Location**: `features/thinking_chain.py` lines 66-71

```python
except Exception as e:
    logging.error(f"处理思维链内容时出错：{str(e)}")
    try:
        return text.replace("<", "&lt;").replace(">", "&gt;")
    except:
        return "处理内容时出错"
```

## Integration

### Usage in Generation Flow

**Location**: `core/generator.py` line 262

```python
# Process thinking content before returning
from features.thinking_chain import process_thinking_content
result = process_thinking_content(result)
return result
```

### Gradio UI Display

The Gradio interface renders the HTML output directly, allowing users to click the "思考过程（点击展开）" summary to view or hide the reasoning.

## Configuration

No configuration options for thinking chain processing. It is automatically applied to all LLM responses.

## Focused Tests

No dedicated tests for thinking chain processing in current test suite.

**Test Coverage Gap**: `tests/` directory does not include tests for:
- `features/thinking_chain.py`

## Change Recipes

### Customizing Thinking Summary Text

To change the collapsible summary text:

```python
# features/thinking_chain.py line 38
processed_text = (
    before +
    "\n\n<details>\n<summary>Click to view reasoning</summary>\n\n" +  # Changed
    thinking_content +
    "\n\n</details>\n\n" +
    after
)
```

### Adding Thinking Content Styling

To add CSS styling for thinking content:

```python
# features/thinking_chain.py line 38
processed_text = (
    before +
    "\n\n<details style='background: #f5f5f5; padding: 10px; margin: 10px 0;'>\n" +
    "<summary style='cursor: pointer; font-weight: bold;'>思考过程（点击展开）</summary>\n\n" +
    "<div style='padding: 10px; border-left: 3px solid #ccc;'>\n" +
    thinking_content +
    "\n</div>\n\n</details>\n\n" +
    after
)
```

### Handling Nested Thinking Tags

For models that might nest thinking tags (rare):

```python
# Add depth tracking
depth = 0
start_idx = 0
while True:
    next_start = processed_text.find("<think>", start_idx)
    next_end = processed_text.find("</think>", start_idx)
    
    if next_start == -1 and next_end == -1:
        break
    
    if next_start != -1 and (next_end == -1 or next_start < next_end):
        depth += 1
        start_idx = next_start + 7
    else:
        depth -= 1
        if depth == 0:
            # Extract complete thinking section
            thinking_content = processed_text[start_idx-7:next_end]
            # Process...
            start_idx = next_end + 8
```

## Related Components

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Answer Generation](/core/generation.md) - Thinking chain integration in generation flow
