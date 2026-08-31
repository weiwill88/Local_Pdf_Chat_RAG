---
type: component
title: Gradio Web UI
description: Gradio interface layout, tab structure, and user interaction patterns
tags: [gradio, ui, interface]
---

# Gradio Web UI

This document covers the Gradio web interface implementation, including tab layouts, components, and user interactions.

## Overview

The Gradio UI (`rag_demo.py`) provides a web-based interface with three main tabs:
1. **问答对话** (Q&A Dialogue) - Document upload and question answering
2. **分块可视化** (Chunk Visualization) - Inspect processed document chunks
3. **系统监控** (System Monitoring) - System resource monitoring

## Application Entry Point

**Location**: `rag_demo.py` line 271

```python
if __name__ == "__main__":
    # Port selection logic
    base_port = 17995
    available_port = None
    for port in range(base_port, base_port + 5):
        if is_port_available(port):
            available_port = port
            break
    
    if available_port:
        print(f"启动 Gradio Web UI 于 http://127.0.0.1:{available_port}")
        demo.launch(server_name="127.0.0.1", port=available_port)
    else:
        print("所有可用端口都被占用")
```

### Port Selection

The application tries ports 17995-17999 sequentially if the default is occupied.

## Tab 1: 问答对话 (Q&A Dialogue)

### Layout Structure

**Location**: `rag_demo.py` lines 228-270

```python
with gr.Tab("问答对话"):
    with gr.Row():
        with gr.Column(scale=3):
            file_input = gr.File(label="上传文档", file_count="multiple")
            process_btn = gr.Button("处理文档", variant="primary")
            
            model_choice = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULT_MODEL_CHOICE,
                label="选择模型后端"
            )
            
            web_search_checkbox = gr.Checkbox(
                label="启用联网搜索",
                value=False
            )
            
            question_input = gr.Textbox(
                label="问题",
                placeholder="请输入您的问题...",
                lines=3
            )
            
            ask_btn = gr.Button("提问", variant="primary")
            
            answer_output = gr.Markdown(label="回答")
        
        with gr.Column(scale=2):
            file_list = gr.File(label="已上传文件")
            status_output = gr.Textbox(label="处理状态", interactive=False)
```

### Components

| Component | Type | Purpose |
|-----------|------|---------|
| `file_input` | File | Multi-file upload for document processing |
| `process_btn` | Button | Trigger document processing pipeline |
| `model_choice` | Dropdown | Select LLM backend (siliconflow/magick/ollama) |
| `web_search_checkbox` | Checkbox | Enable/disable web search |
| `question_input` | Textbox | User question input |
| `ask_btn` | Button | Submit question for answer generation |
| `answer_output` | Markdown | Display generated answer with formatting |
| `file_list` | File | Show uploaded files |
| `status_output` | Textbox | Show processing status messages |

### Event Handlers

**Document Processing** (lines 273-276):
```python
process_btn.click(
    fn=process_multiple_files,
    inputs=[file_input],
    outputs=[status_output, file_list]
)
```

**Question Answering** (lines 278-284):
```python
ask_btn.click(
    fn=lambda q, model, web_search: query_answer(q, web_search, model),
    inputs=[question_input, model_choice, web_search_checkbox],
    outputs=[answer_output]
)
```

## Tab 2: 分块可视化 (Chunk Visualization)

### Overview

This tab allows users to inspect how documents were chunked during processing.

### Layout Structure

**Location**: `rag_demo.py` lines 173-226

```python
with gr.Tab("分块可视化"):
    with gr.Row():
        chunk_preview = gr.Dataframe(
            headers=["Source", "Chunk ID", "Characters", "Tokens", "Preview"],
            label="文档分块预览"
        )
    
    with gr.Row():
        chunk_detail = gr.Markdown(label="分块详情")
```

### Key Functions

#### `get_document_chunks()`

**Location**: `rag_demo.py` line 110

**Signature**:
```python
def get_document_chunks():
    """
    Retrieve and format chunk data for visualization.
    
    Returns:
        Dataframe-ready chunk data with source, ID, size, preview
    """
```

**Implementation**:
```python
# rag_demo.py lines 110-140
def get_document_chunks():
    if not vector_store.id_order:
        return []
    
    data = []
    for chunk_id in vector_store.id_order:
        content = vector_store.contents_map.get(chunk_id, "")
        metadata = vector_store.metadatas_map.get(chunk_id, {})
        
        source = metadata.get('source', 'Unknown')
        doc_id = metadata.get('doc_id', 'Unknown')
        
        # Calculate preview (first 100 chars)
        preview = content[:100] + "..." if len(content) > 100 else content
        
        data.append({
            "Source": source,
            "Chunk ID": chunk_id,
            "Characters": len(content),
            "Tokens": len(content) // 4,  # Approximate
            "Preview": preview
        })
    
    return data
```

#### `show_chunk_details()`

**Location**: `rag_demo.py` line 143

**Signature**:
```python
def show_chunk_details(chunk_id: str):
    """
    Display detailed information for a specific chunk.
    
    Args:
        chunk_id: Selected chunk identifier
    
    Returns:
        Markdown-formatted chunk details
    """
```

**Implementation**:
```python
# rag_demo.py lines 143-161
def show_chunk_details(chunk_id):
    if not chunk_id:
        return "请选择一个分块"
    
    content = vector_store.contents_map.get(chunk_id, "")
    metadata = vector_store.metadatas_map.get(chunk_id, {})
    
    source = metadata.get('source', 'Unknown')
    doc_id = metadata.get('doc_id', 'Unknown')
    
    details = f"""
    ## 分块详情
    
    - **分块 ID**: `{chunk_id}`
    - **文档 ID**: `{doc_id}`
    - **来源文件**: `{source}`
    - **字符数**: {len(content)}
    - **估计 Token 数**: {len(content) // 4}
    
    ### 完整内容
    
    ```
    {content}
    ```
    """
    return details
```

### Chunk Data Cache

**Location**: `rag_demo.py` line 164

```python
chunk_data_cache = gr.State(value=[])
```

The cache stores chunk data to avoid repeated lookups during UI interactions.

### Event Handlers

**Load Chunk Preview** (lines 286-289):
```python
chunk_preview.select(
    fn=lambda selected: show_chunk_details(selected[0]) if selected else "请选择一个分块",
    inputs=[chunk_preview],
    outputs=[chunk_detail]
)
```

## Tab 3: 系统监控 (System Monitoring)

### Overview

This tab displays system resource usage and application status.

### Layout Structure

**Location**: `rag_demo.py` lines 167-171

```python
with gr.Tab("系统监控"):
    system_info = gr.Markdown(label="系统信息")
    refresh_monitor = gr.Button("刷新监控数据")
```

### System Information

**Location**: `rag_demo.py` lines 167-170

```python
def get_system_info():
    """Get system resource information."""
    import psutil
    import platform
    
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    info = f"""
    ## 系统信息
    
    - **操作系统**: {platform.system()} {platform.release()}
    - **Python 版本**: {platform.python_version()}
    - **CPU 使用率**: {cpu}%
    - **内存使用率**: {memory.percent}%
    - **可用内存**: {memory.available / 1024 / 1024 / 1024:.2f} GB
    - **总内存**: {memory.total / 1024 / 1024 / 1024:.2f} GB
    
    ## RAG 状态
    
    - **向量存储**: {'就绪' if vector_store.is_ready else '空'}
    - **文本块数量**: {vector_store.total_chunks}
    """
    return info
```

### Event Handlers

**Refresh Monitor** (lines 291-293):
```python
refresh_monitor.click(
    fn=get_system_info,
    outputs=[system_info]
)
```

## Theme and Styling

### Theme Toggle

**Location**: `rag_demo.py` lines 245-255

```python
# Light/dark theme toggle
theme_switch = gr.Checkbox(
    label="深色模式",
    value=False,
    elem_id="theme-switch"
)

# JavaScript for theme persistence
theme_js = """
<script>
const themeSwitch = document.getElementById('theme-switch');
if (localStorage.getItem('dark_mode') === 'true') {
    themeSwitch.checked = true;
    document.body.classList.add('dark');
}
themeSwitch.addEventListener('change', () => {
    localStorage.setItem('dark_mode', themeSwitch.checked);
    document.body.classList.toggle('dark', themeSwitch.checked);
});
</script>
"""
gr.HTML(theme_js)
```

### Design Rationale

- **LocalStorage**: Theme preference persists across browser sessions
- **CSS Class**: `.dark` class applied to `<body>` for Gradio theme override
- **Checkbox**: Simple toggle UI for theme switching

## Progress Tracking

### Gradio Progress

**Location**: `rag_demo.py` lines 48-97

```python
def process_multiple_files(files, progress=gr.Progress()):
    """Process multiple files with progress tracking."""
    if not files:
        return "请选择要上传的文件...", []
    
    try:
        progress(0.1, desc="清理历史数据...")
        vector_store.clear()
        bm25_manager.clear()
        
        total_files = len(files)
        for idx, file in enumerate(files, 1):
            progress((idx - 1) / total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
            # ... processing ...
        
        progress(0.8, desc="生成文本嵌入...")
        # ... embeddings ...
        
        progress(0.9, desc="构建 FAISS 索引...")
        # ... index building ...
        
        progress(0.95, desc="构建 BM25 检索索引...")
        # ... BM25 ...
        
        progress(1.0, desc="完成!")
        return result, files
```

### Progress Stages

| Progress | Description | Action |
|----------|-------------|--------|
| 0.1 | 清理历史数据... | Clear vector_store and bm25_manager |
| (idx-1)/total | 处理文件 X/N | Process each file |
| 0.8 | 生成文本嵌入... | Encode texts to embeddings |
| 0.9 | 构建 FAISS 索引... | Build FAISS index |
| 0.95 | 构建 BM25 检索索引... | Build BM25 index |
| 1.0 | 完成 | Processing complete |

## State Management

### Gradio State

| State Variable | Type | Purpose |
|----------------|------|---------|
| `chunk_data_cache` | gr.State | Cache chunk data for visualization |
| `theme_switch` | Checkbox | Theme toggle state |

### Application State

| Variable | Source | Purpose |
|----------|--------|---------|
| `vector_store` | `core/vector_store.py` | FAISS index and mappings |
| `bm25_manager` | `core/bm25_index.py` | BM25 index |

## Focused Tests

No dedicated tests for Gradio UI in current test suite.

**Test Coverage Gap**: `tests/` directory does not include tests for:
- `rag_demo.py` UI layout
- Event handler functions

## Change Recipes

### Adding a New Tab

1. Add tab block after existing tabs:
```python
with gr.Tab("新标签"):
    with gr.Row():
        new_component = gr.Textbox(label="新组件")
```

2. Add event handler:
```python
new_component.change(fn=new_handler, inputs=[...], outputs=[...])
```

### Customizing Progress Messages

Modify progress calls in `process_multiple_files()`:

```python
progress(0.5, desc="自定义进度消息...")
```

### Adding Chunk Statistics

Enhance `get_document_chunks()` to include statistics:

```python
def get_document_chunks():
    # ... existing code ...
    
    # Add statistics row
    total_chars = sum(len(vector_store.contents_map.get(cid, "")) 
                      for cid in vector_store.id_order)
    data.insert(0, {
        "Source": "**总计**",
        "Chunk ID": f"{len(vector_store.id_order)} chunks",
        "Characters": total_chars,
        "Tokens": total_chars // 4,
        "Preview": ""
    })
    
    return data
```

### Adding Export Functionality

To export chunks:

```python
def export_chunks():
    import json
    data = []
    for chunk_id in vector_store.id_order:
        data.append({
            "id": chunk_id,
            "content": vector_store.contents_map.get(chunk_id, ""),
            "metadata": vector_store.metadatas_map.get(chunk_id, {})
        })
    
    with open("chunks_export.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return "已导出 chunks_export.json"

export_btn = gr.Button("导出分块")
export_btn.click(fn=export_chunks, outputs=[status_output])
```

## Related Components

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Answer Generation](/core/generation.md) - Backend for Q&A tab
<!-- openwiki: broken internal link [/core/embeddings-and-vector-store.md] file "/core/embeddings-and-vector-store.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Embeddings and Vector Store](/core/embeddings-and-vector-store.md) - Data for chunk visualization
<!-- openwiki: broken internal link [/features/thinking-chain.md] file "/features/thinking-chain.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Thinking Chain Processing](/features/thinking-chain.md) - Formatted output in answer display
