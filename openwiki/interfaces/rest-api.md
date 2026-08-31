---
type: component
title: REST API
description: FastAPI REST endpoints for document upload, querying, and status
tags: [rest-api, fastapi, interface]
---

# REST API

This document covers the FastAPI REST API implementation, including endpoints, request/response schemas, and async patterns.

## Overview

The REST API (`api_router.py`) provides programmatic access to the RAG system via HTTP endpoints:
- `GET /api/status` - System status and configuration
- `POST /api/upload` - Upload and process documents
- `POST /api/ask` - Query documents and get answers

## Application Setup

**Location**: `api_router.py` lines 38-56

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API 服务启动")
    yield
    logger.info("API 服务已关闭")


app = FastAPI(
    title="本地 RAG API 服务",
    description="提供基于本地大模型、云端模型服务和 SERPAPI 的文档问答 API 接口",
    version=__version__,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### CORS Configuration

The API allows all origins (`allow_origins=["*"]`), making it accessible from any frontend. For production use, restrict to specific domains.

### Lifespan Events

- **Startup**: Logs "API 服务启动"
- **Shutdown**: Logs "API 服务已关闭"

## Request/Response Models

### Pydantic Models

**Location**: `api_router.py` lines 59-75

```python
class QuestionRequest(BaseModel):
    """Question request schema."""
    question: str
    enable_web_search: bool = False
    model_choice: str = "siliconflow"


class AnswerResponse(BaseModel):
    """Answer response schema."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class FileProcessResult(BaseModel):
    """File processing result schema."""
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None
```

### Field Descriptions

#### QuestionRequest

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | str | Required | User's question |
| `enable_web_search` | bool | False | Enable web search supplement |
| `model_choice` | str | "siliconflow" | LLM backend (siliconflow/magick/ollama) |

#### AnswerResponse

| Field | Type | Description |
|-------|------|-------------|
| `answer` | str | Generated answer with thinking content |
| `sources` | List[Dict] | Source documents with metadata |
| `metadata` | Dict | Additional response metadata |

#### FileProcessResult

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | "success" or "error" |
| `message` | str | Human-readable message |
| `file_info` | Dict | File metadata (optional) |

## Endpoints

### GET /api/status

**Location**: `api_router.py` lines 103-125

**Signature**:
```python
@app.get("/api/status")
async def get_status():
    """
    Get system status and configuration.
    
    Returns:
        Status information including model configuration and vector store state
    """
```

**Response Schema**:
```json
{
    "status": "ok",
    "version": "2.1.0",
    "model_configured": true,
    "model_choice": "siliconflow",
    "vector_store_ready": false,
    "total_chunks": 0
}
```

**Implementation**:
```python
# api_router.py lines 103-125
@app.get("/api/status")
async def get_status():
    from config import (
        SILICONFLOW_API_KEY, MAGICK_API_KEY,
        choose_default_model, MODEL_CHOICES
    )
    from core.vector_store import vector_store
    
    model_configured = (
        is_configured_api_key(SILICONFLOW_API_KEY) or
        is_configured_api_key(MAGICK_API_KEY)
    )
    
    return {
        "status": "ok",
        "version": __version__,
        "model_configured": model_configured,
        "model_choice": choose_default_model(
            SILICONFLOW_API_KEY, MAGICK_API_KEY
        ),
        "vector_store_ready": vector_store.is_ready,
        "total_chunks": vector_store.total_chunks
    }
```

### POST /api/upload

**Location**: `api_router.py` lines 77-101

**Signature**:
```python
@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process document.
    
    Args:
        file: Uploaded file
    
    Returns:
        File processing result
    """
```

**Request**:
- `Content-Type`: `multipart/form-data`
- Body: `file` (binary file data)

**Response**:
```json
{
    "status": "success",
    "message": "✅ document.pdf: 成功处理 25 个文本块\n总计处理 1 个文件，25 个文本块",
    "file_info": {
        "filename": "document.pdf",
        "chunks": 25
    }
}
```

**Implementation**:
```python
# api_router.py lines 77-101
@app.post("/api/upload", response_model=FileProcessResult)
async def upload_file(file: UploadFile = File(...)):
    """Process document and store in vector database."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, 
                                         suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Import processing function
        from rag_demo import process_multiple_files
        
        # Wrap with progress callback
        progress = ProgressCallback()
        
        # Execute processing in thread pool
        result_text = await asyncio.to_thread(
            process_multiple_files,
            [type('obj', (object,), {"name": tmp_path})],
            progress
        )
        
        # Cleanup temp file
        os.unlink(tmp_path)
        
        # Parse result
        result = result_text[0] if isinstance(result_text, tuple) else result_text
        chunk_match = re.search(r'(\d+) 个文本块', result)
        chunks = int(chunk_match.group(1)) if chunk_match else 0
        
        return {
            "status": "success",
            "message": result,
            "file_info": {
                "filename": file.filename,
                "chunks": chunks
            }
        }
    
    except Exception as e:
        logger.error(f"文件处理失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

### POST /api/ask

**Location**: `api_router.py` lines 127-160

**Signature**:
```python
@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask question about documents.
    
    Args:
        request: QuestionRequest with question, enable_web_search, model_choice
    
    Returns:
        AnswerResponse with answer, sources, and metadata
    """
```

**Request**:
```json
{
    "question": "What is RAG?",
    "enable_web_search": false,
    "model_choice": "siliconflow"
}
```

**Response**:
```json
{
    "answer": "RAG (Retrieval-Augmented Generation) 是一种结合检索和生成的技术...",
    "sources": [
        {
            "content": "RAG combines retrieval and generation...",
            "metadata": {"source": "document.pdf", "doc_id": "doc_123_chunk_0"}
        }
    ],
    "metadata": {
        "model_used": "siliconflow",
        "web_search_used": false
    }
}
```

**Implementation**:
```python
# api_router.py lines 127-160
@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask question against processed documents."""
    try:
        from core.generator import query_answer
        from core.vector_store import vector_store
        
        # Check if vector store has data
        if not vector_store.is_ready and not request.enable_web_search:
            raise HTTPException(
                status_code=400,
                detail="知识库为空，请先上传文档"
            )
        
        # Generate answer
        answer = await asyncio.to_thread(
            query_answer,
            request.question,
            request.enable_web_search,
            request.model_choice
        )
        
        # Extract sources from vector store
        sources = []
        for chunk_id in vector_store.id_order[:5]:  # Top 5
            sources.append({
                "content": vector_store.contents_map.get(chunk_id, ""),
                "metadata": vector_store.metadatas_map.get(chunk_id, {})
            })
        
        return AnswerResponse(
            answer=answer,
            sources=sources,
            metadata={
                "model_used": request.model_choice,
                "web_search_used": request.enable_web_search
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问答处理失败：{str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

## Async Patterns

### ProgressCallback Class

**Location**: `api_router.py` lines 27-35

```python
class ProgressCallback:
    """Progress callback for API file processing."""
    
    def __init__(self):
        self.progress = 0
        self.description = ""
    
    def __call__(self, progress, desc=None):
        self.progress = progress
        self.description = desc or ""
        return self
```

**Usage**: Wraps Gradio's progress interface for API compatibility.

### Thread Pool Execution

Blocking operations run in thread pool to avoid blocking async event loop:

```python
# api_router.py lines 89-93
result_text = await asyncio.to_thread(
    process_multiple_files,
    [type('obj', (object,), {"name": tmp_path})],
    progress
)
```

### Temporary File Handling

```python
# api_router.py lines 81-85
with tempfile.NamedTemporaryFile(delete=False, 
                                 suffix=os.path.splitext(file.filename)[1]) as tmp:
    content = await file.read()
    tmp.write(content)
    tmp_path = tmp.name

# ... processing ...

# Cleanup
os.unlink(tmp_path)
```

## Error Handling

### HTTP Exceptions

```python
# api_router.py line 141-145
if not vector_store.is_ready and not request.enable_web_search:
    raise HTTPException(
        status_code=400,
        detail="知识库为空，请先上传文档"
    )
```

### General Exception Handling

```python
# api_router.py lines 100-101, 158-160
except Exception as e:
    logger.error(f"文件处理失败：{str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

## Configuration

### Port Selection

**Location**: `api_router.py` lines 163-175

```python
if __name__ == "__main__":
    import uvicorn
    from utils.network import is_port_available
    
    base_port = 17995
    for port in range(base_port, base_port + 5):
        if is_port_available(port):
            print(f"启动 REST API 于 http://127.0.0.1:{port}")
            uvicorn.run(app, host="127.0.0.1", port=port)
            break
    else:
        print("所有可用端口都被占用")
```

## Focused Tests

**File**: `tests/test_api_status.py`

### Test Cases

**`test_status_endpoint()`**:
```python
def test_status_endpoint():
    from fastapi.testclient import TestClient
    from api_router import app
    
    client = TestClient(app)
    response = client.get("/api/status")
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_configured" in data
```

## Change Recipes

### Adding a New Endpoint

1. Define request/response models:
```python
class NewRequest(BaseModel):
    field1: str
    field2: int

class NewResponse(BaseModel):
    result: str
```

2. Add endpoint handler:
```python
@app.post("/api/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest):
    # Implementation
    return NewResponse(result="...")
```

3. Add tests in `tests/`

### Adding Streaming Response

For streaming answers:

```python
from fastapi.responses import StreamingResponse

@app.post("/api/ask-stream")
async def ask_stream(request: QuestionRequest):
    def generate():
        for chunk in stream_answer(request.question):
            yield f"data: {json.dumps(chunk)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### Adding Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/api/ask")
async def ask_question(request: QuestionRequest, 
                       credentials: HTTPAuthorizationCredentials = Depends(security)):
    # Validate token
    if not validate_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # ... rest of handler
```

## Related Components

<!-- openwiki: broken internal link [/core/generation.md] file "/core/generation.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Answer Generation](/core/generation.md) - Backend for /api/ask
<!-- openwiki: broken internal link [/interfaces/gradio-ui.md] file "/interfaces/gradio-ui.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Gradio Web UI](/interfaces/gradio-ui.md) - Alternative interface
<!-- openwiki: broken internal link [/configuration/environment-and-models.md] file "/configuration/environment-and-models.md" does not exist. Fix the href or restore the target, then delete this comment. -->
- [Configuration](/configuration/environment-and-models.md) - Model configuration
