"""
REST API 模块（使用FastAPI实现）
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import re
from typing import Dict, Any, List, Optional
import logging
import json
import asyncio
from contextlib import asynccontextmanager

# 从rag_demo导入所需功能
import rag_demo
from io import StringIO

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rag-api")

# 定义类用于模拟Gradio进度回调
class ProgressCallback:
    def __init__(self):
        self.progress = 0
        self.description = ""
        
    def __call__(self, progress: float, desc: str = None):
        self.progress = progress
        self.description = desc or ""
        logger.info(f"进度: {progress:.2f} - {desc}")
        return self

# 启动时确保模型和向量存储准备就绪
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 检查环境
    if not rag_demo.check_environment():
        logger.error("环境检查失败！请确保Ollama服务已启动且所需模型已加载")
    yield
    # 清理工作（如果需要）
    logger.info("API服务已关闭")

# 初始化FastAPI应用
app = FastAPI(
    title="本地RAG API服务",
    description="提供基于本地大模型和SERPAPI的文档问答API接口",
    version="1.0.0",
    lifespan=lifespan
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    enable_web_search: bool = False

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class FileProcessResult(BaseModel):
    status: str
    message: str
    file_info: Optional[Dict[str, Any]] = None

async def process_answer_stream(question: str, enable_web_search: bool):
    """处理流式回答，模拟同步函数的异步版本"""
    progress = ProgressCallback()
    answer = ""
    
    # 创建生成器函数的包装器
    def run_stream():
        for response, status in rag_demo.stream_answer(question, enable_web_search, progress):
            nonlocal answer
            answer = response
            yield response, status
    
    # 在异步上下文中运行同步代码
    loop = asyncio.get_event_loop()
    generator = run_stream()
    
    # 消费生成器直到最后一个结果
    try:
        while True:
            resp, status = await loop.run_in_executor(None, next, generator)
            if status == "完成!":
                break
    except StopIteration:
        pass
    
    return answer

@app.post("/api/upload", response_model=FileProcessResult)
async def upload_pdf(file: UploadFile = File(...)):
    """
    处理PDF文档并存入向量数据库
    - 支持格式：application/pdf
    - 最大文件大小：50MB
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "仅支持PDF文件")

    try:
        # 保存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # 创建一个进度回调
        progress = ProgressCallback()
        
        # 使用rag_demo中的处理函数
        result_text = await asyncio.to_thread(
            rag_demo.process_multiple_pdfs,
            [type('obj', (object,), {"name": tmp_path})],
            progress
        )
        
        # 清理临时文件
        os.unlink(tmp_path)
        
        # 解析结果
        result = result_text[0] if isinstance(result_text, tuple) else result_text
        chunk_match = re.search(r'(\d+) 个文本块', result)
        chunks = int(chunk_match.group(1)) if chunk_match else 0
        
        return {
            "status": "success" if "成功" in result else "error",
            "message": result,
            "file_info": {
                "filename": file.filename,
                "chunks": chunks
            }
        }
    except Exception as e:
        logger.error(f"PDF处理失败: {str(e)}")
        raise HTTPException(500, f"文档处理失败: {str(e)}") from e

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    """
    问答接口
    - question: 问题内容
    - enable_web_search: 是否启用网络搜索增强（默认False）
    """
    if not req.question:
        raise HTTPException(400, "问题不能为空")
    
    try:
        # 使用流式回答生成结果
        answer = await process_answer_stream(req.question, req.enable_web_search)
        
        # 提取可能的来源信息
        sources = []
        
        # 尝试提取标记的URL内容
        url_matches = re.findall(r'\[(网络来源|本地文档):[^\]]+\]\s*(?:\(URL:\s*([^)]+)\))?', answer)
        for source_type, url in url_matches:
            if url:
                sources.append({"type": source_type, "url": url})
            else:
                sources.append({"type": source_type})
        
        # 如果没有找到标记的URL，尝试解析其他格式
        if not sources:
            if "来源:" in answer or "来源：" in answer:
                source_sections = re.findall(r'来源[:|：](.*?)(?=来源[:|：]|$)', answer, re.DOTALL)
                for section in source_sections:
                    sources.append({"type": "引用", "content": section.strip()})
        
        return {
            "answer": answer,
            "sources": sources,
            "metadata": {
                "enable_web_search": req.enable_web_search,
                "model": "deepseek-r1:1.5b"
            }
        }
    except Exception as e:
        logger.error(f"问答失败: {str(e)}")
        raise HTTPException(500, f"问答处理失败: {str(e)}") from e

@app.get("/api/status")
async def check_status():
    """检查API服务状态和环境配置"""
    ollama_status = rag_demo.check_environment()
    serpapi_status = rag_demo.check_serpapi_key()
    
    return {
        "status": "healthy" if ollama_status else "degraded",
        "ollama_service": ollama_status,
        "serpapi_configured": serpapi_status,
        "version": "1.0.0",
        "models": ["deepseek-r1:1.5b", "deepseek-r1:7b"]
    }

@app.get("/api/web_search_status")
async def check_web_search():
    """检查网络搜索功能是否可用"""
    serpapi_key = rag_demo.SERPAPI_KEY
    return {
        "web_search_available": bool(serpapi_key and serpapi_key.strip()),
        "serpapi_configured": bool(serpapi_key and serpapi_key.strip())
    }

if __name__ == "__main__":
    import uvicorn
    port = 17995
    
    # 尝试使用rag_demo中的端口检测逻辑
    ports = [17995, 17996, 17997, 17998, 17999]
    for p in ports:
        if rag_demo.is_port_available(p):
            port = p
            break
    
    logger.info(f"正在启动API服务，端口: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port) 