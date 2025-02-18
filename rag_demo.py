import gradio as gr
from pdfminer.high_level import extract_text_to_fp
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import json
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import socket
import webbrowser
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime

# 在文件开头添加超时设置
import requests
requests.adapters.DEFAULT_RETRIES = 3  # 增加重试次数

# 在文件开头添加环境变量设置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

# 在文件最开头添加代理配置
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'  # 新增代理绕过设置

# 初始化组件
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
CHROMA_CLIENT = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)
COLLECTION = CHROMA_CLIENT.get_or_create_collection("rag_docs")

logging.basicConfig(level=logging.INFO)

print("Gradio version:", gr.__version__)  # 添加版本输出

# 在初始化组件后添加：
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount('http://', HTTPAdapter(max_retries=retries))

# 添加文件处理状态跟踪
class FileProcessor:
    def __init__(self):
        self.processed_files = {}  # 存储已处理文件的状态
        
    def clear_files(self):
        """清空所有文件记录"""
        self.processed_files = {}
        
    def add_file(self, file_name):
        self.processed_files[file_name] = {
            'status': '等待处理',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chunks': 0
        }
        
    def update_status(self, file_name, status, chunks=None):
        if file_name in self.processed_files:
            self.processed_files[file_name]['status'] = status
            if chunks is not None:
                self.processed_files[file_name]['chunks'] = chunks
                
    def get_file_list(self):
        return [
            f"📄 {fname} | {info['status']}"
            for fname, info in self.processed_files.items()
        ]

file_processor = FileProcessor()

def extract_text(filepath):
    """改进的PDF文本提取方法"""
    output = StringIO()
    with open(filepath, 'rb') as file:
        extract_text_to_fp(file, output)
    return output.getvalue()

def process_multiple_pdfs(files, progress=gr.Progress()):
    """处理多个PDF文件"""
    if not files:
        return "请选择要上传的PDF文件", []
    
    try:
        # 清空向量数据库
        progress(0.1, desc="清理历史数据...")
        try:
            # 获取所有现有文档的ID
            existing_data = COLLECTION.get()
            if existing_data and existing_data['ids']:
                COLLECTION.delete(ids=existing_data['ids'])
            logging.info("成功清理历史向量数据")
        except Exception as e:
            logging.error(f"清理历史数据时出错: {str(e)}")
            return f"清理历史数据失败: {str(e)}", []
        
        # 清空文件处理状态
        file_processor.clear_files()
        
        total_files = len(files)
        processed_results = []
        total_chunks = 0
        
        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx-1)/total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
                
                # 添加文件到处理器
                file_processor.add_file(file_name)
                
                # 处理单个文件
                text = extract_text(file.name)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_text(text)
                
                if not chunks:
                    raise ValueError("文档内容为空或无法提取文本")
                
                # 生成文档唯一标识符
                doc_id = f"doc_{int(time.time())}_{idx}"
                
                # 生成嵌入
                embeddings = EMBED_MODEL.encode(chunks)
                
                # 存储向量，添加文档源信息
                ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
                
                COLLECTION.add(
                    ids=ids,
                    embeddings=embeddings.tolist(),
                    documents=chunks,
                    metadatas=metadatas
                )
                
                # 更新处理状态
                total_chunks += len(chunks)
                file_processor.update_status(file_name, "处理完成", len(chunks))
                processed_results.append(f"✅ {file_name}: 成功处理 {len(chunks)} 个文本块")
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"处理文件 {file_name} 时出错: {error_msg}")
                file_processor.update_status(file_name, f"处理失败: {error_msg}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")
        
        # 添加总结信息
        summary = f"\n总计处理 {total_files} 个文件，{total_chunks} 个文本块"
        processed_results.append(summary)
        
        # 获取更新后的文件列表
        file_list = file_processor.get_file_list()
        
        return "\n".join(processed_results), file_list
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"整体处理过程出错: {error_msg}")
        return f"处理过程出错: {error_msg}", []

def stream_answer(question, progress=gr.Progress()):
    """改进的流式问答处理流程"""
    try:
        progress(0.3, desc="生成问题嵌入...")
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.5, desc="检索相关内容...")
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=3,
            include=['documents', 'metadatas']
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            source = metadata.get('source', '未知来源')
            context_with_sources.append(f"[来源: {source}]\n{doc}")
        
        context = "\n\n".join(context_with_sources)
        prompt = f"""基于以下上下文：
        {context}
        
        问题：{question}
        请用中文给出详细回答，并在回答末尾标注信息来源："""
        
        progress(0.7, desc="生成回答...")
        full_answer = ""
        
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": True
            },
            timeout=120,
            stream=True
        )
        
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode()).get("response", "")
                full_answer += chunk
                yield full_answer, "生成回答中..."
                
        yield full_answer, "完成!"
        
    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"

def query_answer(question, progress=gr.Progress()):
    """问答处理流程"""
    try:
        logging.info(f"收到问题：{question}")
        progress(0.3, desc="生成问题嵌入...")
        # 生成问题嵌入
        query_embedding = EMBED_MODEL.encode([question]).tolist()
        
        progress(0.5, desc="检索相关内容...")
        # Chroma检索
        results = COLLECTION.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        # 构建提示词
        context = "\n".join(results['documents'][0])
        prompt = f"""基于以下上下文：
        {context}
        
        问题：{question}
        请用中文给出详细回答："""
        
        progress(0.7, desc="生成回答...")
        # 调用Ollama
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:7b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120,  # 延长到2分钟
            headers={'Connection': 'close'}  # 添加连接头
        )
        response.raise_for_status()  # 检查HTTP状态码
        
        progress(1.0, desc="完成!")
        # 确保返回字符串并处理空值
        result = response.json()
        return str(result.get("response", "未获取到有效回答"))
    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except KeyError:
        return "响应格式异常，请检查模型服务"
    except Exception as e:
        progress(1.0, desc="遇到错误")  # 确保进度条完成
        return f"系统错误: {str(e)}"

# 修改界面布局部分
with gr.Blocks(
    title="本地RAG问答系统",
    css="""
    /* 全局主题变量 */
    :root[data-theme="light"] {
        --text-color: #2c3e50;
        --bg-color: #ffffff;
        --panel-bg: #f8f9fa;
        --border-color: #e9ecef;
        --success-color: #4CAF50;
        --error-color: #f44336;
        --primary-color: #2196F3;
        --secondary-bg: #ffffff;
        --hover-color: #e9ecef;
        --chat-user-bg: #e3f2fd;
        --chat-assistant-bg: #f5f5f5;
    }

    :root[data-theme="dark"] {
        --text-color: #e0e0e0;
        --bg-color: #1a1a1a;
        --panel-bg: #2d2d2d;
        --border-color: #404040;
        --success-color: #81c784;
        --error-color: #e57373;
        --primary-color: #64b5f6;
        --secondary-bg: #2d2d2d;
        --hover-color: #404040;
        --chat-user-bg: #1e3a5f;
        --chat-assistant-bg: #2d2d2d;
    }

    /* 全局样式 */
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    .gradio-container {
        max-width: 1200px !important;
        color: var(--text-color);
        background-color: var(--bg-color);
    }

    /* 主题切换按钮 */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        background: var(--panel-bg);
        color: var(--text-color);
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .theme-toggle:hover {
        background: var(--hover-color);
    }

    /* 面板样式 */
    .left-panel {
        padding-right: 20px;
        border-right: 1px solid var(--border-color);
        background: var(--bg-color);
    }

    .right-panel {
        height: 100vh;
        background: var(--bg-color);
    }

    /* 文件列表样式 */
    .file-list {
        margin-top: 10px;
        padding: 12px;
        background: var(--panel-bg);
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 答案框样式 */
    .answer-box {
        min-height: 500px !important;
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        font-size: 15px;
        line-height: 1.6;
        border: 1px solid var(--border-color);
    }

    /* 输入框样式 */
    textarea {
        background: var(--panel-bg) !important;
        color: var(--text-color) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        font-size: 14px !important;
    }

    /* 按钮样式 */
    button.primary {
        background: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    button.primary:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* 标题和文本样式 */
    h1, h2, h3 {
        color: var(--text-color) !important;
        font-weight: 600 !important;
    }

    .footer-note {
        color: var(--text-color);
        opacity: 0.8;
        font-size: 13px;
        margin-top: 12px;
    }

    /* 加载和进度样式 */
    #loading, .progress-text {
        color: var(--text-color);
    }

    /* 聊天记录样式 */
    .chat-container {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        margin-bottom: 16px;
        max-height: 600px;
        overflow-y: auto;
        background: var(--bg-color);
    }

    .chat-message {
        padding: 12px 16px;
        margin: 8px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
    }

    .chat-message.user {
        background: var(--chat-user-bg);
        margin-left: 32px;
        border-top-right-radius: 4px;
    }

    .chat-message.assistant {
        background: var(--chat-assistant-bg);
        margin-right: 32px;
        border-top-left-radius: 4px;
    }

    .chat-message .timestamp {
        font-size: 12px;
        color: var(--text-color);
        opacity: 0.7;
        margin-bottom: 4px;
    }

    .chat-message .content {
        white-space: pre-wrap;
    }

    /* 按钮组样式 */
    .button-row {
        display: flex;
        gap: 8px;
        margin-top: 8px;
    }

    .clear-button {
        background: var(--error-color) !important;
    }
    """
) as demo:
    gr.Markdown("# 🧠 智能文档问答系统")
    
    with gr.Row():
        # 左侧操作面板
        with gr.Column(scale=1, elem_classes="left-panel"):
            gr.Markdown("## 📂 文档处理区")
            with gr.Group():
                file_input = gr.File(
                    label="上传PDF文档",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("🚀 开始处理", variant="primary")
                upload_status = gr.Textbox(
                    label="处理状态",
                    interactive=False,
                    lines=2
                )
                file_list = gr.Textbox(
                    label="已处理文件",
                    interactive=False,
                    lines=3,
                    elem_classes="file-list"
                )

        # 右侧对话区
        with gr.Column(scale=3, elem_classes="right-panel"):
            gr.Markdown("## 📝 对话记录")
            
            # 对话记录显示区
            chatbot = gr.Chatbot(
                label="对话历史",
                height=500,
                elem_classes="chat-container",
                show_label=False
            )
            
            # 问题输入区
            with gr.Group():
                question_input = gr.Textbox(
                    label="输入问题",
                    lines=3,
                    placeholder="请输入您的问题...",
                    elem_id="question-input"
                )
                with gr.Row():
                    ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button", scale=1)
                status_display = gr.HTML("", elem_id="status-display")
            
            gr.Markdown("""
            <div class="footer-note">
                *回答生成可能需要1-2分钟，请耐心等待<br>
                *支持多轮对话，可基于前文继续提问
            </div>
            """)

    # 调整后的加载提示
    gr.HTML("""
    <div id="loading" style="text-align:center;padding:20px;">
        <h3>🔄 系统初始化中，请稍候...</h3>
    </div>
    """)

    # 进度显示组件调整到左侧面板下方
    with gr.Row(visible=False) as progress_row:
        gr.HTML("""
        <div class="progress-text">
            <span>当前进度：</span>
            <span id="current-step" style="color: #2b6de3;">初始化...</span>
            <span id="progress-percent" style="margin-left:15px;color: #e32b2b;">0%</span>
        </div>
        """)

    def clear_chat_history():
        return [], ""  # 清空对话历史和输入框

    # 修改问答处理函数
    def process_chat(question, history):
        if not question:
            return history, ""
        
        history = history or []
        history.append([question, None])
        
        try:
            for response, status in stream_answer(question):
                if status != "遇到错误":
                    history[-1][1] = response
                    yield history, ""
                else:
                    history[-1][1] = f"❌ {response}"
                    yield history, ""
        except Exception as e:
            history[-1][1] = f"❌ 系统错误: {str(e)}"
            yield history, ""

    # 更新事件处理
    ask_btn.click(
        fn=process_chat,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
        show_progress=False
    ).then(
        fn=lambda: "",
        outputs=status_display
    )

    clear_btn.click(
        fn=clear_chat_history,
        outputs=[chatbot, question_input],
        show_progress=False
    )

    # 添加文件处理按钮事件
    upload_btn.click(
        fn=process_multiple_pdfs,
        inputs=file_input,
        outputs=[upload_status, file_list]
    )

# 修改JavaScript注入部分
demo._js = """
function gradioApp() {
    // 设置默认主题为暗色
    document.documentElement.setAttribute('data-theme', 'dark');
    
    const observer = new MutationObserver((mutations) => {
        document.getElementById("loading").style.display = "none";
        const progress = document.querySelector('.progress-text');
        if (progress) {
            const percent = document.querySelector('.progress > div')?.innerText || '';
            const step = document.querySelector('.progress-description')?.innerText || '';
            document.getElementById('current-step').innerText = step;
            document.getElementById('progress-percent').innerText = percent;
        }
    });
    observer.observe(document.body, {childList: true, subtree: true});
}

function toggleTheme() {
    const root = document.documentElement;
    const currentTheme = root.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    root.setAttribute('data-theme', newTheme);
}

// 初始化主题
document.addEventListener('DOMContentLoaded', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
});
"""

# 修改端口检查函数
def is_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0  # 更可靠的检测方式

def check_environment():
    """环境依赖检查"""
    try:
        # 添加模型存在性检查
        model_check = session.post(
            "http://localhost:11434/api/show",
            json={"name": "deepseek-r1:7b"},
            timeout=10
        )
        if model_check.status_code != 200:
            print("模型未加载！请先执行：")
            print("ollama pull deepseek-r1:7b")
            return False
            
        # 原有检查保持不变...
        response = session.get(
            "http://localhost:11434/api/tags",
            proxies={"http": None, "https": None},  # 禁用代理
            timeout=5
        )
        if response.status_code != 200:
            print("Ollama服务异常，返回状态码:", response.status_code)
            return False
        return True
    except Exception as e:
        print("Ollama连接失败:", str(e))
        return False

# 方案2：禁用浏览器缓存（添加meta标签）
gr.HTML("""
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
""")

# 恢复主程序启动部分
if __name__ == "__main__":
    if not check_environment():
        exit(1)
    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)
    
    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)
        
    try:
        ollama_check = session.get("http://localhost:11434", timeout=5)
        if ollama_check.status_code != 200:
            print("Ollama服务未正常启动！")
            print("请先执行：ollama serve 启动服务")
            exit(1)
            
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port,
            server_name="0.0.0.0",
            show_error=True,
            ssl_verify=False
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")

