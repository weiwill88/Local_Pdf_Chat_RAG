"""
🧠 本地化智能问答系统（FAISS版）—— 主入口

本文件职责：
- Gradio Web UI 的布局与事件绑定
- 文档处理的编排（调用 core/ 模块完成各步骤）
- 系统监控面板
- 应用启动

核心 RAG 逻辑已拆分到 core/ 和 features/ 模块中，
请按照 core/__init__.py 中的学习路线逐模块阅读。
"""

import os
import time
import logging
import webbrowser
import gradio as gr
import jieba
from typing import List, Tuple, Optional
from datetime import datetime

# 导入配置
from config import (
    DEFAULT_MODEL_CHOICE, SILICONFLOW_API_KEY, MAGICK_API_KEY,
    OLLAMA_MODEL_NAME, SILICONFLOW_MODEL_NAME, MAGICK_MODEL_NAME,
    MODEL_CHOICES, MODEL_DISPLAY_NAMES, is_configured_api_key
)

# 导入核心模块
from utils.document_loader import load_document, is_ocr_available
from core.text_splitter import split_text
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.generator import query_answer, call_siliconflow_api, call_magick_api

# 导入知识库管理器
from core.knowledge_base_manager import kb_manager, BASE_DIR

# 导入对话历史管理
from utils.chat_history import init_db, save_message, get_history, export_to_markdown, clear_history

# 导入工具
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO)
print("Gradio version:", gr.__version__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 启动时初始化数据库 + 加载已有的知识库
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
init_db()
print("🗄️ 对话历史数据库已初始化")

initial_kb_list = kb_manager.list_kbs()
initial_chat_history = []
if initial_kb_list:
    kb_manager.load_kb(initial_kb_list[0])
    initial_chat_history = [
        {"role": r, "content": c}
        for r, c, _ in get_history(initial_kb_list[0])
    ]
    print(f"📚 已加载知识库「{initial_kb_list[0]}」({vector_store.total_chunks} 个文本块, {len(initial_chat_history)} 条对话记录)")
else:
    print("📚 暂无知识库，请创建新知识库")

# Tesseract OCR 前置检测
ocr_avail = is_ocr_available()
_tesseract_status_html = (
    """<div class="tesseract-status" style="padding:8px 12px;border-radius:6px;font-size:13px;margin-top:4px;background:rgba(76,175,80,0.1);border:1px solid #4CAF50;color:#4CAF50">🖼️ Tesseract OCR 引擎: <strong>可用</strong>（扫描版 PDF 可正常识别）</div>"""
    if ocr_avail else
    """<div class="tesseract-status" style="padding:8px 12px;border-radius:6px;font-size:13px;margin-top:4px;background:rgba(244,67,54,0.1);border:1px solid #f44336;color:#f44336">🖼️ Tesseract OCR 引擎: <strong>未安装</strong> — 扫描版 PDF 将跳过 OCR。<br>安装: <a href="https://github.com/tesseract-ocr/tesseract" target="_blank">https://github.com/tesseract-ocr/tesseract</a></div>"""
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文档处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_multiple_files(files, kb_name_input="", kb_selector_val="", progress=gr.Progress()):
    """处理多个文件：提取文本 → 分块 → 向量化 → 构建索引 → 保存知识库"""
    if not files:
        return "请选择要上传的文件", [], gr.update()

    # 确定目标知识库名称
    target_kb = kb_name_input.strip() if kb_name_input and kb_name_input.strip() else (kb_selector_val or "")
    if not target_kb:
        target_kb = "default"
        logging.info("未指定知识库名称，使用默认知识库「default」")

    is_new_kb = target_kb not in kb_manager.list_kbs()

    try:
        # 如果是已有知识库，先加载（追加模式）
        if not is_new_kb:
            progress(0.05, desc=f"加载已有知识库「{target_kb}」...")
            kb_manager.load_kb(target_kb)
        else:
            progress(0.05, desc=f"创建新知识库「{target_kb}」...")
            # 如果是新知识库且有切换过其他库，先清空
            if kb_manager.current_kb != target_kb:
                vector_store.clear()
                bm25_manager.clear()

        # 收集已有 + 新文件的全部文本块
        all_chunks = list(vector_store.contents_map.values())
        all_metadatas = list(vector_store.metadatas_map.values())
        all_ids = list(vector_store.id_order)

        total_files = len(files)
        processed_results = []
        batch_chunks, batch_metadatas, batch_ids = [], [], []

        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx - 1) / total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")

                docs = load_document(file.name)
                text = "\n\n".join([doc.page_content for doc in docs])
                if not text.strip():
                    raise ValueError("文档内容为空或无法提取文本")

                chunks = split_text(text)
                doc_id = f"doc_{int(time.time())}_{idx}"
                metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                batch_chunks.extend(chunks)
                batch_metadatas.extend(metadatas)
                batch_ids.extend(chunk_ids)
                processed_results.append(f"✅ {file_name}: 成功处理 {len(chunks)} 个文本块")

            except Exception as e:
                logging.error(f"处理文件 {file_name} 时出错: {str(e)}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {str(e)}")

        # 合并新旧数据
        all_chunks.extend(batch_chunks)
        all_metadatas.extend(batch_metadatas)
        all_ids.extend(batch_ids)

        if all_chunks:
            progress(0.75, desc="生成文本嵌入向量...")
            embeddings = encode_texts(all_chunks, show_progress=True)

            progress(0.85, desc="构建FAISS索引...")
            vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)

            progress(0.90, desc="构建BM25检索索引...")
            bm25_manager.build_index(all_chunks, all_ids)

            progress(0.95, desc=f"保存知识库「{target_kb}」到磁盘...")
            kb_manager.save_current_kb(target_kb)

        action = "新建" if is_new_kb else "更新"
        summary = f"\n{action}知识库「{target_kb}」完成，共 {len(all_chunks)} 个文本块"
        processed_results.append(summary)

        # 刷新下拉框
        updated_choices = kb_manager.list_kbs()

        return "\n".join(processed_results), [f"📄 {os.path.basename(f.name)}" for f in files], gr.update(choices=updated_choices, value=target_kb)

    except Exception as e:
        logging.error(f"处理过程出错: {str(e)}")
        return f"处理过程出错: {str(e)}", [], gr.update()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分块可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chunk_data_cache = {}


def get_document_chunks(progress=gr.Progress()):
    """获取文档分块结果用于可视化"""
    global chunk_data_cache
    try:
        progress(0.1, desc="加载数据...")
        chunk_data_cache.clear()

        if not vector_store.id_order:
            return [], "知识库中没有文档，请先上传并处理文档。"

        table_data = []
        for idx, chunk_id in enumerate(vector_store.id_order):
            content = vector_store.contents_map.get(chunk_id, "")
            meta = vector_store.metadatas_map.get(chunk_id, {})
            if not content:
                continue
            chunk_data = {
                "row_id": idx, "chunk_id": chunk_id,
                "source": meta.get("source", "未知来源"), "content": content,
                "preview": content[:200] + "..." if len(content) > 200 else content,
                "char_count": len(content),
                "token_count": len(list(jieba.cut(content)))
            }
            chunk_data_cache[idx] = chunk_data
            table_data.append([
                chunk_data["source"], f"{idx + 1}/{len(vector_store.id_order)}",
                chunk_data["char_count"], chunk_data["token_count"], chunk_data["preview"]
            ])

        progress(1.0, desc="完成!")
        return table_data, f"共 {len(table_data)} 个文本块"
    except Exception as e:
        chunk_data_cache.clear()
        return [], f"获取分块数据失败: {str(e)}"


def show_chunk_details(evt: gr.SelectData):
    """显示选中分块的详细内容"""
    try:
        if not evt.index or evt.index[0] is None:
            return "未选择有效行"
        selected = chunk_data_cache.get(evt.index[0])
        if not selected:
            return "未找到对应的分块数据"
        return f"""[来源] {selected['source']}
[ID] {selected['chunk_id']}
[字符数] {selected['char_count']}
[分词数] {selected['token_count']}
----------------------------
{selected['content']}"""
    except Exception as e:
        return f"加载失败: {str(e)}"


def get_system_models_info():
    """返回系统使用的各种模型信息"""
    return {
        "嵌入模型": "all-MiniLM-L6-v2",
        "分块方法": "RecursiveCharacterTextSplitter (chunk_size=400, overlap=40)",
        "检索方法": "向量检索 + BM25混合检索 (α=0.7)",
        "重排序模型": "交叉编码器 (distiluse-base-multilingual-cased-v2)",
        "生成模型(Ollama)": OLLAMA_MODEL_NAME,
        "生成模型(SiliconFlow)": SILICONFLOW_MODEL_NAME,
        "生成模型(Magick API)": MAGICK_MODEL_NAME,
        "分词工具": "jieba (中文分词)"
    }


def get_model_display_name(model_choice_val):
    """返回 UI 中展示的模型服务名称。"""
    return MODEL_DISPLAY_NAMES.get(model_choice_val, f"未知模型服务({model_choice_val})")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 知识库切换
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def switch_knowledge_base(kb_name):
    """切换当前问答所用的知识库，同时加载该知识库的对话历史"""
    if not kb_name:
        return "请选择知识库", []
    success = kb_manager.load_kb(kb_name)
    if success:
        history = get_history(kb_name)
        chat_history = [{"role": r, "content": c} for r, c, _ in history]
        return f"✅ 已切换到知识库「{kb_name}」（{vector_store.total_chunks} 个文本块，{len(chat_history)} 条对话记录）", chat_history
    else:
        return f"⚠️ 知识库「{kb_name}」加载失败", []


def get_kb_dropdown_choices():
    """获取知识库下拉选项（带文本块数信息）"""
    kbs = kb_manager.list_kbs()
    if not kbs:
        return []
    return kbs


def export_chat_markdown():
    """将当前知识库的对话导出为 .md 文件"""
    kb_name = kb_manager.current_kb
    if not kb_name:
        return None, "⚠️ 没有当前知识库，无法导出"
    md = export_to_markdown(kb_name)
    # 写入临时文件提供下载
    import tempfile
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    tmp.write(md)
    tmp.close()
    return tmp.name, f"✅ 对话已导出：{kb_name}.md ({len(md)} 字符)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio UI（Gradio 6.x 兼容）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CSS = """
/* 补充性样式 —— 不覆盖 Gradio 6 核心组件，只做细节增强 */
.gradio-container { max-width:100%!important; width:100%!important; }
.left-panel { padding:16px; border-radius:12px; }
.right-panel { border-radius:12px; }
.file-list { margin-top:10px; }
.footer-note { opacity:0.7; font-size:13px; margin-top:12px; }
.chunk-detail-box { min-height:200px; font-family:monospace; white-space:pre-wrap; }
.monitor-panel { border-radius:12px; padding:20px; margin-bottom:20px; }
.metric-title { font-size:14px; margin-bottom:10px; }
.metric-value { font-size:24px; font-weight:700; margin-bottom:5px; }
.metric-trend { font-size:12px; color:#4CAF50; }
.progress-container { width:100%; background:rgba(128,128,128,0.2); border-radius:10px; margin:10px 0; }
.progress-bar { height:8px; border-radius:10px;
    background:linear-gradient(90deg, #00bcd4, #7b1fa2); transition:width 0.3s ease; }
.log-container { max-height:300px; overflow-y:auto; border-radius:8px; padding:15px;
    font-family:monospace; font-size:13px; }
.theme-toggle-btn { min-width:40px!important; font-size:20px!important; padding:4px 8px!important; }
"""

# 主题切换 JS（Gradio 6 通过 body.classList.toggle('dark') 切换暗色模式）
THEME_JS = """
function() {
    // 读取上次保存的主题偏好，默认白色
    const saved = localStorage.getItem('rag-theme');
    if (saved === 'dark') {
        document.querySelector('body').classList.add('dark');
    }
}
"""

def toggle_theme():
    """返回切换主题的 JS 代码（通过 Gradio 的 js 参数执行）"""
    return gr.update()

with gr.Blocks(title="本地RAG问答系统") as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("# 🧠 智能文档问答系统")
        with gr.Column(scale=1, min_width=60):
            theme_btn = gr.Button("🌓", min_width=40, elem_classes="theme-toggle-btn")

    with gr.Tabs() as tabs:
        # ━━━ 问答对话标签页 ━━━
        with gr.TabItem("💬 问答对话"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes="left-panel"):
                    gr.Markdown("## 📂 文档处理区")
                    with gr.Group():
                        with gr.Row(equal_height=True):
                            kb_name_input = gr.Textbox(
                                label="📁 新建知识库名称（输入名称创建新库）",
                                placeholder="如：学习资料、面试题库…",
                                scale=2
                            )
                            kb_selector = gr.Dropdown(
                                choices=initial_kb_list,
                                value=initial_kb_list[0] if initial_kb_list else None,
                                label="📚 当前知识库（下拉切换）",
                                interactive=True,
                                scale=1
                            )
                        file_input = gr.File(
                            label="上传文档 (支持PDF, Word, Excel, PPT, TXT, Markdown等)",
                            file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"],
                            file_count="multiple"
                        )
                        tesseract_status = gr.HTML("")
                        upload_btn = gr.Button("🚀 开始处理", variant="primary")
                        upload_status = gr.Textbox(label="处理状态", interactive=False, lines=2)
                        file_list = gr.Textbox(label="已处理文件", interactive=False, lines=3, elem_classes="file-list")

                    gr.Markdown("## ❓ 输入问题")
                    with gr.Group():
                        question_input = gr.Textbox(label="输入问题", lines=3, placeholder="请输入您的问题...")
                        with gr.Row():
                            web_search_checkbox = gr.Checkbox(
                                label="启用联网搜索", value=False,
                                info="打开后将同时搜索网络内容（需配置SERPAPI_KEY）"
                            )
                            model_choice = gr.Dropdown(
                                choices=MODEL_CHOICES,
                                value=DEFAULT_MODEL_CHOICE,
                                label="模型选择", info="选择使用本地模型或云端模型"
                            )
                        with gr.Row():
                            ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button", scale=1)
                    api_info = gr.HTML("")

                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## 📝 对话记录")
                    chatbot = gr.Chatbot(label="对话历史", height=520, elem_classes="chat-container",
                                         show_label=False, value=initial_chat_history)
                    with gr.Row():
                        export_btn = gr.Button("📥 导出对话(.md)", variant="secondary", scale=1)
                        export_file = gr.File(label="下载对话记录", visible=True, scale=2)
                    status_display = gr.HTML("")
                    gr.Markdown("""<div class="footer-note">
                        *回答生成可能需要1-2分钟，请耐心等待<br>*支持多轮对话，可基于前文继续提问
                    </div>""")

        # ━━━ 分块可视化标签页 ━━━
        with gr.TabItem("📊 分块可视化"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💡 系统模型信息")
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### 核心模型与技术")
                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:")
                                gr.Markdown(f"{value}")
                with gr.Column(scale=2):
                    gr.Markdown("## 📄 文档分块统计")
                    refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")
                    chunks_status = gr.Markdown("点击按钮查看分块统计")
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["来源", "序号", "字符数", "分词数", "内容预览"],
                    elem_classes="chunk-table", interactive=False, wrap=True, row_count=(10, "dynamic")
                )
            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="分块详情", placeholder="点击表格中的行查看完整内容...",
                    lines=8, elem_classes="chunk-detail-box"
                )

        # ━━━ 系统监控标签页 ━━━
        with gr.TabItem("📈 系统监控"):
            with gr.Column():
                with gr.Group(elem_classes="monitor-panel"):
                    with gr.Row():
                        gr.Markdown("## 🖥️ 系统资源监控")
                        refresh_monitor_btn = gr.Button("🔄 刷新数据", variant="primary")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("CPU使用率", elem_classes="metric-title")
                            cpu_value = gr.Markdown("加载中...", elem_classes="metric-value")
                            cpu_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            cpu_info = gr.Markdown("核心数: 加载中...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("内存使用", elem_classes="metric-title")
                            memory_value = gr.Markdown("加载中...", elem_classes="metric-value")
                            memory_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            memory_info = gr.Markdown("总内存: 加载中...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("磁盘空间", elem_classes="metric-title")
                            disk_value = gr.Markdown("加载中...", elem_classes="metric-value")
                            disk_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            disk_info = gr.Markdown("总空间: 加载中...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("向量数据库", elem_classes="metric-title")
                            vector_db_value = gr.Markdown("分块数: 0", elem_classes="metric-value")
                            vector_db_info = gr.Markdown("向量数: 0", elem_classes="metric-trend")
                            with gr.Row():
                                kb_info_display = gr.Markdown("当前知识库: 无", elem_classes="metric-trend")

                with gr.Group(elem_classes="monitor-panel"):
                    gr.Markdown("## 📝 系统日志")
                    with gr.Row():
                        log_level = gr.Dropdown(choices=["所有级别", "信息", "警告", "错误"], value="所有级别", label="日志级别")
                        clear_logs_btn = gr.Button("🗑️ 清空日志", variant="secondary")
                    log_display = gr.HTML("", elem_classes="log-container")

    # ━━━ 事件处理函数 ━━━
    def clear_chat_history():
        kb_name = kb_manager.current_kb
        if kb_name:
            clear_history(kb_name)
        return [], "对话已清空（数据库已同步清除）"

    def process_chat(question, history, enable_web_search, model_choice_val):
        if history is None or not isinstance(history, list):
            history = []

        api_text = """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong></p>
        </div>""" % (
            "已启用" if enable_web_search else "未启用",
            get_model_display_name(model_choice_val)
        )

        if not question or question.strip() == "":
            history.append({"role": "assistant", "content": "问题不能为空，请输入有效问题。"})
            return history, "", api_text

        try:
            answer = query_answer(question, enable_web_search, model_choice_val)
        except Exception as e:
            answer = f"系统错误: {str(e)}"
            logging.error(f"问答处理异常: {str(e)}")

        # 保存到数据库
        kb_name = kb_manager.current_kb or "default"
        save_message(kb_name, "user", question)
        save_message(kb_name, "assistant", answer)

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", api_text

    def update_api_info(enable_web_search, model_choice_val):
        return """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong></p>
        </div>""" % (
            "已启用" if enable_web_search else "未启用",
            get_model_display_name(model_choice_val)
        )

    def get_system_metrics():
        """获取系统监控数据"""
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=1)
            cpu_cnt = psutil.cpu_count(logical=False)
            mem = psutil.virtual_memory()
            mem_total = round(mem.total / (1024 ** 3), 1)
            mem_used = round(mem.used / (1024 ** 3), 1)
            disk = psutil.disk_usage('/')
            disk_total = round(disk.total / (1024 ** 3), 1)
            disk_used = round(disk.used / (1024 ** 3), 1)

            doc_count = len(vector_store.contents_map)
            vec_count = vector_store.total_chunks
            current_kb_name = kb_manager.current_kb or "无"

            def bar(pct, color="var(--tech-cyan)"):
                return f'<div class="progress-container"><div class="progress-bar" style="width:{pct}%;background:{color}"></div></div>'

            c_color = "#4CAF50" if cpu_pct < 50 else "#FFC107" if cpu_pct < 80 else "#f44336"
            m_color = "#4CAF50" if mem.percent < 50 else "#FFC107" if mem.percent < 80 else "#f44336"
            d_color = "#4CAF50" if disk.percent < 50 else "#FFC107" if disk.percent < 80 else "#f44336"

            now = datetime.now().strftime("%H:%M:%S")
            log = f'<div class="log-entry"><span style="color:var(--tech-cyan)">[{now}]</span> <span style="color:#4CAF50">[INFO]</span> 监控数据已更新</div>'

            return (
                f"{cpu_pct}%", bar(cpu_pct, c_color), f"物理核心: {cpu_cnt}",
                f"{mem_used}GB / {mem_total}GB", bar(mem.percent, m_color), f"使用率: {mem.percent}%",
                f"{disk_used}GB / {disk_total}GB", bar(disk.percent, d_color), f"使用率: {disk.percent}%",
                f"分块数: {doc_count}", f"向量数: {vec_count}",
                f"当前知识库: **{current_kb_name}**", log
            )
        except Exception as e:
            err = f"监控错误: {str(e)}"
            return ("错误", "", err, "错误", "", err, "错误", "", err, "错误", err, "错误",
                    f"<div style='color:#f44336'>[ERROR] {err}</div>")

    # ━━━ 页面加载/刷新时加载当前知识库的对话历史 ━━━
    def load_current_kb_history():
        """页面加载/刷新时动态加载当前知识库的对话历史"""
        kb_name = kb_manager.current_kb
        if not kb_name:
            return []
        history = get_history(kb_name)
        return [{"role": r, "content": c} for r, c, _ in history]

    # ━━━ 绑定事件 ━━━
    upload_btn.click(
        process_multiple_files,
        inputs=[file_input, kb_name_input, kb_selector],
        outputs=[upload_status, file_list, kb_selector],
        show_progress=True
    )
    kb_selector.change(
        fn=switch_knowledge_base,
        inputs=[kb_selector],
        outputs=[upload_status, chatbot]
    )
    export_btn.click(
        fn=export_chat_markdown,
        inputs=[],
        outputs=[export_file, status_display]
    )
    ask_btn.click(process_chat, inputs=[question_input, chatbot, web_search_checkbox, model_choice],
                  outputs=[chatbot, question_input, api_info])
    clear_btn.click(clear_chat_history, inputs=[], outputs=[chatbot, status_display])
    web_search_checkbox.change(update_api_info, inputs=[web_search_checkbox, model_choice], outputs=[api_info])
    model_choice.change(update_api_info, inputs=[web_search_checkbox, model_choice], outputs=[api_info])
    refresh_chunks_btn.click(fn=get_document_chunks, outputs=[chunks_data, chunks_status])
    chunks_data.select(fn=show_chunk_details, outputs=chunk_detail_text)
    refresh_monitor_btn.click(fn=get_system_metrics, outputs=[
        cpu_value, cpu_progress, cpu_info,
        memory_value, memory_progress, memory_info,
        disk_value, disk_progress, disk_info,
        vector_db_value, vector_db_info, kb_info_display, log_display
    ])
    clear_logs_btn.click(fn=lambda: "<div style='color:#4CAF50'>日志已清空</div>", outputs=[log_display])
    theme_btn.click(fn=toggle_theme, inputs=[], outputs=[], js="""
        () => {
            document.querySelector('body').classList.toggle('dark');
            const isDark = document.querySelector('body').classList.contains('dark');
            localStorage.setItem('rag-theme', isDark ? 'dark' : 'light');
        }
    """)

    # 页面刷新时重新加载对话历史（解决刷新后历史清空的问题）
    demo.load(
        fn=lambda: (load_current_kb_history(), _tesseract_status_html),
        inputs=[], outputs=[chatbot, tesseract_status]
    )


def check_environment():
    """环境依赖检查"""
    if is_configured_api_key(SILICONFLOW_API_KEY):
        print("✅ SiliconFlow API 密钥已配置")
        try:
            result = call_siliconflow_api("你好，请回复'连接成功'", temperature=0.1, max_tokens=50)
            if isinstance(result, str) and ("连接成功" in result or "你好" in result):
                print("✅ SiliconFlow API 连接测试成功")
            else:
                print("⚠️ SiliconFlow API 响应异常，但继续运行")
            return True
        except Exception as e:
            print(f"⚠️ SiliconFlow API 测试失败: {e}")
            return True

    if is_configured_api_key(MAGICK_API_KEY):
        print("✅ Magick API 密钥已配置")
        try:
            result = call_magick_api("你好，请回复'连接成功'", temperature=0.1, max_tokens=50)
            if isinstance(result, str) and ("连接成功" in result or "你好" in result):
                print("✅ Magick API 连接测试成功")
            else:
                print("⚠️ Magick API 响应异常，但继续运行")
            return True
        except Exception as e:
            print(f"⚠️ Magick API 测试失败: {e}")
            return True

    print("⚠️ 未配置云端 API 密钥，将尝试使用本地 Ollama")
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            print("✅ 本地 Ollama 服务可用")
            return True
    except Exception:
        pass
    print("❌ 未找到任何可用的 LLM 后端")
    print("   请在 .env 中配置 SILICONFLOW_API_KEY / MAGICK_API_KEY 或启动 Ollama 服务")
    return False


if __name__ == "__main__":
    if not check_environment():
        exit(1)

    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("所有端口都被占用，请手动释放端口")
        exit(1)

    try:
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port, server_name="0.0.0.0",
            show_error=True, ssl_verify=False, height=900,
            css=CSS, js=THEME_JS
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")
