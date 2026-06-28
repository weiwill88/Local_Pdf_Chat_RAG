"""
🧠 Sistem Tanya Jawab Dokumen Pintar Lokal (Versi FAISS) —— Entri Utama

Tanggung jawab file ini:
- Tata letak Gradio Web UI dan pengikatan event
- Orkestrasi pemrosesan dokumen (memanggil modul core/ untuk menyelesaikan setiap langkah)
- Panel pemantauan sistem
- Peluncuran aplikasi

Logika inti RAG telah dipisahkan ke dalam modul core/ dan features/,
silakan baca modul demi modul sesuai dengan jalur pembelajaran di core/__init__.py.
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
    DEFAULT_MODEL_CHOICE, SILICONFLOW_API_KEY,
    OLLAMA_MODEL_NAME, SILICONFLOW_MODEL_NAME
)

# 导入核心模块
from core.document_loader import extract_text
from core.text_splitter import split_text
from core.embeddings import encode_texts
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.generator import query_answer, call_siliconflow_api

# 导入工具
from utils.network import is_port_available

logging.basicConfig(level=logging.INFO)
print("Gradio version:", gr.__version__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 文档处理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def process_multiple_files(files, progress=gr.Progress()):
    """处理多个文件：提取文本 → 分块 → 向量化 → 构建索引"""
    if not files:
        return "Silakan pilih file yang akan diunggah (mendukung PDF, Word, Excel, PPT, TXT, Markdown, dll.)", []

    try:
        progress(0.1, desc="Membersihkan data riwayat...")
        vector_store.clear()
        bm25_manager.clear()

        total_files = len(files)
        processed_results = []
        all_chunks, all_metadatas, all_ids = [], [], []

        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx - 1) / total_files, desc=f"Memproses file {idx}/{total_files}: {file_name}")

                text = extract_text(file.name)
                if not text:
                    raise ValueError("Konten dokumen kosong atau teks tidak dapat diekstrak")

                chunks = split_text(text)
                doc_id = f"doc_{int(time.time())}_{idx}"
                metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]
                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]

                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                all_ids.extend(chunk_ids)
                processed_results.append(f"✅ {file_name}: Berhasil memproses {len(chunks)} blok teks")

            except Exception as e:
                logging.error(f"Terjadi kesalahan saat memproses file {file_name}: {str(e)}")
                processed_results.append(f"❌ {file_name}: Gagal memproses - {str(e)}")

        if all_chunks:
            progress(0.8, desc="Menghasilkan embeddings teks...")
            embeddings = encode_texts(all_chunks, show_progress=True)

            progress(0.9, desc="Membangun indeks FAISS...")
            vector_store.build_index(all_chunks, all_ids, all_metadatas, embeddings)

        progress(0.95, desc="Membangun indeks pencarian BM25...")
        bm25_manager.build_index(all_chunks, all_ids)

        summary = f"\nTotal memproses {total_files} file, {len(all_chunks)} blok teks"
        processed_results.append(summary)
        return "\n".join(processed_results), [f"📄 {os.path.basename(f.name)}" for f in files]

    except Exception as e:
        logging.error(f"Terjadi kesalahan dalam proses pemrosesan: {str(e)}")
        return f"Terjadi kesalahan dalam proses pemrosesan: {str(e)}", []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 分块可视化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chunk_data_cache = {}


def get_document_chunks(progress=gr.Progress()):
    """获取文档分块结果用于可视化"""
    global chunk_data_cache
    try:
        progress(0.1, desc="Memuat data...")
        chunk_data_cache.clear()

        if not vector_store.id_order:
            return [], "Tidak ada dokumen di basis pengetahuan, silakan unggah dan proses dokumen terlebih dahulu。"

        table_data = []
        for idx, chunk_id in enumerate(vector_store.id_order):
            content = vector_store.contents_map.get(chunk_id, "")
            meta = vector_store.metadatas_map.get(chunk_id, {})
            if not content:
                continue
            chunk_data = {
                "row_id": idx, "chunk_id": chunk_id,
                "source": meta.get("source", "Sumber tidak diketahui"), "content": content,
                "preview": content[:200] + "..." if len(content) > 200 else content,
                "char_count": len(content),
                "token_count": len(list(jieba.cut(content)))
            }
            chunk_data_cache[idx] = chunk_data
            table_data.append([
                chunk_data["source"], f"{idx + 1}/{len(vector_store.id_order)}",
                chunk_data["char_count"], chunk_data["token_count"], chunk_data["preview"]
            ])

        progress(1.0, desc="Selesai!")
        return table_data, f"Total {len(table_data)} blok teks"
    except Exception as e:
        chunk_data_cache.clear()
        return [], f"Gagal mendapatkan data blok teks: {str(e)}"


def show_chunk_details(evt: gr.SelectData):
    """显示选中分块的详细内容"""
    try:
        if not evt.index or evt.index[0] is None:
            return "Baris tidak valid terpilih"
        selected = chunk_data_cache.get(evt.index[0])
        if not selected:
            return "Data blok teks tidak ditemukan"
        return f"""[Sumber] {selected['source']}
[ID] {selected['chunk_id']}
[Jumlah Karakter] {selected['char_count']}
[Jumlah Token] {selected['token_count']}
----------------------------
{selected['content']}"""
    except Exception as e:
        return f"Gagal memuat: {str(e)}"


def get_system_models_info():
    """返回系统使用的各种模型信息"""
    return {
        "Model Embedding": "all-MiniLM-L6-v2",
        "Metode Pembagian": "RecursiveCharacterTextSplitter (chunk_size=400, overlap=40)",
        "Metode Pencarian": "Pencarian Vektor + Pencarian Hibrida BM25 (α=0.7)",
        "Model Re-ranking": "Cross-Encoder (distiluse-base-multilingual-cased-v2)",
        "Model Generatif (Ollama)": OLLAMA_MODEL_NAME,
        "Model Generatif (SiliconFlow)": SILICONFLOW_MODEL_NAME,
        "Alat Tokenisasi": "jieba (Segmentasi Kata Mandarin)"
    }


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

with gr.Blocks(title="Sistem Tanya Jawab RAG Lokal") as demo:
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown("# 🧠 Sistem Tanya Jawab Dokumen Pintar")
        with gr.Column(scale=1, min_width=60):
            theme_btn = gr.Button("🌓", min_width=40, elem_classes="theme-toggle-btn")

    with gr.Tabs() as tabs:
        # ━━━ Tab Percakapan Tanya Jawab ━━━
        with gr.TabItem("💬 Tanya Jawab"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=5, elem_classes="left-panel"):
                    gr.Markdown("## 📂 Area Pemrosesan Dokumen")
                    with gr.Group():
                        file_input = gr.File(
                            label="Unggah dokumen (mendukung PDF, Word, Excel, PPT, TXT, Markdown, dll.)",
                            file_types=[".pdf", ".txt", ".docx", ".xlsx", ".xls", ".pptx", ".md"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("🚀 Mulai Proses", variant="primary")
                        upload_status = gr.Textbox(label="Status Pemrosesan", interactive=False, lines=2)
                        file_list = gr.Textbox(label="File yang Diproses", interactive=False, lines=3, elem_classes="file-list")

                    gr.Markdown("## ❓ Masukkan Pertanyaan")
                    with gr.Group():
                        question_input = gr.Textbox(label="Masukkan Pertanyaan", lines=3, placeholder="Ketik pertanyaan Anda di sini...")
                        with gr.Row():
                            web_search_checkbox = gr.Checkbox(
                                label="Aktifkan Pencarian Web", value=False,
                                info="Jika diaktifkan, konten web juga akan dicari (memerlukan SERPAPI_KEY)"
                            )
                            model_choice = gr.Dropdown(
                                choices=["ollama", "siliconflow"],
                                value=DEFAULT_MODEL_CHOICE,
                                label="Pilihan Model", info="Pilih menggunakan model lokal atau model cloud"
                            )
                        with gr.Row():
                            ask_btn = gr.Button("🔍 Mulai Bertanya", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ Bersihkan Percakapan", variant="secondary", elem_classes="clear-button", scale=1)
                    api_info = gr.HTML("")

                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## 📝 Riwayat Percakapan")
                    chatbot = gr.Chatbot(label="Riwayat Obrolan", height=600, elem_classes="chat-container",
                                         show_label=False)
                    status_display = gr.HTML("")
                    gr.Markdown("""<div class="footer-note">
                        *Menghasilkan jawaban mungkin memerlukan waktu 1-2 menit, mohon tunggu dengan sabar<br>*Mendukung percakapan multi-putaran, Anda dapat terus bertanya berdasarkan konteks sebelumnya
                    </div>""")

        # ━━━ Tab Visualisasi Blok Teks ━━━
        with gr.TabItem("📊 Visualisasi Blok Teks"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💡 Informasi Model Sistem")
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### Model Inti & Teknologi")
                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:")
                                gr.Markdown(f"{value}")
                with gr.Column(scale=2):
                    gr.Markdown("## 📄 Statistik Blok Teks Dokumen")
                    refresh_chunks_btn = gr.Button("🔄 Segarkan Data Blok", variant="primary")
                    chunks_status = gr.Markdown("Klik tombol untuk melihat statistik blok")
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["Sumber", "Nomor", "Jumlah Karakter", "Jumlah Token", "Pratinjau Konten"],
                    elem_classes="chunk-table", interactive=False, wrap=True, row_count=(10, "dynamic")
                )
            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="Detail Blok Teks", placeholder="Klik baris pada tabel untuk melihat konten lengkap...",
                    lines=8, elem_classes="chunk-detail-box"
                )

        # ━━━ Tab Pemantauan Sistem ━━━
        with gr.TabItem("📈 Pemantauan"):
            with gr.Column():
                with gr.Group(elem_classes="monitor-panel"):
                    with gr.Row():
                        gr.Markdown("## 🖥️ Pemantauan Sumber Daya Sistem")
                        refresh_monitor_btn = gr.Button("🔄 Segarkan Data", variant="primary")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("Penggunaan CPU", elem_classes="metric-title")
                            cpu_value = gr.Markdown("Memuat...", elem_classes="metric-value")
                            cpu_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            cpu_info = gr.Markdown("Jumlah Core: Memuat...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Penggunaan Memori", elem_classes="metric-title")
                            memory_value = gr.Markdown("Memuat...", elem_classes="metric-value")
                            memory_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            memory_info = gr.Markdown("Total Memori: Memuat...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Ruang Disk", elem_classes="metric-title")
                            disk_value = gr.Markdown("Memuat...", elem_classes="metric-value")
                            disk_progress = gr.HTML('<div class="progress-container"><div class="progress-bar" style="width:0%"></div></div>')
                            disk_info = gr.Markdown("Total Ruang: Memuat...", elem_classes="metric-trend")
                        with gr.Column():
                            gr.Markdown("Database Vektor", elem_classes="metric-title")
                            vector_db_value = gr.Markdown("Jumlah Blok: 0", elem_classes="metric-value")
                            vector_db_info = gr.Markdown("Jumlah Vektor: 0", elem_classes="metric-trend")

                with gr.Group(elem_classes="monitor-panel"):
                    gr.Markdown("## 📝 Log Sistem")
                    with gr.Row():
                        log_level = gr.Dropdown(choices=["Semua Level", "Info", "Peringatan", "Error"], value="Semua Level", label="Level Log")
                        clear_logs_btn = gr.Button("🗑️ Bersihkan Log", variant="secondary")
                    log_display = gr.HTML("", elem_classes="log-container")

    # ━━━ Fungsi Penanganan Event ━━━
    def clear_chat_history():
        return [], "Riwayat obrolan dibersihkan"

    def process_chat(question, history, enable_web_search, model_choice_val):
        if history is None or not isinstance(history, list):
            history = []

        api_text = """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>Deskripsi Fungsi:</strong></p>
            <p>1. <strong>Pencarian Web</strong>: %s</p>
            <p>2. <strong>Pilihan Model</strong>: Saat ini menggunakan <strong>%s</strong></p>
        </div>""" % (
            "Aktif" if enable_web_search else "Nonaktif",
            "Model Cloud DeepSeek-R1" if model_choice_val == "siliconflow" else "Model Ollama Lokal"
        )

        if not question or question.strip() == "":
            history.append({"role": "assistant", "content": "Pertanyaan tidak boleh kosong, silakan masukkan pertanyaan yang valid."})
            return history, "", api_text

        try:
            answer = query_answer(question, enable_web_search, model_choice_val)
        except Exception as e:
            answer = f"Kesalahan Sistem: {str(e)}"
            logging.error(f"Pengecualian pemrosesan tanya jawab: {str(e)}")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        return history, "", api_text

    def update_api_info(enable_web_search, model_choice_val):
        return """<div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;
            background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>Deskripsi Fungsi:</strong></p>
            <p>1. <strong>Pencarian Web</strong>: %s</p>
            <p>2. <strong>Pilihan Model</strong>: Saat ini menggunakan <strong>%s</strong></p>
        </div>""" % (
            "Aktif" if enable_web_search else "Nonaktif",
            "Model Cloud DeepSeek-R1" if model_choice_val == "siliconflow" else "Model Ollama Lokal"
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

            def bar(pct, color="var(--tech-cyan)"):
                return f'<div class="progress-container"><div class="progress-bar" style="width:{pct}%;background:{color}"></div></div>'

            c_color = "#4CAF50" if cpu_pct < 50 else "#FFC107" if cpu_pct < 80 else "#f44336"
            m_color = "#4CAF50" if mem.percent < 50 else "#FFC107" if mem.percent < 80 else "#f44336"
            d_color = "#4CAF50" if disk.percent < 50 else "#FFC107" if disk.percent < 80 else "#f44336"

            now = datetime.now().strftime("%H:%M:%S")
            log = f'<div class="log-entry"><span style="color:var(--tech-cyan)">[{now}]</span> <span style="color:#4CAF50">[INFO]</span> Data pemantauan telah diperbarui</div>'

            return (
                f"{cpu_pct}%", bar(cpu_pct, c_color), f"Core Fisik: {cpu_cnt}",
                f"{mem_used}GB / {mem_total}GB", bar(mem.percent, m_color), f"Tingkat Penggunaan: {mem.percent}%",
                f"{disk_used}GB / {disk_total}GB", bar(disk.percent, d_color), f"Tingkat Penggunaan: {disk.percent}%",
                f"Jumlah Blok: {doc_count}", f"Jumlah Vektor: {vec_count}", log
            )
        except Exception as e:
            err = f"Kesalahan Pemantauan: {str(e)}"
            return ("Error", "", err, "Error", "", err, "Error", "", err, "Error", err,
                    f"<div style='color:#f44336'>[ERROR] {err}</div>")

    # ━━━ 绑定事件 ━━━
    upload_btn.click(process_multiple_files, inputs=[file_input], outputs=[upload_status, file_list], show_progress=True)
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
        vector_db_value, vector_db_info, log_display
    ])
    clear_logs_btn.click(fn=lambda: "<div style='color:#4CAF50'>Log telah dibersihkan</div>", outputs=[log_display])
    theme_btn.click(fn=toggle_theme, inputs=[], outputs=[], js="""
        () => {
            document.querySelector('body').classList.toggle('dark');
            const isDark = document.querySelector('body').classList.contains('dark');
            localStorage.setItem('rag-theme', isDark ? 'dark' : 'light');
        }
    """)


def check_environment():
    """Pemeriksaan dependensi lingkungan"""
    if SILICONFLOW_API_KEY and not SILICONFLOW_API_KEY.startswith("Your"):
        print("✅ SiliconFlow API Key telah dikonfigurasi")
        try:
            result = call_siliconflow_api("Halo, silakan balas dengan 'Koneksi Berhasil'", temperature=0.1, max_tokens=50)
            if isinstance(result, str) and any(w in result for w in ["Koneksi", "Berhasil", "Halo", "连接成功", "你好"]):
                print("✅ Pengujian koneksi SiliconFlow API berhasil")
            else:
                print("⚠️ Respons SiliconFlow API tidak normal, tetapi tetap melanjutkan proses")
            return True
        except Exception as e:
            print(f"⚠️ Pengujian SiliconFlow API gagal: {e}")
            return True
    else:
        print("⚠️ SiliconFlow API Key belum dikonfigurasi, mencoba menggunakan Ollama lokal")
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                print("✅ Layanan Ollama lokal tersedia")
                return True
        except Exception:
            pass
        print("❌ Tidak menemukan backend LLM aktif yang tersedia")
        print("   Silakan konfigurasi SILICONFLOW_API_KEY di .env atau jalankan layanan Ollama")
        return False


if __name__ == "__main__":
    if not check_environment():
        exit(1)

    ports = [17995, 17996, 17997, 17998, 17999]
    selected_port = next((p for p in ports if is_port_available(p)), None)

    if not selected_port:
        print("Semua port terpakai, silakan bebaskan port secara manual")
        exit(1)

    try:
        webbrowser.open(f"http://127.0.0.1:{selected_port}")
        demo.launch(
            server_port=selected_port, server_name="0.0.0.0",
            show_error=True, ssl_verify=False, height=900,
            css=CSS, js=THEME_JS
        )
    except Exception as e:
        print(f"Gagal meluncurkan: {str(e)}")
