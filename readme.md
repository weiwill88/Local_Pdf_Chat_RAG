<div align="center">
<h1>🧠 Local Pdf Chat RAG</h1>
<p><strong>本地化智能问答系统 — 检索增强生成（RAG）v3.0</strong></p>
<p>
<img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python">
<img src="https://img.shields.io/badge/License-MIT-green" alt="License">
<img src="https://img.shields.io/badge/UI-Gradio_6.20.0-violet" alt="Gradio">
<img src="https://img.shields.io/badge/Vector_Store-FAISS-yellow" alt="FAISS">
<img src="https://img.shields.io/badge/LLM-SiliconFlow_|_Ollama_|_Magick-orange" alt="LLM">
<img src="https://img.shields.io/badge/Status-v3.0_Complete-brightgreen" alt="Status">
</p>
<p>📄 DOCX / PDF / XLSX / PPTX / MD / TXT &nbsp;·&nbsp; 🔍 语义 + BM25 混合检索 &nbsp;·&nbsp; 🌐 联网搜索自动兜底</p>
</div>

---

## 📋 项目简介

**Local Pdf Chat RAG** 是一个从零构建的、功能完备的本地化 RAG 问答系统。支持多种格式文档解析、双路混合检索、Cross-Encoder 重排序、Parent Document Retriever 父子块映射、MMR 多样性排序、联网搜索自动兜底、多知识库物理隔离、Ollama 本地大模型一键切换等完整能力链。

> 🎯 **设计目标**：提供一个可直接投入使用的 RAG 系统，同时保持模块化解耦与代码可读性，便于二次开发与学习。

---

## 🏗️ 技术栈

| 层级 | 技术选型 | 用途 |
|------|----------|------|
| **前端 UI** | Gradio 6.20.0 | Web 交互界面（问答/文件管理/可视化/监控） |
| **向量引擎** | FAISS (IndexFlatL2/IVFFlat/IVFPQ) | 语义相似度检索，自动按数据量选型 |
| **文本编码** | Sentence-Transformers all-MiniLM-L6-v2 | 384 维文本嵌入 |
| **稀疏检索** | rank-bm25 (BM25Okapi) + jieba | 中文关键词检索 |
| **重排序** | Cross-Encoder distiluse-base-multilingual-cased-v2 | 精排 Top-5 |
| **LLM 推理** | SiliconFlow API / Ollama 本地 / Magick API | 回答生成 |
| **联网搜索** | SerpAPI (Google) | 本地召回不足时自动补充 |
| **文档解析** | PyMuPDF / python-docx / openpyxl / python-pptx / Tesseract OCR | PDF/DOCX/XLSX/PPTX/MD/TXT + 扫描件 |
| **数据持久化** | SQLite + JSON + FAISS 序列化 | 对话历史 / 索引元数据 / 向量存储 |
| **REST 接口** | FastAPI | 可选 API 服务 |
| **系统监控** | psutil | CPU / 内存 / 磁盘实时监控 |

---

## ✨ 三阶段功能清单

### 🔵 第一阶段 — 基础 RAG 能力

| 功能 | 描述 |
|------|------|
| ✅ **多格式文档解析** | PDF / DOCX / XLSX / PPTX / MD / TXT 统一加载 |
| ✅ **扫描件 OCR** | Tesseract 中英文识别 + 二级引擎自检 |
| ✅ **递归文本分割** | `chunk_size=400`, `overlap=40` |
| ✅ **FAISS 向量索引** | 按数据量自动选型（FlatL2 / IVFFlat / IVFPQ）|
| ✅ **BM25 关键词检索** | jieba 分词 + BM25Okapi |
| ✅ **混合检索** | 语义 0.7 + BM25 0.3 加权融合 |
| ✅ **Cross-Encoder 重排序** | 精排保留 Top-5 |
| ✅ **递归检索** | LLM 改写查询，最多 3 轮 |
| ✅ **多知识库管理** | 创建 / 切换 / 重命名 / 删除 |
| ✅ **对话历史持久化** | SQLite 存储，按知识库隔离 |
| ✅ **Markdown 导出** | 对话记录 → .md 文件 |
| ✅ **DeepSeek 思维链** | `<think>` 解析 → 可折叠 HTML |
| ✅ **联网搜索** | SerpAPI Google 搜索 |
| ✅ **FastAPI REST** | `/api/upload`, `/api/ask`, `/api/status` |
| ✅ **暗色模式** | 主题偏好 localStorage 持久化 |

### 🟡 第二阶段 — 检索升级 + 文件管理

| 功能 | 描述 |
|------|------|
| ✅ **Parent Document Retriever** | 父块 800ch + 子块 200ch 双层分割，子块检索→父块上下文 |
| ✅ **MMR 重排序** | 余弦相似度贪心选择，λ 平衡相关性与多样性 |
| ✅ **Alpha 权重滑块** | UI 实时调节混合检索权重（0=纯BM25, 1=纯向量） |
| ✅ **文件列表展示** | Dataframe 显示文件名/类型/上传时间/分块数 |
| ✅ **单文件删除** | 彻底清除向量/父块/BM25/文件注册 |
| ✅ **一键清空** | 清空内存 + 删除磁盘残留 |
| ✅ **SHA256 去重** | 上传文件哈希校验，重复自动跳过 |
| ✅ **溯源引用** | 回答末尾 `📄 来源 N: 文档名` + 可折叠原文 |
| ✅ **单条对话删除** | 精确删除用户-助手消息对 |
| ✅ **指数退避重试** | API 调用异常自动重试 |

### 🟢 第三阶段 — 联网兜底 + 模型切换 + 隔离架构 + 监控交互

| 模块 | 功能 | 描述 |
|------|------|------|
| 🌐 **M1 联网增强** | 自动兜底 | 本地召回=0 或 相似度<0.3 → 自动触发 SerpAPI |
| | 来源分区 | 回答严格区分【本地文档参考】/【网络检索参考】 |
| | 异常友好提示 | 超时/密钥无效/额度用尽 → 中文提示不崩溃 |
| | 配置容错 | `.env` 空格/空值自动 strip |
| 🖥️ **M2 Ollama 适配** | 动态模型列表 | 自动加载本机已安装 Ollama 模型到下拉框 |
| | 参数面板 | 折叠式 num_ctx / temperature / top_p 滑块 |
| | 一键切换 | 云端 ↔ 本地模型自由切换，面板自动显隐 |
| | 服务检测 | Ollama 未启动时优雅降级，不阻塞启动 |
| 🔒 **M3 物理隔离** | 独立 FAISS | 每个知识库独立 `index.faiss` 文件 |
| | 独立 SQLite 表 | 每库独立数据表 `chat_history_{sanitized_kb}` |
| | 文件隔离 | 文件列表/删除/上传强制归属当前选中库 |
| 📊 **M4 监控交互** | 全局统计 | 总知识库数/文档数/分块数/父块数/磁盘占用 |
| | KB 拆解 | 下拉选库 → 该库文档/分块/索引大小 |
| | 运行自检 | OCR / SerpAPI / 云端模型 / Ollama 状态 |
| | 日志查看器 | 最近 10 条运行日志 |
| | 工程收尾 | `.gitignore` 完善 / 日志写入文件 |

---

## 🖼️ 使用截图

<!-- 截图是项目展示的重要部分，建议以下场景各截一张图替换 -->

| 页面 | 说明 |
|------|------|
| **💬 问答对话** | 文档上传区 + 模型选择 + Alpha 滑块 + 联网搜索勾选框 + 对话记录 + 溯源引用 |
| **📊 分块可视化** | 表格展示所有文本块（来源/序号/字符数/分词数/内容预览）+ 点击查看详情 |
| **📂 文件管理** | 按知识库展示文件列表 + 删除 / 清空操作 |
| **📈 系统监控** | 系统资源 + 全局统计 + 按库拆解 + 运行自检 + 日志查看器 |

<!-- TODO: 请将截图放入 docs/screenshots/ 目录并替换下方路径 -->
<details>
<summary>📸 点击查看示例截图占位</summary>

```
docs/screenshots/
├── chat_interface.png      # 问答对话页面
├── chunk_visualization.png # 分块可视化页面
├── file_management.png     # 文件管理页面
└── system_monitor.png      # 系统监控页面
```

</details>

---

## 🚀 快速开始

### 环境要求

- **Python** 3.10+
- **操作系统** Windows / macOS / Linux
- **可选** Tesseract OCR（扫描版 PDF 需要）
- **可选** Ollama（使用本地模型需要）

### 1. 克隆项目

```bash
git clone https://github.com/CCC481568794/Local_Pdf_Chat_RAG.git
cd Local_Pdf_Chat_RAG
```

### 2. 创建虚拟环境

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

> **关键依赖一览**：`gradio==6.20.0` `sentence-transformers` `faiss-cpu` `rank-bm25` `jieba` `pypdf2` `python-docx` `openpyxl` `python-pptx` `pytesseract` `pdf2image` `psutil` `python-dotenv` `requests`

### 4. 配置环境变量

```bash
# 从模板创建
cp example.env .env
```

编辑 `.env`，至少配置一个 LLM 后端：

```ini
# ━━━ LLM API（三选一或全配） ━━━
SILICONFLOW_API_KEY=sk-xxx          # SiliconFlow 云端（推荐，无需 GPU）

# ━━━ 联网搜索（可选） ━━━
SERPAPI_KEY=your_serpapi_key        # 需要在 serpapi.com 注册

# ━━━ OCR（可选，扫描版 PDF 需要） ━━━
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe

# ━━━ Ollama 本地模型（可选） ━━━
OLLAMA_MODEL_NAME=deepseek-r1:8b
```

### 5. 启动服务

```bash
python rag_demo.py
```

> **Windows 中文系统注意**（避免 emoji 和控制台编码问题）：
> ```powershell
> $env:PYTHONIOENCODING='utf-8'
> python -X utf8 rag_demo.py
> ```

启动后浏览器打开 `http://127.0.0.1:17995`（端口 17995→17999 自动选取可用端口）。

> ⏰ 首次启动会下载嵌入模型 all-MiniLM-L6-v2（约 80MB），请耐心等待。

---

## ⚙️ .env 完整配置说明

```ini
# ═══════════════════════════════════════
# LLM API 密钥（至少配置一项）
# ═══════════════════════════════════════
SILICONFLOW_API_KEY=sk-xxx
SILICONFLOW_API_URL=https://api.deepseek.com/v1/chat/completions
SILICONFLOW_MODEL_NAME=deepseek-v4-flash

MAGICK_API_KEY=Your_Magick_API_Key
MAGICK_API_URL=https://api.magickapi.com/v1/chat/completions
MAGICK_MODEL_NAME=gpt-4o-mini

OLLAMA_MODEL_NAME=deepseek-r1:8b

# ═══════════════════════════════════════
# 联网搜索
# ═══════════════════════════════════════
SERPAPI_KEY=your_serpapi_key_here

# ═══════════════════════════════════════
# Tesseract OCR（扫描 PDF 需要）
# ═══════════════════════════════════════
TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe

# ═══════════════════════════════════════
# Ollama 参数（可选）
# ═══════════════════════════════════════
OLLAMA_NUM_CTX=4096
OLLAMA_TEMPERATURE=0.7
OLLAMA_TOP_P=0.9
```

---

## 📂 项目结构

```
Local_Pdf_Chat_RAG/
├── rag_demo.py                # 🖥️ Gradio UI 主入口（事件绑定 + 布局）
├── config.py                  # ⚙️ 配置中心（环境变量 / 超参数 / 检测机制）
├── api_router.py              # 🔌 FastAPI REST API
├── .env                       # 🔑 密钥配置（不纳入版本控制）
├── requirements.txt           # 📦 依赖清单
├── chat_history.db            # 🗄️ SQLite 对话历史（自动生成）
├── rag_app_output.log         # 📝 运行日志（自动生成）
│
├── core/                      # 🧠 RAG 核心引擎
│   ├── embeddings.py          #    文本→向量（SentenceTransformer）
│   ├── text_splitter.py       #    单层/双层分块（RecursiveCharacter）
│   ├── vector_store.py        #    FAISS 索引 + 文件管理 + 持久化 + 相似度检索
│   ├── bm25_index.py          #    BM25 稀疏检索 + 持久化
│   ├── retriever.py           #    混合检索 + 递归检索 + 联网兜底触发
│   ├── parent_retriever.py    #    父文档映射 + MMR 重排序
│   ├── reranker.py            #    Cross-Encoder 重排序
│   ├── generator.py           #    LLM 调用 + Prompt 构建 + 溯源引用
│   └── knowledge_base_manager.py  # 多知识库 save/load/delete
│
├── features/                  # ✨ 功能扩展
│   ├── web_search.py          #    SerpAPI 联网搜索
│   ├── thinking_chain.py      #    DeepSeek 思维链处理
│   └── conflict_detector.py   #    多来源矛盾检测
│
├── utils/                     # 🔧 工具模块
│   ├── document_loader.py     #    多格式文档加载 + OCR
│   ├── chat_history.py        #    SQLite 独立数据表 + Markdown 导出
│   ├── retry.py               #    指数退避重试
│   └── network.py             #    HTTP Session + 端口检测
│
└── knowledge_bases/           # 📚 知识库数据（自动生成，不纳入版本控制）
    └── {kb_name}/
        ├── index.faiss
        ├── contents_map.json
        ├── metadatas_map.json
        ├── parent_chunks_map.json
        ├── file_index.json
        └── ... (其他元数据)
```

---

## 🧩 核心架构流程

```
用户提问
   │
   ▼
┌─ 1. 语义检索 (FAISS + all-MiniLM-L6-v2) ──┐
│    + BM25 关键词检索 (jieba + BM25Okapi)    │
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 2. Hybrid 混合 (α × 语义 + (1-α) × BM25) ┐
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 3. Parent 映射 (子块 → 父块) ────────────┐
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 4. MMR 多样性重排 ───────────────────────┐
│   λ × Sim(q,d) - (1-λ) × max Sim(dᵢ,dⱼ)  │
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 5. Cross-Encoder 精排 Top-5 ─────────────┐
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 6. 联网兜底判断 ─────────────────────────┐
│   召回=0 或 最高相似度<0.3 → SerpAPI 补充   │
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 7. LLM 生成 ─────────────────────────────┐
│   Prompt = 上下文 + 来源标注 + 问题 → 模型  │
└──────────────────────────────────────────────┘
   │
   ▼
┌─ 8. 溯源引用 ─────────────────────────────┐
│   ▶ 本地知识库参考 + ▶ 网络检索补充内容     │
└──────────────────────────────────────────────┘
```

---

## 📊 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `CHUNK_SIZE` | 400 | 单层分块字符数 |
| `PARENT_CHUNK_SIZE` | 800 | 父文档块大小 |
| `CHILD_CHUNK_SIZE` | 200 | 子文档块大小 |
| `HYBRID_ALPHA` | 0.7 | 语义检索权重 |
| `MMR_LAMBDA` | 0.5 | MMR 多样性（1=纯相关） |
| `RETRIEVAL_TOP_K` | 10 | 初始检索候选数 |
| `RERANK_TOP_K` | 5 | 重排序后保留数 |
| `LOCAL_SCORE_THRESHOLD` | 0.3 | 联网搜索触发阈值 |
| `WEB_SEARCH_MAX_RESULTS` | 5 | 联网搜索最大条数 |
| `OLLAMA_NUM_CTX` | 4096 | Ollama 上下文窗口 |

---

## 🤝 贡献指南

欢迎提交 Issue 和 PR！

1. Fork 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交变更：`git commit -m 'feat: add amazing feature'`
4. 推送分支：`git push origin feature/amazing-feature`
5. 提交 Pull Request

### 开发约定

- 新增功能写在独立文件中（`core/` 或 `utils/` 目录）
- 配置参数放在 `config.py`，不硬编码
- 所有空列表/空索引增加前置判断，杜绝除零异常
- 保持向后兼容，不改动已稳定的解析/OCR 逻辑

---

## 📄 许可证

本项目基于 MIT 许可证开源。

---

<div align="center">
<p>如果这个项目对你有帮助，欢迎 ⭐ Star 支持！</p>
<p>
<a href="https://github.com/CCC481568794/Local_Pdf_Chat_RAG">GitHub 仓库</a>
</p>
</div>
