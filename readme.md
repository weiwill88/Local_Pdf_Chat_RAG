# 本地RAG问答系统

🚀 基于本地大模型的智能文档问答系统，支持PDF文档解析与自然语言问答，🎯支持联网搜索增强能力，✨新增多文档处理与多轮对话功能

## 为什么选择本地RAG？

🔒 **私有数据安全**：全程本地处理，敏感文档无需上传第三方服务

⚡ **实时响应**：基于本地向量数据库实现毫秒级语义检索

💡 **领域适配**：可针对专业领域文档定制知识库

🌐 **离线/在线双模式**：支持本地文档与网络结果智能融合

💰 **成本可控**：避免云服务按次计费，长期使用成本更低

## 功能特性

### 核心功能
📄 PDF文档解析与向量化存储
🧠 基于DeepSeek-7B本地大模型
⚡ 流式回答生成
🔍 语义检索与上下文理解
🌐 联网搜索增强（SerpAPI集成）
🔗 多源结果智能整合与矛盾检测
🖥️ 友好的Web交互界面

### ✨ 新增特性
- 🔄 **多文档处理**：支持同时上传和处理多个PDF文件
- 💬 **多轮对话**：支持基于上下文的连续对话
- 📊 **状态追踪**：实时显示文档处理进度和状态
- 🌓 **暗色主题**：默认暗色主题，提供更好的阅读体验
- 🔍 **源文档溯源**：回答中自动标注信息来源

## 环境要求

- Python 3.9+
- 内存：至少8GB
- 显存：至少4GB（推荐8GB）
- SerpAPI账号（免费额度可用）

## 安装步骤

1. 克隆仓库：
```bash
git clone https://github.com/weiwill88/Local_Pdf_Chat_RAG
cd Local_Pdf_Chat_RAG
```

2. 创建虚拟环境：
```bash
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
rag_env\Scripts\activate  # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 配置环境变量：
```bash
# 复制示例文件
cp .env.example .env
# 编辑.env文件添加你的API密钥
SERPAPI_KEY=your_serpapi_key_here
```

5. 安装Ollama服务：
```bash
curl -fsSL https://ollama.com/install.sh | sh  # Linux/Mac
winget install ollama  # Windows（需要管理员权限）

ollama pull deepseek-r1:7b

# 启动Ollama服务（Windows会自动注册服务）
ollama serve &
```

## 使用方法

1. 启动服务：
```bash
.\rag_env\Scripts\activate
python rag_demo.py
```

2. 访问浏览器打开的本地地址（通常是`http://localhost:17995`）

3. 操作流程：
   - 上传单个或多个PDF文档（支持批量处理）
   - 等待处理完成，查看处理状态
   - 在提问区输入问题
     - 时间敏感问题自动获取最新网络结果
     - 支持多轮连续对话
   - 查看整合本地文档与网络搜索的智能回答
   - 可随时清空对话重新开始

### 新增功能说明

#### 多文档处理
- 支持同时上传多个PDF文件
- 自动分割文本并生成向量嵌入
- 实时显示每个文件的处理进度
- 每次上传新文档会自动清理历史数据

#### 多轮对话
- 保留完整对话历史
- 支持基于上下文的连续提问
- 实时流式输出回答
- 清晰的对话气泡界面

#### 界面优化
- 响应式布局设计
- 默认暗色主题支持
- 文件处理状态实时显示
- 优雅的对话展示效果

## 配置说明

1. 模型配置：
   - 修改`rag_demo.py`中的模型名称：
   ```python
   "model": "deepseek-r1:7b"  # 可替换为其他支持的模型
   ```

2. 性能调优：
   - 调整`process_pdf`函数中的文本分割参数：
   ```python
   chunk_size=800  # 文本块大小
   chunk_overlap=50  # 块间重叠
   ```

3. 网络搜索设置：
```python
# 在combined_rag.py中调整搜索参数
SEARCH_ENGINE = "google"  # 可选：bing, duckduckgo
NUM_RESULTS = 5           # 默认获取5条网络结果
```

## 🛠️ 技术栈

- 向量数据库：ChromaDB
- 文本嵌入：Sentence-Transformers (all-MiniLM-L6-v2)
- LLM模型：Ollama (deepseek-r1:1.5b)
- 前端框架：Gradio
- PDF处理：pdfminer
- 文本分割：LangChain

## 📦 安装依赖

推荐使用以下方式安装依赖，可以显著提升安装速度：

```bash
# 创建虚拟环境
python -m venv rag_env
source rag_env/bin/activate  # Linux/Mac
rag_env\Scripts\activate  # Windows

# 更新 pip
python -m pip install --upgrade pip

# 方式1：使用预编译包安装（推荐，速度最快）
pip install --only-binary :all: -r requirements.txt

# 方式2：使用国内镜像源安装
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 方式3：如果上述方法都不行，可以尝试
pip install --no-cache-dir -r requirements.txt
```

注意事项：
1. 建议使用 Python 3.9+ 版本
2. Windows 用户可能需要安装 Visual C++ Build Tools
3. 如果安装过程中遇到问题，可以尝试逐个安装依赖包

## 🚀 使用方法

1. **启动Ollama服务**
```bash
ollama serve
```

2. **拉取模型**
```bash
ollama pull deepseek-r1:1.5b
```

3. **运行应用**
```bash
python rag_demo.py
```

## 💡 功能说明

### 文档处理
- 支持批量上传PDF文件
- 自动分割文本并生成向量嵌入
- 实时显示处理进度和状态
- 每次上传新文档会自动清理历史数据

### 问答功能
- 支持多轮对话，保留对话历史
- 实时流式输出回答
- 显示信息来源，支持溯源
- 可随时清空对话重新开始

### 界面特性
- 响应式布局设计
- 默认暗色主题
- 清晰的文件处理状态显示
- 优雅的对话气泡界面

## ⚙️ 配置说明

- 文本分块大小：800字符
- 块重叠大小：50字符
- 检索相关片段数：3
- 请求超时时间：120秒

## 📝 注意事项

1. 确保本地已安装并启动Ollama服务
2. 首次使用需要下载模型，可能需要一些时间
3. 处理大文件时可能需要较长时间，请耐心等待
4. 建议每次问答会话使用相关的文档集

### 📚 文档上传限制

系统的文档处理能力主要受以下因素限制：

1. **内存限制**：
   - 文档向量化过程在内存中进行
   - 建议单个PDF文件不超过50MB
   - 总处理文档量建议不超过系统可用内存的1/4
   - 8GB内存建议同时处理文档总量不超过500MB

2. **向量数据库限制**：
   - ChromaDB对单个集合的向量数量无硬性限制
   - 建议单次会话文档块总数不超过10,000个
   - 按默认分块大小(800字符)计算，约等于800万字符

3. **处理时间考虑**：
   - 文档越大，处理时间越长
   - 建议单次上传文档数不超过10个
   - 大型文档建议分批处理

4. **性能优化建议**：
   - 对于大型文档，可以调整chunk_size参数（增大分块大小）
   - 可以根据实际需求调整chunk_overlap参数
   - 必要时可以预处理PDF，去除非必要内容

5. **最佳实践**：
   - 建议按主题分批上传相关文档
   - 处理完一批文档后再上传新批次
   - 可以通过调整配置参数优化处理性能

注意：以上限制是基于一般使用场景的建议值，实际限制取决于系统配置和具体使用场景。

## 🤝 贡献指南

欢迎提交Issue和Pull Request来帮助改进项目。

## �� 许可证

MIT License
