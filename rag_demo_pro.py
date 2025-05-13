import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 以上两行添加的Hugging Face镜像设置，是为了解决没有科学上网环境的盆友下载向量模型的问题
import gradio as gr
from pdfminer.high_level import extract_text_to_fp
from sentence_transformers import SentenceTransformer
# 导入交叉编码器
from sentence_transformers import CrossEncoder
import faiss # Новый импорт
import requests
import json
from io import StringIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import socket
import webbrowser
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
from datetime import datetime
import hashlib
import re
from dotenv import load_dotenv
# 导入BM25算法库
from rank_bm25 import BM25Okapi
import numpy as np # Убедимся, что numpy импортирован
import jieba
import threading
from functools import lru_cache

# 加载环境变量
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  # В .env файле установите SERPAPI_KEY
SEARCH_ENGINE = "google"  # Можно изменить на другую поисковую систему при необходимости
# Новое: Конфигурация метода переранжирования (кросс-энкодер или LLM)
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")  # "cross_encoder" или "llm"
# Новое: Конфигурация SiliconFlow API
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")

# В начале файла добавляем настройки таймаута
import requests
requests.adapters.DEFAULT_RETRIES = 3  # Увеличиваем количество попыток

# В начале файла добавляем настройки переменных окружения
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Отключаем оптимизацию oneDNN

# В самом начале файла добавляем конфигурацию прокси
import os
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'  # Новая настройка обхода прокси

# Инициализация компонентов
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
# Модель для эмбеддингов также можно переключить на модель, оптимизированную для китайского языка, например:
# EMBED_MODEL = SentenceTransformer('shibing624/text2vec-base-chinese')

# FAISS相关的 глобальные переменные
faiss_index = None
faiss_contents_map = {}  # original_id -> content
faiss_metadatas_map = {} # original_id -> metadata
faiss_id_order_for_index = [] # Сохраняет порядок ID, как они были добавлены в FAISS

# Новое: Инициализация кросс-энкодера (отложенная загрузка)
cross_encoder = None
cross_encoder_lock = threading.Lock()

def get_cross_encoder():
    """延迟加载交叉编码器模型"""
    global cross_encoder
    if cross_encoder is None:
        with cross_encoder_lock:
            if cross_encoder is None:
                try:
                    # 使用多语言交叉编码器，更适合中文
                    cross_encoder = CrossEncoder('sentence-transformers/distiluse-base-multilingual-cased-v2')
                    logging.info("交叉编码器加载成功")
                except Exception as e:
                    logging.error(f"加载交叉编码器失败: {str(e)}")
                    # 设置为None，下次调用会重试
                    cross_encoder = None
    return cross_encoder

# 新增：BM25索引管理
def recursive_retrieval(initial_query, max_iterations=3, enable_web_search=False, model_choice="ollama"):
    """
    实现递归检索与迭代查询功能
    通过分析当前查询结果，确定是否需要进一步查询
    
    Args:
        initial_query: 初始查询
        max_iterations: 最大迭代次数
        enable_web_search: 是否启用网络搜索
        model_choice: 使用的模型选择("ollama"或"siliconflow")
        
    Returns:
        包含所有检索内容的列表
    """
    query = initial_query
    all_contexts = []
    all_doc_ids = []  # 使用原始ID
    all_metadata = []
    
    global faiss_index, faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
    
    for i in range(max_iterations):
        logging.info(f"递归检索迭代 {i+1}/{max_iterations}，当前查询: {query}")
        
        web_results_texts = [] # Store text from web results for context building
        if enable_web_search and check_serpapi_key():
            try:
                # update_web_results now needs to handle FAISS directly or be adapted
                # For now, let's assume it returns texts to be added to context
                web_search_raw_results = update_web_results(query) # This function needs to be adapted for FAISS
                for res in web_search_raw_results:
                    text = f"标题：{res.get('title', '')}\\n摘要：{res.get('snippet', '')}"
                    web_results_texts.append(text)
                    # We would also need to add these to faiss_index, faiss_contents_map etc.
                    # and get their FAISS indices if we want them to be part of semantic search.
                    # This part is complex due to dynamic addition and potential ID clashes.
                    # For now, web results are added as pure text context, not searched semantically *again* within this loop's FAISS query.
            except Exception as e:
                logging.error(f"网络搜索错误: {str(e)}")
        
        query_embedding = EMBED_MODEL.encode([query])
        query_embedding_np = np.array(query_embedding).astype('float32')
        
        semantic_results_docs = []
        semantic_results_metadatas = []
        semantic_results_ids = []

        if faiss_index and faiss_index.ntotal > 0:
            try:
                D, I = faiss_index.search(query_embedding_np, k=10) # D: distances, I: indices
                # I contains the internal FAISS indices. We need to map them back to original IDs.
                for faiss_idx in I[0]: # I[0] because query_embedding_np was a batch of 1
                    if faiss_idx != -1 and faiss_idx < len(faiss_id_order_for_index):
                        original_id = faiss_id_order_for_index[faiss_idx]
                        semantic_results_docs.append(faiss_contents_map.get(original_id, ""))
                        semantic_results_metadatas.append(faiss_metadatas_map.get(original_id, {}))
                        semantic_results_ids.append(original_id)
        except Exception as e:
                logging.error(f"FAISS 检索错误: {str(e)}")
        
        bm25_results = BM25_MANAGER.search(query, top_k=10) # BM25_MANAGER.search returns list of dicts
        
        # Adapt hybrid_merge to work with current data structures
        # It expects semantic_results in a specific format if we pass it directly
        # For now, prepare a structure similar to old semantic_results for hybrid_merge
        prepared_semantic_results_for_hybrid = {
            "ids": [semantic_results_ids],
            "documents": [semantic_results_docs],
            "metadatas": [semantic_results_metadatas]
        }

        hybrid_results = hybrid_merge(prepared_semantic_results_for_hybrid, bm25_results, alpha=0.7)
        
        doc_ids_current_iter = []
        docs_current_iter = []
        metadata_list_current_iter = []
        
        if hybrid_results:
            for doc_id, result_data in hybrid_results[:10]: # doc_id here is the original_id
                doc_ids_current_iter.append(doc_id)
                docs_current_iter.append(result_data['content'])
                metadata_list_current_iter.append(result_data['metadata'])
        
        if docs_current_iter:
            try:
                reranked_results = rerank_results(query, docs_current_iter, doc_ids_current_iter, metadata_list_current_iter, top_k=5)
            except Exception as e:
                logging.error(f"重排序错误: {str(e)}")
                reranked_results = [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0}) 
                                  for doc_id, doc, meta in zip(doc_ids_current_iter, docs_current_iter, metadata_list_current_iter)]
        else:
            reranked_results = []
        
        current_contexts_for_llm = web_results_texts[:] # Start with web results for LLM context
        for doc_id, result_data in reranked_results:
            doc = result_data['content']
            metadata = result_data['metadata']
            
            if doc_id not in all_doc_ids:  
                all_doc_ids.append(doc_id)
                all_contexts.append(doc)
                all_metadata.append(metadata)
            current_contexts_for_llm.append(doc) # Add reranked local docs for LLM context
        
        if i == max_iterations - 1:
            break
            
        if current_contexts_for_llm: # Use combined web and local context for deciding next query
            current_summary = "\\n".join(current_contexts_for_llm[:3]) if current_contexts_for_llm else "未找到相关信息"
            
            next_query_prompt = f"""基于原始问题: {initial_query}
以及已检索信息: 
{current_summary}

分析是否需要进一步查询。如果需要，请提供新的查询问题，使用不同角度或更具体的关键词。
如果已经有充分信息，请回复'不需要进一步查询'。

新查询(如果需要):"""
            
            try:
                if model_choice == "siliconflow":
                    logging.info("使用SiliconFlow API分析是否需要进一步查询")
                    next_query_result = call_siliconflow_api(next_query_prompt, temperature=0.7, max_tokens=256)
                    if "<think>" in next_query_result:
                        next_query = next_query_result.split("<think>")[0].strip()
                    else:
                        next_query = next_query_result
                else:
                    logging.info("使用本地Ollama模型分析是否需要进一步查询")
                    response = session.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "deepseek-r1:1.5b",
                            "prompt": next_query_prompt,
                            "stream": False
                        },
                        timeout=30
                    )
                    next_query = response.json().get("response", "")
                
                if "不需要" in next_query or "不需要进一步查询" in next_query or len(next_query.strip()) < 5:
                    logging.info("LLM判断不需要进一步查询，结束递归检索")
                    break
                    
                # 使用新查询继续迭代
                query = next_query
                logging.info(f"生成新查询: {query}")
            except Exception as e:
                logging.error(f"生成新查询时出错: {str(e)}")
                break
        else:
            # 如果当前迭代没有检索到内容，结束迭代
            break
    
    return all_contexts, all_doc_ids, all_metadata

class BM25IndexManager:
    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}  # 映射BM25索引位置到文档ID
        self.tokenized_corpus = []
        self.raw_corpus = []
        
    def build_index(self, documents, doc_ids):
        """构建BM25索引"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        
        # 对文档进行分词，使用jieba分词器更适合中文
        self.tokenized_corpus = []
        for doc in documents:
            # 对中文文档进行分词
            tokens = list(jieba.cut(doc))
            self.tokenized_corpus.append(tokens)
        
        # 创建BM25索引
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        return True
        
    def search(self, query, top_k=5):
        """使用BM25检索相关文档"""
        if not self.bm25_index:
            return []
        
        # 对查询进行分词
        tokenized_query = list(jieba.cut(query))
        
        # 获取BM25得分
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 获取得分最高的文档索引
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # 返回结果
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # 只返回有相关性的结果
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        
        return results
    
    def clear(self):
        """清空索引"""
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

# 初始化BM25索引管理器
BM25_MANAGER = BM25IndexManager()

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

#########################################
# SerpAPI 网络查询及向量化处理函数
#########################################
def serpapi_search(query: str, num_results: int = 5) -> list:
    """
    执行 SerpAPI 搜索，并返回解析后的结构化结果
    """
    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量。请在.env文件中设置您的 API 密钥。")
    try:
        params = {
            "engine": SEARCH_ENGINE,
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": num_results,
            "hl": "zh-CN",  # 中文界面
            "gl": "cn"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        search_data = response.json()
        return _parse_serpapi_results(search_data)
    except Exception as e:
        logging.error(f"网络搜索失败: {str(e)}")
        return []

def _parse_serpapi_results(data: dict) -> list:
    """解析 SerpAPI 返回的原始数据"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            result = {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
                "timestamp": item.get("date")  # 若有时间信息，可选
            }
            results.append(result)
    # 如果有知识图谱信息，也可以添加置顶（可选）
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"),
            "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"),
            "source": "knowledge_graph"
        })
    return results

def update_web_results(query: str, num_results: int = 5) -> list:
    """
    基于 SerpAPI 搜索结果。注意：此版本不将结果存入FAISS。
    它仅返回原始搜索结果。
    """
    results = serpapi_search(query, num_results)
    if not results:
        logging.info("网络搜索没有返回结果或发生错误")
        return []
    
    # 之前这里有删除旧网络结果和添加到ChromaDB的逻辑。
    # 由于FAISS IndexFlatL2不支持按ID删除，并且动态添加涉及复杂ID管理，
    # 此简化版本不将网络结果添加到FAISS索引。
    # 返回原始结果，供调用者决定如何使用（例如，仅作为文本上下文）。
    logging.info(f"网络搜索返回 {len(results)} 条结果，这些结果不会被添加到FAISS索引中。")
    return results # 返回原始SerpAPI结果列表

# 检查是否配置了SERPAPI_KEY
def check_serpapi_key():
    """检查是否配置了SERPAPI_KEY"""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != ""

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

#########################################
# 矛盾检测函数
#########################################
def detect_conflicts(sources):
    """精准矛盾检测算法"""
    key_facts = {}
    for item in sources:
        facts = extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
        for fact, value in facts.items():
            if fact in key_facts:
                if key_facts[fact] != value:
                    return True
            else:
                key_facts[fact] = value
    return False

def extract_facts(text):
    """从文本提取关键事实（示例逻辑）"""
    facts = {}
    # 提取数值型事实
    numbers = re.findall(r'\b\d{4}年|\b\d+%', text)
    if numbers:
        facts['关键数值'] = numbers
    # 提取技术术语
    if "产业图谱" in text:
        facts['技术方法'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法', text)))
    return facts

def evaluate_source_credibility(source):
    """评估来源可信度"""
    credibility_scores = {
        "gov.cn": 0.9,
        "edu.cn": 0.85,
        "weixin": 0.7,
        "zhihu": 0.6,
        "baidu": 0.5
    }
    
    url = source.get('url', '')
    if not url:
        return 0.5  # 默认中等可信度
    
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    
    domain = domain_match.group(1)
    
    # 检查是否匹配任何已知域名
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    
    return 0.5  # 默认中等可信度

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
        # 清空向量数据库和相关存储
        progress(0.1, desc="清理历史数据...")
        global faiss_index, faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
        faiss_index = None
        faiss_contents_map = {}
        faiss_metadatas_map = {}
        faiss_id_order_for_index = []
        
            # 清空BM25索引
            BM25_MANAGER.clear()
        logging.info("成功清理历史FAISS数据和BM25索引")
        
        # 清空文件处理状态
        file_processor.clear_files()
        
        total_files = len(files)
        processed_results = []
        total_chunks = 0
        
        all_new_chunks = []
        all_new_metadatas = []
        all_new_original_ids = []
        
        for idx, file in enumerate(files, 1):
            try:
                file_name = os.path.basename(file.name)
                progress((idx-1)/total_files, desc=f"处理文件 {idx}/{total_files}: {file_name}")
                
                file_processor.add_file(file_name)
                text = extract_text(file.name)
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,        
                    chunk_overlap=40,     
                    separators=["\\n\\n", "\\n", "。", "，", "；", "：", " ", ""]
                )
                chunks = text_splitter.split_text(text)
                
                if not chunks:
                    raise ValueError("文档内容为空或无法提取文本")
                
                doc_id = f"doc_{int(time.time())}_{idx}"
                
                # Store chunks and metadatas temporarily before batch embedding
                current_file_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                current_file_metadatas = [{"source": file_name, "doc_id": doc_id} for _ in chunks]

                all_new_chunks.extend(chunks)
                all_new_metadatas.extend(current_file_metadatas)
                all_new_original_ids.extend(current_file_ids)
                
                total_chunks += len(chunks)
                file_processor.update_status(file_name, "处理完成", len(chunks))
                processed_results.append(f"✅ {file_name}: 成功处理 {len(chunks)} 个文本块")
                
            except Exception as e:
                error_msg = str(e)
                logging.error(f"处理文件 {file_name} 时出错: {error_msg}")
                file_processor.update_status(file_name, f"处理失败: {error_msg}")
                processed_results.append(f"❌ {file_name}: 处理失败 - {error_msg}")
        
        if all_new_chunks:
            progress(0.8, desc="生成文本嵌入...")
            embeddings = EMBED_MODEL.encode(all_new_chunks, show_progress_bar=True)
            embeddings_np = np.array(embeddings).astype('float32')
            
            progress(0.9, desc="构建FAISS索引...")
            if faiss_index is None: # Should always be None here due to clearing
                dimension = embeddings_np.shape[1]
                faiss_index = faiss.IndexFlatL2(dimension)
            
            faiss_index.add(embeddings_np)
            
            for i, original_id in enumerate(all_new_original_ids):
                faiss_contents_map[original_id] = all_new_chunks[i]
                faiss_metadatas_map[original_id] = all_new_metadatas[i]
            faiss_id_order_for_index.extend(all_new_original_ids) # Keep track of order for FAISS indices
            logging.info(f"FAISS索引构建完成，共索引 {faiss_index.ntotal} 个文本块")

        summary = f"\\n总计处理 {total_files} 个文件，{total_chunks} 个文本块"
        processed_results.append(summary)
        
        progress(0.95, desc="构建BM25检索索引...")
        update_bm25_index() # This will need to use faiss_contents_map
        
        file_list = file_processor.get_file_list()
        
        return "\\n".join(processed_results), file_list
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"整体处理过程出错: {error_msg}")
        return f"处理过程出错: {error_msg}", []

# 新增：交叉编码器重排序函数
def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """
    使用交叉编码器对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        top_k: 返回结果数量
        
    返回:
        重排序后的结果列表 [(doc_id, {'content': doc, 'metadata': metadata, 'score': score}), ...]
    """
    if not docs:
        return []
        
    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("交叉编码器不可用，跳过重排序")
        # 返回原始顺序（按索引排序）
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
    
    # 准备交叉编码器输入
    cross_inputs = [[query, doc] for doc in docs]
    
    try:
        # 计算相关性得分
        scores = encoder.predict(cross_inputs)
        
        # 组合结果
        results = [
            (doc_id, {
                'content': doc, 
                'metadata': meta,
                'score': float(score)  # 确保是Python原生类型
            }) 
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        
        # 按得分排序
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        
        # 返回前K个结果
        return results[:top_k]
    except Exception as e:
        logging.error(f"交叉编码器重排序失败: {str(e)}")
        # 出错时返回原始顺序
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

# 新增：LLM相关性评分函数
@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """
    使用LLM对查询和文档的相关性进行评分（带缓存）
    
    参数:
        query: 查询字符串
        doc: 文档内容
        
    返回:
        相关性得分 (0-10)
    """
    try:
        # 构建评分提示词
        prompt = f"""给定以下查询和文档片段，评估它们的相关性。
        评分标准：0分表示完全不相关，10分表示高度相关。
        只需返回一个0-10之间的整数分数，不要有任何其他解释。
        
        查询: {query}
        
        文档片段: {doc}
        
        相关性分数(0-10):"""
        
        # 调用本地LLM
        response = session.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-r1:1.5b",  # 使用较小模型进行评分
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        
        # 提取得分
        result = response.json().get("response", "").strip()
        
        # 尝试解析为数字
        try:
            score = float(result)
            # 确保分数在0-10范围内
            score = max(0, min(10, score))
            return score
        except ValueError:
            # 如果无法解析为数字，尝试从文本中提取数字
            match = re.search(r'\b([0-9]|10)\b', result)
            if match:
                return float(match.group(1))
            else:
                # 默认返回中等相关性
                return 5.0
                
    except Exception as e:
        logging.error(f"LLM评分失败: {str(e)}")
        # 默认返回中等相关性
        return 5.0

def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """
    使用LLM对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        top_k: 返回结果数量
    
    返回:
        重排序后的结果列表
    """
    if not docs:
        return []
    
    results = []
    
    # 对每个文档进行评分
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        # 获取LLM评分
        score = get_llm_relevance_score(query, doc)
        
        # 添加到结果列表
        results.append((doc_id, {
            'content': doc, 
            'metadata': meta,
            'score': score / 10.0  # 归一化到0-1
        }))
    
    # 按得分排序
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    
    # 返回前K个结果
    return results[:top_k]

# 新增：通用重排序函数
def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """
    对检索结果进行重排序
    
    参数:
        query: 查询字符串
        docs: 文档内容列表
        doc_ids: 文档ID列表
        metadata_list: 元数据列表
        method: 重排序方法 ("cross_encoder", "llm" 或 None)
        top_k: 返回结果数量
        
    返回:
        重排序后的结果
    """
    # 如果未指定方法，使用全局配置
    if method is None:
        method = RERANK_METHOD
    
    # 根据方法选择重排序函数
    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        # 默认不进行重排序，按原始顺序返回
        return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx/len(docs)}) 
                for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]

def stream_answer(question, enable_web_search=False, model_choice="ollama", progress=gr.Progress()):
    """改进的流式问答处理流程，支持联网搜索、混合检索和重排序，以及多种模型选择"""
    global faiss_index # 确保可以访问
    try:
        # 检查向量数据库是否为空
        knowledge_base_exists = faiss_index is not None and faiss_index.ntotal > 0
        if not knowledge_base_exists:
                if not enable_web_search:
                    yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
                    return
                else:
                    logging.warning("知识库为空，将仅使用网络搜索结果")
        
        progress(0.3, desc="执行递归检索...")
        # 使用递归检索获取更全面的答案上下文
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question,
            max_iterations=3,
            enable_web_search=enable_web_search,
            model_choice=model_choice
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        # 使用检索到的结果构建上下文
        for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        # 检测矛盾
        conflict_detected = detect_conflicts(sources_for_conflict_detection)
        
        # 获取可信源
        if conflict_detected:
            credible_sources = [s for s in sources_for_conflict_detection 
                               if s['type'] == 'web' and evaluate_source_credibility(s) > 0.7]
        
        context = "\n\n".join(context_with_sources)
        
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        # 改进提示词模板，提高回答质量
        prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""
        
        prompt = prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else ("网络搜索结果" if enable_web_search else "本地文档"),
            context=context if context else ("网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
            question=question,
            time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
            conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
        )
        
        progress(0.7, desc="生成回答...")
        full_answer = ""
        
        # 根据模型选择使用不同的API
        if model_choice == "siliconflow":
            # 对于SiliconFlow API，不支持流式响应，所以一次性获取
            progress(0.8, desc="通过SiliconFlow API生成回答...")
            full_answer = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            
            # 处理思维链
            if "<think>" in full_answer and "</think>" in full_answer:
                processed_answer = process_thinking_content(full_answer)
            else:
                processed_answer = full_answer
                
            yield processed_answer, "完成!"
        else:
            # 使用本地Ollama模型的流式响应
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
                    
                    # 检查是否有完整的思维链标签可以处理
                    if "<think>" in full_answer and "</think>" in full_answer:
                        # 需要确保完整收集一个思维链片段后再显示
                        processed_answer = process_thinking_content(full_answer)
                    else:
                        processed_answer = full_answer
                    
                    yield processed_answer, "生成回答中..."
                    
            # 处理最终输出，确保应用思维链处理
            final_answer = process_thinking_content(full_answer)
            yield final_answer, "完成!"
        
    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"

def query_answer(question, enable_web_search=False, model_choice="ollama", progress=gr.Progress()):
    """问答处理流程，支持联网搜索、混合检索和重排序，以及多种模型选择"""
    global faiss_index # 确保可以访问
    try:
        logging.info(f"收到问题：{question}，联网状态：{enable_web_search}，模型选择：{model_choice}")
        
        # 检查向量数据库是否为空
        knowledge_base_exists = faiss_index is not None and faiss_index.ntotal > 0
        if not knowledge_base_exists:
                if not enable_web_search:
                    return "⚠️ 知识库为空，请先上传文档。"
                else:
                    logging.warning("知识库为空，将仅使用网络搜索结果")
        
        progress(0.3, desc="执行递归检索...")
        # 使用递归检索获取更全面的答案上下文
        all_contexts, all_doc_ids, all_metadata = recursive_retrieval(
            initial_query=question,
            max_iterations=3,
            enable_web_search=enable_web_search,
            model_choice=model_choice
        )
        
        # 组合上下文，包含来源信息
        context_with_sources = []
        sources_for_conflict_detection = []
        
        # 使用检索到的结果构建上下文
        for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
            source_type = metadata.get('source', '本地文档')
            
            source_item = {
                'text': doc,
                'type': source_type
            }
            
            if source_type == 'web':
                url = metadata.get('url', '未知URL')
                title = metadata.get('title', '未知标题')
                context_with_sources.append(f"[网络来源: {title}] (URL: {url})\n{doc}")
                source_item['url'] = url
                source_item['title'] = title
            else:
                source = metadata.get('source', '未知来源')
                context_with_sources.append(f"[本地文档: {source}]\n{doc}")
                source_item['source'] = source
            
            sources_for_conflict_detection.append(source_item)
        
        # 检测矛盾
        conflict_detected = detect_conflicts(sources_for_conflict_detection)
        
        # 获取可信源
        if conflict_detected:
            credible_sources = [s for s in sources_for_conflict_detection 
                              if s['type'] == 'web' and evaluate_source_credibility(s) > 0.7]
        
        context = "\n\n".join(context_with_sources)
        
        # 添加时间敏感检测
        time_sensitive = any(word in question for word in ["最新", "今年", "当前", "最近", "刚刚"])
        
        # 改进提示词模板，提高回答质量
        prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. 在回答末尾标注信息来源{time_instruction}{conflict_instruction}

请现在开始回答："""
        
        prompt = prompt_template.format(
            context_type="本地文档和网络搜索结果" if enable_web_search and knowledge_base_exists else ("网络搜索结果" if enable_web_search else "本地文档"),
            context=context if context else ("网络搜索结果将用于回答。" if enable_web_search and not knowledge_base_exists else "知识库为空或未找到相关内容。"),
            question=question,
            time_instruction="，优先使用最新的信息" if time_sensitive and enable_web_search else "",
            conflict_instruction="，并明确指出不同来源的差异" if conflict_detected else ""
        )
        
        progress(0.8, desc="生成回答...")
        
        # 根据模型选择使用不同的API
        if model_choice == "siliconflow":
            # 使用SiliconFlow API
            result = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            
            # 处理思维链
            processed_result = process_thinking_content(result)
            return processed_result
        else:
            # 使用本地Ollama
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
            return process_thinking_content(str(result.get("response", "未获取到有效回答")))
            
    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except KeyError:
        return "响应格式异常，请检查模型服务"
    except Exception as e:
        progress(1.0, desc="遇到错误")  # 确保进度条完成
        return f"系统错误: {str(e)}"

def process_thinking_content(text):
    """处理包含<think>标签的内容，将其转换为Markdown格式"""
    # 检查输入是否为有效文本
    if text is None:
        return ""
    
    # 确保输入是字符串
    if not isinstance(text, str):
        try:
            processed_text = str(text)
        except:
            return "无法处理的内容格式"
    else:
        processed_text = text
    
    # 处理思维链标签
    try:
        while "<think>" in processed_text and "</think>" in processed_text:
            start_idx = processed_text.find("<think>")
            end_idx = processed_text.find("</think>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = processed_text[start_idx + 7:end_idx]
                before_think = processed_text[:start_idx]
                after_think = processed_text[end_idx + 8:]
                
                # 使用可折叠详情框显示思维链
                processed_text = before_think + "\n\n<details>\n<summary>思考过程（点击展开）</summary>\n\n" + thinking_content + "\n\n</details>\n\n" + after_think
        
        # 处理其他HTML标签，但保留details和summary标签
        processed_html = []
        i = 0
        while i < len(processed_text):
            if processed_text[i:i+8] == "<details" or processed_text[i:i+9] == "</details" or \
               processed_text[i:i+8] == "<summary" or processed_text[i:i+9] == "</summary":
                # 保留这些标签
                tag_end = processed_text.find(">", i)
                if tag_end != -1:
                    processed_html.append(processed_text[i:tag_end+1])
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
    except Exception as e:
        logging.error(f"处理思维链内容时出错: {str(e)}")
        # 出错时至少返回原始文本，但确保安全处理HTML标签
        try:
            return text.replace("<", "&lt;").replace(">", "&gt;")
        except:
            return "处理内容时出错"
    
    return processed_text

def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """
    调用SiliconFlow API获取回答
    
    Args:
        prompt: 提示词
        temperature: 温度参数
        max_tokens: 最大生成token数
        
    Returns:
        生成的回答文本和思维链内容
    """
    try:
        payload = {
            "model": "Pro/deepseek-ai/DeepSeek-R1",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "max_tokens": max_tokens,
            "stop": None,
            "temperature": temperature,
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
        
        headers = {
            "Authorization": f"Bearer sk-lkyhoqsulzscivpiogbesxhougfjaqlthmakpnkmsousyqep",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            SILICONFLOW_API_URL,
            json=payload,
            headers=headers,
            timeout=60  # 延长超时时间
        )
        
        response.raise_for_status()
        result = response.json()
        
        # 提取回答内容和思维链
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            
            # 如果有思维链，则添加特殊标记，以便前端处理
            if reasoning:
                # 添加思维链标记
                full_response = f"{content}<think>{reasoning}</think>"
                return full_response
            else:
                return content
        else:
            return "API返回结果格式异常，请检查"
            
    except requests.exceptions.RequestException as e:
        logging.error(f"调用SiliconFlow API时出错: {str(e)}")
        return f"调用API时出错: {str(e)}"
    except json.JSONDecodeError:
        logging.error("SiliconFlow API返回非JSON响应")
        return "API响应解析失败"
    except Exception as e:
        logging.error(f"调用SiliconFlow API时发生未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"

def hybrid_merge(semantic_results, bm25_results, alpha=0.7):
    """
    合并语义搜索和BM25搜索结果
    
    参数:
        semantic_results: 向量检索结果 (字典格式，包含ids, documents, metadatas)
        bm25_results: BM25检索结果 (字典列表，包含id, score, content)
        alpha: 语义搜索权重 (0-1)
        
    返回:
        合并后的结果列表 [(doc_id, {'score': score, 'content': content, 'metadata': metadata}), ...]
    """
    merged_dict = {}
    global faiss_metadatas_map # Ensure we can access the global map
    
    # 处理语义搜索结果
    if (semantic_results and 
        isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
        isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
        isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
        isinstance(semantic_results['documents'][0], list) and 
        isinstance(semantic_results['metadatas'][0], list) and 
        isinstance(semantic_results['ids'][0], list) and
        len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(semantic_results['ids'][0])):
        
        num_results = len(semantic_results['documents'][0])
        # Assuming semantic_results are already ordered by relevance (higher is better)
        # A simple rank-based score, can be replaced if actual scores/distances are available and preferred
        for i, (doc_id, doc, meta) in enumerate(zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            score = 1.0 - (i / max(1, num_results)) # Higher rank (smaller i) gets higher score
            merged_dict[doc_id] = {
                'score': alpha * score, 
                'content': doc,
                'metadata': meta
            }
    else:
        logging.warning("Semantic results are missing, have an unexpected format, or are empty. Skipping semantic part in hybrid merge.")
    
    # 处理BM25结果
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
        
    valid_bm25_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
    max_bm25_score = max(valid_bm25_scores) if valid_bm25_scores else 1.0
    
    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            logging.warning(f"Skipping invalid BM25 result item: {result}")
            continue
            
        doc_id = result['id']
        # Normalize BM25 score
        normalized_score = result['score'] / max_bm25_score if max_bm25_score > 0 else 0
        
        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * normalized_score
        else:
            metadata = faiss_metadatas_map.get(doc_id, {}) # Get metadata from our global map
            merged_dict[doc_id] = {
                'score': (1 - alpha) * normalized_score,
                'content': result['content'],
                'metadata': metadata
            }
    
    merged_results = sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)
    return merged_results

# 新增：更新本地文档的BM25索引
def update_bm25_index():
    """更新BM25索引，从内存中的映射加载所有文档"""
    global faiss_contents_map, faiss_id_order_for_index
    try:
        # Use the ordered list of IDs to ensure consistency
        doc_ids = faiss_id_order_for_index
        if not doc_ids:
            logging.warning("没有可索引的文档 (FAISS ID列表为空)")
            BM25_MANAGER.clear()
            return False

        # Retrieve documents in the correct order
        documents = [faiss_contents_map.get(doc_id, "") for doc_id in doc_ids]
        
        # Filter out any potential empty documents if necessary, though map access should be safe
        valid_docs_with_ids = [(doc_id, doc) for doc_id, doc in zip(doc_ids, documents) if doc]
        if not valid_docs_with_ids:
             logging.warning("没有有效的文档内容可用于BM25索引")
            BM25_MANAGER.clear()
            return False
            
        # Separate IDs and documents again for building the index
        final_doc_ids = [item[0] for item in valid_docs_with_ids]
        final_documents = [item[1] for item in valid_docs_with_ids]
            
        BM25_MANAGER.build_index(final_documents, final_doc_ids)
        logging.info(f"BM25索引更新完成，共索引 {len(final_doc_ids)} 个文档")
        return True
    except Exception as e:
        logging.error(f"更新BM25索引失败: {str(e)}")
        return False

# 新增函数：获取系统使用的模型信息
def get_system_models_info():
    """返回系统使用的各种模型信息"""
    models_info = {
        "嵌入模型": "all-MiniLM-L6-v2",
        "分块方法": "RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)",
        "检索方法": "向量检索 + BM25混合检索 (α=0.7)",
        "重排序模型": "交叉编码器 (sentence-transformers/distiluse-base-multilingual-cased-v2)",
        "生成模型": "deepseek-r1 (7B/1.5B)",
        "分词工具": "jieba (中文分词)"
    }
    return models_info

# 新增函数：获取文档分块可视化数据
def get_document_chunks(progress=gr.Progress()):
    """获取文档分块结果用于可视化"""
    global faiss_contents_map, faiss_metadatas_map, faiss_id_order_for_index
    global chunk_data_cache # Ensure we can update the global cache
    try:
        progress(0.1, desc="正在从内存加载数据...")
        
        if not faiss_id_order_for_index:
            chunk_data_cache = []
            return [], "知识库中没有文档，请先上传并处理文档。"
            
        progress(0.5, desc="正在组织分块数据...")
        
        doc_groups = {}
        # Iterate using the ordered IDs to reflect the FAISS index order conceptually
        for doc_id in faiss_id_order_for_index:
            doc = faiss_contents_map.get(doc_id, "")
            meta = faiss_metadatas_map.get(doc_id, {})
            if not doc: # Skip if content is somehow empty
                continue

            source = meta.get('source', '未知来源')
            if source not in doc_groups:
                doc_groups[source] = []
            
            doc_id_meta = meta.get('doc_id', '未知ID') # Get the original document ID from meta
            chunk_info = {
                "original_id": doc_id, # Keep the chunk-specific ID
                "doc_id": doc_id_meta,
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "full_content": doc,
                "token_count": len(list(jieba.cut(doc))),
                "char_count": len(doc)
            }
            doc_groups[source].append(chunk_info)
        
        result_dicts = []
        result_lists = []
        
        # Keep track of chunks per source for indexing
        source_chunk_counters = {source: 0 for source in doc_groups.keys()}
        total_chunks = 0
        
        for source, chunks in doc_groups.items():
            num_chunks_in_source = len(chunks)
            for chunk in chunks:
                source_chunk_counters[source] += 1
                total_chunks += 1
                result_dict = {
                    "来源": source,
                    "序号": f"{source_chunk_counters[source]}/{num_chunks_in_source}",
                    "字符数": chunk["char_count"],
                    "分词数": chunk["token_count"],
                    "内容预览": chunk["content"],
                    "完整内容": chunk["full_content"],
                    "原始分块ID": chunk["original_id"] # Add original ID for potential debugging
                }
                result_dicts.append(result_dict)
                
                result_lists.append([
                    source,
                    f"{source_chunk_counters[source]}/{num_chunks_in_source}",
                    chunk["char_count"],
                    chunk["token_count"],
                    chunk["content"]
                ])
        
        progress(1.0, desc="数据加载完成!")
        
        chunk_data_cache = result_dicts # Update the global cache
        summary = f"总计 {total_chunks} 个文本块，来自 {len(doc_groups)} 个不同来源。"
        
        return result_lists, summary
    except Exception as e:
        chunk_data_cache = []
        return [], f"获取分块数据失败: {str(e)}"

# 添加全局缓存变量
chunk_data_cache = []

# 新增函数：显示分块详情
def show_chunk_details(evt: gr.SelectData, chunks):
    """显示选中分块的详细内容"""
    try:
        if evt.index[0] < len(chunk_data_cache):
            selected_chunk = chunk_data_cache[evt.index[0]]
            return selected_chunk.get("完整内容", "内容加载失败")
        return "未找到选中的分块"
    except Exception as e:
        return f"加载分块详情失败: {str(e)}"

# 修改布局部分，添加一个新的标签页
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
        margin: 0;
        padding: 0;
        overflow-x: hidden;
        width: 100vw;
        height: 100vh;
    }

    .gradio-container {
        max-width: 100% !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 1% !important;
        color: var(--text-color);
        background-color: var(--bg-color);
        min-height: 100vh;
    }
    
    /* 确保标签内容撑满 */
    .tabs.svelte-710i53 {
        margin: 0 !important;
        padding: 0 !important;
        width: 100% !important;
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
        width: 100%;
    }

    .right-panel {
        height: 100vh;
        background: var(--bg-color);
        width: 100%;
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
        max-height: 80vh;
        height: 80vh !important;
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

    /* API配置提示样式 */
    .api-info {
        margin-top: 10px;
        padding: 10px;
        border-radius: 5px;
        background: var(--panel-bg);
        border: 1px solid var(--border-color);
    }

    /* 新增: 数据可视化卡片样式 */
    .model-card {
        background: var(--panel-bg);
        border-radius: 8px;
        padding: 16px;
        border: 1px solid var(--border-color);
        margin-bottom: 16px;
    }

    .model-card h3 {
        margin-top: 0;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }

    .model-item {
        display: flex;
        margin-bottom: 8px;
    }

    .model-item .label {
        flex: 1;
        font-weight: 500;
    }

    .model-item .value {
        flex: 2;
    }

    /* 数据表格样式 */
    .chunk-table {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid var(--border-color);
    }

    .chunk-table th, .chunk-table td {
        border: 1px solid var(--border-color);
        padding: 8px;
    }

    .chunk-detail-box {
        min-height: 200px;
        padding: 16px;
        background: var(--panel-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
        font-family: monospace;
        white-space: pre-wrap;
        overflow-y: auto;
    }
    """
) as demo:
    gr.Markdown("# 🧠 智能文档问答系统")
    
    with gr.Tabs() as tabs:
        # 第一个选项卡：问答对话
        with gr.TabItem("💬 问答对话"):
            with gr.Row(equal_height=True):
                # 左侧操作面板 - 调整比例为合适的大小
                with gr.Column(scale=5, elem_classes="left-panel"):
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
                    
                    # 将问题输入区移至左侧面板底部
                    gr.Markdown("## ❓ 输入问题")
                    with gr.Group():
                        question_input = gr.Textbox(
                            label="输入问题",
                            lines=3,
                            placeholder="请输入您的问题...",
                            elem_id="question-input"
                        )
                        with gr.Row():
                            # 添加联网开关
                            web_search_checkbox = gr.Checkbox(
                                label="启用联网搜索", 
                                value=False,
                                info="打开后将同时搜索网络内容（需配置SERPAPI_KEY）"
                            )
                            
                            # 添加模型选择下拉框
                            model_choice = gr.Dropdown(
                                choices=["ollama", "siliconflow"],
                                value="ollama",
                                label="模型选择",
                                info="选择使用本地模型或云端模型"
                            )
                            
                        with gr.Row():
                            ask_btn = gr.Button("🔍 开始提问", variant="primary", scale=2)
                            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", elem_classes="clear-button", scale=1)
                    
                    # 添加API配置提示信息
                    api_info = gr.HTML(
                        """
                        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
                            <p>📢 <strong>功能说明：</strong></p>
                            <p>1. <strong>联网搜索</strong>：%s</p>
                            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
                        </div>
                        """
                    )

                # 右侧对话区 - 调整比例
                with gr.Column(scale=7, elem_classes="right-panel"):
                    gr.Markdown("## 📝 对话记录")
                    
                    # 对话记录显示区
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=600,  # 增加高度
                        elem_classes="chat-container",
                        show_label=False
                    )
                    
                    status_display = gr.HTML("", elem_id="status-display")
                    gr.Markdown("""
                    <div class="footer-note">
                        *回答生成可能需要1-2分钟，请耐心等待<br>
                        *支持多轮对话，可基于前文继续提问
                    </div>
                    """)
        
        # 第二个选项卡：分块可视化
        with gr.TabItem("📊 分块可视化"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 💡 系统模型信息")
                    
                    # 显示系统模型信息卡片
                    models_info = get_system_models_info()
                    with gr.Group(elem_classes="model-card"):
                        gr.Markdown("### 核心模型与技术")
                        
                        for key, value in models_info.items():
                            with gr.Row():
                                gr.Markdown(f"**{key}**:", elem_classes="label")
                                gr.Markdown(f"{value}", elem_classes="value")
                
                with gr.Column(scale=2):
                    gr.Markdown("## 📄 文档分块统计")
                    refresh_chunks_btn = gr.Button("🔄 刷新分块数据", variant="primary")
                    chunks_status = gr.Markdown("点击按钮查看分块统计")
            
            # 分块数据表格和详情
            with gr.Row():
                chunks_data = gr.Dataframe(
                    headers=["来源", "序号", "字符数", "分词数", "内容预览"],
                    elem_classes="chunk-table",
                    interactive=False,
                    wrap=True,
                    row_count=(10, "dynamic")
                )
            
            with gr.Row():
                chunk_detail_text = gr.Textbox(
                    label="分块详情",
                    placeholder="点击表格中的行查看完整内容...",
                    lines=8,
                    elem_classes="chunk-detail-box"
                )
                
            gr.Markdown("""
            <div class="footer-note">
                * 点击表格中的行可查看该分块的完整内容<br>
                * 分词数表示使用jieba分词后的token数量
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

    # 定义函数处理事件
    def clear_chat_history():
        return None, "对话已清空"

    def process_chat(question, history, enable_web_search, model_choice):
        if history is None:
            history = []
        
        # 更新模型选择信息的显示
        api_text = """
        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
        </div>
        """ % (
            "已启用" if enable_web_search else "未启用", 
            "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
            "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
        )
        
        # 如果问题为空，不处理
        if not question or question.strip() == "":
            history.append(("", "问题不能为空，请输入有效问题。"))
            return history, "", api_text
        
        # 添加用户问题到历史
        history.append((question, ""))
        
        # 创建生成器
        resp_generator = stream_answer(question, enable_web_search, model_choice)
        
        # 流式更新回答
        for response, status in resp_generator:
            history[-1] = (question, response)
            yield history, "", api_text

    def update_api_info(enable_web_search, model_choice):
        api_text = """
        <div class="api-info" style="margin-top:10px;padding:10px;border-radius:5px;background:var(--panel-bg);border:1px solid var(--border-color);">
            <p>📢 <strong>功能说明：</strong></p>
            <p>1. <strong>联网搜索</strong>：%s</p>
            <p>2. <strong>模型选择</strong>：当前使用 <strong>%s</strong> %s</p>
        </div>
        """ % (
            "已启用" if enable_web_search else "未启用", 
            "Cloud DeepSeek-R1 模型" if model_choice == "siliconflow" else "本地 Ollama 模型",
            "(需要在.env文件中配置SERPAPI_KEY)" if enable_web_search else ""
        )
        return api_text

    # 绑定UI事件
    upload_btn.click(
        process_multiple_pdfs,
        inputs=[file_input],
        outputs=[upload_status, file_list],
        show_progress=True
    )

    # 绑定提问按钮
    ask_btn.click(
        process_chat,
        inputs=[question_input, chatbot, web_search_checkbox, model_choice],
        outputs=[chatbot, question_input, api_info]
    )

    # 绑定清空按钮
    clear_btn.click(
        clear_chat_history,
        inputs=[],
        outputs=[chatbot, status_display]
    )

    # 当切换联网搜索或模型选择时更新API信息
    web_search_checkbox.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )
    
    model_choice.change(
        update_api_info,
        inputs=[web_search_checkbox, model_choice],
        outputs=[api_info]
    )
    
    # 新增：分块可视化刷新按钮事件
    refresh_chunks_btn.click(
        fn=get_document_chunks,
        outputs=[chunks_data, chunks_status]
    )
    
    # 新增：分块表格点击事件
    chunks_data.select(
        fn=show_chunk_details,
        inputs=chunks_data,
        outputs=chunk_detail_text
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
            ssl_verify=False,
            height=900
        )
    except Exception as e:
        print(f"启动失败: {str(e)}")

