"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
import json
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第一步：加载环境变量
# 优先加载 .env（用户配置），不存在则回退到 example.env（示例配置）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ 未找到 .env 文件，已回退加载 example.env。建议：cp example.env .env 并填入真实 API Key")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第二步：API 密钥配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = (os.getenv("SERPAPI_KEY") or "").strip()
SEARCH_ENGINE = "google"

SILICONFLOW_API_KEY = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
SILICONFLOW_API_URL = (os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.deepseek.com/v1/chat/completions"
) or "").strip()
MAGICK_API_KEY = (os.getenv("MAGICK_API_KEY") or "").strip()
MAGICK_API_URL = (os.getenv(
    "MAGICK_API_URL",
    "https://api.magickapi.com/v1/chat/completions"
) or "").strip()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = (os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b") or "").strip()
SILICONFLOW_MODEL_NAME = (os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-v4-flash") or "").strip()
MAGICK_MODEL_NAME = (os.getenv("MAGICK_MODEL_NAME", "gpt-4o-mini") or "").strip()
RERANK_METHOD = (os.getenv("RERANK_METHOD", "cross_encoder") or "").strip()

# ━━━ Ollama 可配置参数（默认值） ━━━
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "0.9"))

# ━━━ 固定的模型选项底座（Ollama 具体模型名在 rag_demo.py 中动态加载） ━━━
MODEL_CHOICES = ["siliconflow", "ollama"]
MODEL_DISPLAY_NAMES = {
    "siliconflow": "Cloud SiliconFlow",
    "ollama": "Ollama 本地模型",
}


def is_configured_api_key(api_key):
    """判断 API Key 是否为用户实际配置值（已自动 strip 空格）。"""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))


def is_serpapi_configured():
    """判断 SERPAPI_KEY 是否有效配置（含空格容错）。"""
    return bool(SERPAPI_KEY and len(SERPAPI_KEY) > 10 and not SERPAPI_KEY.startswith("Your"))


def get_ollama_models():
    """
    从本地 Ollama 服务获取已安装模型列表。

    Returns:
        list[str]: 模型名称列表，如 ["deepseek-r1:8b", "llama3:8b"]
        若 Ollama 未运行或出错，返回空列表。
    """
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            return []
        data = resp.json()
        models = data.get("models", [])
        return sorted([m["name"] for m in models if "name" in m])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        logging.warning("Ollama 服务未启动或无法连接")
        return []
    except (json.JSONDecodeError, KeyError, Exception) as e:
        logging.error(f"获取 Ollama 模型列表失败: {str(e)}")
        return []


def is_ollama_running():
    """检测本地 Ollama 服务是否运行（11434 端口）"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400          # 文本分块大小（字符数）
CHUNK_OVERLAP = 40        # 相邻分块的重叠字符数

# ━━━ Parent Document Retriever 参数 ━━━
PARENT_CHUNK_SIZE = 800       # 父文档块大小（字符数）
PARENT_CHUNK_OVERLAP = 80     # 父文档块重叠
CHILD_CHUNK_SIZE = 200        # 子文档块大小（字符数）
CHILD_CHUNK_OVERLAP = 20      # 子文档块重叠
USE_PARENT_RETRIEVER = True   # 是否启用父文档检索

# ━━━ MMR 参数 ━━━
MMR_LAMBDA = 0.5              # MMR 多样性阈值：1=纯相关，0=纯多样
MMR_TOP_K = 8                 # MMR 重排后保留的候选数

HYBRID_ALPHA = 0.7        # 混合检索中语义检索的权重（0-1）
RETRIEVAL_TOP_K = 10      # 检索返回的候选文档数量

# ━━━ 联网搜索参数 ━━━
WEB_SEARCH_MAX_RESULTS = 5       # 联网搜索返回的最大结果条数
WEB_SEARCH_TIMEOUT = 15          # 联网搜索超时时间（秒）
WEB_SEARCH_AUTO_FALLBACK = True  # 本地检索不足时自动联网补充
LOCAL_SCORE_THRESHOLD = 0.3      # 本地检索相关性阈值，低于此值触发联网
RERANK_TOP_K = 5          # 重排序后保留的文档数量
MAX_RETRIEVAL_ITERATIONS = 3  # 递归检索的最大迭代轮数

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：Tesseract OCR 路径配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TESSERACT_CMD = (os.getenv("TESSERACT_CMD") or "").strip()
# 如果配置了路径且文件存在，提前设置 pytesseract 路径
if TESSERACT_CMD:
    if os.path.isfile(TESSERACT_CMD):
        try:
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
            logging.info(f"✅ 使用配置的 Tesseract 路径: {TESSERACT_CMD}")
        except ImportError:
            pass  # pytesseract 尚未安装，后续再设置
    else:
        logging.warning(f"⚠️ 配置的 TESSERACT_CMD 路径不存在: {TESSERACT_CMD}，将使用系统默认检测")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：运行时环境配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第六步：LLM 后端自动检测
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    自动检测可用的 LLM 后端，返回默认模型选择

    检测优先级：
    1. SiliconFlow API Key 已配置 → 默认使用云端 API
    2. Magick API Key 已配置 → 默认使用 Magick API
    3. 本地 Ollama 服务可用 → 默认使用本地模型
    4. 都不可用 → 返回 siliconflow 并提示用户配置
    """
    if is_configured_api_key(SILICONFLOW_API_KEY):
        logging.info("✅ 检测到 SiliconFlow API Key，默认使用云端模型")
        return "siliconflow"

    if is_configured_api_key(MAGICK_API_KEY):
        logging.info("✅ 检测到 Magick API Key，默认使用 Magick API 模型")
        return "magick"

    try:
        if is_ollama_running():
            ollama_models = get_ollama_models()
            if ollama_models:
                logging.info(f"✅ 检测到本地 Ollama 服务（模型: {ollama_models[0]}），默认使用本地模型")
                return "ollama"
    except Exception:
        pass

    logging.warning("⚠️ 未检测到可用 LLM 后端，请配置 SiliconFlow/Magick API Key 或启动 Ollama")
    return "siliconflow"

DEFAULT_MODEL_CHOICE = detect_default_model()
