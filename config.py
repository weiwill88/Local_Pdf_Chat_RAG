"""
配置中心 —— 环境变量加载、模型参数、自动检测机制

学习要点：
- 了解如何通过 .env 文件管理敏感配置（API Key）
- 了解 RAG 系统中的关键超参数及其作用
- 理解 LLM 后端的自动检测与回退机制
"""

import os
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
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.deepseek.com/v1/chat/completions"
)
MAGICK_API_KEY = os.getenv("MAGICK_API_KEY")
MAGICK_API_URL = os.getenv(
    "MAGICK_API_URL",
    "https://api.magickapi.com/v1/chat/completions"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第三步：模型名称配置
# Ollama 格式: deepseek-r1:8b
# SiliconFlow/Magick API 格式: 使用对应平台提供的模型 ID
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
SILICONFLOW_MODEL_NAME = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-v4-flash")
MAGICK_MODEL_NAME = os.getenv("MAGICK_MODEL_NAME", "gpt-4o-mini")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

MODEL_CHOICES = ["ollama", "siliconflow", "magick"]
MODEL_DISPLAY_NAMES = {
    "ollama": "本地 Ollama 模型",
    "siliconflow": "Cloud DeepSeek-R1 模型",
    "magick": "Magick API 模型"
}


def is_configured_api_key(api_key):
    """判断 API Key 是否为用户实际配置值。"""
    return bool(api_key and api_key.strip() and not api_key.strip().startswith("Your"))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第四步：RAG 超参数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400          # 文本分块大小（字符数）
CHUNK_OVERLAP = 40        # 相邻分块的重叠字符数
HYBRID_ALPHA = 0.7        # 混合检索中语义检索的权重（0-1）
RETRIEVAL_TOP_K = 10      # 检索返回的候选文档数量
RERANK_TOP_K = 5          # 重排序后保留的文档数量
MAX_RETRIEVAL_ITERATIONS = 3  # 递归检索的最大迭代轮数

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 第五步：Tesseract OCR 路径配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()
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
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logging.info("✅ 检测到本地 Ollama 服务，默认使用本地模型")
            return "ollama"
    except Exception:
        pass

    logging.warning("⚠️ 未检测到可用 LLM 后端，请配置 SiliconFlow/Magick API Key 或启动 Ollama")
    return "siliconflow"

DEFAULT_MODEL_CHOICE = detect_default_model()
