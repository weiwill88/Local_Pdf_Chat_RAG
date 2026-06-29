"""
Pusat Konfigurasi —— Pemuatan variabel lingkungan, parameter model, mekanisme deteksi otomatis

Poin Pembelajaran:
- Memahami cara mengelola konfigurasi sensitif (API Key) melalui file .env
- Memahami hyperparameter kunci dalam sistem RAG beserta fungsinya
- Memahami mekanisme deteksi otomatis dan rollback pada backend LLM
"""

import os
import logging
import requests
from pathlib import Path
from dotenv import load_dotenv

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 1: Memuat variabel lingkungan
# Memuat .env (konfigurasi pengguna) terlebih dahulu, jika tidak ada maka kembali ke example.env (konfigurasi contoh)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
dotenv_path = Path(__file__).parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path(__file__).parent / "example.env"
    logging.warning("⚠️ File .env tidak ditemukan, memuat example.env sebagai cadangan. Saran: cp example.env .env dan masukkan API Key yang sebenarnya")
load_dotenv(dotenv_path)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 2: Konfigurasi API Key
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SEARCH_ENGINE = "google"

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    "https://api.siliconflow.cn/v1/chat/completions"
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 3: Konfigurasi Nama Model
# Format Ollama: deepseek-r1:8b | Format SiliconFlow: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "deepseek-r1:8b")
SILICONFLOW_MODEL_NAME = os.getenv("SILICONFLOW_MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
RERANK_METHOD = os.getenv("RERANK_METHOD", "cross_encoder")

# Konfigurasi Model Embedding
# Pilihan provider: 'sentence_transformers' (offline lokal) atau 'ollama' (lokal menggunakan layanan Ollama)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 4: Hyperparameter RAG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK_SIZE = 400          # Ukuran blok teks (jumlah karakter)
CHUNK_OVERLAP = 60        # Jumlah karakter tumpang tindih antar blok yang berdekatan
HYBRID_ALPHA = 0.7        # Bobot pencarian semantik dalam pencarian hibrida (0-1)
RETRIEVAL_TOP_K = 15      # Jumlah dokumen kandidat yang dikembalikan oleh pencarian (factoid default)
RERANK_TOP_K = 7          # Jumlah dokumen yang dipertahankan setelah pengurutan ulang (factoid default)
MAX_RETRIEVAL_ITERATIONS = 1  # Jumlah iterasi maksimum untuk pencarian rekursif

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Dynamic Top-K per query type
# factoid   : single-answer questions ("what year", "who is")
# comparison: multi-entity questions ("compare X and Y", "difference between")
# enumeration: exhaustive list questions ("list all", "apa saja", "sebutkan semua")
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FACTOID_RETRIEVAL_TOP_K = 15
COMPARISON_RETRIEVAL_TOP_K = 25
ENUMERATION_RETRIEVAL_TOP_K = 40

FACTOID_RERANK_TOP_K = 7
COMPARISON_RERANK_TOP_K = 15
ENUMERATION_RERANK_TOP_K = 25

# Keywords used to detect query type (extend as needed)
ENUMERATION_SIGNALS = [
    "list", "all", "every", "each", "enumerate",
    "sebutkan", "semua", "apa saja", "daftarkan",
    "berapa banyak", "which", "what are", "what were",
    "how many",
]
COMPARISON_SIGNALS = [
    "compare", "comparison", "difference", "vs", "versus",
    "bandingkan", "perbedaan", "dibandingkan",
]

# Page-level smart OCR fallback thresholds
IMAGE_PLACEHOLDER_THRESHOLD = 8
MIN_VISIBLE_RATIO = 0.20

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 5: Konfigurasi Lingkungan Runtime
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'
requests.adapters.DEFAULT_RETRIES = 3

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Langkah 6: Deteksi Otomatis Backend LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def detect_default_model():
    """
    Mendeteksi secara otomatis backend LLM yang tersedia, mengembalikan pilihan model default

    Prioritas Deteksi:
    1. SiliconFlow API Key telah dikonfigurasi → Default menggunakan API cloud
    2. Layanan Ollama lokal tersedia → Default menggunakan model lokal
    3. Keduanya tidak tersedia → Mengembalikan siliconflow dan meminta pengguna untuk mengonfigurasi
    """
    if SILICONFLOW_API_KEY and SILICONFLOW_API_KEY.strip() and not SILICONFLOW_API_KEY.startswith("Your"):
        logging.info("✅ Terdeteksi SiliconFlow API Key, menggunakan model cloud secara default")
        return "siliconflow"

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            logging.info("✅ Terdeteksi layanan Ollama lokal, menggunakan model lokal secara default")
            return "ollama"
    except Exception:
        pass

    logging.warning("⚠️ Backend LLM yang aktif tidak terdeteksi, silakan konfigurasi SiliconFlow API Key atau jalankan Ollama")
    return "siliconflow"

DEFAULT_MODEL_CHOICE = detect_default_model()
