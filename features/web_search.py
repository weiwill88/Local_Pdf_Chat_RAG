"""
Pencarian Web —— Mendapatkan informasi internet waktu nyata melalui SerpAPI

Poin Pembelajaran:
- "R" dalam RAG tidak terbatas pada dokumen lokal, tetapi juga dapat memperoleh informasi waktu nyata dari internet
- SerpAPI adalah pembungkus API untuk Google Search, perlu mendaftar untuk mendapatkan Kunci API (API Key)
- Hasil pencarian web tidak masuk ke indeks FAISS, hanya disediakan untuk LLM sebagai konteks teks
"""

import logging
import requests
from config import SERPAPI_KEY, SEARCH_ENGINE


def check_serpapi_key():
    """Memeriksa apakah SERPAPI_KEY yang valid telah dikonfigurasi"""
    return SERPAPI_KEY is not None and SERPAPI_KEY.strip() != "" and not SERPAPI_KEY.startswith("Your")


def serpapi_search(query, num_results=5):
    """Menjalankan pencarian SerpAPI"""
    if not SERPAPI_KEY:
        raise ValueError("Variabel lingkungan SERPAPI_KEY tidak diatur")
    try:
        params = {
            "engine": SEARCH_ENGINE, "q": query, "api_key": SERPAPI_KEY,
            "num": num_results, "hl": "id", "gl": "id"
        }
        response = requests.get("https://serpapi.com/search", params=params, timeout=15)
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except Exception as e:
        logging.error(f"Pencarian web gagal: {str(e)}")
        return []


def _parse_serpapi_results(data):
    """Mengurai data mentah yang dikembalikan oleh SerpAPI"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"), "url": item.get("link"),
                "snippet": item.get("snippet"), "timestamp": item.get("date")
            })
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"), "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"), "source": "knowledge_graph"
        })
    return results


def search_web(query, num_results=5):
    """Menjalankan pencarian web (hasil tidak ditambahkan ke indeks FAISS, hanya digunakan sebagai konteks)"""
    results = serpapi_search(query, num_results)
    if not results:
        logging.info("Pencarian web tidak mengembalikan hasil")
    else:
        logging.info(f"Pencarian web mengembalikan {len(results)} hasil")
    return results
