"""
Re-ranker —— Melakukan pengurutan ulang halus kedua pada hasil pencarian

Poin Pembelajaran:
- Pencarian dua tahap (Recall + Rerank) adalah paradigma yang umum digunakan di industri
- Tahap Recall menggunakan pencarian efisien (FAISS/BM25) untuk memanggil kandidat dari sejumlah besar dokumen
- Tahap Rerank menggunakan model yang lebih presisi (Cross-Encoder/LLM) untuk mengurutkan kandidat secara halus
- Cross-Encoder lebih presisi daripada model Bi-Encoder (Dual-Tower), tetapi lebih lambat (cocok untuk pengurutan halus sejumlah kecil kandidat)
"""

import logging
import re
import threading
from functools import lru_cache
from config import OLLAMA_MODEL_NAME, RERANK_METHOD

# Cross-Encoder (Lazy Loading + Thread Safe)
_cross_encoder = None
_cross_encoder_lock = threading.Lock()


def get_cross_encoder():
    """Memuat model Cross-Encoder secara lazy (Double-Checked Locking, Thread Safe)"""
    global _cross_encoder
    if _cross_encoder is None:
        with _cross_encoder_lock:
            if _cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                    _cross_encoder = CrossEncoder(
                        'sentence-transformers/distiluse-base-multilingual-cased-v2'
                    )
                    logging.info("Cross-Encoder berhasil dimuat")
                except Exception as e:
                    logging.error(f"Gagal memuat Cross-Encoder: {str(e)}")
                    _cross_encoder = None
    return _cross_encoder


def rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k=5):
    """Menggunakan Cross-Encoder untuk mengurutkan ulang hasil pencarian"""
    if not docs:
        return []

    encoder = get_cross_encoder()
    if encoder is None:
        logging.warning("Cross-Encoder tidak tersedia, melewati pengurutan ulang")
        return _fallback_results(doc_ids, docs, metadata_list)

    cross_inputs = [[query, doc] for doc in docs]
    try:
        scores = encoder.predict(cross_inputs)
        results = [
            (doc_id, {'content': doc, 'metadata': meta, 'score': float(score)})
            for doc_id, doc, meta, score in zip(doc_ids, docs, metadata_list, scores)
        ]
        results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
        return results[:top_k]
    except Exception as e:
        logging.error(f"Gagal melakukan pengurutan ulang dengan Cross-Encoder: {str(e)}")
        return _fallback_results(doc_ids, docs, metadata_list)


@lru_cache(maxsize=32)
def get_llm_relevance_score(query, doc):
    """Menggunakan LLM untuk menilai relevansi antara kueri dan dokumen (dengan cache)"""
    from utils.network import get_session
    try:
        prompt = f"""Diberikan kueri dan fragmen dokumen berikut, evaluasi relevansi keduanya.
        Kriteria penilaian: Nilai 0 berarti sama sekali tidak relevan, nilai 10 berarti sangat relevan.
        Cukup kembalikan satu angka skor bulat antara 0-10, tanpa penjelasan lainnya.

        Kueri: {query}
        Fragmen Dokumen: {doc}
        Skor Relevansi (0-10):"""

        response = get_session().post(
            "http://localhost:11434/api/generate",
            json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
            timeout=180
        )
        result = response.json().get("response", "").strip()
        try:
            return max(0, min(10, float(result)))
        except ValueError:
            match = re.search(r'\b([0-9]|10)\b', result)
            return float(match.group(1)) if match else 5.0
    except Exception as e:
        logging.error(f"Penilaian relevansi LLM gagal: {str(e)}")
        return 5.0


def rerank_with_llm(query, docs, doc_ids, metadata_list, top_k=5):
    """Menggunakan LLM untuk mengurutkan ulang hasil pencarian dengan penilaian satu per satu"""
    if not docs:
        return []
    results = []
    for doc_id, doc, meta in zip(doc_ids, docs, metadata_list):
        score = get_llm_relevance_score(query, doc)
        results.append((doc_id, {'content': doc, 'metadata': meta, 'score': score / 10.0}))
    results = sorted(results, key=lambda x: x[1]['score'], reverse=True)
    return results[:top_k]


def rerank_results(query, docs, doc_ids, metadata_list, method=None, top_k=5):
    """Mengurutkan ulang hasil pencarian (Entri Tunjuk/Tunggal)"""
    if method is None:
        method = RERANK_METHOD

    if method == "llm":
        return rerank_with_llm(query, docs, doc_ids, metadata_list, top_k)
    elif method == "cross_encoder":
        return rerank_with_cross_encoder(query, docs, doc_ids, metadata_list, top_k)
    else:
        return _fallback_results(doc_ids, docs, metadata_list)


def _fallback_results(doc_ids, docs, metadata_list):
    """Rencana cadangan (Fallback): Mengembalikan sesuai dengan urutan asli"""
    return [(doc_id, {'content': doc, 'metadata': meta, 'score': 1.0 - idx / len(docs)})
            for idx, (doc_id, doc, meta) in enumerate(zip(doc_ids, docs, metadata_list))]
