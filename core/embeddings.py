"""
Model Vektorisasi —— Memetakan teks ke ruang vektor berdimensi tinggi

Poin Pembelajaran:
- Embedding mengonversi teks menjadi vektor dengan dimensi tetap, membuat teks dengan makna semantik serupa lebih dekat di ruang vektor
- all-MiniLM-L6-v2 adalah model teroptimasi Bahasa Inggris (384 dimensi), Bahasa Mandarin dapat menggunakan text2vec-base-chinese
- Saat pertama kali dijalankan, model akan diunduh secara otomatis (sekitar 80MB), memerlukan koneksi internet
"""

import logging
import numpy as np
from functools import lru_cache
from config import EMBEDDING_PROVIDER, EMBEDDING_MODEL_NAME


def _normalize_embeddings(embeddings):
    array = np.asarray(embeddings, dtype='float32')
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return array / norms


class OllamaEmbedder:
    """
    Penyedia embedding menggunakan API lokal Ollama.
    """
    def __init__(self, model_name="nomic-embed-text", base_url="http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self._dimension = None

    def get_sentence_embedding_dimension(self) -> int:
        if self._dimension is None:
            try:
                # Mengambil dimensi secara dinamis dengan menguji teks pendek
                test_embed = self.encode(["test"])
                self._dimension = len(test_embed[0])
            except Exception as e:
                logging.error(f"Gagal mendeteksi dimensi embedding Ollama secara otomatis: {str(e)}")
                # Fallback ke dimensi standar jika nomic-embed-text
                if "nomic" in self.model_name:
                    self._dimension = 768
                else:
                    self._dimension = 384
        return self._dimension

    def encode(self, texts, show_progress_bar=False):
        from utils.network import get_session
        session = get_session()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        url = f"{self.base_url}/api/embed"
        
        try:
            # Batching request ke Ollama (default 100 per batch)
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                payload = {
                    "model": self.model_name,
                    "input": batch_texts
                }
                response = session.post(url, json=payload, timeout=120)
                
                # Fallback ke /api/embeddings jika menggunakan versi Ollama lama yang tidak memiliki /api/embed
                if response.status_code == 404:
                    logging.warning("/api/embed tidak ditemukan, mencoba fallback ke /api/embeddings")
                    for text in batch_texts:
                        fb_url = f"{self.base_url}/api/embeddings"
                        fb_payload = {
                            "model": self.model_name,
                            "prompt": text
                        }
                        fb_resp = session.post(fb_url, json=fb_payload, timeout=120)
                        fb_resp.raise_for_status()
                        embeddings.append(fb_resp.json()["embedding"])
                else:
                    response.raise_for_status()
                    res_json = response.json()
                    embeddings.extend(res_json["embeddings"])
        except Exception as e:
            logging.error(f"Terjadi kesalahan saat memanggil API embedding Ollama: {str(e)}")
            raise e

        return np.array(embeddings)


@lru_cache(maxsize=1)
def get_embed_model():
    """
    Mendapatkan model vektorisasi (Singleton + Cache)

    Memuat model saat pertama kali dipanggil, panggilan berikutnya langsung mengembalikan instansi dari cache.
    """
    if EMBEDDING_PROVIDER == "ollama":
        logging.info(f"Menggunakan penyedia embedding lokal Ollama dengan model: {EMBEDDING_MODEL_NAME}")
        model = OllamaEmbedder(model_name=EMBEDDING_MODEL_NAME)
    else:
        from sentence_transformers import SentenceTransformer
        logging.info(f"Memuat model vektorisasi offline (SentenceTransformer): {EMBEDDING_MODEL_NAME}")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    logging.info(f"Pemuatan model vektorisasi selesai, dimensi output: {model.get_sentence_embedding_dimension()}")
    return model



def encode_texts(texts, show_progress=False):
    """
    Mengodekan daftar teks menjadi vektor

    Args:
        texts: Daftar teks
        show_progress: Apakah menampilkan bilah kemajuan

    Returns:
        Array numpy, bentuk (n_texts, embedding_dim)
    """
    model = get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    return _normalize_embeddings(embeddings)


def encode_query(query):
    """
    Mengodekan teks kueri tunggal menjadi vektor

    Returns:
        Array numpy, bentuk (1, embedding_dim)
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return _normalize_embeddings(embedding)
