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

# Deskripsi Pilihan Model:
# - all-MiniLM-L6-v2: Teroptimasi Bahasa Inggris, 384 dimensi, ringan dan cepat (default)
# - shibing624/text2vec-base-chinese: Teroptimasi Bahasa Mandarin
# - BAAI/bge-small-zh-v1.5: Teroptimasi Bahasa Mandarin, performa lebih baik
EMBED_MODEL_NAME = 'all-MiniLM-L6-v2'


@lru_cache(maxsize=1)
def get_embed_model():
    """
    Mendapatkan model vektorisasi (Singleton + Cache)

    Memuat model saat pertama kali dipanggil, panggilan berikutnya langsung mengembalikan instansi dari cache.
    """
    from sentence_transformers import SentenceTransformer
    logging.info(f"Memuat model vektorisasi: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)
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
    return np.array(embeddings).astype('float32')


def encode_query(query):
    """
    Mengodekan teks kueri tunggal menjadi vektor

    Returns:
        Array numpy, bentuk (1, embedding_dim)
    """
    model = get_embed_model()
    embedding = model.encode([query])
    return np.array(embedding).astype('float32')
