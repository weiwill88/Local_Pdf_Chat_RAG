"""
Indeks Pencarian Jarang BM25 —— Pencarian tradisional berbasis kata kunci

Poin Pembelajaran:
- BM25 (Best Matching 25) adalah algoritma temu kembali informasi klasik
- Saling melengkapi dengan pencarian semantik vektor: pencarian semantik ahli dalam memahami niat/intent, BM25 ahli dalam pencocokan kata kunci yang tepat
- Bahasa Mandarin perlu disegmentasi terlebih dahulu (jieba), Bahasa Inggris dapat langsung dipisahkan dengan spasi
- Penggunaan hibrida dari keduanya (Hybrid Search) dapat secara signifikan meningkatkan efek pencarian
"""

import logging
import numpy as np
import jieba
from rank_bm25 import BM25Okapi


class BM25IndexManager:
    """
    Manajer Indeks Pencarian BM25

    Bertanggung jawab untuk membangun, mencari, dan mengelola indeks BM25.
    Menggunakan tokenisasi jieba untuk mendukung pencarian bahasa Mandarin.
    """

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        """Membangun indeks BM25"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logging.info(f"Pembangunan indeks BM25 selesai, total mengindeks {len(documents)} dokumen")
        return True

    def search(self, query, top_k=5):
        """Menggunakan BM25 untuk mencari dokumen relevan"""
        if not self.bm25_index:
            return []

        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:
                results.append({
                    'id': self.doc_mapping[idx],
                    'score': float(bm25_scores[idx]),
                    'content': self.raw_corpus[idx]
                })
        return results

    def clear(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []


# 模块级单例
bm25_manager = BM25IndexManager()
