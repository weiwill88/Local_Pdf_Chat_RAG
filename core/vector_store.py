"""
Penyimpanan Vektor (Vector Store) —— Manajemen Indeks Vektor FAISS

Poin Pembelajaran:
- FAISS (Facebook AI Similarity Search) adalah pustaka pencarian kemiripan vektor yang efisien
- IndexFlatL2: Pencarian brute force, presisi tetapi lambat. Cocok untuk himpunan data kecil (<10 ribu)
- IndexIVFFlat: Indeks terbalik (Inverted File Index), melakukan pengelompokan (clustering) sebelum mencari. Cocok untuk himpunan data sedang
- IndexIVFPQ: Kuantisasi produk (Product Quantization), mengorbankan akurasi demi efisiensi. Cocok untuk himpunan data besar (>100 ribu)
- Proyek ini secara otomatis memilih tipe indeks optimal berdasarkan jumlah vektor
"""

import logging
import numpy as np
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ


class AutoFaissIndex:
    """
    Kelas pembungkus untuk pemilihan tipe indeks FAISS secara otomatis

    Secara otomatis memilih tipe indeks optimal berdasarkan jumlah data:
    - Himpunan data kecil (<10 ribu): FlatL2 (pencarian presisi)
    - Himpunan data sedang (10 ribu - 100 ribu): IVFFlat (pencarian perkiraan)
    - Himpunan data besar (>100 ribu): IVFPQ (pencarian perkiraan efisiensi tinggi)
    """

    def __init__(self, dimension=384):
        self.dimension = dimension
        self.index = None
        self.index_type = None
        self.nlist = None
        self.m = None
        self.nprobe = None
        self.small_dataset_threshold = 10_000
        self.medium_dataset_threshold = 100_000

    @property
    def ntotal(self):
        return self.index.ntotal if self.index else 0

    def select_index_type(self, num_vectors):
        """Secara otomatis memilih tipe indeks optimal berdasarkan jumlah vektor"""
        if num_vectors <= self.small_dataset_threshold:
            self.index_type = "FlatL2"
            self.index = IndexFlatL2(self.dimension)
            self.nprobe = 1
        elif num_vectors <= self.medium_dataset_threshold:
            self.index_type = "IVFFlat"
            self.nlist = min(100, int(np.sqrt(num_vectors)))
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFFlat(quantizer, self.dimension, self.nlist)
            self.nprobe = min(10, max(1, int(self.nlist * 0.1)))
        else:
            self.index_type = "IVFPQ"
            self.nlist = min(256, int(np.sqrt(num_vectors)))
            self.m = min(8, self.dimension // 4)
            quantizer = IndexFlatL2(self.dimension)
            self.index = IndexIVFPQ(quantizer, self.dimension, self.nlist, self.m, 8)
            self.nprobe = min(32, max(1, int(self.nlist * 0.05)))

        logging.info(f"Tipe indeks terpilih: {self.index_type}, Jumlah vektor: {num_vectors}")
        return self.index_type

    def train(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.train(vectors)

    def add(self, vectors):
        if self.index_type in ["IVFFlat", "IVFPQ"] and not self.index.is_trained:
            self.train(vectors)
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        if self.index_type in ["IVFFlat", "IVFPQ"]:
            self.index.nprobe = self.nprobe
        return self.index.search(query_vectors, k)

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    Manajer Penyimpanan Vektor

    Membungkus indeks FAISS beserta konten dokumen dan pemetaan metadata terkait.
    Menyelesaikan masalah pengelolaan 4 variabel global pada kode asli.
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        Membangun indeks FAISS

        Args:
            chunks: Daftar fragmen teks
            chunk_ids: Daftar ID fragmen
            metadatas: Daftar metadata
            embeddings: Array vektor (numpy, float32)
        """
        dimension = embeddings.shape[1]
        num_vectors = len(chunks)

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index
        logging.info(f"Pembangunan indeks FAISS selesai, total {self.index.ntotal} blok teks, Tipe: {auto_index.index_type}")

    def search(self, query_embedding, k=10):
        """
        Mencari vektor paling mirip

        Returns:
            (docs, doc_ids, metadatas)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
            return docs, doc_ids, metadatas
        except Exception as e:
            logging.error(f"Kesalahan pencarian FAISS: {str(e)}")
            return [], [], []

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        logging.info("Penyimpanan vektor telah dikosongkan")


# 模块级单例
vector_store = VectorStore()
