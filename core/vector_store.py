"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型
"""

import os
import json
import logging
import numpy as np
from faiss import IndexFlatL2, IndexIVFFlat, IndexIVFPQ, serialize_index, deserialize_index


class AutoFaissIndex:
    """
    自动选择 FAISS 索引类型的封装类

    根据数据量自动选择最优索引类型：
    - 小数据集（<1万）: FlatL2（精确搜索）
    - 中等数据集（1万-10万）: IVFFlat（近似搜索）
    - 大数据集（>10万）: IVFPQ（高效近似搜索）
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
        """根据向量数量自动选择最优索引类型"""
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

        logging.info(f"选择索引类型: {self.index_type}，向量数: {num_vectors}")
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

    def save(self, directory):
        """将 FAISS 索引保存到磁盘（使用序列化，支持中文路径）"""
        os.makedirs(directory, exist_ok=True)
        index_path = os.path.join(directory, "index.faiss")
        # 使用 serialize_index 避免 FAISS C++ 层中文路径问题
        idx_bytes = serialize_index(self.index)
        # serialize_index 返回 numpy.ndarray，转为 bytes 写入
        if hasattr(idx_bytes, 'tobytes'):
            idx_bytes = idx_bytes.tobytes()
        with open(index_path, "wb") as f:
            f.write(idx_bytes)
        meta = {
            "dimension": self.dimension,
            "index_type": self.index_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "m": self.m,
        }
        with open(os.path.join(directory, "index_meta.json"), "w") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, directory):
        """从磁盘加载 FAISS 索引"""
        index_path = os.path.join(directory, "index.faiss")
        meta_path = os.path.join(directory, "index_meta.json")
        if not os.path.exists(index_path):
            return None
        with open(index_path, "rb") as f:
            raw = f.read()
        # deserialize_index 需要 numpy 数组
        index = deserialize_index(np.frombuffer(raw, dtype='uint8'))
        auto_index = cls(dimension=384)
        auto_index.index = index
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            auto_index.index_type = meta.get("index_type")
            auto_index.nlist = meta.get("nlist")
            auto_index.nprobe = meta.get("nprobe", 1)
            auto_index.m = meta.get("m")
        else:
            auto_index.index_type = "FlatL2"
            auto_index.nprobe = 1
        return auto_index

    def get_index_info(self):
        return {
            "index_type": self.index_type, "dimension": self.dimension,
            "nlist": self.nlist, "nprobe": self.nprobe, "size": self.ntotal
        }


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    解决原代码中 4 个全局变量的管理问题。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表

    def build_index(self, chunks, chunk_ids, metadatas, embeddings):
        """
        构建 FAISS 索引

        Args:
            chunks: 文本片段列表
            chunk_ids: 片段 ID 列表
            metadatas: 元数据列表
            embeddings: 向量数组 (numpy, float32)
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
        logging.info(f"FAISS 索引构建完成，共 {self.index.ntotal} 个文本块，类型: {auto_index.index_type}")

    def search(self, query_embedding, k=10):
        """
        搜索最相似的向量

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
            logging.error(f"FAISS 检索错误: {str(e)}")
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
        logging.info("向量存储已清空")

    def save(self, directory):
        """将整个向量存储保存到磁盘"""
        os.makedirs(directory, exist_ok=True)
        # 保存 FAISS 索引
        if self.index is not None and self.index.ntotal > 0:
            self.index.save(directory)
        # 保存文本内容和元数据
        with open(os.path.join(directory, "contents_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.contents_map, f, ensure_ascii=False)
        with open(os.path.join(directory, "metadatas_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadatas_map, f, ensure_ascii=False)
        with open(os.path.join(directory, "id_order.json"), "w", encoding="utf-8") as f:
            json.dump(self.id_order, f, ensure_ascii=False)
        logging.info(f"向量存储已保存到 {directory}（{len(self.id_order)} 个文本块）")

    def load(self, directory):
        """从磁盘加载向量存储"""
        self.clear()
        # 加载 FAISS 索引
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            auto_index = AutoFaissIndex.load(directory)
            if auto_index is not None:
                self.index = auto_index
        # 加载文本内容和元数据
        cm_path = os.path.join(directory, "contents_map.json")
        mm_path = os.path.join(directory, "metadatas_map.json")
        io_path = os.path.join(directory, "id_order.json")
        if os.path.exists(cm_path):
            with open(cm_path, "r", encoding="utf-8") as f:
                self.contents_map = json.load(f)
        if os.path.exists(mm_path):
            with open(mm_path, "r", encoding="utf-8") as f:
                self.metadatas_map = json.load(f)
        if os.path.exists(io_path):
            with open(io_path, "r", encoding="utf-8") as f:
                self.id_order = json.load(f)
        logging.info(f"向量存储已从 {directory} 加载（{len(self.id_order)} 个文本块）")
        return self.is_ready


# 模块级单例
vector_store = VectorStore()
