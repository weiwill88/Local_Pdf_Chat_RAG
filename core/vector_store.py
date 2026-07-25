"""
向量存储 —— FAISS 向量索引管理

学习要点：
- FAISS (Facebook AI Similarity Search) 是高效的向量相似度检索库
- IndexFlatL2: 暴力搜索，精确但慢。适合小数据集（<1万）
- IndexIVFFlat: 倒排索引，先聚类再搜索。适合中等数据集
- IndexIVFPQ: 乘积量化，牺牲精度换效率。适合大数据集（>10万）
- 本项目根据向量数量自动选择最优索引类型

Parent Document Retriever 支持：
- 子块（child chunks）存入向量库用于检索
- 父块（parent chunks）保留完整原文供 LLM 阅读
- 检索命中子块后向上映射回父块

文件管理支持：
- file_index: 记录每个文件对应的 chunk_id 列表
- file_hashes: 记录文件哈希值，用于上传去重
"""

import os
import json
import logging
import hashlib
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

    def get_embeddings_by_indices(self, indices):
        """
        从 FAISS 索引中恢复指定位置的向量（仅支持 FlatL2）
        用于 MMR 重排序时计算候选向量
        """
        if self.index_type != "FlatL2":
            logging.warning(f"MMR 在 {self.index_type} 索引上可能较慢，建议使用 FlatL2")
        try:
            # FAISS IndexFlatL2 支持 reconstruct
            vectors = []
            for idx in indices:
                if idx != -1:
                    vec = self.index.reconstruct(int(idx))
                    vectors.append(vec)
            return np.array(vectors).astype('float32') if vectors else np.array([])
        except Exception as e:
            logging.error(f"恢复向量失败: {str(e)}")
            return np.array([])


class VectorStore:
    """
    向量存储管理器

    封装 FAISS 索引及其关联的文档内容和元数据映射。
    支持 Parent Document Retriever 父子块映射。
    支持文件级索引管理。
    """

    def __init__(self):
        self.index = None           # AutoFaissIndex 实例
        self.contents_map = {}      # chunk_id -> 文本内容
        self.metadatas_map = {}     # chunk_id -> 元数据
        self.id_order = []          # 按顺序记录的 chunk_id 列表

        # Parent Document Retriever 支持
        self.parent_chunks_map = {}     # parent_id -> parent_text
        self.child_to_parent_map = {}   # child_id -> parent_id

        # 文件管理支持
        self.file_index = {}        # file_name -> [chunk_ids]
        self.file_hashes = {}       # file_hash -> file_name
        self.file_meta = {}         # file_name -> {upload_time, type, chunk_count}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 文件哈希与去重
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """计算文件的 SHA256 哈希值"""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logging.error(f"计算文件哈希失败: {str(e)}")
            return ""

    def is_duplicate_file(self, file_hash: str) -> bool:
        """检查文件哈希是否已存在"""
        return file_hash in self.file_hashes

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 文件管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def get_file_list(self):
        """获取当前知识库的文件列表"""
        files = []
        for file_name, chunk_ids in self.file_index.items():
            meta = self.file_meta.get(file_name, {})
            files.append({
                "name": file_name,
                "type": meta.get("type", "未知"),
                "upload_time": meta.get("upload_time", "未知"),
                "chunk_count": len(chunk_ids),
                "chunk_ids": chunk_ids,
            })
        return sorted(files, key=lambda f: f.get("upload_time", ""), reverse=True)

    def delete_file(self, file_name: str) -> bool:
        """
        从向量库中删除指定文件的所有分块
        返回 True 表示成功删除
        """
        if file_name not in self.file_index:
            logging.warning(f"文件不存在于索引中: {file_name}")
            return False

        chunk_ids_to_remove = set(self.file_index[file_name])
        if not chunk_ids_to_remove:
            del self.file_index[file_name]
            self.file_meta.pop(file_name, None)
            return True

        # 重建 id_order、contents_map、metadatas_map
        new_id_order = []
        for cid in self.id_order:
            if cid not in chunk_ids_to_remove:
                new_id_order.append(cid)

        for cid in chunk_ids_to_remove:
            self.contents_map.pop(cid, None)
            self.metadatas_map.pop(cid, None)

        self.id_order = new_id_order

        # 清理 parent 映射
        child_ids_to_remove = {cid for cid in chunk_ids_to_remove if cid in self.child_to_parent_map}
        parent_ids_to_remove = set()
        for cid in child_ids_to_remove:
            parent_id = self.child_to_parent_map.pop(cid, None)
            if parent_id:
                parent_ids_to_remove.add(parent_id)
        for pid in parent_ids_to_remove:
            self.parent_chunks_map.pop(pid, None)

        # 删除文件索引
        del self.file_index[file_name]
        self.file_meta.pop(file_name, None)

        # 同时删除对应的文件哈希
        file_hash_to_remove = None
        for f_hash, f_name in self.file_hashes.items():
            if f_name == file_name:
                file_hash_to_remove = f_hash
                break
        if file_hash_to_remove:
            del self.file_hashes[file_hash_to_remove]

        # 重建 FAISS 索引（如果有剩余向量）
        self._rebuild_faiss_index()

        logging.info(f"已从向量库删除文件: {file_name}，移除了 {len(chunk_ids_to_remove)} 个分块")
        return True

    def clear_all_files(self):
        """清空所有文件数据"""
        # 保存所有索引文件名用于日志
        file_count = len(self.file_index)
        self.parent_chunks_map.clear()
        self.child_to_parent_map.clear()
        self.file_index.clear()
        self.file_hashes.clear()
        self.file_meta.clear()
        self.clear()
        logging.info(f"已清空所有文件数据（{file_count} 个文件）")

    def _rebuild_faiss_index(self):
        """根据当前的 id_order 和 contents_map 重建 FAISS 索引"""
        if not self.id_order:
            self.index = None
            logging.info("向量库为空，跳过 FAISS 重建")
            return

        # 收集剩余块的文本和元数据
        remaining_chunks = []
        remaining_ids = []
        remaining_metas = []
        for cid in self.id_order:
            if cid in self.contents_map:
                remaining_chunks.append(self.contents_map[cid])
                remaining_ids.append(cid)
                remaining_metas.append(self.metadatas_map.get(cid, {}))

        if not remaining_chunks:
            self.index = None
            return

        # 重新编码
        try:
            from core.embeddings import encode_texts
            embeddings = encode_texts(remaining_chunks, show_progress=False)
        except Exception as e:
            logging.error(f"重建 FAISS 时编码失败: {str(e)}")
            self.index = None
            return

        # 重建索引
        dimension = embeddings.shape[1]
        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(len(remaining_chunks))
        auto_index.add(embeddings)
        self.index = auto_index

        # 更新 id_order
        self.id_order = remaining_ids
        # 重建 contents_map / metadatas_map (清理孤儿键)
        new_contents = {}
        new_metas = {}
        for cid, chunk, meta in zip(remaining_ids, remaining_chunks, remaining_metas):
            new_contents[cid] = chunk
            new_metas[cid] = meta
        self.contents_map = new_contents
        self.metadatas_map = new_metas

        logging.info(f"FAISS 索引重建完成，共 {len(self.id_order)} 个文本块")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 索引构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

        # 清空并重建
        self.clear()

        auto_index = AutoFaissIndex(dimension=dimension)
        auto_index.select_index_type(num_vectors)

        for chunk_id, chunk, meta in zip(chunk_ids, chunks, metadatas):
            self.contents_map[chunk_id] = chunk
            self.metadatas_map[chunk_id] = meta
            self.id_order.append(chunk_id)

        auto_index.add(embeddings)
        self.index = auto_index

        logging.info(
            f"FAISS 索引构建完成，共 {self.index.ntotal} 个文本块，"
            f"类型: {auto_index.index_type}"
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 检索
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    def search_with_scores(self, query_embedding, k=10):
        """
        搜索最相似的向量并返回实际相似度分数（用于自动触发联网搜索的判断）

        将 FAISS L2 距离转换为 0~1 相似度分数。
        对于 L2 归一化向量：cos_sim = 1 - L2²/2

        Returns:
            (docs, doc_ids, metadatas, scores)
            scores: [0~1] 相似度列表，值越高越相关
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], [], []
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas, scores = [], [], [], []
            for faiss_idx, l2_dist in zip(I[0], D[0]):
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
                        # L2 → 余弦相似度（假设向量已 L2 归一化）
                        sim = max(0.0, 1.0 - (l2_dist * l2_dist) / 2.0)
                        scores.append(round(sim, 4))
            return docs, doc_ids, metadatas, scores
        except Exception as e:
            logging.error(f"FAISS 检索错误(带分数): {str(e)}")
            return [], [], [], []

    def search_with_embeddings(self, query_embedding, k=10):
        """
        搜索最相似的向量并返回向量值（用于 MMR 重排序）

        Returns:
            (docs, doc_ids, metadatas, doc_embeddings)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], [], np.array([])
        try:
            D, I = self.index.search(query_embedding, k=k)
            docs, doc_ids, metadatas = [], [], []
            valid_indices = []
            for faiss_idx in I[0]:
                if faiss_idx != -1 and faiss_idx < len(self.id_order):
                    original_id = self.id_order[faiss_idx]
                    if original_id in self.contents_map:
                        docs.append(self.contents_map[original_id])
                        doc_ids.append(original_id)
                        metadatas.append(self.metadatas_map.get(original_id, {}))
                        valid_indices.append(int(faiss_idx))

            # 获取向量用于 MMR
            doc_embeddings = np.array([])
            if valid_indices:
                doc_embeddings = self.index.get_embeddings_by_indices(valid_indices)

            return docs, doc_ids, metadatas, doc_embeddings
        except Exception as e:
            logging.error(f"FAISS 检索错误(带向量): {str(e)}")
            return [], [], [], np.array([])

    def search_with_parent(self, query_embedding, k=10):
        """
        搜索子块并向上映射回父块

        1. 向量检索子块
        2. 将命中的子块按父块去重归并
        3. 返回父块的完整内容

        Returns:
            (parent_docs, parent_ids, parent_metadatas, child_info)
            child_info: [{"child_id": ..., "parent_id": ..., "score": ...}]
        """
        docs, doc_ids, metadatas = self.search(query_embedding, k=k)
        if not docs or not self.child_to_parent_map:
            return docs, doc_ids, metadatas, []

        # 子块→父块映射
        parent_set = {}  # parent_id -> (parent_text, [child_ids])
        child_info = []

        for doc, cid, meta in zip(docs, doc_ids, metadatas):
            parent_id = self.child_to_parent_map.get(cid)
            if parent_id and parent_id in self.parent_chunks_map:
                parent_text = self.parent_chunks_map[parent_id]
                if parent_id not in parent_set:
                    parent_set[parent_id] = {
                        "text": parent_text,
                        "child_ids": [],
                        "metadata": meta.copy(),
                    }
                parent_set[parent_id]["child_ids"].append(cid)
                child_info.append({
                    "child_id": cid,
                    "parent_id": parent_id,
                    "content": doc[:100] + "..." if len(doc) > 100 else doc,
                })
            else:
                # 没有父块映射的块，直接返回
                parent_id = cid
                if parent_id not in parent_set:
                    parent_set[parent_id] = {
                        "text": doc,
                        "child_ids": [cid],
                        "metadata": meta.copy(),
                    }

        # 排序并返回父块结果
        parent_docs = []
        parent_ids = []
        parent_metadatas = []
        for pid, info in parent_set.items():
            parent_docs.append(info["text"])
            parent_ids.append(pid)
            meta = info["metadata"]
            meta["child_count"] = len(info["child_ids"])
            parent_metadatas.append(meta)

        return parent_docs, parent_ids, parent_metadatas, child_info

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 状态
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @property
    def is_ready(self):
        return self.index is not None and self.index.ntotal > 0

    @property
    def total_chunks(self):
        return self.index.ntotal if self.index is not None else 0

    def clear(self):
        """清空所有数据"""
        self.index = None
        self.contents_map.clear()
        self.metadatas_map.clear()
        self.id_order.clear()
        self.parent_chunks_map.clear()
        self.child_to_parent_map.clear()
        self.file_index.clear()
        self.file_hashes.clear()
        self.file_meta.clear()
        logging.info("向量存储已清空")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 持久化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

        # 保存 Parent 映射
        with open(os.path.join(directory, "parent_chunks_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.parent_chunks_map, f, ensure_ascii=False)
        with open(os.path.join(directory, "child_to_parent_map.json"), "w", encoding="utf-8") as f:
            json.dump(self.child_to_parent_map, f, ensure_ascii=False)

        # 保存文件索引
        with open(os.path.join(directory, "file_index.json"), "w", encoding="utf-8") as f:
            json.dump(self.file_index, f, ensure_ascii=False)
        with open(os.path.join(directory, "file_hashes.json"), "w", encoding="utf-8") as f:
            json.dump(self.file_hashes, f, ensure_ascii=False)
        with open(os.path.join(directory, "file_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.file_meta, f, ensure_ascii=False)

        logging.info(
            f"向量存储已保存到 {directory} "
            f"（{len(self.id_order)} 个子块, {len(self.parent_chunks_map)} 个父块, "
            f"{len(self.file_index)} 个文件）"
        )

    def load(self, directory):
        """从磁盘加载向量存储"""
        self.clear()
        # 加载 FAISS 索引
        index_path = os.path.join(directory, "index.faiss")
        if os.path.exists(index_path):
            auto_index = AutoFaissIndex.load(directory)
            if auto_index is not None:
                self.index = auto_index

        def _load_json(filename):
            path = os.path.join(directory, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}

        # 加载基础映射
        self.contents_map = _load_json("contents_map.json")
        self.metadatas_map = _load_json("metadatas_map.json")
        self.id_order = _load_json("id_order.json")

        # 加载 Parent 映射
        self.parent_chunks_map = _load_json("parent_chunks_map.json")
        self.child_to_parent_map = _load_json("child_to_parent_map.json")

        # 加载文件索引
        self.file_index = _load_json("file_index.json")
        self.file_hashes = _load_json("file_hashes.json")
        self.file_meta = _load_json("file_meta.json")

        logging.info(
            f"向量存储已从 {directory} 加载"
            f"（{len(self.id_order)} 个子块, {len(self.parent_chunks_map)} 个父块, "
            f"{len(self.file_index)} 个文件）"
        )
        return self.is_ready


# 模块级单例
vector_store = VectorStore()
