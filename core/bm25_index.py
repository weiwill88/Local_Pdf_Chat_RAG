"""
BM25 稀疏检索索引 —— 基于关键词的传统检索

学习要点：
- BM25 (Best Matching 25) 是经典的信息检索算法
- 与向量语义检索互补：语义检索擅长理解意图，BM25 擅长精确关键词匹配
- 中文需要先分词（jieba），英文可直接按空格分
- 两者混合使用（Hybrid Search）可以显著提升检索效果
"""

import os
import json
import logging
import numpy as np
import jieba
from rank_bm25 import BM25Okapi


class BM25IndexManager:
    """
    BM25 检索索引管理器

    负责构建、搜索和管理 BM25 索引。
    使用 jieba 分词以支持中文检索。
    """

    def __init__(self):
        self.bm25_index = None
        self.doc_mapping = {}
        self.tokenized_corpus = []
        self.raw_corpus = []

    def build_index(self, documents, doc_ids):
        """构建 BM25 索引"""
        self.raw_corpus = documents
        self.doc_mapping = {i: doc_id for i, doc_id in enumerate(doc_ids)}
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logging.info(f"BM25 索引构建完成，共索引 {len(documents)} 个文档")
        return True

    def search(self, query, top_k=5):
        """使用 BM25 检索相关文档"""
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

    def save(self, directory):
        """将 BM25 索引保存到磁盘"""
        os.makedirs(directory, exist_ok=True)
        # doc_mapping 的 key 是 int，转为 str 以支持 JSON 序列化
        str_mapping = {str(k): v for k, v in self.doc_mapping.items()}
        with open(os.path.join(directory, "bm25_doc_mapping.json"), "w", encoding="utf-8") as f:
            json.dump(str_mapping, f, ensure_ascii=False)
        with open(os.path.join(directory, "bm25_raw_corpus.json"), "w", encoding="utf-8") as f:
            json.dump(self.raw_corpus, f, ensure_ascii=False)
        with open(os.path.join(directory, "bm25_tokenized_corpus.json"), "w", encoding="utf-8") as f:
            json.dump(self.tokenized_corpus, f, ensure_ascii=False)

    def load(self, directory):
        """从磁盘加载 BM25 索引"""
        paths = [
            os.path.join(directory, "bm25_doc_mapping.json"),
            os.path.join(directory, "bm25_raw_corpus.json"),
            os.path.join(directory, "bm25_tokenized_corpus.json"),
        ]
        if not all(os.path.exists(p) for p in paths):
            return False
        with open(paths[0], "r", encoding="utf-8") as f:
            str_mapping = json.load(f)
            self.doc_mapping = {int(k): v for k, v in str_mapping.items()}
        with open(paths[1], "r", encoding="utf-8") as f:
            self.raw_corpus = json.load(f)
        with open(paths[2], "r", encoding="utf-8") as f:
            self.tokenized_corpus = json.load(f)
        self.bm25_index = BM25Okapi(self.tokenized_corpus)
        logging.info(f"BM25 索引已加载（{len(self.raw_corpus)} 个文档）")
        return True


# 模块级单例
bm25_manager = BM25IndexManager()
