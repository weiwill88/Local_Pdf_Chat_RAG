"""
父文档检索器 + MMR 重排序

Parent Document Retriever：
- 文档做双层分割（大块父文档 + 小块子文档）
- 子块存入向量库用于相似度检索
- 命中子块后向上映射回完整父块作为 LLM 上下文
- 解决切块导致的上下文断裂、信息缺失问题

MMR (Max Marginal Relevance)：
- 平衡相关性与多样性 λ * Sim(q, d_i) - (1-λ) * max Sim(d_i, d_j)
- λ=1 纯相关、λ=0 纯多样
- 过滤高度重复语义块
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

from config import MMR_LAMBDA, MMR_TOP_K
from core.vector_store import vector_store
from core.embeddings import encode_query


def mmr_rerank(query_embedding: np.ndarray,
               candidate_embeddings: np.ndarray,
               candidate_texts: List[str],
               candidate_ids: List[str],
               lambda_val: float = None,
               top_k: int = None) -> List[Tuple[str, str, float]]:
    """
    MMR 最大边际相关性重排序

    算法：MMR = argmax[λ * Sim(q, d_i) - (1-λ) * max_{d_j in S} Sim(d_i, d_j)]
    - λ 接近 1：倾向于相关性强但与已选结果相似度高的结果（高精度）
    - λ 接近 0：倾向于多样性好、与已选结果差异大的结果（高多样性）

    Args:
        query_embedding: 查询向量，shape (1, dim)
        candidate_embeddings: 候选向量，shape (N, dim)
        candidate_texts: 候选文本列表
        candidate_ids: 候选 ID 列表
        lambda_val: 多样性参数，1=纯相关，0=纯多样
        top_k: 返回数量

    Returns:
        [(id, text, score)] 按 MMR 分数降序排列
    """
    if lambda_val is None:
        lambda_val = MMR_LAMBDA
    if top_k is None:
        top_k = MMR_TOP_K

    if len(candidate_texts) == 0:
        return []

    if len(candidate_texts) <= top_k:
        # 候选不够多，直接返回原序
        return list(zip(candidate_ids, candidate_texts,
                        [1.0 - i / len(candidate_texts) for i in range(len(candidate_texts))]))

    n = len(candidate_embeddings)
    if n == 0:
        return list(zip(candidate_ids, candidate_texts, [0.0] * len(candidate_texts)))

    # 确保向量是二维的
    if candidate_embeddings.ndim == 1:
        candidate_embeddings = candidate_embeddings.reshape(1, -1)

    # 计算查询与每个候选的余弦相似度
    query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)
    cand_norm = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-10)

    sim_q = np.dot(cand_norm, query_norm.T).flatten()  # 每个候选与查询的相似度
    sim_matrix = np.dot(cand_norm, cand_norm.T)  # 候选间余弦相似度矩阵

    # MMR 贪心选择
    selected = []
    remaining = list(range(n))

    for _ in range(min(top_k, n)):
        if not remaining:
            break

        mmr_scores = []
        for idx in remaining:
            relevance = lambda_val * sim_q[idx]
            if selected:
                diversity = (1 - lambda_val) * np.max(sim_matrix[idx, selected])
            else:
                diversity = 0
            mmr_scores.append(relevance - diversity)

        best_idx = remaining[np.argmax(mmr_scores)]
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [
        (candidate_ids[i], candidate_texts[i], float(sim_q[i]))
        for i in selected
    ]


def parent_retrieve(query: str, k: int = 10, use_mmr: bool = True,
                    mmr_lambda: float = None, mmr_top_k: int = None):
    """
    父文档检索入口

    流程：
    1. 向量检索子块（带 embedding 返回）
    2. 可选 MMR 重排序
    3. 子块→父块映射
    4. 返回父块完整内容

    Args:
        query: 查询文本
        k: 初始检索数量
        use_mmr: 是否启用 MMR 重排序
        mmr_lambda: MMR 多样性参数
        mmr_top_k: MMR 保留候选数

    Returns:
        (contexts, doc_ids, metadatas, sources_info)
    """
    if not vector_store.is_ready:
        return [], [], [], []

    query_embedding = encode_query(query)

    # 步骤 1: 检索子块（带向量）
    docs, doc_ids, metadatas, doc_embeddings = vector_store.search_with_embeddings(
        query_embedding, k=k
    )

    if not docs:
        return [], [], [], []

    # 步骤 2: MMR 重排序（可选）
    if use_mmr and len(doc_embeddings) > 0:
        mmr_results = mmr_rerank(
            query_embedding, doc_embeddings, docs, doc_ids,
            lambda_val=mmr_lambda, top_k=mmr_top_k or k
        )
        if mmr_results:
            doc_ids = [r[0] for r in mmr_results]
            docs = [r[1] for r in mmr_results]
            # 重建 metadatas 顺序
            metadatas = [vector_store.metadatas_map.get(cid, {}) for cid in doc_ids]

    # 步骤 3: 子块→父块映射
    parent_docs, parent_ids, parent_metadatas, child_info = \
        vector_store.search_with_parent(query_embedding, k=k)

    # 如果父文档映射启用且有结果，使用父块内容
    contexts = parent_docs if parent_docs else docs
    final_ids = parent_ids if parent_ids else doc_ids
    final_metas = parent_metadatas if parent_metadatas else metadatas

    # 构建来源信息
    sources_info = []
    for cid, doc in zip(final_ids, contexts):
        meta = vector_store.metadatas_map.get(cid, {}) if cid in vector_store.metadatas_map else {}
        sources_info.append({
            "source": meta.get("source", "未知"),
            "chunk_id": cid,
            "preview": doc[:200] + "..." if len(doc) > 200 else doc,
        })

    return contexts, final_ids, final_metas, sources_info
