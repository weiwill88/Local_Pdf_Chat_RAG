"""
检索器 —— 混合检索 + 递归检索策略 + Parent Document Retriever

学习要点：
- 混合检索（Hybrid Search）结合语义检索和关键词检索的优势
- alpha 参数控制两者权重（0 = 纯向量，1 = 纯 BM25）
- Parent Document Retriever：子块检索 → 父块上下文
- MMR 最大边际相关性：平衡相关性与多样性

Parent Retriever 数据流：
  查询 → 向量+BM25检索子块 → 混合排序 → 子块→父块映射 → MMR重排 → 重排序 → LLM
"""

import logging
from config import (
    HYBRID_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_K, MAX_RETRIEVAL_ITERATIONS,
    WEB_SEARCH_AUTO_FALLBACK, LOCAL_SCORE_THRESHOLD, WEB_SEARCH_MAX_RESULTS
)
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.embeddings import encode_query, encode_texts
from core.reranker import rerank_results
from core.parent_retriever import parent_retrieve, mmr_rerank
from features.web_search import check_serpapi_key, search_web


def hybrid_merge(semantic_results, bm25_results, alpha=None):
    """
    合并语义检索和 BM25 检索结果

    使用加权分数：语义分数 × alpha + BM25分数 × (1-alpha)

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: 语义检索权重，0=纯BM25，1=纯向量

    Returns:
        排序后的 [(doc_id, {'score': ..., 'content': ..., 'metadata': ...})]
    """
    if alpha is None:
        alpha = HYBRID_ALPHA

    merged_dict = {}

    # 处理语义检索结果
    if (semantic_results and
            isinstance(semantic_results.get('documents'), list) and len(semantic_results['documents']) > 0 and
            isinstance(semantic_results.get('metadatas'), list) and len(semantic_results['metadatas']) > 0 and
            isinstance(semantic_results.get('ids'), list) and len(semantic_results['ids']) > 0 and
            isinstance(semantic_results['documents'][0], list) and
            len(semantic_results['documents'][0]) == len(semantic_results['metadatas'][0]) == len(
                semantic_results['ids'][0])):
        num_results = len(semantic_results['documents'][0])
        for i, (doc_id, doc, meta) in enumerate(
                zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            score = 1.0 - (i / max(1, num_results))
            merged_dict[doc_id] = {'score': alpha * score, 'content': doc, 'metadata': meta}
    else:
        logging.warning("语义检索结果为空或格式异常")

    # 处理 BM25 结果
    if not bm25_results:
        return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)

    valid_scores = [r['score'] for r in bm25_results if isinstance(r, dict) and 'score' in r]
    max_bm25 = max(valid_scores) if valid_scores else 1.0

    for result in bm25_results:
        if not (isinstance(result, dict) and 'id' in result and 'score' in result and 'content' in result):
            continue
        doc_id = result['id']
        norm_score = result['score'] / max_bm25 if max_bm25 > 0 else 0

        if doc_id in merged_dict:
            merged_dict[doc_id]['score'] += (1 - alpha) * norm_score
        else:
            metadata = vector_store.metadatas_map.get(doc_id, {})
            merged_dict[doc_id] = {
                'score': (1 - alpha) * norm_score,
                'content': result['content'], 'metadata': metadata
            }

    return sorted(merged_dict.items(), key=lambda x: x[1]['score'], reverse=True)


def map_child_to_parent(hybrid_results):
    """
    将子块检索结果映射回父块

    子块去重归并：同一父块的多个子块只保留最高分

    Args:
        hybrid_results: [(child_id, {'score': ..., 'content': ..., 'metadata': ...})]

    Returns:
        [(parent_or_child_id, {'score': ..., 'content': ..., 'metadata': ...,
                               'child_ids': [...]})]
        如果存在父块映射，content 为父块完整内容
    """
    if not vector_store.child_to_parent_map:
        # 没有父块映射，直接返回原始结果
        return hybrid_results

    parent_groups = {}
    for doc_id, data in hybrid_results:
        parent_id = vector_store.child_to_parent_map.get(doc_id)

        if parent_id and parent_id in vector_store.parent_chunks_map:
            parent_content = vector_store.parent_chunks_map[parent_id]
            if parent_id not in parent_groups:
                parent_groups[parent_id] = {
                    'score': data['score'],
                    'content': parent_content,
                    'metadata': data['metadata'].copy(),
                    'child_ids': [doc_id],
                }
            else:
                parent_groups[parent_id]['score'] = max(
                    parent_groups[parent_id]['score'], data['score']
                )
                parent_groups[parent_id]['child_ids'].append(doc_id)
        else:
            # 没有父块的子块，直接作为独立结果
            if doc_id not in parent_groups:
                parent_groups[doc_id] = {
                    'score': data['score'],
                    'content': data['content'],
                    'metadata': data['metadata'],
                    'child_ids': [doc_id],
                }

    sorted_parents = sorted(
        parent_groups.items(), key=lambda x: x[1]['score'], reverse=True
    )
    return sorted_parents


def recursive_retrieval(initial_query, max_iterations=None, enable_web_search=False,
                        model_choice="siliconflow", alpha=None, use_parent_retriever=True,
                        use_mmr=True, mmr_lambda=None,
                        ollama_model_name=None, ollama_num_ctx=None,
                        ollama_temperature=None, ollama_top_p=None):
    """
    递归检索与查询优化 + 联网搜索兜底（严格按规则触发）

    执行流程：
      第 1 轮：完整本地 RAG（向量+BM25 → 混合 → 父文档映射 → MMR → 重排序）
      → 判定是否触发联网搜索（二选一）
        ① 本地召回文本块数 == 0
        ② 最高余弦相似度 < 0.3
      → 若触发：用用户原始提问调 SerpAPI，结果标记为【网络检索参考】
      → 若失败：友好提示「联网搜索失败，仅基于本地文档作答」
      第 2+ 轮：递归优化查询（仅本地 RAG，不再重复联网）

    Args:
        initial_query: 初始查询（用户原始提问）
        max_iterations: 递归迭代次数
        enable_web_search: 是否启用联网搜索
        model_choice: 模型选择
        alpha: 混合检索权重（0=纯BM25, 1=纯向量）
        use_parent_retriever: 是否启用父文档检索
        use_mmr: 是否启用 MMR 重排序
        mmr_lambda: MMR 多样性参数

    Returns:
        (all_contexts, all_doc_ids, all_metadata, sources_info)
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    query = initial_query
    all_contexts, all_doc_ids, all_metadata, all_sources_info = [], [], [], []

    # ━━━ 联网搜索状态跟踪 ━━━
    web_search_triggered = False       # 是否已触发过联网搜索
    web_search_failed_msg = None       # 联网搜索失败时的错误消息（非 None 表示失败）
    orig_query = initial_query         # 保留用户原始提问，用于联网搜索

    for i in range(max_iterations):
        logging.info(f"递归检索第 {i + 1}/{max_iterations} 轮，Query: {query}")

        # ━━━ 1. 语义检索（带真实相似度分数） ━━━
        query_embedding = encode_query(query)
        sem_docs, sem_ids, sem_metas, sem_scores = vector_store.search_with_scores(
            query_embedding, k=RETRIEVAL_TOP_K
        )
        # 记录第一轮的原始相似度，用于后续触发判断
        if i == 0:
            first_round_scores = list(sem_scores)
            first_round_count = len(sem_docs)

        # ━━━ 2. BM25 检索 ━━━
        bm25_res = bm25_manager.search(query, top_k=RETRIEVAL_TOP_K) if bm25_manager.bm25_index else []

        # ━━━ 3. 混合排序 ━━━
        prepared = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}
        hybrid = hybrid_merge(prepared, bm25_res, alpha=alpha)

        # ━━━ 4. 父文档映射（Parent Retriever） ━━━
        if use_parent_retriever:
            hybrid = map_child_to_parent(hybrid)

        # ━━━ 5. 准备重排序输入 ━━━
        ids_iter, docs_iter, meta_iter = [], [], []
        for doc_id, data in hybrid[:RETRIEVAL_TOP_K]:
            ids_iter.append(doc_id)
            docs_iter.append(data['content'])
            meta_iter.append(data['metadata'])

        # ━━━ 6. MMR 多样性重排序 ━━━
        if use_mmr and docs_iter and query_embedding is not None:
            try:
                cand_embeddings = encode_texts(docs_iter)
                mmr_results = mmr_rerank(
                    query_embedding, cand_embeddings, docs_iter, ids_iter,
                    lambda_val=mmr_lambda, top_k=len(docs_iter)
                )
                if mmr_results:
                    ids_iter = [r[0] for r in mmr_results]
                    docs_iter = [r[1] for r in mmr_results]
                    meta_iter = [vector_store.metadatas_map.get(pid, {}) for pid in ids_iter]
            except Exception as e:
                logging.error(f"MMR 重排序失败: {str(e)}")

        # ━━━ 7. Cross-Encoder 重排序 ━━━
        if docs_iter:
            try:
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
            except Exception as e:
                logging.error(f"重排序失败: {str(e)}")
                reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                            for did, d, m in zip(ids_iter, docs_iter, meta_iter)]
        else:
            reranked = []

        # ━━━ 8. 收集本地召回结果 ━━━
        current_contexts = []
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                all_contexts.append(data['content'])
                metadata_with_type = dict(data['metadata'])
                metadata_with_type['source_type'] = 'local'
                all_metadata.append(metadata_with_type)
                source_name = data['metadata'].get('source', '未知来源')
                all_sources_info.append({
                    "source": source_name,
                    "chunk_id": doc_id,
                    "preview": data['content'][:200] + "..." if len(data['content']) > 200 else data['content'],
                    "source_type": "local",
                })

        # ━━━ 9. 联网搜索兜底（仅第 1 轮、未触发过、勾选了且配置了 Key） ━━━
        if i == 0 and enable_web_search and check_serpapi_key() and not web_search_triggered:
            max_sim = max(first_round_scores) if first_round_scores else 0.0
            # 触发条件：① 本地召回 = 0  ② 最高相似度 < 0.3
            trigger_web = (first_round_count == 0) or (max_sim < LOCAL_SCORE_THRESHOLD)
            logging.info(
                f"联网搜索判定: 本地召回{first_round_count}条, 最高相似度{max_sim:.4f}, "
                f"阈值{LOCAL_SCORE_THRESHOLD}, 触发={'是' if trigger_web else '否'}"
            )
            if trigger_web:
                try:
                    logging.info(f"→ 触发联网搜索，使用原始提问: {orig_query}")
                    web_results = search_web(orig_query, num_results=WEB_SEARCH_MAX_RESULTS)
                    logging.info(f"→ SerpAPI 返回 {len(web_results)} 条结果")

                    web_texts = []
                    search_has_error = False
                    for res in web_results:
                        # 搜索自身错误（密钥无效、超时、额度用尽）
                        if res.get("error"):
                            web_search_failed_msg = res.get("message", "联网搜索失败")
                            logging.warning(f"→ 联网搜索报错: {web_search_failed_msg}")
                            search_has_error = True
                            break
                        snippet = res.get('snippet', '')
                        title = res.get('title', '')
                        url = res.get('url', '')
                        if snippet:
                            web_texts.append(snippet)
                            all_sources_info.append({
                                "source": "【网络检索参考】",
                                "chunk_id": url,
                                "preview": snippet[:200] if len(snippet) > 200 else snippet,
                                "source_type": "web",
                                "title": title,
                                "url": url,
                            })

                    if search_has_error:
                        # 搜索报错：不标记 triggered，让下游追加错误消息
                        pass
                    elif web_texts:
                        combined_web = "\n\n".join(
                            f"[网络检索 {j+1}] {t}" for j, t in enumerate(web_texts)
                        )
                        all_contexts.append(combined_web)
                        all_doc_ids.append(f"web_search_{i}")
                        all_metadata.append({
                            "source": "【网络检索参考】",
                            "source_type": "web",
                            "title": f"联网搜索补充（来自 {orig_query}）",
                        })
                        logging.info(f"→ 联网搜索成功，新增 {len(web_texts)} 条网络摘要")
                        web_search_triggered = True
                    else:
                        logging.info("→ 联网搜索返回 0 条有效摘要")
                        web_search_triggered = True

                except Exception as e:
                    web_search_failed_msg = f"联网搜索失败，仅基于本地文档作答: {str(e)}"
                    logging.error(f"→ {web_search_failed_msg}")

        # 无 SERPAPI_KEY 时仅记录日志
        if i == 0 and enable_web_search and not check_serpapi_key():
            logging.warning("联网搜索未启用：未配置 SERPAPI_KEY")
        # ━━━ 结束联网搜索逻辑 ━━━

        # 如果第一轮联网搜索失败，追加错误消息到来源中
        if i == 0 and web_search_failed_msg and not web_search_triggered:
            all_sources_info.append({
                "source": "【网络检索参考】",
                "chunk_id": "web_search_failed",
                "preview": web_search_failed_msg,
                "source_type": "web_error",
                "title": "联网搜索提示",
                "url": "",
            })

        if i == max_iterations - 1:
            break

        # ━━━ 10. LLM 判断是否需要继续递归检索 ━━━
        if all_contexts:
            summary = "\n".join([c[:200] for c in all_contexts[:3]])
            prompt = f"""你是一个查询优化助手。根据以下信息判断是否需要新的查询。

[初始问题]
{initial_query}

[检索结果摘要]
{summary}

要求：
1. 如果信息已足够，直接回复：不需要进一步查询
2. 否则返回一个更精准的新查询，仅包含查询词
"""
            try:
                from core.generator import call_llm_simple
                next_query = call_llm_simple(
                    prompt, model_choice,
                    ollama_model_name=ollama_model_name,
                    ollama_num_ctx=ollama_num_ctx,
                    ollama_temperature=ollama_temperature,
                    ollama_top_p=ollama_top_p
                )
                if "不需要" in next_query:
                    logging.info("LLM 判断无需更多查询")
                    break
                if len(next_query) > 100:
                    logging.warning("生成内容过长，不视为有效查询")
                    break
                query = next_query
                logging.info(f"生成下一轮查询: {query}")
            except Exception as e:
                logging.error(f"生成新查询失败: {str(e)}")
                break
        else:
            break

    logging.info(
        f"检索完成: 本地{len([s for s in all_sources_info if s.get('source_type') == 'local'])}条"
        f", 网络{len([s for s in all_sources_info if s.get('source_type') == 'web'])}条"
        f"{', 搜索失败' if web_search_failed_msg else ''}"
    )
    return all_contexts, all_doc_ids, all_metadata, all_sources_info
