"""
检索器 —— 混合检索 + 递归检索策略

学习要点：
- 混合检索（Hybrid Search）结合语义检索和关键词检索的优势
- alpha 参数控制两者权重（0.7 = 70% 语义 + 30% 关键词）
- 递归检索通过多轮迭代，利用 LLM 改写查询获取更全面的信息
"""

import logging
from config import HYBRID_ALPHA, RETRIEVAL_TOP_K, RERANK_TOP_K, MAX_RETRIEVAL_ITERATIONS
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.embeddings import encode_query
from core.reranker import rerank_results
from features.web_search import check_serpapi_key, search_web


def hybrid_merge(semantic_results, bm25_results, alpha=None):
    """
    合并语义检索和 BM25 检索结果

    使用加权分数：语义分数 × alpha + BM25分数 × (1-alpha)

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]]}
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: 语义检索权重

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


def recursive_retrieval(initial_query, max_iterations=None, enable_web_search=False, model_choice="siliconflow"):
    """
    递归检索与查询优化

    流程：1.语义+BM25检索 → 2.混合排序 → 3.重排序 → 4.LLM判断是否改写query继续

    Returns:
        (all_contexts, all_doc_ids, all_metadata)
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    query = initial_query
    all_contexts, all_doc_ids, all_metadata = [], [], []
    seen_web_sources = set()

    for i in range(max_iterations):
        logging.info(f"递归检索 {i + 1}/{max_iterations}，当前 Query: {query}")

        # 网络搜索补充
        web_texts = []
        if enable_web_search and check_serpapi_key():
            try:
                for res in search_web(query):
                    title = res.get('title') or ''
                    url = res.get('url') or ''
                    snippet = res.get('snippet') or ''
                    web_texts.append(f"标题：{title}\n摘要：{snippet}")
                    source_key = url or f"{title}\n{snippet}"
                    if snippet and source_key not in seen_web_sources:
                        seen_web_sources.add(source_key)
                        all_contexts.append(snippet)
                        all_doc_ids.append(f"web:{source_key}")
                        all_metadata.append({
                            'source': 'web',
                            'title': title,
                            'url': url,
                            'timestamp': res.get('timestamp'),
                        })
            except Exception as e:
                logging.error(f"网络搜索出错: {str(e)}")

        # 语义检索
        query_embedding = encode_query(query)
        sem_docs, sem_ids, sem_metas = vector_store.search(query_embedding, k=RETRIEVAL_TOP_K)

        prepared = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas]}

        # BM25 检索
        bm25_res = bm25_manager.search(query, top_k=RETRIEVAL_TOP_K) if bm25_manager.bm25_index else []

        # 混合排序 → 重排序
        hybrid = hybrid_merge(prepared, bm25_res)
        ids_iter, docs_iter, meta_iter = [], [], []
        for doc_id, data in hybrid[:RETRIEVAL_TOP_K]:
            ids_iter.append(doc_id)
            docs_iter.append(data['content'])
            meta_iter.append(data['metadata'])

        if docs_iter:
            try:
                reranked = rerank_results(query, docs_iter, ids_iter, meta_iter, top_k=RERANK_TOP_K)
            except Exception as e:
                logging.error(f"重排序失败: {str(e)}")
                reranked = [(did, {'content': d, 'metadata': m, 'score': 1.0})
                            for did, d, m in zip(ids_iter, docs_iter, meta_iter)]
        else:
            reranked = []

        # 整合结果
        current_contexts = web_texts[:]
        for doc_id, data in reranked:
            if doc_id not in all_doc_ids:
                all_doc_ids.append(doc_id)
                all_contexts.append(data['content'])
                all_metadata.append(data['metadata'])
            current_contexts.append(data['content'])

        if i == max_iterations - 1:
            break

        # LLM 判断是否需要继续
        if current_contexts:
            summary = "\n".join(current_contexts[:3])
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
                next_query = call_llm_simple(prompt, model_choice)
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

    return all_contexts, all_doc_ids, all_metadata
