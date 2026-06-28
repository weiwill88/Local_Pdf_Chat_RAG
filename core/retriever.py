"""
Pencari (Retriever) —— Pencarian Hibrida + Strategi Pencarian Rekursif

Poin Pembelajaran:
- Pencarian Hibrida (Hybrid Search) menggabungkan keunggulan pencarian semantik dan pencarian kata kunci
- Parameter alpha mengontrol bobot keduanya (0.7 = 70% semantik + 30% kata kunci)
- Pencarian rekursif melalui beberapa putaran iterasi menggunakan LLM untuk menulis ulang kueri guna mendapatkan informasi yang lebih komprehensif
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
    Menggabungkan hasil pencarian semantik dan pencarian BM25

    Menggunakan skor terbobot: Skor Semantik × alpha + Skor BM25 × (1-alpha)

    Args:
        semantic_results: {'ids': [[...]], 'documents': [[...]], 'metadatas': [[...]], 'distances': [[...]]}
        bm25_results: [{'id': ..., 'score': ..., 'content': ...}]
        alpha: Bobot pencarian semantik

    Returns:
        [(doc_id, {'score': ..., 'content': ..., 'metadata': ...})] yang diurutkan
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
        distances = semantic_results.get('distances', [[]])
        semantic_distances = distances[0] if isinstance(distances, list) and len(distances) > 0 else []
        if semantic_distances and len(semantic_distances) != len(semantic_results['documents'][0]):
            logging.warning("Hasil pencarian semantik tidak memiliki jumlah jarak yang selaras dengan dokumen")
            semantic_distances = []

        for i, (doc_id, doc, meta) in enumerate(
                zip(semantic_results['ids'][0], semantic_results['documents'][0], semantic_results['metadatas'][0])):
            if semantic_distances:
                distance = float(semantic_distances[i])
                score = max(0.0, 1.0 - (distance / 2.0))
            else:
                num_results = len(semantic_results['documents'][0])
                score = 1.0 - (i / max(1, num_results))
            merged_dict[doc_id] = {'score': alpha * score, 'content': doc, 'metadata': meta}
    else:
        logging.warning("Hasil pencarian semantik kosong atau format tidak normal")

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
    Pencarian rekursif dan optimasi kueri

    Alur: 1.Pencarian Semantik + BM25 → 2.Pengurutan Hibrida → 3.Pengurutan Ulang (Rerank) → 4.Penilaian LLM apakah kueri perlu ditulis ulang untuk melanjutkan

    Returns:
        (all_contexts, all_doc_ids, all_metadata)
    """
    if max_iterations is None:
        max_iterations = MAX_RETRIEVAL_ITERATIONS

    query = initial_query
    all_contexts, all_doc_ids, all_metadata = [], [], []

    for i in range(max_iterations):
        logging.info(f"Pencarian rekursif {i + 1}/{max_iterations}, Kueri saat ini: {query}")

        # 网络搜索补充
        web_texts = []
        if enable_web_search and check_serpapi_key():
            try:
                for res in search_web(query):
                    web_texts.append(f"Judul: {res.get('title', '')}\nAbstrak: {res.get('snippet', '')}")
            except Exception as e:
                logging.error(f"Terjadi kesalahan pada pencarian web: {str(e)}")

        # 语义检索
        query_embedding = encode_query(query)
        sem_docs, sem_ids, sem_metas, sem_distances = vector_store.search(query_embedding, k=RETRIEVAL_TOP_K)

        prepared = {"ids": [sem_ids], "documents": [sem_docs], "metadatas": [sem_metas], "distances": [sem_distances]}

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
                logging.error(f"Pengurutan ulang gagal: {str(e)}")
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
            prompt = f"""Anda adalah asisten optimasi kueri. Nilai apakah kueri baru diperlukan berdasarkan informasi berikut.

[Pertanyaan Awal]
{initial_query}

[Ringkasan Hasil Pencarian]
{summary}

Persyaratan:
1. Jika informasi sudah cukup, langsung jawab: tidak perlu kueri lebih lanjut
2. Jika tidak, kembalikan kueri baru yang lebih presisi, hanya berisi kata pencarian saja
"""
            try:
                from core.generator import call_llm_simple
                next_query = call_llm_simple(prompt, model_choice)
                if "tidak perlu" in next_query.lower() or "不需要" in next_query:
                    logging.info("LLM menilai tidak perlu kueri tambahan")
                    break
                if len(next_query) > 100:
                    logging.warning("Konten yang dihasilkan terlalu panjang, tidak dianggap sebagai kueri yang valid")
                    break
                query = next_query
                logging.info(f"Menghasilkan kueri putaran berikutnya: {query}")
            except Exception as e:
                logging.error(f"Gagal menghasilkan kueri baru: {str(e)}")
                break
        else:
            break

    return all_contexts, all_doc_ids, all_metadata
