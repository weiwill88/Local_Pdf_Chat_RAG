import numpy as np
import pytest

import core.retriever as retriever
from core.bm25_index import BM25IndexManager
from core.generator import _build_context
from core.retriever import hybrid_merge


def test_bm25_returns_relevant_document_first():
    manager = BM25IndexManager()
    manager.build_index(
        [
            "FAISS supports dense vector retrieval",
            "BM25 supports exact keyword retrieval",
            "RAG combines retrieval with generation",
        ],
        ["dense", "sparse", "rag"],
    )

    results = manager.search("BM25 keyword", top_k=2)

    assert results
    assert results[0]["id"] == "sparse"


def test_hybrid_merge_combines_semantic_and_sparse_scores():
    semantic = {
        "ids": [["doc-a", "doc-b"]],
        "documents": [["semantic result", "shared result"]],
        "metadatas": [[{"source": "a"}, {"source": "b"}]],
    }
    sparse = [
        {"id": "doc-b", "score": 4.0, "content": "shared result"},
        {"id": "doc-c", "score": 2.0, "content": "keyword result"},
    ]

    merged = hybrid_merge(semantic, sparse, alpha=0.5)
    merged_by_id = dict(merged)

    assert set(merged_by_id) == {"doc-a", "doc-b", "doc-c"}
    assert merged[0][0] == "doc-b"
    assert merged_by_id["doc-b"]["score"] > merged_by_id["doc-a"]["score"]


def test_recursive_retrieval_returns_web_results_with_source_metadata(monkeypatch):
    web_result = {
        "title": "RAG retrieval update",
        "url": "https://example.test/rag-update",
        "snippet": "The new retrieval pipeline preserves web source metadata.",
        "timestamp": "2026-08-14",
    }
    monkeypatch.setattr(retriever, "check_serpapi_key", lambda: True)
    monkeypatch.setattr(retriever, "search_web", lambda query: [web_result])
    monkeypatch.setattr(
        retriever,
        "encode_query",
        lambda query: np.zeros((1, 384), dtype="float32"),
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "search",
        lambda query_embedding, k: (
            ["Local retrieval context."],
            ["doc-local"],
            [{"source": "local.pdf"}],
        ),
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)
    monkeypatch.setattr(
        retriever,
        "rerank_results",
        lambda query, docs, ids, metadata, top_k: [
            (
                doc_id,
                {"content": doc, "metadata": meta, "score": 1.0},
            )
            for doc_id, doc, meta in zip(ids, docs, metadata)
        ],
    )

    contexts, doc_ids, metadata = retriever.recursive_retrieval(
        "What changed in retrieval?",
        max_iterations=1,
        enable_web_search=True,
    )

    assert contexts == [web_result["snippet"], "Local retrieval context."]
    assert doc_ids == ["web:https://example.test/rag-update", "doc-local"]
    assert metadata == [
        {
            "source": "web",
            "title": web_result["title"],
            "url": web_result["url"],
            "timestamp": web_result["timestamp"],
        },
        {"source": "local.pdf"},
    ]
    final_context, sources = _build_context(
        contexts,
        doc_ids,
        metadata,
        enable_web_search=True,
    )
    assert web_result["snippet"] in final_context
    assert web_result["url"] in final_context
    assert "Local retrieval context." in final_context
    assert sources == [
        {
            "text": web_result["snippet"],
            "type": "web",
            "url": web_result["url"],
            "title": web_result["title"],
        },
        {
            "text": "Local retrieval context.",
            "type": "local.pdf",
            "source": "local.pdf",
        },
    ]


@pytest.mark.parametrize("url", ["https://example.test/stable-source", ""])
def test_recursive_retrieval_deduplicates_web_results_across_iterations(monkeypatch, url):
    web_result = {
        "title": "Stable source",
        "url": url,
        "snippet": "This result is returned for both retrieval queries.",
        "timestamp": None,
    }
    monkeypatch.setattr(retriever, "check_serpapi_key", lambda: True)
    monkeypatch.setattr(retriever, "search_web", lambda query: [web_result])
    monkeypatch.setattr(
        retriever,
        "encode_query",
        lambda query: np.zeros((1, 384), dtype="float32"),
    )
    monkeypatch.setattr(
        retriever.vector_store,
        "search",
        lambda query_embedding, k: ([], [], []),
    )
    monkeypatch.setattr(retriever.bm25_manager, "bm25_index", None)
    monkeypatch.setattr(
        "core.generator.call_llm_simple",
        lambda prompt, model_choice: "refined retrieval query",
    )

    contexts, doc_ids, metadata = retriever.recursive_retrieval(
        "initial query",
        max_iterations=2,
        enable_web_search=True,
    )

    assert contexts == [web_result["snippet"]]
    source_key = url or f'{web_result["title"]}\n{web_result["snippet"]}'
    assert doc_ids == [f"web:{source_key}"]
    assert len(metadata) == 1
