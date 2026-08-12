from core.bm25_index import BM25IndexManager
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
