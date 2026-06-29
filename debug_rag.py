#!/usr/bin/env python3
"""
debug_rag.py — RAG Pipeline Inspector
======================================

Run this script to inspect every stage of the pipeline for any query.
No need to touch the main application — this is a standalone debug tool.

QUICK START
-----------
First upload your PDF via the Gradio UI (http://localhost:17995), then:

    # Run all 3 built-in failing test queries
    python debug_rag.py

    # Run your own question
    python debug_rag.py --query "What is PROMIA?"

    # Check if a keyword is even IN the index (fastest check)
    python debug_rag.py --keyword PROMIA

    # Check multiple keywords at once
    python debug_rag.py --keyword PROMIA --keyword Rockwell --keyword "Our Services"

    # Skip LLM (fast mode — only shows retrieval stages)
    python debug_rag.py --no-llm

    # Stop at a specific stage
    python debug_rag.py --stage faiss      # only FAISS
    python debug_rag.py --stage hybrid     # FAISS + BM25 + hybrid merge
    python debug_rag.py --stage rerank     # up to reranking
    python debug_rag.py --stage context    # up to prompt context (no LLM)
    python debug_rag.py --stage all        # full pipeline (default)

    # Browse chunks by section name
    python debug_rag.py --browse "OUR SERVICE"
    python debug_rag.py --browse "TRAINING"

    # Ingest a PDF directly (skips Gradio, uses cache if already ingested)
    python debug_rag.py --pdf path/to/file.pdf --query "..."

    # Change how many candidates to inspect before reranking (default 15)
    python debug_rag.py --top-n 20

STAGES
------
1. PDF Parsing   → Is the answer text in the document at all?   Use --keyword
2. Chunking      → Is it split well?                            Use --browse
3. Metadata      → Does the chunk have correct heading?         Use --browse
4. FAISS         → Does semantic search find it?                Use --stage faiss
5. BM25          → Does keyword search find it?                 Use --stage hybrid
6. Hybrid Merge  → What is the combined ranking?                Use --stage hybrid
7. Reranker      → Did Cross-Encoder keep it?                   Use --stage rerank
8. Prompt        → Did the answer reach the LLM context?        Use --stage context
9. LLM           → Did the model answer correctly?              (default)
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RETRIEVAL_TOP_K, RERANK_TOP_K, DEFAULT_MODEL_CHOICE, OLLAMA_MODEL_NAME, HYBRID_ALPHA
from core.cache import CACHE_DIR
from core.vector_store import vector_store
from core.bm25_index import bm25_manager
from core.embeddings import encode_query, encode_texts
from core.retriever import hybrid_merge
from core.reranker import rerank_results
from core.generator import _build_context, _build_prompt, call_siliconflow_api
from core.document_loader import extract_text
from core.text_splitter import split_text
from core.cache import load_document_cache, save_document_cache
from utils.network import get_session

# ── Built-in failing test cases ───────────────────────────────────────────────
TEST_QUERIES = [
    "What are the four main service categories provided by PT. SCADA PRIMA CIPTA?",
    "What proprietary RBI software did SPC use?",
    "Sebutkan mitra teknologi",
]

STAGES = ("faiss", "bm25", "hybrid", "rerank", "context", "all")


# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_caches() -> bool:
    """Rebuild FAISS + BM25 from every .pkl in cache/"""
    cache_files = sorted(Path(CACHE_DIR).glob("*.pkl"))
    if not cache_files:
        return False
    all_chunks, all_ids, all_metas, all_embeddings = [], [], [], []
    for pkl_file in cache_files:
        try:
            with open(pkl_file, "rb") as fh:
                chunks, metadatas, chunk_ids, embeddings = pickle.load(fh)
            all_chunks.extend(chunks)
            all_ids.extend(chunk_ids)
            all_metas.extend(metadatas)
            all_embeddings.append(np.asarray(embeddings, dtype="float32"))
            print(f"  [cache] {pkl_file.name[:24]}  {len(chunks)} chunks")
        except Exception as exc:
            print(f"  [cache] SKIP {pkl_file.name}: {exc}")
    if not all_chunks:
        return False
    vector_store.build_index(all_chunks, all_ids, all_metas, np.vstack(all_embeddings))
    bm25_manager.build_index(all_chunks, all_ids)
    print(f"  [cache] Done — {len(all_chunks)} chunks total\n")
    return True


def ingest_pdf(pdf_path: str) -> bool:
    """Parse → chunk → embed → index one PDF (uses cache if unchanged)."""
    import os
    print(f"  [ingest] {pdf_path}")
    cached, file_hash = load_document_cache(pdf_path)
    if cached:
        chunks, metadatas, chunk_ids, embeddings = cached
        print(f"  [ingest] Cache hit — {len(chunks)} chunks")
    else:
        text = extract_text(pdf_path)
        if not text:
            print("  [ingest] ERROR: Docling returned empty text")
            return False
        chunks = split_text(text)
        file_name = os.path.basename(pdf_path)
        doc_id = f"doc_{file_hash[:12]}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_name, "doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]
        print(f"  [ingest] Embedding {len(chunks)} chunks...")
        embeddings = encode_texts(chunks, show_progress=True)
        save_document_cache(file_hash, (chunks, metadatas, chunk_ids, embeddings))
    arr = np.asarray(embeddings, dtype="float32")
    vector_store.build_index(chunks, chunk_ids, metadatas, arr)
    bm25_manager.build_index(chunks, chunk_ids)
    print(f"  [ingest] Indexed {len(chunks)} chunks\n")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1+2: Keyword scanner  →  Is the answer text in the index at all?
# ─────────────────────────────────────────────────────────────────────────────

def scan_keyword(keyword: str) -> bool:
    """Case-insensitive search across every indexed chunk.

    FOUND  → parsing + chunking OK, problem is in retrieval stages 4-8
    MISSING → problem is at PDF parsing (stage 1). Enable OCR or check PDF.
    """
    hits = [
        (cid, content, vector_store.metadatas_map.get(cid, {}))
        for cid, content in vector_store.contents_map.items()
        if keyword.lower() in content.lower()
    ]
    W = "=" * 60
    if hits:
        print(f"\n{W}")
        print(f"  KEYWORD '{keyword}'  →  FOUND in {len(hits)} chunk(s)")
        print(f"  Conclusion: text exists → problem is in retrieval, not parsing")
        print(W)
        for cid, content, meta in hits[:5]:
            idx = content.lower().find(keyword.lower())
            snippet = content[max(0, idx - 40): idx + 100].replace("\n", " ").strip()
            chunk_no = meta.get("chunk_index", "?")
            sub = meta.get("subsection") or meta.get("section") or "—"
            print(f"\n  chunk_index : {chunk_no}")
            print(f"  chunk_id    : {cid}")
            print(f"  source      : {meta.get('source', '?')}")
            print(f"  subsection  : {sub}")
            print(f"  context     : ...{snippet!r}...")
        return True
    else:
        print(f"\n{'=' * 60}")
        print(f"  KEYWORD '{keyword}'  →  NOT FOUND")
        print(f"  Conclusion: PDF parsing failed — text was never extracted.")
        print(f"  Fix: enable OCR in document_loader.py or check the PDF manually.")
        print("=" * 60)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2+3: Browse chunks by section  →  Is chunking + metadata correct?
# ─────────────────────────────────────────────────────────────────────────────

def browse_chunks(pattern: str) -> None:
    """List all chunks whose section/subsection/heading contains PATTERN.

    Use this to verify:
    - The chunk boundaries are sensible
    - The heading metadata is correct
    - The answer isn't split across too many small chunks
    """
    hits = []
    for cid, content in vector_store.contents_map.items():
        meta = vector_store.metadatas_map.get(cid, {})
        labels = " ".join(str(v) for v in [
            meta.get("section"), meta.get("subsection"), meta.get("heading")
        ] if v).lower()
        if pattern.lower() in labels or pattern.lower() in content[:150].lower():
            hits.append((meta.get("chunk_index", 0), cid, content, meta))

    hits.sort(key=lambda x: x[0])
    print(f"\n  [browse] Pattern '{pattern}' → {len(hits)} chunk(s)\n")
    for chunk_no, cid, content, meta in hits:
        sub = meta.get("subsection") or meta.get("section") or "—"
        hdg = meta.get("heading") or "—"
        print(f"  ── chunk #{chunk_no}  id={cid}")
        print(f"     subsection : {sub}")
        print(f"     heading    : {hdg}")
        print(f"     chars      : {len(content)}")
        print(f"     content    :\n{content[:300]!r}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Stages 4-9: Full pipeline trace
# ─────────────────────────────────────────────────────────────────────────────

def _heading(meta: dict) -> str:
    return meta.get("heading") or meta.get("subsection") or meta.get("section") or "—"


def run_debug(question: str, model_choice: str = DEFAULT_MODEL_CHOICE,
              top_n: int = 15, stage: str = "all", no_llm: bool = False) -> None:
    """Full pipeline trace for one question, stopping at `stage`."""
    W = "=" * 70
    T = "-" * 70

    print(f"\n{W}")
    print(f"  QUESTION  : {question}")
    print(f"  stage     : {stage}   top_n={top_n}   model={model_choice}")
    print(W)

    if not vector_store.is_ready:
        print("  [ERROR] Index is empty. Upload a PDF via Gradio or use --pdf.")
        return

    # ── Stage 4: FAISS ────────────────────────────────────────────────────
    query_emb = encode_query(question)
    sem_docs, sem_ids, sem_metas, sem_distances = vector_store.search(query_emb, k=top_n)

    print(f"\n  [STAGE 4 – FAISS]  cosine score (higher = more similar)")
    print(T)
    print(f"  {'Rank':<5} {'CosScore':>9}  {'Subsection / Heading':<35}  Preview")
    print(T)
    for rank, (sid, sdoc, smeta, sdist) in enumerate(
            zip(sem_ids, sem_docs, sem_metas, sem_distances), 1):
        label = _heading(smeta)[:34]
        preview = sdoc[:60].replace("\n", " ").strip()
        print(f"  {rank:<5} {sdist:>+9.4f}  {label:<35}  {preview!r}")

    if stage == "faiss":
        return

    # ── Stage 5+6: BM25 + Hybrid ─────────────────────────────────────────
    bm25_res = bm25_manager.search(question, top_k=top_n) if bm25_manager.bm25_index else []

    print(f"\n  [STAGE 5 – BM25]  keyword score")
    print(T)
    print(f"  {'Rank':<5} {'BM25Score':>9}  {'Subsection / Heading':<35}  Preview")
    print(T)
    if not bm25_res:
        print("  (no BM25 results — index may be empty)")
    for rank, r in enumerate(bm25_res, 1):
        meta = vector_store.metadatas_map.get(r["id"], {})
        label = _heading(meta)[:34]
        preview = r["content"][:60].replace("\n", " ").strip()
        print(f"  {rank:<5} {r['score']:>9.4f}  {label:<35}  {preview!r}")

    prepared = {"ids": [sem_ids], "documents": [sem_docs],
                "metadatas": [sem_metas], "distances": [sem_distances]}
    hybrid = hybrid_merge(prepared, bm25_res)

    print(f"\n  [STAGE 6 – HYBRID]  alpha={HYBRID_ALPHA}  "
          f"({HYBRID_ALPHA:.0%} semantic + {1-HYBRID_ALPHA:.0%} BM25)")
    print(T)
    print(f"  {'Rank':<5} {'HybridScore':>11}  {'Subsection / Heading':<35}  Preview")
    print(T)
    for rank, (doc_id, data) in enumerate(hybrid[:top_n], 1):
        meta = data.get("metadata", {})
        label = _heading(meta)[:34]
        preview = data["content"][:60].replace("\n", " ").strip()
        print(f"  {rank:<5} {data['score']:>11.4f}  {label:<35}  {preview!r}")

    if stage in ("bm25", "hybrid"):
        return

    # ── Stage 7: Reranker ─────────────────────────────────────────────────
    ids_pre   = [doc_id for doc_id, _ in hybrid[:top_n]]
    docs_pre  = [data["content"]  for _, data in hybrid[:top_n]]
    metas_pre = [data["metadata"] for _, data in hybrid[:top_n]]

    try:
        reranked = rerank_results(question, docs_pre, ids_pre, metas_pre, top_k=RERANK_TOP_K)
    except Exception as exc:
        print(f"\n  [RERANKER ERROR] {exc}")
        reranked = [(did, {"content": d, "metadata": m, "score": 1.0})
                    for did, d, m in zip(ids_pre[:RERANK_TOP_K],
                                          docs_pre[:RERANK_TOP_K],
                                          metas_pre[:RERANK_TOP_K])]

    print(f"\n  [STAGE 7 – RERANKER]  Cross-Encoder top-{RERANK_TOP_K}")
    print(T)
    print(f"  {'Rank':<5} {'CE Score':>9}  {'Subsection / Heading':<35}  Preview")
    print(T)
    for rank, (doc_id, data) in enumerate(reranked, 1):
        meta = data.get("metadata", {})
        label = _heading(meta)[:34]
        preview = data["content"][:60].replace("\n", " ").strip()
        print(f"  {rank:<5} {data['score']:>9.4f}  {label:<35}  {preview!r}")

    if stage == "rerank":
        return

    # ── Stage 8: Prompt context ───────────────────────────────────────────
    all_ids  = [doc_id for doc_id, _ in reranked]
    all_docs = [data["content"]  for _, data in reranked]
    all_meta = [data["metadata"] for _, data in reranked]

    context, _ = _build_context(all_docs, all_ids, all_meta, enable_web_search=False)
    prompt = _build_prompt(question, context,
                           enable_web_search=False, knowledge_base_exists=True,
                           time_sensitive=False, conflict_detected=False)

    print(f"\n  [STAGE 8 – PROMPT CONTEXT]  {len(context)} chars total")
    print(T)
    print(context[:2500])
    if len(context) > 2500:
        print(f"\n  ... [{len(context)-2500} chars hidden, use --stage context to stop here and read in full]")

    if stage == "context" or no_llm:
        if no_llm:
            print("\n  [LLM skipped — remove --no-llm to run generation]")
        return

    # ── Stage 9: LLM answer ───────────────────────────────────────────────
    print(f"\n  [STAGE 9 – LLM ANSWER]  model={model_choice}")
    print(T)
    try:
        if model_choice == "siliconflow":
            answer = call_siliconflow_api(prompt, temperature=0.1, max_tokens=1024)
        else:
            resp = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": False},
                timeout=180,
            )
            answer = resp.json().get("response", "").strip()
        print(answer)
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")

    print(f"\n{W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdf",     metavar="PATH",
        help="Ingest this PDF before running (skips embedding if already cached)")
    parser.add_argument("--model",   default=DEFAULT_MODEL_CHOICE,
        help="LLM backend: siliconflow | ollama  (default: auto-detected)")
    parser.add_argument("--query",   metavar="TEXT", action="append",
        help="Run this question (can repeat: --query Q1 --query Q2)")
    parser.add_argument("--keyword", metavar="WORD", action="append",
        help="Check if WORD is in the index (can repeat)")
    parser.add_argument("--browse",  metavar="SECTION",
        help="List all chunks whose section/heading contains this text")
    parser.add_argument("--stage",   default="all", choices=STAGES,
        help="Stop pipeline at this stage  (default: all)")
    parser.add_argument("--no-llm",  action="store_true",
        help="Skip LLM generation — show retrieval stages only (fast)")
    parser.add_argument("--top-n",   type=int, default=15, metavar="N",
        help="Candidates before reranking  (default: 15)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  RAG Pipeline Inspector")
    print("=" * 70 + "\n")

    # ── Load indexes ──────────────────────────────────────────────────────
    if args.pdf:
        ok = ingest_pdf(args.pdf)
    else:
        print("  Loading indexes from cache/...")
        ok = load_all_caches()

    if not ok:
        print("\n  No documents loaded.")
        print("  Upload a PDF via the Gradio UI first, then run this script.")
        print("  Or: python debug_rag.py --pdf path/to/document.pdf")
        sys.exit(1)

    print(f"  Index: {vector_store.total_chunks} chunks, "
          f"{len(vector_store.contents_map)} unique entries")
    print(f"  Config: CHUNK_SIZE={__import__('config').CHUNK_SIZE}  "
          f"CHUNK_OVERLAP={__import__('config').CHUNK_OVERLAP}  "
          f"RETRIEVAL_TOP_K={__import__('config').RETRIEVAL_TOP_K}  "
          f"RERANK_TOP_K={__import__('config').RERANK_TOP_K}  "
          f"HYBRID_ALPHA={HYBRID_ALPHA}\n")

    # ── Keyword scans ─────────────────────────────────────────────────────
    if args.keyword:
        for kw in args.keyword:
            scan_keyword(kw)
        print()

    # ── Browse chunks ─────────────────────────────────────────────────────
    if args.browse:
        browse_chunks(args.browse)

    # ── If only keyword/browse was requested, stop here ───────────────────
    if (args.keyword or args.browse) and not args.query:
        return

    # ── Run queries ───────────────────────────────────────────────────────
    queries = args.query if args.query else TEST_QUERIES
    for q in queries:
        run_debug(q,
                  model_choice=args.model,
                  top_n=args.top_n,
                  stage=args.stage,
                  no_llm=args.no_llm)


if __name__ == "__main__":
    main()
