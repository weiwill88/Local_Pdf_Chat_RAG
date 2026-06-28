"""
Modul Pemrosesan Inti RAG

Jalur Pembelajaran (sesuai urutan alur kerja RAG):
1. document_loader.py  → Memahami cara dokumen didekripsi menjadi teks biasa
2. text_splitter.py    → Memahami cara teks panjang dipecah menjadi fragmen yang ramah pencarian
3. embeddings.py       → Memahami cara teks dipetakan ke ruang vektor
4. vector_store.py     → Memahami cara FAISS menyimpan dan mencari vektor
5. bm25_index.py       → Memahami bagaimana pencarian jarang melengkapi pencarian padat
6. retriever.py        → Memahami rancangan strategi pencarian hibrida
7. reranker.py         → Memahami pencarian dua tahap (recall + rerank)
8. generator.py        → Memahami konstruksi prompt dan panggilan LLM
"""
