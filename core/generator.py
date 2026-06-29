"""
Panggilan LLM —— Pembuatan Jawaban Model Besar (Ollama + SiliconFlow)

Poin Pembelajaran:
- Prompt Engineering: Cara membangun templat prompt berkualitas tinggi
- Perbedaan antara output streaming vs non-streaming
- Adaptasi multi-model: Integrasi Ollama lokal dan API SiliconFlow cloud
"""

import json
import logging
import requests
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL,
    SILICONFLOW_MODEL_NAME, OLLAMA_MODEL_NAME,
    OLLAMA_OPTIONS_GENERATION, OLLAMA_OPTIONS_REWRITE,
)
from utils.network import get_session
from core.retriever import recursive_retrieval
from core.vector_store import vector_store
from features.conflict_detector import detect_conflicts, evaluate_source_credibility
from features.thinking_chain import process_thinking_content


def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """Memanggil antarmuka SiliconFlow Cloud untuk mendapatkan jawaban"""
    if not SILICONFLOW_API_KEY:
        logging.error("SILICONFLOW_API_KEY tidak diatur")
        return "Kesalahan: Kunci API SiliconFlow tidak dikonfigurasi."

    try:
        payload = {
            "model": SILICONFLOW_MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": max_tokens,
            "temperature": temperature, "top_p": 0.7, "top_k": 50,
            "frequency_penalty": 0.5, "n": 1,
            "response_format": {"type": "text"}
        }
        headers = {
            "Authorization": f"Bearer {SILICONFLOW_API_KEY.strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(SILICONFLOW_API_URL, data=json_payload, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "")
            if reasoning:
                return f"{content}<think>{reasoning}</think>"
            return content
        return "Format hasil pengembalian API tidak normal"

    except requests.exceptions.RequestException as e:
        logging.error(f"Terjadi kesalahan saat memanggil SiliconFlow API: {str(e)}")
        return f"Terjadi kesalahan saat memanggil API: {str(e)}"
    except Exception as e:
        logging.error(f"Kesalahan tidak diketahui pada SiliconFlow API: {str(e)}")
        return f"Terjadi kesalahan tidak diketahui: {str(e)}"


def call_llm_simple(prompt, model_choice="siliconflow", temperature=0.0):
    """Panggilan LLM sederhana (digunakan untuk penentuan penulisan ulang kueri dalam pencarian rekursif)
    Menyediakan parameter `temperature` agar pemanggilan deterministik ketika diperlukan.
    """
    if model_choice == "siliconflow":
        result = call_siliconflow_api(prompt, temperature=temperature)
        result = result.strip() if isinstance(result, str) else result[0].strip()
        if "<think>" in result:
            result = result.split("<think>")[0].strip()
        return result
    else:
        response = get_session().post(
            "http://localhost:11434/api/chat",
            json={
                "model": OLLAMA_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
                "options": {**OLLAMA_OPTIONS_REWRITE, "temperature": temperature},
            },
            timeout=180
        )
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "").strip()


def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected, query_type="factoid"):
    """Membangun prompt"""
    extra = ""
    if time_sensitive and enable_web_search:
        extra += " Prioritise the most recent information."
    if conflict_detected:
        extra += " Clearly highlight discrepancies between sources."
    if query_type == "enumeration":
        extra += (
            " This is an enumeration question. You MUST scan every retrieved chunk and collect"
            " ALL matching items without stopping early. Present your answer as a numbered list."
            " If an item appears in multiple chunks, list it only once."
        )

    prompt_template = (
        "You are a document Q&A assistant. Answer ONLY using the retrieved context below."
        " Do not use any external knowledge."
        " Answer in the same language as the question.{extra}\n"
        "When the question asks for a list or categories, enumerate all relevant items found across all retrieved chunks.\n"
        "If the context does not contain sufficient information, respond with exactly:\n"
        "\"I cannot find this information in the provided documents.\"\n"
        "Always cite the source label (e.g. [Document: filename | Section: heading]) "
        "inline next to each claim.\n\n"
        "[Retrieved Context]\n{context}\n\n"
        "[Question]\n{question}\n\n"
        "Answer:"
    )

    resolved_context = context if context else (
        "No web search results available." if enable_web_search and not knowledge_base_exists
        else "No relevant context found in the provided documents."
    )

    return prompt_template.format(
        extra=extra,
        context=resolved_context,
        question=question,
    )


def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """Membangun konteks dan informasi sumber"""
    context_parts = []
    sources_for_conflict = []

    for doc, doc_id, metadata in zip(all_contexts, all_doc_ids, all_metadata):
        source_type = metadata.get('source', 'Dokumen Lokal')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            url = metadata.get('url', 'URL tidak diketahui')
            title = metadata.get('title', 'Judul tidak diketahui')
            context_parts.append(f"[Sumber Internet: {title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
        else:
            source = metadata.get('source', 'Sumber tidak diketahui')
            section = metadata.get('section')
            subsection = metadata.get('subsection')
            heading = metadata.get('heading')
            chunk_index = metadata.get('chunk_index')
            source_label = f"[Dokumen Lokal: {source}"
            if section:
                source_label += f" | Section: {section}"
            if subsection:
                source_label += f" | Subsection: {subsection}"
            if heading:
                source_label += f" | Heading: {heading}"
            if chunk_index is not None:
                source_label += f" | Chunk #{chunk_index + 1}"
            source_label += "]"
            context_parts.append(f"{source_label}\n{doc}")
            source_item['source'] = source
            if section:
                source_item['section'] = section
            if subsection:
                source_item['subsection'] = subsection
            if heading:
                source_item['heading'] = heading
            if chunk_index is not None:
                source_item['chunk_index'] = chunk_index

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict


def query_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None):
    """
    Alur utama pemrosesan tanya jawab (non-streaming)

    完整流程：递归检索 → 构建上下文 → 矛盾检测 → 构建Prompt → LLM生成
    """
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ Basis pengetahuan kosong, silakan unggah dokumen terlebih dahulu."

        if progress:
            progress(0.3, desc="Menjalankan pencarian rekursif...")

        all_contexts, all_doc_ids, all_metadata, query_type = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question.lower() for w in ["terbaru", "tahun ini", "saat ini", "baru-baru ini", "baru saja"])

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected, query_type)

        if progress:
            progress(0.8, desc="Menghasilkan jawaban...")

        if model_choice == "siliconflow":
            result = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
        else:
            response = get_session().post(
                "http://localhost:11434/api/chat",
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,
                    "options": OLLAMA_OPTIONS_GENERATION,
                },
                timeout=180,
                headers={'Connection': 'close'}
            )
            response.raise_for_status()
            result = response.json().get("message", {}).get("content", "Gagal mendapatkan jawaban yang valid")

        return process_thinking_content(result)

    except json.JSONDecodeError:
        return "Gagal mengurai respons, silakan coba lagi"
    except Exception as e:
        return f"Kesalahan sistem: {str(e)}"


def stream_answer(question, enable_web_search=False, model_choice="siliconflow", progress=None):
    """Alur utama pemrosesan tanya jawab (streaming, digunakan untuk mode generator Gradio)"""
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            yield "⚠️ Basis pengetahuan kosong, silakan unggah dokumen terlebih dahulu.", "Terjadi kesalahan"
            return

        if progress:
            progress(0.3, desc="Menjalankan pencarian rekursif...")

        all_contexts, all_doc_ids, all_metadata, query_type = recursive_retrieval(
            initial_query=question, enable_web_search=enable_web_search, model_choice=model_choice
        )

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question.lower() for w in ["terbaru", "tahun ini", "saat ini", "baru-baru ini", "baru saja"])

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected, query_type)

        if model_choice == "siliconflow":
            full_answer = call_siliconflow_api(prompt, temperature=0.7, max_tokens=1536)
            yield process_thinking_content(full_answer), "Selesai!"
        else:
            response = get_session().post(
                "http://localhost:11434/api/chat",
                json={
                    "model": OLLAMA_MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "think": False,
                    "options": OLLAMA_OPTIONS_GENERATION,
                },
                timeout=120,
                stream=True
            )
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    chunk = data.get("message", {}).get("content", "")
                    full_answer += chunk
                    yield full_answer, "Menghasilkan jawaban..."

            yield process_thinking_content(full_answer), "Selesai!"

    except Exception as e:
        yield f"Kesalahan sistem: {str(e)}", "Terjadi kesalahan"
