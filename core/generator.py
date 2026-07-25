"""
LLM 调用 —— 大模型回答生成（Ollama + SiliconFlow + Magick API）

学习要点：
- Prompt Engineering：如何构建高质量的提示词模板
- 流式输出 vs 非流式输出的区别
- 多模型适配：本地 Ollama、云端 SiliconFlow API 和 Magick API 的对接
- 溯源引用：在回答末尾标注来源文档名和文本块编号
"""

import json
import logging
import re
import requests
from config import (
    SILICONFLOW_API_KEY, SILICONFLOW_API_URL,
    SILICONFLOW_MODEL_NAME, OLLAMA_MODEL_NAME, OLLAMA_NUM_CTX,
    OLLAMA_TEMPERATURE, OLLAMA_TOP_P,
    MAGICK_API_KEY, MAGICK_API_URL, MAGICK_MODEL_NAME
)
from utils.network import get_session
from core.retriever import recursive_retrieval
from core.vector_store import vector_store
from features.conflict_detector import detect_conflicts, evaluate_source_credibility
from features.thinking_chain import process_thinking_content


def _normalize_chat_completions_url(api_url):
    """兼容用户填写 base URL 或完整 chat completions URL。"""
    if not api_url:
        return ""
    url = api_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _extract_openai_compatible_content(result):
    """从 OpenAI-compatible 响应中提取回答文本和推理内容。"""
    if "choices" not in result or not result["choices"]:
        return "API返回结果格式异常"

    message = result["choices"][0].get("message", {})
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")

    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    if reasoning:
        return f"{content}<think>{reasoning}</think>"
    return content


def _call_openai_compatible_api(provider_name, api_key, api_url, model_name,
                                prompt, temperature=0.7, max_tokens=1024,
                                extra_payload=None, timeout=180):
    """调用 OpenAI-compatible Chat Completions API 获取回答。"""
    if not api_key:
        logging.error(f"未设置 {provider_name} API Key")
        return f"错误：未配置 {provider_name} API Key。"
    if not api_url:
        logging.error(f"未设置 {provider_name} API URL")
        return f"错误：未配置 {provider_name} API URL。"

    chat_url = _normalize_chat_completions_url(api_url)
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        if extra_payload:
            payload.update(extra_payload)
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json; charset=utf-8"
        }
        json_payload = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        response = requests.post(chat_url, data=json_payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        return _extract_openai_compatible_content(result)

    except requests.exceptions.Timeout:
        return f"⏱️ {provider_name} 请求超时（{timeout}秒），请检查网络连接后重试。"
    except requests.exceptions.ConnectionError:
        return f"🔌 {provider_name} 网络连接失败，请检查网络设置。"
    except requests.exceptions.HTTPError as e:
        logging.error(f"调用{provider_name} API时出错: {str(e)}")
        return (
            f"调用{provider_name} API时出错: {str(e)}。"
            f"请检查 API Key、API URL 和模型名称 {model_name} 是否可用。"
        )
    except requests.exceptions.RequestException as e:
        logging.error(f"调用{provider_name} API时出错: {str(e)}")
        return f"调用{provider_name} API时出错: {str(e)}"
    except Exception as e:
        logging.error(f"{provider_name} API 未知错误: {str(e)}")
        return f"发生未知错误: {str(e)}"


def call_siliconflow_api(prompt, temperature=0.7, max_tokens=1024):
    """调用 SiliconFlow 云端 API 获取回答"""
    return _call_openai_compatible_api(
        "SiliconFlow",
        SILICONFLOW_API_KEY,
        SILICONFLOW_API_URL,
        SILICONFLOW_MODEL_NAME,
        prompt,
        temperature,
        max_tokens,
        extra_payload={
            "top_p": 0.7,
            "top_k": 50,
            "frequency_penalty": 0.5,
            "n": 1,
            "response_format": {"type": "text"}
        }
    )


def call_magick_api(prompt, temperature=0.7, max_tokens=1024):
    """调用 Magick API 获取回答。"""
    return _call_openai_compatible_api(
        "Magick API",
        MAGICK_API_KEY,
        MAGICK_API_URL,
        MAGICK_MODEL_NAME,
        prompt,
        temperature,
        max_tokens
    )


def call_cloud_api(prompt, model_choice="siliconflow", temperature=0.7, max_tokens=1024):
    """统一调用云端 OpenAI-compatible 模型服务。"""
    if model_choice == "siliconflow":
        return call_siliconflow_api(prompt, temperature, max_tokens)
    if model_choice == "magick":
        return call_magick_api(prompt, temperature, max_tokens)
    raise ValueError(f"未知云端模型服务: {model_choice}")


def call_ollama_api(prompt, model_name=None, num_ctx=None, temperature=None,
                     top_p=None, max_tokens=2048, timeout=180):
    """
    调用本地 Ollama API（支持可配置参数）。

    Args:
        prompt: 输入提示词
        model_name: Ollama 模型名，默认使用 OLLAMA_MODEL_NAME
        num_ctx: 上下文窗口长度
        temperature: 温度 (0~1)
        top_p: 核采样系数
        max_tokens: 最大生成长度
        timeout: 超时秒数

    Returns:
        str: 模型回答文本
    """
    if model_name is None:
        model_name = OLLAMA_MODEL_NAME
    if num_ctx is None:
        num_ctx = OLLAMA_NUM_CTX
    if temperature is None:
        temperature = OLLAMA_TEMPERATURE
    if top_p is None:
        top_p = OLLAMA_TOP_P

    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        }
    }
    try:
        response = get_session().post(
            "http://localhost:11434/api/generate",
            json=payload, timeout=timeout,
            headers={'Connection': 'close'}
        )
        response.raise_for_status()
        result = response.json().get("response", "")
        logging.info(f"Ollama [{model_name}] 生成完成, {len(result)} 字符")
        return result.strip()
    except requests.exceptions.Timeout:
        return f"⏱️ Ollama 请求超时（{timeout}秒），请检查模型大小或增加超时时间。"
    except requests.exceptions.ConnectionError:
        return "🔌 无法连接到 Ollama 服务（localhost:11434），请确认 Ollama 是否已启动。"
    except Exception as e:
        return f"🤖 Ollama 调用失败: {str(e)}"


def call_llm_simple(prompt, model_choice="siliconflow", ollama_model_name=None,
                     ollama_num_ctx=None, ollama_temperature=None, ollama_top_p=None):
    """
    简单的 LLM 调用（用于递归检索中的查询改写判断）

    支持传入 Ollama 参数以保持配置一致。
    """
    if model_choice in ("siliconflow", "magick"):
        result = call_cloud_api(prompt, model_choice)
        result = result.strip() if isinstance(result, str) else result[0].strip()
        if "<think>" in result:
            result = result.split("<think>")[0].strip()
        return result
    elif model_choice == "ollama":
        return call_ollama_api(
            prompt, model_name=ollama_model_name,
            num_ctx=ollama_num_ctx, temperature=ollama_temperature,
            top_p=ollama_top_p, max_tokens=512
        )
    raise ValueError(f"未知模型选择: {model_choice}")


def _build_prompt(question, context, enable_web_search, knowledge_base_exists,
                  time_sensitive, conflict_detected, has_web_results=False):
    """构建提示词（明确区分本地文档与网络检索来源）"""
    prompt_template = """作为一个专业的问答助手，你需要基于以下{context_type}回答用户问题。

提供的参考内容：
{context}

用户问题：{question}

请遵循以下回答原则：
1. 仅基于提供的参考内容回答问题，不要使用你自己的知识
2. 如果参考内容中没有足够信息，请坦诚告知你无法回答
3. 回答应该全面、准确、有条理，并使用适当的段落和结构
4. 请用中文回答
5. {source_instruction}{time_instruction}{conflict_instruction}

请现在开始回答："""

    # 确定来源类型描述
    if has_web_results and knowledge_base_exists:
        context_type = "【本地文档参考】和【网络检索参考】资料"
        source_instruction = "对于参考内容，请明确区分【本地文档参考】来源和【网络检索参考】来源，在回答中标注哪些信息来自本地文档、哪些来自网络搜索。回答末尾按'【本地文档参考】'和'【网络检索参考】'两类分别列出参考资料。"
    elif has_web_results:
        context_type = "【网络检索参考】资料"
        source_instruction = "所有信息均来自网络搜索，请在回答末尾列出网络来源链接。"
    elif knowledge_base_exists:
        context_type = "【本地文档参考】"
        source_instruction = "所有信息均来自上传的本地文档，请在回答末尾标注信息来源文档名。"
    else:
        context_type = "空知识库"
        source_instruction = ""

    return prompt_template.format(
        context_type=context_type,
        context=context if context else (
            "知识库为空，请基于你自己的知识回答。" if not enable_web_search else "网络搜索结果将用于回答。"),
        question=question,
        source_instruction=source_instruction,
        time_instruction="优先使用最新的信息。" if time_sensitive and enable_web_search else "",
        conflict_instruction="如果不同来源之间存在差异或矛盾，请明确指出。" if conflict_detected else ""
    )


def _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search):
    """构建上下文和来源信息（区分【本地文档参考】/【网络检索参考】）"""
    context_parts = []
    sources_for_conflict = []
    local_count, web_count = 0, 0

    for idx, (doc, doc_id, metadata) in enumerate(zip(all_contexts, all_doc_ids, all_metadata)):
        source_type = metadata.get('source_type', 'local')
        source_item = {'text': doc, 'type': source_type}

        if source_type == 'web':
            web_count += 1
            url = metadata.get('url', '未知URL')
            title = metadata.get('title', '未知标题')
            context_parts.append(f"[【网络检索参考】来源 {web_count}: {title}] (URL: {url})\n{doc}")
            source_item['url'] = url
            source_item['title'] = title
            source_item['source'] = "【网络检索参考】"
        else:
            local_count += 1
            source = metadata.get('source', '未知来源')
            chunk_label = metadata.get('chunk_id', doc_id)
            context_parts.append(f"[【本地文档参考】来源 {local_count}: {source}] [块: {chunk_label}]\n{doc}")
            source_item['source'] = f"【本地文档参考】{source}"

        sources_for_conflict.append(source_item)

    return "\n\n".join(context_parts), sources_for_conflict


def _build_citation_section(sources_info):
    """
    构建溯源引用 HTML 片段（区分【本地文档参考】/【网络检索参考】）

    在回答末尾追加参考资料列表，每个引用包含：
    - 来源类型徽章（本地文档参考/网络检索参考）
    - 来源文档名
    - 文本块编号
    - 可折叠的原文预览

    Args:
        sources_info: [{"source": ..., "chunk_id": ..., "preview": ..., "source_type": ...}]

    Returns:
        HTML 格式的引用片段
    """
    if not sources_info:
        return ""

    local_citations = []
    web_citations = []
    error_msg = None
    seen_local = {}

    for idx, src in enumerate(sources_info, 1):
        source_type = src.get("source_type", "local")
        source_name = src.get("source", "未知来源")
        preview = src.get("preview", "")

        # 联网搜索失败消息单独处理
        if source_type == "web_error":
            error_msg = preview
            continue

        escaped_preview = preview.replace("<", "&lt;").replace(">", "&gt;")

        if source_type == "web":
            title = src.get("title", source_name)
            web_citations.append(
                f'  - 🌐 搜索结果 [{idx}]: <details><summary>{title}</summary>\n\n{escaped_preview}\n\n</details>'
            )
        else:
            if source_name not in seen_local:
                seen_local[source_name] = len(seen_local) + 1
            local_citations.append(
                f'  - 📄 片段 [{idx}]: '
                f'<details><summary>查看原文片段</summary>\n\n{escaped_preview}\n\n</details>'
            )

    parts = []
    if local_citations:
        parts.append("**▶ 本地知识库参考**\n" + "\n".join(local_citations))
    if web_citations:
        parts.append("**▶ 网络检索补充内容**\n" + "\n".join(web_citations))
    if error_msg:
        parts.append(f"**⚠️ 网络搜索提示**\n\n> {error_msg}")

    return "\n\n---\n\n" + "\n\n".join(parts)


def query_answer(question, enable_web_search=False, model_choice="siliconflow",
                 progress=None, alpha=None, use_parent_retriever=True,
                 use_mmr=True, mmr_lambda=None,
                 ollama_model_name=None, ollama_num_ctx=None,
                 ollama_temperature=None, ollama_top_p=None):
    """
    问答处理主流程（非流式）

    完整流程：递归检索 → 构建上下文 → 矛盾检测 → 构建Prompt → LLM生成 → 溯源引用

    Args:
        question: 用户问题
        enable_web_search: 是否启用联网搜索
        model_choice: 模型选择
        progress: 进度回调
        alpha: 混合检索权重（0=纯BM25, 1=纯向量）
        use_parent_retriever: 是否启用父文档检索
        use_mmr: 是否启用 MMR 重排序
        mmr_lambda: MMR 多样性参数
        ollama_model_name: Ollama 具体模型名（下拉框选择）
        ollama_num_ctx: Ollama 上下文窗口
        ollama_temperature: Ollama 温度
        ollama_top_p: Ollama top_p

    Returns:
        str: 回答文本（含溯源引用）
    """
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            return "⚠️ 知识库为空，请先上传文档。"

        if progress:
            progress(0.3, desc="执行递归检索...")

        # 检索阶段
        try:
            all_contexts, all_doc_ids, all_metadata, sources_info = recursive_retrieval(
                initial_query=question, enable_web_search=enable_web_search,
                model_choice=model_choice, alpha=alpha,
                use_parent_retriever=use_parent_retriever,
                use_mmr=use_mmr, mmr_lambda=mmr_lambda,
                ollama_model_name=ollama_model_name,
                ollama_num_ctx=ollama_num_ctx,
                ollama_temperature=ollama_temperature,
                ollama_top_p=ollama_top_p,
            )
        except Exception as e:
            logging.error(f"检索阶段失败: {str(e)}")
            return f"🔍 文档检索失败: {str(e)}。请确认知识库包含有效内容。"

        if not all_contexts and not enable_web_search:
            return "⚠️ 未在知识库中找到相关答案，请尝试换一种方式提问或上传更多文档。"

        if not all_contexts and enable_web_search:
            return "⚠️ 本地知识库和网络搜索均未找到有效内容，请尝试换一种方式提问。"

        # 构建上下文
        try:
            context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        except Exception as e:
            return f"📄 上下文构建失败: {str(e)}"

        # 检测是否有网络检索结果
        has_web_results = any(
            m.get("source_type") == "web" or m.get("source") == "【网络检索参考】"
            for m in all_metadata
        )

        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])

        # 构建 Prompt
        try:
            prompt = _build_prompt(question, context, enable_web_search,
                                   knowledge_base_exists, time_sensitive, conflict_detected,
                                   has_web_results=has_web_results)
        except Exception as e:
            return f"📝 提示词构建失败: {str(e)}"

        if progress:
            progress(0.8, desc="生成回答...")

        # 调用 LLM
        try:
            if model_choice in ("siliconflow", "magick"):
                result = call_cloud_api(prompt, model_choice, temperature=0.7, max_tokens=1536)
            elif model_choice == "ollama":
                result = call_ollama_api(
                    prompt, model_name=ollama_model_name,
                    num_ctx=ollama_num_ctx, temperature=ollama_temperature,
                    top_p=ollama_top_p, max_tokens=2048
                )
            else:
                return f"错误：未知模型选择 {model_choice}"
        except requests.exceptions.Timeout:
            return "⏱️ 模型请求超时（180秒），请稍后重试或选择其他模型。"
        except requests.exceptions.ConnectionError:
            return "🔌 无法连接到模型服务，请检查模型配置和网络连接。"
        except Exception as e:
            return f"🤖 模型生成失败: {str(e)}。请检查模型配置。"

        # 处理思考链
        answer = process_thinking_content(result)

        # 构建来源摘要头部
        local_count = sum(1 for m in all_metadata if m.get("source_type") == "local")
        web_count = sum(1 for m in all_metadata if m.get("source_type") == "web")
        has_error = any(m.get("source_type") == "web_error" for m in all_metadata)
        source_summary_parts = []
        if local_count > 0:
            source_summary_parts.append(f"▶ 本地知识库参考：{local_count} 个相关片段")
        if web_count > 0:
            source_summary_parts.append(f"▶ 网络检索补充内容：{web_count} 条搜索结果")
        if source_summary_parts:
            answer = f"> **📋 回答来源分区**\n\n" + "\n".join(source_summary_parts) + "\n\n---\n\n" + answer

        # 追加溯源引用
        if all_doc_ids and sources_info:
            citation = _build_citation_section(sources_info)
            answer += citation

        return answer

    except json.JSONDecodeError:
        return "响应解析失败，请重试"
    except MemoryError:
        return "⚠️ 内存不足，请减少文档大小或重启应用"
    except Exception as e:
        return f"系统错误: {str(e)}"


def stream_answer(question, enable_web_search=False, model_choice="siliconflow",
                  progress=None, alpha=None, use_parent_retriever=True,
                  use_mmr=True, mmr_lambda=None):
    """问答处理主流程（流式，用于 Gradio generator 模式）"""
    try:
        knowledge_base_exists = vector_store.is_ready
        if not knowledge_base_exists and not enable_web_search:
            yield "⚠️ 知识库为空，请先上传文档。", "遇到错误"
            return

        if progress:
            progress(0.3, desc="执行递归检索...")

        try:
            all_contexts, all_doc_ids, all_metadata, sources_info = recursive_retrieval(
                initial_query=question, enable_web_search=enable_web_search,
                model_choice=model_choice, alpha=alpha,
                use_parent_retriever=use_parent_retriever,
                use_mmr=use_mmr, mmr_lambda=mmr_lambda,
                ollama_model_name=ollama_model_name,
                ollama_num_ctx=ollama_num_ctx,
                ollama_temperature=ollama_temperature,
                ollama_top_p=ollama_top_p,
            )
        except Exception as e:
            yield f"🔍 文档检索失败: {str(e)}", "遇到错误"
            return

        context, sources = _build_context(all_contexts, all_doc_ids, all_metadata, enable_web_search)
        conflict_detected = detect_conflicts(sources)
        time_sensitive = any(w in question for w in ["最新", "今年", "当前", "最近", "刚刚"])
        has_web_results = any(
            m.get("source_type") == "web" or m.get("source") == "【网络检索参考】"
            for m in all_metadata
        )

        prompt = _build_prompt(question, context, enable_web_search,
                               knowledge_base_exists, time_sensitive, conflict_detected,
                               has_web_results=has_web_results)

        if model_choice in ("siliconflow", "magick"):
            full_answer = call_cloud_api(prompt, model_choice, temperature=0.7, max_tokens=1536)
            answer = process_thinking_content(full_answer)
            if all_doc_ids and sources_info:
                answer += _build_citation_section(sources_info)
            yield answer, "完成!"
        elif model_choice == "ollama":
            response = get_session().post(
                "http://localhost:11434/api/generate",
                json={"model": OLLAMA_MODEL_NAME, "prompt": prompt, "stream": True},
                timeout=120, stream=True
            )
            full_answer = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode()).get("response", "")
                    full_answer += chunk
                    if "<think>" in full_answer and "</think>" in full_answer:
                        yield process_thinking_content(full_answer), "生成回答中..."
                    else:
                        yield full_answer, "生成回答中..."

            answer = process_thinking_content(full_answer)
            if all_doc_ids and sources_info:
                answer += _build_citation_section(sources_info)
            yield answer, "完成!"
        else:
            yield f"错误：未知模型选择 {model_choice}", "遇到错误"

    except Exception as e:
        yield f"系统错误: {str(e)}", "遇到错误"
