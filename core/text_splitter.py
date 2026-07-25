"""
文本分块器 —— 将长文本切分为检索友好的片段

学习要点：
- chunk_size：每个片段的最大字符数。过大则检索粒度粗，过小则上下文缺失
- chunk_overlap：相邻片段的重叠字符数。避免关键信息被切断
- separators：按优先级尝试的分割符。中文文档应包含中文标点

Parent Document Retriever 双层分块：
- 父文档块（parent_chunks）：大块保存完整上下文，供 LLM 阅读
- 子文档块（child_chunks）：极小粒度存入向量库，用于相似度检索
- 检索命中子块后向上映射回父块，保留完整语义
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP


def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    将长文本切分为多个片段（单层分块，兼容旧调用）

    使用 RecursiveCharacterTextSplitter 递归切分：
    先尝试按段落分割，若片段仍过大则按句子分割，以此类推。

    Args:
        text: 待切分的长文本
        chunk_size: 每个片段的最大字符数（默认使用配置值 400）
        chunk_overlap: 相邻片段的重叠字符数（默认使用配置值 40）

    Returns:
        切分后的文本片段列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)


def split_text_parent_child(text, parent_chunk_size=None, parent_overlap=None,
                             child_chunk_size=None, child_overlap=None):
    """
    双层分块：大块父文档 + 小块子文档

    先切出较大的父文档块作为完整上下文单元，
    再将每个父块内部细分为小子块用于向量检索。
    子块元数据中记录 parent_index，检索命中子块后可向上映射回父块。

    Args:
        text: 待切分的长文本
        parent_chunk_size: 父块大小（默认 800）
        parent_overlap: 父块重叠（默认 80）
        child_chunk_size: 子块大小（默认 200）
        child_overlap: 子块重叠（默认 20）

    Returns:
        parent_chunks: List[str] 父文档块列表
        child_chunks: List[str] 子文档块列表
        child_to_parent: List[int] 子块→父块索引映射（长度=子块数，值=父块索引）
        parent_ids: List[str] 每个父块对应的 ID 列表
        child_ids: List[str] 每个子块对应的 ID 列表
    """
    p_size = parent_chunk_size or PARENT_CHUNK_SIZE
    p_overlap = parent_overlap or PARENT_CHUNK_OVERLAP
    c_size = child_chunk_size or CHILD_CHUNK_SIZE
    c_overlap = child_overlap or CHILD_CHUNK_OVERLAP

    # 1. 切父块
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=p_size,
        chunk_overlap=p_overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    parent_chunks = parent_splitter.split_text(text)

    # 2. 对每个父块内部切子块
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )

    child_chunks = []
    child_to_parent = []
    parent_ids = []
    child_ids = []

    for p_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"parent_{p_idx}"
        parent_ids.append(parent_id)

        # 对该父块内部切子块
        inner_children = child_splitter.split_text(parent_text)
        for c_idx, child_text in enumerate(inner_children):
            child_chunks.append(child_text)
            child_to_parent.append(p_idx)
            child_ids.append(f"{parent_id}_child_{c_idx}")

    # 如果父块没有切出任何子块（极短文本），至少保留一个子块
    if not child_chunks and parent_chunks:
        child_chunks = parent_chunks
        child_to_parent = list(range(len(parent_chunks)))
        child_ids = [f"parent_{i}_child_0" for i in range(len(parent_chunks))]

    return parent_chunks, child_chunks, child_to_parent, parent_ids, child_ids
