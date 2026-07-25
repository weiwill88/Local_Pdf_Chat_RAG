"""
统一的文档加载器 —— 支持 PDF / DOCX / MD / TXT 多格式解析
返回 LangChain Document 对象列表

用法：
    from utils.document_loader import load_document
    docs = load_document("example.docx")
    text = docs[0].page_content
    source = docs[0].metadata["source"]
"""

import os
import logging
from typing import List

from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    """
    统一文档加载入口，根据文件后缀自动分发到对应解析器

    Args:
        file_path: 文件绝对路径

    Returns:
        List[Document]: LangChain Document 对象列表

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    file_name = os.path.basename(file_path)

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".md": _load_markdown,
        ".markdown": _load_markdown,
        ".txt": _load_txt,
    }

    loader = loaders.get(ext)
    if loader is None:
        raise ValueError(f"不支持的文件格式: {ext}，支持的格式：{list(loaders.keys())}")

    logging.info(f"正在解析文档: {file_name}")
    docs = loader(file_path)

    # 确保每个 Document 都携带基本元数据
    for doc in docs:
        if "source" not in doc.metadata:
            doc.metadata["source"] = file_name
        if "file_path" not in doc.metadata:
            doc.metadata["file_path"] = file_path
        if "file_type" not in doc.metadata:
            doc.metadata["file_type"] = ext.lstrip(".")

    logging.info(f"文档解析完成: {file_name} → {len(docs)} 个 Document")
    return docs


def _load_pdf(file_path: str) -> List[Document]:
    """加载 PDF 文件（沿用现有的 pdfminer 方式）"""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from io import StringIO

        output = StringIO()
        with open(file_path, "rb") as f:
            extract_text_to_fp(f, output)
        text = output.getvalue()

        if not text.strip():
            logging.warning(f"PDF 文件为空或无法提取文本: {file_path}")
            return []

        return [
            Document(
                page_content=text,
                metadata={"source": os.path.basename(file_path), "file_path": file_path, "file_type": "pdf"},
            )
        ]
    except Exception as e:
        logging.error(f"PDF 解析失败: {file_path}, 错误: {e}")
        raise


def _load_docx(file_path: str) -> List[Document]:
    """加载 DOCX 文件（使用 python-docx 库）"""
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(file_path)

        # 提取所有段落文本
        paragraphs = [p.text for p in doc.paragraphs]

        # 提取表格内容
        tables_text = []
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                tables_text.append(" | ".join(cells))

        all_text = "\n".join(paragraphs)
        if tables_text:
            all_text += "\n\n" + "\n".join(tables_text)

        if not all_text.strip():
            logging.warning(f"DOCX 文件内容为空: {file_path}")
            return []

        return [
            Document(
                page_content=all_text,
                metadata={
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": "docx",
                    "paragraphs": len(paragraphs),
                    "tables": len(tables_text),
                },
            )
        ]
    except ImportError:
        logging.error("处理 DOCX 文件需要安装 python-docx 库")
        raise
    except Exception as e:
        logging.error(f"DOCX 解析失败: {file_path}, 错误: {e}")
        raise


def _load_markdown(file_path: str) -> List[Document]:
    """加载 Markdown 文件（直接读取 UTF-8 文本）"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            logging.warning(f"Markdown 文件内容为空: {file_path}")
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(file_path),
                    "file_path": file_path,
                    "file_type": "markdown",
                },
            )
        ]
    except Exception as e:
        logging.error(f"Markdown 读取失败: {file_path}, 错误: {e}")
        raise


def _load_txt(file_path: str) -> List[Document]:
    """加载 TXT 文件（使用 LangChain TextLoader）"""
    try:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()

        # 补充元数据
        for doc in docs:
            doc.metadata.setdefault("source", os.path.basename(file_path))
            doc.metadata.setdefault("file_path", file_path)
            doc.metadata.setdefault("file_type", "txt")

        return docs
    except ImportError:
        # 没有 langchain_community 时使用内置方式
        logging.warning("langchain_community 未安装，使用内置方式读取 TXT")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                return []

            return [
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(file_path),
                        "file_path": file_path,
                        "file_type": "txt",
                    },
                )
            ]
        except Exception as e:
            logging.error(f"TXT 读取失败: {file_path}, 错误: {e}")
            raise
    except Exception as e:
        logging.error(f"TXT 读取失败: {file_path}, 错误: {e}")
        raise
