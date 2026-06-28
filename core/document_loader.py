"""
Pemuat Dokumen —— Ekstraksi Teks Dokumen Multi-Format

Poin Pembelajaran:
- Memahami metode parsing untuk various format dokumen (PDF, Word, Excel, PPT)
- Memahami langkah pertama RAG: mengonversi dokumen tidak terstruktur menjadi teks biasa
"""

import os
import logging


def extract_text(filepath):
    """
    Mengekstrak konten teks biasa dari file

    Format yang didukung: PDF / Word / Excel / PPT / Teks Biasa / Markdown

    Args:
        filepath: Jalur file

    Returns:
        String konten teks yang diekstrak
    """
    file_ext = os.path.splitext(filepath)[1].lower()

    if file_ext == '.pdf':
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # Disable OCR for text-based PDFs
            converter = DocumentConverter(
                format_options={
                    "pdf": PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(filepath)
            return result.document.export_to_markdown()
        except Exception as e:
            logging.error(f"Gagal memproses dokumen PDF menggunakan docling: {e}")
            return ""

    elif file_ext in ['.txt', '.md']:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()

    elif file_ext == '.docx':
        try:
            from docx import Document
            doc = Document(filepath)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            logging.error("Memproses dokumen Word memerlukan instalasi pustaka python-docx")
            return ""

    elif file_ext in ['.xlsx', '.xls']:
        try:
            import pandas as pd
            text = ""
            xl = pd.ExcelFile(filepath)
            for sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                text += f"Lembar Kerja: {sheet_name}\n"
                text += df.to_string(index=False) + "\n\n"
            return text
        except ImportError:
            logging.error("Memproses file Excel memerlukan instalasi pustaka pandas")
            return ""

    elif file_ext == '.pptx':
        try:
            from pptx import Presentation
            prs = Presentation(filepath)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
            return text
        except ImportError:
            logging.error("Memproses file PPT memerlukan instalasi pustaka python-pptx")
            return ""

    else:
        logging.warning(f"Format file tidak didukung: {file_ext}")
        return ""
