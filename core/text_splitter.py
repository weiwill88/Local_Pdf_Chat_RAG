"""
Pembagi Teks (Text Splitter) —— Memecah teks panjang menjadi fragmen yang ramah pencarian

Poin Pembelajaran:
- chunk_size: Jumlah karakter maksimum per fragmen. Jika terlalu besar, granularitas pencarian menjadi kasar; jika terlalu kecil, konteks akan hilang
- chunk_overlap: Jumlah karakter yang tumpang tindih antara fragmen yang berdekatan. Menghindari pemotongan informasi penting
- separators: Pemisah yang dicoba berdasarkan prioritas. Dokumen Bahasa Mandarin harus menyertakan tanda baca Bahasa Mandarin
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_text(text, chunk_size=None, chunk_overlap=None):
    """
    Memecah teks panjang menjadi beberapa fragmen

    Menggunakan RecursiveCharacterTextSplitter untuk pemecahan rekursif:
    Pertama dicoba memecah berdasarkan paragraf, jika fragmen masih terlalu besar maka dipecah berdasarkan kalimat, dan seterusnya.

    Args:
        text: Teks panjang yang akan dipecah
        chunk_size: Jumlah karakter maksimum per fragmen (default menggunakan nilai konfigurasi 400)
        chunk_overlap: Jumlah karakter tumpang tindih antara fragmen yang berdekatan (default menggunakan nilai konfigurasi 40)

    Returns:
        Daftar fragmen teks setelah dipecah
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or CHUNK_SIZE,
        chunk_overlap=chunk_overlap or CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "，", "；", "：", " ", ""]
    )
    return text_splitter.split_text(text)
