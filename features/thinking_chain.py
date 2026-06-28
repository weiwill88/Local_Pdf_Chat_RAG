"""
Pemrosesan Chain of Thought (Berpikir Runtut) —— Pemformatan Tag Chain of Thought DeepSeek-R1

Poin Pembelajaran:
- Model DeepSeek-R1 akan mengeluarkan tag <think>...</think> dalam jawabannya yang berisi proses penalaran
- Modul ini mengonversi konten chain of thought menjadi kotak detail HTML yang dapat dilipat
"""

import logging


def process_thinking_content(text):
    """
    Memproses konten yang mengandung tag <think>, mengonversinya menjadi format HTML yang dapat dilipat

    Mengonversi <think>proses penalaran</think> menjadi tag dapat dilipat <details>.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            processed_text = str(text)
        except:
            return "Format konten tidak dapat diproses"
    else:
        processed_text = text

    try:
        while "<think>" in processed_text and "</think>" in processed_text:
            start_idx = processed_text.find("<think>")
            end_idx = processed_text.find("</think>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                thinking_content = processed_text[start_idx + 7:end_idx]
                before = processed_text[:start_idx]
                after = processed_text[end_idx + 8:]
                processed_text = (
                    before +
                    "\n\n<details>\n<summary>Proses Berpikir (Klik untuk memperluas)</summary>\n\n" +
                    thinking_content +
                    "\n\n</details>\n\n" +
                    after
                )

        # 处理其他 HTML 标签，保留 details 和 summary
        processed_html = []
        i = 0
        while i < len(processed_text):
            if (processed_text[i:i + 8] == "<details" or
                    processed_text[i:i + 9] == "</details" or
                    processed_text[i:i + 8] == "<summary" or
                    processed_text[i:i + 9] == "</summary"):
                tag_end = processed_text.find(">", i)
                if tag_end != -1:
                    processed_html.append(processed_text[i:tag_end + 1])
                    i = tag_end + 1
                    continue
            if processed_text[i] == "<":
                processed_html.append("&lt;")
            elif processed_text[i] == ">":
                processed_html.append("&gt;")
            else:
                processed_html.append(processed_text[i])
            i += 1

        processed_text = "".join(processed_html)
    except Exception as e:
        logging.error(f"Terjadi kesalahan saat memproses konten chain of thought: {str(e)}")
        try:
            return text.replace("<", "&lt;").replace(">", "&gt;")
        except:
            return "Terjadi kesalahan saat memproses konten"

    return processed_text
