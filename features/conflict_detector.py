"""
Detektor Konflik —— Mendeteksi konflik informasi dari berbagai sumber

Poin Pembelajaran:
- Ketika sistem RAG menggunakan dokumen lokal dan pencarian web secara bersamaan, konflik dapat terjadi di antara berbagai sumber tersebut
- Deteksi konflik membantu LLM menandai perbedaan dalam jawaban, meningkatkan kredibilitas jawaban
"""

import re


def detect_conflicts(sources):
    """Mendeteksi konflik dalam informasi dari berbagai sumber"""
    key_facts = {}
    for item in sources:
        facts = _extract_facts(item['text'] if 'text' in item else item.get('excerpt', ''))
        for fact, value in facts.items():
            if fact in key_facts and key_facts[fact] != value:
                return True
            key_facts[fact] = value
    return False


def _extract_facts(text):
    """Mengekstrak fakta penting dari teks"""
    facts = {}
    numbers = re.findall(r'\b\d{4}(?:年)?|\b\d+%', text)
    if numbers:
        facts['Nilai Penting'] = numbers
    if "peta industri" in text.lower() or "产业图谱" in text:
        facts['Metode Teknologi'] = list(set(re.findall(r'[A-Za-z]+模型|[A-Z]{2,}算法|[A-Za-z]+ model|[A-Za-z]+ algoritma', text.lower())))
    return facts


def evaluate_source_credibility(source):
    """Mengevaluasi kredibilitas sumber (berdasarkan aturan sederhana nama domain)"""
    credibility_scores = {
        "gov.cn": 0.9, "edu.cn": 0.85, "weixin": 0.7, "zhihu": 0.6, "baidu": 0.5,
        "gov": 0.9, "edu": 0.85
    }
    url = source.get('url', '')
    if not url:
        return 0.5
    domain_match = re.search(r'//([^/]+)', url)
    if not domain_match:
        return 0.5
    domain = domain_match.group(1)
    for known_domain, score in credibility_scores.items():
        if known_domain in domain:
            return score
    return 0.5
