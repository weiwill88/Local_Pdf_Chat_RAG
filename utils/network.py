"""
Utilitas Jaringan —— Manajemen Sesi HTTP, Deteksi Port
"""

import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = None


def get_session():
    """Mendapatkan HTTP Session dengan mekanisme coba ulang (Singleton)"""
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        _session.mount('http://', HTTPAdapter(max_retries=retries))
    return _session


def is_port_available(port):
    """Memeriksa apakah port tersedia"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(('127.0.0.1', port)) != 0
