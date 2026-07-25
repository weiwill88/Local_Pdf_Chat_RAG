"""
联网搜索 —— 通过 SerpAPI 获取实时网络信息

学习要点：
- RAG 的"R"不限于本地文档，也可以从网络获取实时信息
- SerpAPI 是 Google 搜索的 API 封装，需要注册获取 API Key
- 网络搜索结果不进入 FAISS 索引，仅作为文本上下文提供给 LLM
"""

import json
import logging
import requests
from config import SERPAPI_KEY, SEARCH_ENGINE, WEB_SEARCH_MAX_RESULTS, WEB_SEARCH_TIMEOUT, is_serpapi_configured


def check_serpapi_key():
    """检查是否配置了有效的 SERPAPI_KEY（含空格容错）"""
    return is_serpapi_configured()


def get_serpapi_status_message():
    """返回 SERPAPI 配置状态的友好提示文本"""
    if check_serpapi_key():
        return "✅ 联网搜索已就绪（SERPAPI_KEY 已配置）"
    return "⚠️ 联网搜索不可用：未配置 SERPAPI_KEY，请在 .env 文件中设置 SERPAPI_KEY"


def serpapi_search(query, num_results=None, timeout=None):
    """
    执行 SerpAPI 搜索

    Raises:
        ValueError: 未配置 API Key
        ConnectionError: 网络连接失败
        TimeoutError: 请求超时
        RuntimeError: API 返回错误（密钥无效、额度用尽等）

    Returns:
        list: 搜索结果列表
    """
    if num_results is None:
        num_results = WEB_SEARCH_MAX_RESULTS
    if timeout is None:
        timeout = WEB_SEARCH_TIMEOUT

    if not SERPAPI_KEY:
        raise ValueError("未设置 SERPAPI_KEY 环境变量")
    try:
        params = {
            "engine": SEARCH_ENGINE, "q": query, "api_key": SERPAPI_KEY,
            "num": num_results, "hl": "zh-CN", "gl": "cn"
        }
        response = requests.get(
            "https://serpapi.com/search", params=params, timeout=timeout
        )
        # 细分 HTTP 状态码错误
        if response.status_code == 401:
            raise RuntimeError(
                "🔑 SerpAPI 密钥无效或已过期，请检查 .env 中 SERPAPI_KEY 是否正确"
            )
        if response.status_code == 403:
            raise RuntimeError(
                "🚫 SerpAPI 账户额度已用尽或无权访问，请检查账户状态"
            )
        if response.status_code == 429:
            raise RuntimeError(
                "⏳ SerpAPI 请求过于频繁，请稍后重试"
            )
        response.raise_for_status()
        return _parse_serpapi_results(response.json())
    except requests.exceptions.Timeout:
        logging.error(f"网络搜索超时（{timeout}秒）")
        raise TimeoutError(f"⏱️ 联网搜索请求超时（{timeout}秒），请检查网络连接后重试")
    except requests.exceptions.ConnectionError:
        logging.error("网络搜索连接失败")
        raise ConnectionError("🔌 无法连接到搜索引擎服务器，请检查网络连接")
    except requests.exceptions.HTTPError as e:
        logging.error(f"网络搜索 HTTP 错误: {str(e)}")
        raise RuntimeError(f"🌐 搜索引擎返回错误（HTTP {response.status_code}），请稍后重试")
    except (ValueError, json.JSONDecodeError) as e:
        logging.error(f"网络搜索响应解析失败: {str(e)}")
        return []


def _parse_serpapi_results(data):
    """解析 SerpAPI 返回的原始数据"""
    results = []
    if "organic_results" in data:
        for item in data["organic_results"]:
            results.append({
                "title": item.get("title"), "url": item.get("link"),
                "snippet": item.get("snippet"), "timestamp": item.get("date")
            })
    if "knowledge_graph" in data:
        kg = data["knowledge_graph"]
        results.insert(0, {
            "title": kg.get("title"), "url": kg.get("source", {}).get("link", ""),
            "snippet": kg.get("description"), "source": "knowledge_graph"
        })
    return results


def search_web(query, num_results=None):
    """
    执行网络搜索（结果不加入 FAISS 索引，仅作为上下文）

    返回值统一为 list，异常由调用方处理。
    """
    try:
        results = serpapi_search(query, num_results)
        if not results:
            logging.info("网络搜索没有返回结果")
        else:
            logging.info(f"网络搜索返回 {len(results)} 条结果")
        return results
    except (ValueError, ConnectionError, TimeoutError, RuntimeError) as e:
        logging.error(f"网络搜索失败: {str(e)}")
        # 将异常转换为包含错误信息的列表，让上游能识别
        return [{"error": True, "message": str(e)}]
    except Exception as e:
        logging.error(f"网络搜索未知错误: {str(e)}")
        return [{"error": True, "message": f"⚠️ 网络搜索发生未知错误: {str(e)}"}]
