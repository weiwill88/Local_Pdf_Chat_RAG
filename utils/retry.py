"""
API 调用重试与超时工具

提供带指数退避的重试机制和安全 API 调用包装，
确保网络波动时系统有优雅的容错表现。
"""

import logging
import time
import functools
from typing import Tuple, Any, Optional


def with_retry(func=None, max_retries=3, initial_delay=1.0, backoff_factor=2.0,
               retryable_exceptions=None):
    """
    带指数退避的重试装饰器

    当函数抛出 retryable_exceptions 中的异常时，
    按指数退避策略重试（1s, 2s, 4s...），最多 max_retries 次。

    Args:
        func: 被装饰的函数
        max_retries: 最大重试次数
        initial_delay: 首次重试延迟（秒）
        backoff_factor: 延迟倍增因子
        retryable_exceptions: 可重试的异常元组

    Usage:
        @with_retry(max_retries=3)
        def fetch_data(url):
            return requests.get(url, timeout=10)
    """
    if retryable_exceptions is None:
        try:
            import requests
            retryable_exceptions = (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
            )
        except ImportError:
            retryable_exceptions = (TimeoutError, ConnectionError, OSError)

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay

            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logging.warning(
                            f"{fn.__name__} 第 {attempt + 1}/{max_retries} 次重试失败: {e}。"
                            f"{delay:.1f}s 后重试..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logging.error(
                            f"{fn.__name__} 重试 {max_retries} 次后仍然失败: {e}"
                        )
                except Exception as e:
                    # 非可重试异常直接抛出
                    raise e

            return last_exception

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def safe_api_call(func, *args, user_friendly_message: str = "API 调用异常",
                  default_return: Any = None, **kwargs) -> Tuple[Any, Optional[str]]:
    """
    安全 API 调用包装

    捕获调用过程中的所有异常，返回 (result, error_message)
    正常时 error_message 为 None，异常时 result 为 default_return

    Args:
        func: 要调用的函数
        user_friendly_message: 用户友好的错误消息前缀
        default_return: 异常时的默认返回值
        *args, **kwargs: 传递给 func 的参数

    Returns:
        (result, error) 元组
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_msg = f"{user_friendly_message}: {str(e)}"
        logging.error(error_msg)
        return default_return, error_msg
