"""
对话历史管理 —— SQLite 持久化存储 + Markdown 导出

每个知识库的对话记录互相隔离，按时间排序存储。

函数列表：
    init_db()                  — 初始化数据库和表结构
    save_message(kb, role, content)  — 保存单条消息
    get_history(kb)            — 获取某知识库的全部历史
    get_history_with_ids(kb)   — 获取历史（含数据库 ID）
    export_to_markdown(kb)     — 导出为 Markdown 字符串
    clear_history(kb)          — 清空某知识库的对话记录
    delete_message(msg_id)     — 删除单条消息
    delete_message_pair(kb, idx) — 删除用户-助手消息对
    get_message_count(kb)      — 获取消息数量
"""

import sqlite3
import os
import re
from datetime import datetime
from typing import List, Tuple, Optional

# 数据库文件保存在项目根目录
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chat_history.db")


def _get_conn():
    """获取数据库连接（确保线程安全）"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建 chat_history 表（幂等，可重复调用）"""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                knowledge_base  TEXT    NOT NULL,
                role            TEXT    NOT NULL  CHECK (role IN ('user', 'assistant')),
                content         TEXT    NOT NULL,
                timestamp       TEXT    NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_kb ON chat_history(knowledge_base)")
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"数据库初始化失败: {e}") from e
    finally:
        conn.close()


def save_message(knowledge_base: str, role: str, content: str):
    """
    保存一条对话消息到数据库

    Args:
        knowledge_base: 知识库名称
        role: 'user' 或 'assistant'
        content: 消息内容
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role 必须是 'user' 或 'assistant'，收到: {role}")
    if not content:
        return  # 空内容不保存

    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO chat_history (knowledge_base, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (knowledge_base, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"保存消息失败: {e}") from e
    finally:
        conn.close()


def get_history(knowledge_base: str) -> List[Tuple[str, str, str]]:
    """
    获取指定知识库的全部对话历史，按时间升序排列

    Returns:
        List of (role, content, timestamp) tuples
    """
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT role, content, timestamp FROM chat_history WHERE knowledge_base = ? ORDER BY timestamp ASC, id ASC",
            (knowledge_base,),
        )
        rows = [(row["role"], row["content"], row["timestamp"]) for row in cursor.fetchall()]
        return rows
    except Exception as e:
        raise RuntimeError(f"读取历史失败: {e}") from e
    finally:
        conn.close()


def get_history_with_ids(knowledge_base: str) -> List[Tuple[int, str, str, str]]:
    """
    获取指定知识库的全部对话历史（含数据库 ID），按时间升序排列

    Returns:
        List of (id, role, content, timestamp) tuples
    """
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT id, role, content, timestamp FROM chat_history WHERE knowledge_base = ? ORDER BY timestamp ASC, id ASC",
            (knowledge_base,),
        )
        rows = [(row["id"], row["role"], row["content"], row["timestamp"]) for row in cursor.fetchall()]
        return rows
    except Exception as e:
        raise RuntimeError(f"读取历史失败: {e}") from e
    finally:
        conn.close()


def delete_message(message_id: int) -> bool:
    """
    根据数据库 ID 删除单条消息

    Args:
        message_id: 消息的数据库 ID

    Returns:
        True 表示成功删除
    """
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM chat_history WHERE id = ?", (message_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        raise RuntimeError(f"删除消息失败: {e}") from e
    finally:
        conn.close()


def delete_message_pair(knowledge_base: str, pair_index: int) -> bool:
    """
    删除指定索引处的用户-助手消息对（1 条用户消息 + 1 条助手回复）

    Args:
        knowledge_base: 知识库名称
        pair_index: 从 0 开始的消息对索引

    Returns:
        True 表示成功删除
    """
    rows = get_history_with_ids(knowledge_base)
    # 消息按 user/assistant 交替排列
    user_indices = [i for i, (_, role, _, _) in enumerate(rows) if role == "user"]
    if pair_index < 0 or pair_index >= len(user_indices):
        return False

    msg_idx = user_indices[pair_index]
    # 删除用户消息
    delete_message(rows[msg_idx][0])
    # 如果下一条是 assistant 回复，也删除
    if msg_idx + 1 < len(rows) and rows[msg_idx + 1][1] == "assistant":
        delete_message(rows[msg_idx + 1][0])

    return True


def get_message_count(knowledge_base: str) -> int:
    """获取指定知识库的消息总数"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            "SELECT COUNT(*) as cnt FROM chat_history WHERE knowledge_base = ?",
            (knowledge_base,),
        )
        row = cursor.fetchone()
        return row["cnt"] if row else 0
    except Exception as e:
        raise RuntimeError(f"获取消息数失败: {e}") from e
    finally:
        conn.close()


def export_to_markdown(knowledge_base: str) -> str:
    """
    将指定知识库的对话记录导出为 Markdown 格式字符串（增强版）

    自动提取消息中的参考资料，在文档末尾汇总

    Returns:
        Markdown 格式的完整对话记录
    """
    rows = get_history(knowledge_base)
    if not rows:
        return f"# 对话记录：{knowledge_base}\n\n*暂无对话记录*\n"

    lines = [
        f"# 对话记录：{knowledge_base}",
        "",
        f"> 共 {len(rows)} 条消息 | 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]

    # 收集所有参考资料
    all_sources = set()

    for role, content, ts in rows:
        if role == "user":
            lines.append(f"## 🧑 用户 ({ts})")
        else:
            lines.append(f"## 🤖 助手 ({ts})")
        lines.append("")
        lines.append(content)
        lines.append("")

        # 从回答中提取来源文档名
        for match in re.finditer(r'📄 \*\*来源 \d+: ([^*]+)\*\*', content):
            all_sources.add(match.group(1).strip())

    # 追加参考来源汇总
    if all_sources:
        lines.append("---")
        lines.append("")
        lines.append("## 📚 参考来源汇总")
        lines.append("")
        for src in sorted(all_sources):
            lines.append(f"- {src}")
        lines.append("")

    return "\n".join(lines)


def clear_history(knowledge_base: str):
    """清空指定知识库的全部对话记录"""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM chat_history WHERE knowledge_base = ?", (knowledge_base,))
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"清空历史失败: {e}") from e
    finally:
        conn.close()
