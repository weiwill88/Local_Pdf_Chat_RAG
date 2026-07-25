"""
对话历史管理 —— SQLite 持久化存储 + Markdown 导出（【架构升级】独立数据表）

每个知识库拥有独立的 SQLite 数据表 `chat_history_{sanitized_kb}`，
切换知识库时自动切换对应表，确保数据完全物理隔离。

函数列表：
    init_db()                  — 初始化数据库和表结构
    save_message(kb, role, content)  — 保存单条消息
    get_history(kb)            — 获取某知识库的全部历史
    get_history_with_ids(kb)   — 获取历史（含数据库 ID）
    export_to_markdown(kb)     — 导出为 Markdown 字符串
    clear_history(kb)          — 清空某知识库的对话记录
    delete_message(kb, msg_id) — 删除单条消息
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


def _sanitize_kb_name(name: str) -> str:
    """将知识库名称转为安全的 SQLite 表名"""
    safe = re.sub(r'[^a-zA-Z0-9_一-鿿]', '_', str(name))
    if not safe:
        safe = "default"
    return safe


def _table_name(knowledge_base: str) -> str:
    """返回知识库对应的 SQLite 表名"""
    safe = _sanitize_kb_name(knowledge_base)
    return f"chat_history_{safe}"


def _ensure_table(conn, knowledge_base: str):
    """确保知识库对应的数据表存在（幂等）"""
    table = _table_name(knowledge_base)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table} (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
            content         TEXT    NOT NULL,
            timestamp       TEXT    NOT NULL
        )
    """)
    conn.commit()


def _get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库（无需全局表，每 KB 首次使用时按需创建）"""
    # 只创建元数据表，记录所有知识库对应的表名
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_history_registry (
                kb_name     TEXT PRIMARY KEY,
                table_name  TEXT NOT NULL,
                created_at  TEXT NOT NULL
            )
        """)
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"数据库初始化失败: {e}") from e
    finally:
        conn.close()


def _get_or_create_table(knowledge_base: str):
    """获取或创建知识库对应的数据表，返回表名"""
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        _ensure_table(conn, knowledge_base)
        conn.execute(
            "INSERT OR IGNORE INTO chat_history_registry (kb_name, table_name, created_at) VALUES (?, ?, ?)",
            (knowledge_base, table, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"创建知识库数据表失败: {e}") from e
    finally:
        conn.close()
    return table


def save_message(knowledge_base: str, role: str, content: str):
    """
    保存一条对话消息到知识库独立的 SQLite 数据表

    Args:
        knowledge_base: 知识库名称
        role: 'user' 或 'assistant'
        content: 消息内容
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role 必须是 'user' 或 'assistant'，收到: {role}")
    if not content:
        return

    table = _get_or_create_table(knowledge_base)
    conn = _get_conn()
    try:
        conn.execute(
            f"INSERT INTO {table} (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
    except Exception as e:
        raise RuntimeError(f"保存消息失败: {e}") from e
    finally:
        conn.close()


def get_history(knowledge_base: str) -> List[Tuple[str, str, str]]:
    """
    获取指定知识库的全部对话历史（从独立数据表读取）

    Returns:
        List of (role, content, timestamp) tuples
    """
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        # 表可能不存在，不存在视为空历史
        cursor = conn.execute(
            f"SELECT role, content, timestamp FROM {table} ORDER BY timestamp ASC, id ASC"
        )
        rows = [(row["role"], row["content"], row["timestamp"]) for row in cursor.fetchall()]
        return rows
    except sqlite3.OperationalError:
        # 表不存在 → 空历史
        return []
    except Exception as e:
        raise RuntimeError(f"读取历史失败: {e}") from e
    finally:
        conn.close()


def get_history_with_ids(knowledge_base: str) -> List[Tuple[int, str, str, str]]:
    """
    获取指定知识库的全部对话历史（含数据库 ID），从独立数据表读取

    Returns:
        List of (id, role, content, timestamp) tuples
    """
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        cursor = conn.execute(
            f"SELECT id, role, content, timestamp FROM {table} ORDER BY timestamp ASC, id ASC"
        )
        rows = [(row["id"], row["role"], row["content"], row["timestamp"]) for row in cursor.fetchall()]
        return rows
    except sqlite3.OperationalError:
        return []
    except Exception as e:
        raise RuntimeError(f"读取历史失败: {e}") from e
    finally:
        conn.close()


def delete_message(knowledge_base: str, message_id: int) -> bool:
    """
    从知识库的独立数据表中删除单条消息

    Args:
        knowledge_base: 知识库名称
        message_id: 消息的数据库 ID

    Returns:
        True 表示成功删除
    """
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (message_id,))
        conn.commit()
        return cursor.rowcount > 0
    except Exception as e:
        raise RuntimeError(f"删除消息失败: {e}") from e
    finally:
        conn.close()


def delete_message_pair(knowledge_base: str, pair_index: int) -> bool:
    """
    删除指定索引处的用户-助手消息对（从独立数据表操作）

    Args:
        knowledge_base: 知识库名称
        pair_index: 从 0 开始的消息对索引

    Returns:
        True 表示成功删除
    """
    rows = get_history_with_ids(knowledge_base)
    user_indices = [i for i, (_, role, _, _) in enumerate(rows) if role == "user"]
    if pair_index < 0 or pair_index >= len(user_indices):
        return False

    msg_idx = user_indices[pair_index]
    delete_message(knowledge_base, rows[msg_idx][0])
    if msg_idx + 1 < len(rows) and rows[msg_idx + 1][1] == "assistant":
        delete_message(knowledge_base, rows[msg_idx + 1][0])

    return True


def get_message_count(knowledge_base: str) -> int:
    """获取指定知识库的消息总数"""
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        row = cursor.fetchone()
        return row["cnt"] if row else 0
    except sqlite3.OperationalError:
        return 0
    except Exception as e:
        raise RuntimeError(f"获取消息数失败: {e}") from e
    finally:
        conn.close()


def export_to_markdown(knowledge_base: str) -> str:
    """
    将指定知识库的对话记录导出为 Markdown 格式字符串（增强版）

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

    all_sources = set()

    for role, content, ts in rows:
        if role == "user":
            lines.append(f"## 🧑 用户 ({ts})")
        else:
            lines.append(f"## 🤖 助手 ({ts})")
        lines.append("")
        lines.append(content)
        lines.append("")

        for match in re.finditer(r'📄 \*\*来源 \d+: ([^*]+)\*\*', content):
            all_sources.add(match.group(1).strip())

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
    """清空指定知识库的独立数据表中的全部对话记录"""
    table = _table_name(knowledge_base)
    conn = _get_conn()
    try:
        conn.execute(f"DELETE FROM {table}")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # 表不存在视为已清空
    except Exception as e:
        raise RuntimeError(f"清空历史失败: {e}") from e
    finally:
        conn.close()
