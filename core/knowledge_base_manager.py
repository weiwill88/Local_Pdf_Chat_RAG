"""
知识库管理器 —— 多知识库的创建 / 切换 / 持久化

负责：
- 在 knowledge_bases/ 目录下管理多个独立的知识库
- 协调 vector_store 和 bm25_manager 的保存与加载
- 启动时自动扫描已有知识库
"""

import os
import json
import logging
import shutil

from core.vector_store import vector_store
from core.bm25_index import bm25_manager

BASE_DIR = "knowledge_bases"


class KnowledgeBaseManager:
    """
    多知识库管理器（单例模式）
    所有操作均通过全局 kb_manager 实例进行。
    """

    def __init__(self):
        self.current_kb = None  # 当前激活的知识库名称
        os.makedirs(BASE_DIR, exist_ok=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 知识库列表与状态
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def list_kbs(self):
        """扫描 knowledge_bases/ 目录，返回所有知识库名称列表"""
        if not os.path.isdir(BASE_DIR):
            return []
        return sorted([
            d for d in os.listdir(BASE_DIR)
            if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith(".")
        ])

    def kb_path(self, name):
        """返回知识库的磁盘路径"""
        return os.path.join(BASE_DIR, name)

    def get_kb_info(self, name):
        """获取知识库的摘要信息"""
        path = self.kb_path(name)
        if not os.path.isdir(path):
            return None
        chunks = 0
        meta_path = os.path.join(path, "contents_map.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    chunks = len(json.load(f))
            except Exception:
                pass
        return {
            "name": name,
            "chunks": chunks,
            "path": path,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 保存与加载
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save_current_kb(self, name):
        """将当前内存中的向量库和 BM25 索引保存到指定知识库"""
        path = self.kb_path(name)
        os.makedirs(path, exist_ok=True)
        vector_store.save(path)
        bm25_manager.save(path)
        self.current_kb = name
        logging.info(f"知识库「{name}」已保存（{vector_store.total_chunks} 个文本块）")

    def load_kb(self, name):
        """从磁盘加载知识库到内存（替换当前向量库）"""
        path = self.kb_path(name)
        if not os.path.isdir(path):
            logging.warning(f"知识库不存在: {name}")
            return False

        # 清空当前状态
        vector_store.clear()
        bm25_manager.clear()

        # 加载
        vs_ok = vector_store.load(path)
        bm25_ok = bm25_manager.load(path)

        if vs_ok:
            self.current_kb = name
            logging.info(f"知识库「{name}」已加载（{vector_store.total_chunks} 个文本块）")
            return True
        else:
            self.current_kb = None
            logging.warning(f"知识库「{name}」没有有效的向量索引")
            return False

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 删除
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def delete_kb(self, name):
        """删除指定的知识库"""
        path = self.kb_path(name)
        if not os.path.isdir(path):
            return False
        shutil.rmtree(path)
        if self.current_kb == name:
            self.current_kb = None
            vector_store.clear()
            bm25_manager.clear()
        logging.info(f"知识库「{name}」已删除")
        return True

    def rename_kb(self, old_name, new_name):
        """重命名知识库"""
        old_path = self.kb_path(old_name)
        new_path = self.kb_path(new_name)
        if not os.path.isdir(old_path) or os.path.exists(new_path):
            return False
        os.rename(old_path, new_path)
        if self.current_kb == old_name:
            self.current_kb = new_name
        logging.info(f"知识库「{old_name}」已重命名为「{new_name}」")
        return True


# 全局单例
kb_manager = KnowledgeBaseManager()
