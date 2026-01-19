# CategoryManager - 动态类别管理器
# ============================================================================
# 负责扫描目标库，维护当前的分类树

from pathlib import Path
from typing import List, Set
import logging

logger = logging.getLogger(__name__)


class CategoryManager:
    """管理目标库的分类结构，支持动态类别感知"""

    def __init__(self, target_dir: Path):
        """
        初始化类别管理器

        Args:
            target_dir: 分类目标目录 (20_Classification)
        """
        self.target_dir = Path(target_dir)
        self._categories: Set[str] = set()
        self.refresh()

    def refresh(self) -> None:
        """刷新类别列表 (扫描一级子目录)"""
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建分类目录: {self.target_dir}")

        self._categories = {
            d.name
            for d in self.target_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        }
        logger.info(f"当前类别数量: {len(self._categories)}")

    def get_categories(self) -> List[str]:
        """
        返回当前所有类别名称 (已排序)

        Returns:
            类别名称列表
        """
        return sorted(self._categories)

    def get_categories_with_count(self) -> List[tuple]:
        """
        返回类别及其包含的论文数量

        Returns:
            (类别名, 论文数量) 元组列表
        """
        result = []
        for cat in sorted(self._categories):
            cat_path = self.target_dir / cat
            count = len(list(cat_path.iterdir())) if cat_path.exists() else 0
            result.append((cat, count))
        return result

    def ensure_category(self, name: str) -> Path:
        """
        确保类别目录存在，返回路径

        Args:
            name: 类别名称

        Returns:
            类别目录路径
        """
        category_path = self.target_dir / name
        if not category_path.exists():
            category_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"创建新类别目录: {name}")
        self._categories.add(name)
        return category_path

    def has_category(self, name: str) -> bool:
        """
        检查类别是否存在

        Args:
            name: 类别名称

        Returns:
            是否存在
        """
        return name in self._categories
