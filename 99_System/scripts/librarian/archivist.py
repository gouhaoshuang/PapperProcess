# Archivist - 归档员
# ============================================================================
# 负责创建新目录（如需）并将文件移动到最终位置

import re
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Archivist:
    """负责文件归档和移动"""

    def __init__(self, target_dir: Path):
        """
        初始化归档员

        Args:
            target_dir: 目标分类目录 (20_Classification)
        """
        self.target_dir = Path(target_dir)
        if not self.target_dir.exists():
            self.target_dir.mkdir(parents=True, exist_ok=True)

    def archive(
        self, source_dir: Path, category: str, clean_title: Optional[str] = None
    ) -> Path:
        """
        将整个论文目录移动到分类目录

        Args:
            source_dir: 源论文目录
            category: 目标分类名称
            clean_title: 清洗后的标题 (用于重命名，可选)

        Returns:
            移动后的目录路径
        """
        # 确保分类目录存在
        category_path = self.target_dir / category
        category_path.mkdir(parents=True, exist_ok=True)

        # 确定目标目录名
        if clean_title:
            new_name = self._sanitize_name(clean_title)
        else:
            new_name = self._sanitize_name(source_dir.name)

        dest_path = category_path / new_name

        # 处理重名
        if dest_path.exists():
            dest_path = self._get_unique_path(dest_path)

        # 移动目录
        try:
            shutil.move(str(source_dir), str(dest_path))
            logger.info(f"归档完成: {source_dir.name} -> {category}/{dest_path.name}")
            return dest_path
        except Exception as e:
            logger.error(f"归档失败: {e}")
            raise

    def _sanitize_name(self, name: str) -> str:
        """
        清洗文件/目录名

        Args:
            name: 原始名称

        Returns:
            清洗后的名称
        """
        # 移除 Windows 非法字符
        name = re.sub(r'[<>:"/\\|?*]', "_", name)
        # 移除多余空格和下划线
        name = re.sub(r"[\s_]+", "_", name)
        # 移除首尾下划线
        name = name.strip("_")
        # 截断过长名称 (Windows 路径限制)
        if len(name) > 100:
            name = name[:100].rstrip("_")
        return name or "untitled"

    def _get_unique_path(self, path: Path) -> Path:
        """
        获取唯一路径 (添加数字后缀)

        Args:
            path: 原始路径

        Returns:
            唯一路径
        """
        counter = 1
        stem = path.name
        parent = path.parent

        while path.exists():
            path = parent / f"{stem}_{counter}"
            counter += 1

        return path

    def rollback(self, archived_path: Path, original_dir: Path) -> bool:
        """
        回滚归档操作 (将文件移回原位置)

        Args:
            archived_path: 已归档的路径
            original_dir: 原始目录

        Returns:
            是否成功
        """
        try:
            if archived_path.exists() and not original_dir.exists():
                shutil.move(str(archived_path), str(original_dir))
                logger.info(f"回滚成功: {archived_path} -> {original_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"回滚失败: {e}")
            return False
