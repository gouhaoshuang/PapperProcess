# Scanner - 文件扫描器
# ============================================================================
# 负责遍历输入目录，识别未处理的 Markdown 论文文件

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class Scanner:
    """扫描待处理的 Markdown 论文文件"""

    def __init__(self, staging_dir: Path, min_filename_length: int = 10):
        """
        初始化扫描器

        Args:
            staging_dir: 待处理文件所在目录
            min_filename_length: 论文文件名最小长度 (用于过滤非论文文件)
        """
        self.staging_dir = Path(staging_dir)
        self.min_filename_length = min_filename_length

    def scan(self) -> List[Path]:
        """
        扫描并返回所有符合条件的 MD 文件路径列表

        Returns:
            论文 Markdown 文件路径列表
        """
        if not self.staging_dir.exists():
            logger.warning(f"扫描目录不存在: {self.staging_dir}")
            return []

        papers = []
        for item in self.staging_dir.iterdir():
            if item.is_dir():
                # 每个论文在独立子目录中，查找其中的 .md 文件
                md_files = list(item.glob("*.md"))
                for md_file in md_files:
                    if len(md_file.stem) >= self.min_filename_length:
                        papers.append(md_file)
                        logger.debug(f"发现论文: {md_file.name}")

        logger.info(f"共扫描到 {len(papers)} 篇待处理论文")
        return papers

    def get_paper_dir(self, md_file: Path) -> Path:
        """
        获取论文所在目录 (通常是 MD 文件的父目录)

        Args:
            md_file: Markdown 文件路径

        Returns:
            论文目录路径
        """
        return md_file.parent
