"""
论文总结系统 - 论文校验模块

验证 Markdown 文件是否为有效论文格式。
规则: 第一行必须是 "---" (YAML Frontmatter 起始标记)
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("summarizer")


def is_valid_paper(file_path: str | Path) -> bool:
    """
    校验 Markdown 文件是否为有效论文。

    Args:
        file_path: Markdown 文件路径

    Returns:
        True 如果是有效论文, False 否则
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"文件不存在: {file_path}")
        return False

    if not file_path.suffix.lower() == ".md":
        logger.warning(f"不是 Markdown 文件: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            is_valid = first_line == "---"

            if not is_valid:
                logger.info(f"跳过非论文文件 (第一行不是 '---'): {file_path.name}")

            return is_valid

    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        return False


def find_paper_markdown(
    paper_dir: str | Path, output_suffix: str = "_笔记"
) -> Path | None:
    """
    在论文目录中查找主 Markdown 文件。

    策略:
    1. 优先查找与目录同名的 .md 文件
    2. 否则返回第一个有效的论文 .md 文件
    3. 排除已生成的笔记文件 (包含 output_suffix 后缀)

    Args:
        paper_dir: 论文目录路径
        output_suffix: 输出文件后缀，用于排除已生成的笔记

    Returns:
        论文 Markdown 文件路径, 如果未找到返回 None
    """
    paper_dir = Path(paper_dir)

    if not paper_dir.is_dir():
        logger.warning(f"不是目录: {paper_dir}")
        return None

    # 查找所有 .md 文件
    md_files = list(paper_dir.glob("*.md"))

    # 过滤掉笔记文件
    md_files = [f for f in md_files if output_suffix not in f.stem]

    if not md_files:
        logger.warning(f"目录中没有 Markdown 文件: {paper_dir}")
        return None

    # 优先查找与目录同名的文件
    for md_file in md_files:
        if md_file.stem == paper_dir.name:
            if is_valid_paper(md_file):
                return md_file

    # 否则返回第一个有效的论文文件
    for md_file in md_files:
        if is_valid_paper(md_file):
            return md_file

    logger.warning(f"目录中没有有效的论文文件: {paper_dir}")
    return None
