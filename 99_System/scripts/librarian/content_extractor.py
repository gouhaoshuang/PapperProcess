# ContentExtractor - 内容提取器
# ============================================================================
# 负责从 Markdown 文本中精准提取标题和摘要部分

import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    """提取的论文内容"""

    title: str  # 论文标题
    abstract: str  # 摘要内容
    raw_header: str  # 原始头部文本 (备用)


class ContentExtractor:
    """从 Markdown 中提取标题和摘要"""

    # 摘要匹配模式 (按优先级排序)
    ABSTRACT_PATTERNS = [
        # ## Abstract 或 # Abstract
        r"(?:^|\n)#+\s*Abstract\s*\n(.*?)(?=\n#+|\n\n\n|\Z)",
        # **Abstract:** 或 **Abstract**
        r"(?:^|\n)\*\*Abstract[:\.\s]*\*\*\s*(.*?)(?=\n\n|\Z)",
        # Abstract: 或 Abstract.
        r"(?i)(?:^|\n)Abstract[:\.\s]+(.*?)(?=\n\n\n|\n#+|\Z)",
        # 直接 Abstract 开头的段落
        r"(?i)(?:^|\n)(Abstract.*?)(?=\n\n\n|\n#+|\Z)",
    ]

    def __init__(self, max_chars: int = 2000):
        """
        初始化提取器

        Args:
            max_chars: 读取的最大字符数 (节省 token)
        """
        self.max_chars = max_chars

    def extract(self, file_path: Path) -> ExtractedContent:
        """
        提取论文内容

        Args:
            file_path: Markdown 文件路径

        Returns:
            提取的内容对象
        """
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk", errors="ignore")

        header = content[: self.max_chars]

        title = self._extract_title(content, file_path.stem)
        abstract = self._extract_abstract(header)

        # 如果没有找到摘要，使用前 500 个单词作为代替
        if not abstract:
            abstract = self._get_first_paragraphs(header, 500)
            logger.warning(f"未找到明确摘要，使用前文替代: {file_path.name}")

        return ExtractedContent(title=title, abstract=abstract, raw_header=header)

    def _extract_title(self, content: str, fallback: str) -> str:
        """
        提取标题 (H1 或文件名)

        Args:
            content: 文件内容
            fallback: 备用标题 (文件名)

        Returns:
            标题字符串
        """
        # 尝试匹配 # 开头的一级标题
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # 清理标题中的 markdown 格式
            title = re.sub(r"\*\*|\*|`", "", title)
            return title

        # 尝试匹配粗体标题
        match = re.search(r"^\*\*(.+?)\*\*", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return fallback

    def _extract_abstract(self, text: str) -> Optional[str]:
        """
        提取摘要段落

        Args:
            text: 文本内容

        Returns:
            摘要文本或 None
        """
        for pattern in self.ABSTRACT_PATTERNS:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                abstract = match.group(1).strip()
                # 清理多余空白
                abstract = re.sub(r"\s+", " ", abstract)
                # 限制长度
                return abstract[:1500]
        return None

    def _get_first_paragraphs(self, text: str, max_words: int) -> str:
        """
        获取前 N 个单词

        Args:
            text: 文本内容
            max_words: 最大单词数

        Returns:
            截取的文本
        """
        # 跳过可能的标题和元数据
        lines = text.split("\n")
        content_lines = []
        in_content = False

        for line in lines:
            # 跳过标题和空行
            if not in_content:
                if line.strip() and not line.startswith("#"):
                    in_content = True
            if in_content:
                content_lines.append(line)

        content = " ".join(content_lines)
        words = content.split()[:max_words]
        return " ".join(words)
