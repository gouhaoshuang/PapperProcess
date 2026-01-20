"""
论文总结系统 - 汇总合并模块

将大纲和扩写内容合并为最终 Markdown 文档。
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("summarizer")


def assemble_document(
    outline: dict, expanded_sections: list[dict], output_path: str | Path
) -> bool:
    """
    汇总合并生成最终 Markdown 文档。

    Args:
        outline: 大纲字典
        expanded_sections: 扩写后的段落列表
        output_path: 输出文件路径

    Returns:
        True 如果成功, False 否则
    """
    output_path = Path(output_path)

    try:
        # 生成 YAML Frontmatter
        frontmatter = generate_frontmatter(outline)

        # 生成文档内容
        content_parts = [frontmatter]

        # 添加论文标题
        paper_title = outline.get("paper_title", "论文笔记")
        content_parts.append(f"# {paper_title}\n")

        # 添加元信息摘要
        meta_info = generate_meta_summary(outline)
        if meta_info:
            content_parts.append(meta_info)

        # 添加各段落内容
        for section in expanded_sections:
            title = section.get("title", "")
            content = section.get("content", "")

            # 添加段落标题
            content_parts.append(f"\n## {title}\n")

            # 添加段落内容
            if content:
                content_parts.append(content)
            else:
                content_parts.append("*（内容生成失败）*")

        # 合并所有内容
        final_content = "\n".join(content_parts)

        # 写入文件
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        logger.info(f"文档生成成功: {output_path}")
        logger.info(f"文档长度: {len(final_content)} 字符")
        return True

    except Exception as e:
        logger.error(f"文档生成失败: {e}")
        return False


def generate_frontmatter(outline: dict) -> str:
    """
    生成 YAML Frontmatter。

    Args:
        outline: 大纲字典

    Returns:
        YAML Frontmatter 字符串
    """
    lines = ["---"]

    # 标题
    title = outline.get("paper_title", "论文笔记")
    lines.append(f'title: "{title}"')

    # 作者
    authors = outline.get("authors", "")
    if authors:
        lines.append(f'authors: "{authors}"')

    # 年份
    year = outline.get("year", "")
    if year:
        lines.append(f'year: "{year}"')

    # 生成日期
    created = datetime.now().strftime("%Y-%m-%d")
    lines.append(f'created: "{created}"')

    # 标签
    lines.append("tags: [paper-notes, auto-generated]")

    # 状态
    lines.append('status: "generated"')

    lines.append("---\n")

    return "\n".join(lines)


def generate_meta_summary(outline: dict) -> str:
    """
    生成元信息摘要区块。

    Args:
        outline: 大纲字典

    Returns:
        元信息摘要 Markdown 字符串
    """
    parts = []

    authors = outline.get("authors", "")
    year = outline.get("year", "")
    one_liner = outline.get("one_liner", "")

    if authors or year or one_liner:
        parts.append("> **论文信息**")
        if authors:
            parts.append(f"> - **作者**: {authors}")
        if year:
            parts.append(f"> - **年份**: {year}")
        if one_liner:
            parts.append(f"> - **核心贡献**: {one_liner}")
        parts.append("")

    return "\n".join(parts) if parts else ""


def get_output_path(paper_md: str | Path, output_suffix: str = "_笔记") -> Path:
    """
    根据原论文路径生成输出文件路径。

    Args:
        paper_md: 原论文 Markdown 文件路径
        output_suffix: 输出文件后缀

    Returns:
        输出文件路径
    """
    paper_md = Path(paper_md)
    output_name = paper_md.stem + output_suffix + ".md"
    return paper_md.parent / output_name
