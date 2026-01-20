"""
论文总结系统 - 汇总合并模块

将大纲和扩写内容合并为最终 Markdown 文档。
"""

import re
import sys
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger("summarizer")


def find_image_in_source(source_content: str, figure_id: str) -> str | None:
    """
    在原始论文 Markdown 中搜索 Figure X. 来定位对应图片。

    匹配逻辑：
    1. 在原始论文中搜索 "Figure X." (X 为图片编号)
    2. 检查该行的前面第二行是否包含 "![](" 标志
    3. 如果有，提取图片路径

    Args:
        source_content: 原始论文 Markdown 内容
        figure_id: 图片编号（如 "1", "2", "12"）

    Returns:
        匹配的图片路径（相对路径），或 None
    """
    lines = source_content.split("\n")

    # 搜索 "Figure X." 模式（支持 Figure 1. 或 **Figure 1.** 格式）
    search_pattern = f"Figure {figure_id}."

    for i, line in enumerate(lines):
        if search_pattern in line:
            # 检查前面第二行（i-2）
            if i >= 2:
                prev_line = lines[i - 2]
                # 检查是否包含 Markdown 图片语法 "![]("
                if "![](" in prev_line:
                    # 提取图片路径：匹配 ![...](...) 格式
                    match = re.search(r"!\[.*?\]\(([^)]+)\)", prev_line)
                    if match:
                        image_path = match.group(1)
                        return image_path
    return None


def replace_image_placeholders(
    content: str, paper_dir: Path, source_content: str = None
) -> str:
    """
    将内容中的 `<Figure X>` 占位符替换为实际的 Markdown 图片语法。

    规则:
    - `<Figure X>` 格式表示需要插入图片，替换为 `![Figure X](./image_file.jpeg)`
    - `(Figure X)` 格式表示引用说明，不替换

    匹配方式:
    - 在原始论文 Markdown 中搜索 "Figure X." 来定位图片
    - 检查该行前面第二行是否包含图片语法 "![]("
    - 如果有则提取图片路径

    Args:
        content: 生成的笔记 Markdown 内容
        paper_dir: 论文所在目录
        source_content: 原始论文 Markdown 内容（用于搜索图片位置）

    Returns:
        替换后的 Markdown 内容
    """
    # 如果没有提供原始论文内容，直接返回
    if not source_content:
        logger.warning("未提供原始论文内容，跳过图片替换")
        return content

    # 匹配 <Figure X> 格式（X 为数字）
    pattern = r"<Figure\s*(\d+)>"

    def replace_match(match: re.Match) -> str:
        figure_id = match.group(1)

        # 在原始论文中搜索对应图片
        image_path = find_image_in_source(source_content, figure_id)

        if image_path:
            # 找到匹配图片，使用相对路径，前后空行确保正确渲染
            logger.debug(f"Figure {figure_id} -> {image_path}")
            return f"\n\n![Figure {figure_id}](./{image_path})\n\n"
        else:
            # 未找到匹配图片，添加警告注释
            logger.warning(f"未找到 Figure {figure_id} 的匹配图片")
            return f"\n\n<!-- Figure {figure_id} 未找到匹配图片 -->\n\n"

    replaced_content = re.sub(pattern, replace_match, content)

    # 清理多余的空行（连续超过 3 个空行压缩为 2 个）
    replaced_content = re.sub(r"\n{4,}", "\n\n\n", replaced_content)

    return replaced_content


def assemble_document(
    outline: dict,
    expanded_sections: list[dict],
    output_path: str | Path,
    paper_dir: Path | None = None,
    source_content: str | None = None,
) -> bool:
    """
    汇总合并生成最终 Markdown 文档。

    Args:
        outline: 大纲字典
        expanded_sections: 扩写后的段落列表
        output_path: 输出文件路径
        paper_dir: 论文目录（用于图片替换），为 None 则从 output_path 推断
        source_content: 原始论文 Markdown 内容（用于图片匹配）

    Returns:
        True 如果成功, False 否则
    """
    output_path = Path(output_path)
    if paper_dir is None:
        paper_dir = output_path.parent

    try:
        # 生成 YAML Frontmatter
        frontmatter = generate_frontmatter(outline)

        # 生成文档内容
        content_parts = [frontmatter]

        # 添加论文标题
        paper_title = outline.get("paper_title", "论文笔记")
        content_parts.append(f"# {paper_title}\n")

        # 添加元信息摘要
        # meta_info = generate_meta_summary(outline)
        # if meta_info:
        # content_parts.append(meta_info)

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

        # 替换图片占位符 <Figure X> -> ![Figure X](./xxx.jpeg)
        final_content = replace_image_placeholders(
            final_content, paper_dir, source_content
        )

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
