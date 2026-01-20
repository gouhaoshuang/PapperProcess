"""
Markdown 解析器

从 Markdown 文件中提取图片引用。
"""

import re
from pathlib import Path


def extract_image_lines(md_content: str) -> list[tuple[int, str, str, str]]:
    """
    从 Markdown 内容中提取图片引用行。

    Args:
        md_content: Markdown 文件内容

    Returns:
        列表 [(行索引(0-based), 原始行内容, 图片文件名, 原始引用格式), ...]
        格式类型:
        - 'markdown': ![alt](path)
        - 'obsidian': ![[path]]
        - 'img_tag': <img src="path" .../>
    """
    image_refs = []
    lines = md_content.split("\n")

    for line_idx, line in enumerate(lines):
        if ".jpeg" in line.lower() or ".png" in line.lower() or ".jpg" in line.lower():
            # 匹配各种图片引用格式
            patterns = [
                # ![alt](./path.jpeg) 或 ![alt](path.jpeg)
                (r"(!\[.*?\]\()(\.?/?[^)]+\.(jpeg|jpg|png))(\))", "markdown"),
                # ![[path.jpeg]] 或 ![[path.jpeg|size]]
                (r"(!\[\[)(\.?/?[^\]|]+\.(jpeg|jpg|png))([^\]]*\]\])", "obsidian"),
                # <img src="path.jpeg" ... />
                (r'<img[^>]+src=["\']\.?/?([^"\']+\.(jpeg|jpg|png))["\']', "img_tag"),
            ]

            for pattern, fmt in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if fmt == "img_tag":
                        # <img> 标签，提取图片名
                        image_name = match.group(1).lstrip("./")
                        image_refs.append((line_idx, line, image_name, "img_tag"))
                    else:
                        # 提取图片路径 (group 2 是路径)
                        image_path_raw = match.group(2)
                        image_name = image_path_raw.lstrip("./")
                        image_refs.append((line_idx, line, image_name, fmt))
                    break

    return image_refs


def create_img_tag(image_name: str, width: int) -> str:
    """
    创建带 width 属性的 <img> 标签。

    Args:
        image_name: 图片文件名
        width: 宽度值

    Returns:
        HTML img 标签字符串
    """
    return f'<img src="{image_name}" width="{width}" />'


def replace_image_reference(line: str, image_name: str, new_tag: str, fmt: str) -> str:
    """
    替换行中的图片引用为新标签。

    Args:
        line: 原始行内容
        image_name: 图片文件名
        new_tag: 新的 img 标签
        fmt: 原始格式类型 ('markdown' 或 'obsidian')

    Returns:
        替换后的行内容
    """
    if fmt == "markdown":
        # 替换 ![...](path) 格式
        pattern = r"!\[.*?\]\(\.?/?" + re.escape(image_name) + r"\)"
    elif fmt == "obsidian":
        # 替换 ![[path]] 格式
        pattern = r"!\[\[\.?/?" + re.escape(image_name) + r"[^\]]*\]\]"
    elif fmt == "img_tag":
        # 替换 <img src="..." .../> 格式
        pattern = r'<img[^>]+src=["\']' + re.escape(image_name) + r'["\'][^>]*/?>'
    else:
        return line

    return re.sub(pattern, new_tag, line)
