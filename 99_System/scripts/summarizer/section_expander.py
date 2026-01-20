"""
论文总结系统 - 段落扩写模块

根据大纲逐段生成详细笔记内容。
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai

from utils.logger import get_logger

logger = get_logger("summarizer")


def format_subsections_text(subsections: list[dict]) -> str:
    """
    将 subsections 列表格式化为提示词文本。

    Args:
        subsections: 子标题列表，每个包含 subtitle 和 key_points

    Returns:
        格式化后的文本，如:
        - **子标题1**
          - 要点1
          - 要点2
        - **子标题2**
          - 要点1
    """
    lines = []
    for sub in subsections:
        subtitle = sub.get("subtitle", "未命名子标题")
        key_points = sub.get("key_points", [])

        lines.append(f"- **{subtitle}**")
        for point in key_points:
            lines.append(f"  - {point}")

    return "\n".join(lines) if lines else "（无详细子标题）"


def expand_section(
    model: genai.GenerativeModel,
    uploaded_files: list,
    section: dict,
    expansion_prompt_template: str,
) -> str:
    """
    扩写单个段落。

    Args:
        model: Gemini 模型实例
        uploaded_files: 上传的文件列表 (用于图片引用)
        section: 段落信息字典（包含 title, subsections, key_figures, key_formulas）
        expansion_prompt_template: 扩写提示词模板

    Returns:
        生成的 Markdown 内容
    """
    # 格式化 subsections 文本
    subsections = section.get("subsections", [])
    subsections_text = format_subsections_text(subsections)

    # 兼容旧格式：如果没有 subsections 但有 summary，则使用 summary
    if not subsections and section.get("summary"):
        subsections_text = f"- **概述**: {section.get('summary')}"

    # 填充提示词模板变量
    prompt = expansion_prompt_template.format(
        section_title=section.get("title", ""),
        subsections_text=subsections_text,
        key_figures=", ".join(section.get("key_figures", [])) or "无",
        key_formulas="是" if section.get("key_formulas", False) else "否",
    )

    # 构建请求内容
    contents = uploaded_files + [prompt]

    try:
        logger.info(f"扩写段落: {section.get('title', 'Unknown')}")
        response = model.generate_content(contents)

        if response.text:
            logger.info(f"段落扩写成功, 长度: {len(response.text)} 字符")
            return response.text.strip()
        else:
            logger.warning("段落扩写返回空响应")
            return ""

    except Exception as e:
        logger.error(f"段落扩写失败: {e}")
        return ""


def expand_all_sections(
    model: genai.GenerativeModel,
    uploaded_files: list,
    outline: dict,
    expansion_prompt_template: str,
    retry_delay: float = 1.0,
    max_retries: int = 3,
) -> list[dict]:
    """
    扩写所有段落（带报错重试机制）。

    Args:
        model: Gemini 模型实例
        uploaded_files: 上传的文件列表
        outline: 大纲字典
        expansion_prompt_template: 扩写提示词模板
        retry_delay: 请求间隔 (秒)
        max_retries: 单个段落最大重试次数

    Returns:
        包含 title 和 content 的段落列表
    """
    import time

    sections = outline.get("sections", [])
    results = []

    for i, section in enumerate(sections):
        section_title = section.get("title", f"段落 {i + 1}")
        logger.info(f"处理段落 {i + 1}/{len(sections)}: {section_title}")

        content = ""
        # 报错重试机制
        for attempt in range(max_retries + 1):
            content = expand_section(
                model=model,
                uploaded_files=uploaded_files,
                section=section,
                expansion_prompt_template=expansion_prompt_template,
            )

            if content:
                # 成功，跳出重试循环
                break
            elif attempt < max_retries:
                # 失败，等待后重试
                wait_time = retry_delay * (2**attempt)  # 2s, 4s, 8s
                logger.warning(
                    f"段落 '{section_title}' 扩写失败，{wait_time:.0f}秒后重试 ({attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    f"段落 '{section_title}' 扩写失败，已达最大重试次数 ({max_retries})"
                )

        results.append(
            {
                "id": section.get("id", i + 1),
                "title": section_title,
                "content": content,
            }
        )

        # 请求间隔，避免限流
        if i < len(sections) - 1:
            time.sleep(retry_delay)

    logger.info(f"所有段落扩写完成, 共 {len(results)} 个")
    return results
