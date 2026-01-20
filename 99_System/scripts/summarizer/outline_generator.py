"""
论文总结系统 - 大纲生成模块

调用 Gemini API 生成论文 JSON 格式大纲。
"""

import json
import re
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import google.generativeai as genai

from utils.logger import get_logger

logger = get_logger("summarizer")


def load_prompt_template(prompt_file: str | Path) -> str:
    """
    加载提示词模板文件。

    Args:
        prompt_file: 提示词文件路径

    Returns:
        提示词内容
    """
    prompt_file = Path(prompt_file)

    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取 "## 提示词模板" 之后的内容
        if "## 提示词模板" in content:
            content = content.split("## 提示词模板", 1)[1]

        # 移除 "---" 分隔符之后的内容 (如变量说明等)
        if "\n---\n" in content:
            content = content.split("\n---\n")[0]

        return content.strip()
    except Exception as e:
        logger.error(f"加载提示词失败: {prompt_file}, 错误: {e}")
        return ""


def generate_outline(
    model: genai.GenerativeModel,
    uploaded_files: list,
    outline_prompt: str,
    max_retries: int = 3,
) -> dict | None:
    """
    生成论文大纲。

    Args:
        model: Gemini 模型实例
        uploaded_files: 上传的文件列表 (Markdown + 图片)
        outline_prompt: 大纲生成提示词
        max_retries: 最大重试次数

    Returns:
        解析后的 JSON 大纲字典, 失败返回 None
    """
    # 构建请求内容
    contents = uploaded_files + [outline_prompt]

    for attempt in range(max_retries):
        try:
            logger.info(f"生成大纲 (尝试 {attempt + 1}/{max_retries})...")
            response = model.generate_content(contents)

            if not response.text:
                logger.warning("API 返回空响应")
                continue

            # 解析 JSON
            outline = parse_json_response(response.text)

            if outline and validate_outline(outline):
                logger.info(
                    f"大纲生成成功, 包含 {len(outline.get('sections', []))} 个段落"
                )
                return outline
            else:
                logger.warning("大纲验证失败，重试中...")

        except Exception as e:
            logger.error(f"生成大纲失败: {e}")

    logger.error("大纲生成失败，已达最大重试次数")
    return None


def parse_json_response(text: str) -> dict | None:
    """
    解析 API 返回的 JSON 响应。

    支持处理:
    - 纯 JSON 文本
    - 包裹在 Markdown 代码块中的 JSON

    Args:
        text: API 响应文本

    Returns:
        解析后的字典, 失败返回 None
    """
    # 尝试直接解析
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试提取 Markdown 代码块中的 JSON
    json_match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试提取 {...} 块
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error("无法解析 JSON 响应")
    logger.debug(f"原始响应: {text[:500]}...")
    return None


def validate_outline(outline: dict) -> bool:
    """
    验证大纲结构是否完整。

    支持新的 subsections 结构:
    - sections[].subsections[].subtitle
    - sections[].subsections[].key_points

    Args:
        outline: 大纲字典

    Returns:
        True 如果结构有效
    """
    required_fields = ["paper_title", "sections"]

    for field in required_fields:
        if field not in outline:
            logger.warning(f"大纲缺少必要字段: {field}")
            return False

    if not isinstance(outline["sections"], list) or len(outline["sections"]) == 0:
        logger.warning("大纲 sections 为空或格式错误")
        return False

    # 验证每个 section
    for i, section in enumerate(outline["sections"]):
        if "title" not in section:
            logger.warning(f"Section {i} 缺少 title 字段")
            return False

        # 验证 subsections 结构（新格式）
        if "subsections" in section:
            if not isinstance(section["subsections"], list):
                logger.warning(f"Section {i} 的 subsections 格式错误")
                return False
            for j, subsection in enumerate(section["subsections"]):
                if "subtitle" not in subsection:
                    logger.warning(f"Section {i}.subsection {j} 缺少 subtitle 字段")
                    return False

    return True


def save_outline(outline: dict, output_path: str | Path) -> bool:
    """
    保存大纲到 JSON 文件。

    Args:
        outline: 大纲字典
        output_path: 输出文件路径

    Returns:
        True 如果成功, False 否则
    """
    output_path = Path(output_path)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(outline, f, ensure_ascii=False, indent=2)
        logger.info(f"大纲已保存: {output_path}")
        return True
    except Exception as e:
        logger.error(f"保存大纲失败: {e}")
        return False
