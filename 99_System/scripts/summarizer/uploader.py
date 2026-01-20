"""
论文总结系统 - 文件上传模块

收集论文目录下的 Markdown 和图片资源，上传至 Gemini API。
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai

from utils.logger import get_logger

logger = get_logger("summarizer")


def collect_paper_resources(
    paper_dir: str | Path, paper_md: str | Path, image_extensions: list[str] = None
) -> list[Path]:
    """
    收集论文目录下的所有资源文件。

    包括:
    - 指定的 Markdown 文件
    - 目录下的 JPEG/JPG/PNG 图片

    Args:
        paper_dir: 论文目录路径
        paper_md: 论文 Markdown 文件路径
        image_extensions: 图片扩展名列表

    Returns:
        资源文件路径列表
    """
    if image_extensions is None:
        image_extensions = [".jpeg", ".jpg", ".png"]

    paper_dir = Path(paper_dir)
    paper_md = Path(paper_md)

    resources = []

    # 添加 Markdown 文件
    if paper_md.exists():
        resources.append(paper_md)
        logger.info(f"收集 Markdown: {paper_md.name}")

    # 收集图片文件
    for ext in image_extensions:
        for img_file in paper_dir.glob(f"*{ext}"):
            resources.append(img_file)
            logger.info(f"收集图片: {img_file.name}")

    logger.info(f"共收集 {len(resources)} 个资源文件")
    return resources


def upload_to_gemini(client: genai.Client, resources: list[Path]) -> list:
    """
    将资源文件上传至 Gemini API。

    Args:
        client: Gemini API 客户端
        resources: 资源文件路径列表

    Returns:
        上传后的文件对象列表
    """
    uploaded_files = []

    for file_path in resources:
        try:
            logger.info(f"上传文件: {file_path.name}")
            uploaded = client.files.upload(file=str(file_path))
            uploaded_files.append(uploaded)
            logger.info(f"上传成功: {uploaded.name}")
        except Exception as e:
            logger.error(f"上传失败: {file_path.name}, 错误: {e}")

    return uploaded_files


def read_markdown_content(file_path: str | Path) -> str:
    """
    读取 Markdown 文件内容。

    Args:
        file_path: Markdown 文件路径

    Returns:
        文件内容字符串
    """
    file_path = Path(file_path)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.info(f"读取 Markdown 成功: {file_path.name}, 长度: {len(content)} 字符")
        return content
    except Exception as e:
        logger.error(f"读取文件失败: {file_path}, 错误: {e}")
        return ""


def cleanup_uploaded_files(client: genai.Client, uploaded_files: list) -> None:
    """
    清理已上传的文件 (节省存储费用)。

    Args:
        client: Gemini API 客户端
        uploaded_files: 上传后的文件对象列表
    """
    for file in uploaded_files:
        try:
            client.files.delete(name=file.name)
            logger.info(f"已删除上传文件: {file.name}")
        except Exception as e:
            logger.warning(f"删除文件失败: {file.name}, 错误: {e}")
