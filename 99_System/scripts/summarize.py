"""
论文总结系统 - 主入口脚本

扫描论文目录，生成笔记摘要。

使用方式:
    python summarize.py --paper "论文目录路径"   # 处理单篇
    python summarize.py --limit 5               # 处理前5篇
    python summarize.py                         # 处理所有
"""

import argparse
import os
import sys
from pathlib import Path

from google import genai

from config import CONFIG
from utils.logger import get_logger
from summarizer.validator import is_valid_paper, find_paper_markdown
from summarizer.uploader import (
    collect_paper_resources,
    upload_to_gemini,
    read_markdown_content,
    cleanup_uploaded_files,
)
from summarizer.outline_generator import (
    load_prompt_template,
    generate_outline,
    save_outline,
)
from summarizer.section_expander import expand_all_sections
from summarizer.assembler import assemble_document, get_output_path

# 使用统一日志配置
logger = get_logger("summarizer")


def create_client() -> genai.Client:
    """创建 Gemini API 客户端。"""
    api_key = os.environ.get(CONFIG["gemini"]["api_key_env"])
    if not api_key:
        logger.error(f"未设置环境变量: {CONFIG['gemini']['api_key_env']}")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    logger.info("Gemini API 客户端创建成功")
    return client


def process_paper(client: genai.Client, paper_dir: Path) -> bool:
    """
    处理单篇论文。

    Args:
        client: Gemini API 客户端
        paper_dir: 论文目录路径

    Returns:
        True 如果成功, False 否则
    """
    summarizer_config = CONFIG["summarizer"]
    gemini_config = CONFIG["gemini"]

    # 1. 查找论文 Markdown 文件
    paper_md = find_paper_markdown(paper_dir, summarizer_config["output_suffix"])
    if not paper_md:
        logger.warning(f"跳过目录 (未找到有效论文): {paper_dir.name}")
        return False

    # 2. 检查是否已生成笔记
    output_path = get_output_path(paper_md, summarizer_config["output_suffix"])
    if output_path.exists():
        logger.info(f"跳过 (笔记已存在): {paper_dir.name}")
        return True

    logger.info(f"开始处理论文: {paper_md.name}")

    # 3. 收集资源文件
    resources = collect_paper_resources(
        paper_dir, paper_md, summarizer_config["image_extensions"]
    )

    # 4. 上传文件到 Gemini
    uploaded_files = upload_to_gemini(client, resources)
    if not uploaded_files:
        logger.error("文件上传失败")
        return False

    try:
        # 5. 加载提示词
        prompts_dir = Path(summarizer_config["prompts_dir"])

        system_instruction = load_prompt_template(
            prompts_dir / summarizer_config["system_instruction"]
        )
        outline_prompt = load_prompt_template(
            prompts_dir / summarizer_config["outline_prompt"]
        )
        expansion_prompt = load_prompt_template(
            prompts_dir / summarizer_config["expansion_prompt"]
        )

        # 6. 生成大纲
        outline = generate_outline(
            client=client,
            model_name=gemini_config["model"],
            uploaded_files=uploaded_files,
            outline_prompt=outline_prompt,
            system_instruction=system_instruction,
            temperature=gemini_config["temperature"],
            max_retries=gemini_config["max_retries"],
        )

        if not outline:
            logger.error("大纲生成失败")
            return False

        # 7. 保存大纲 JSON 文件
        outline_path = paper_dir / f"{paper_md.stem}_outline.json"
        save_outline(outline, outline_path)

        # 8. 逐段扩写
        expanded_sections = expand_all_sections(
            client=client,
            model_name=gemini_config["model"],
            uploaded_files=uploaded_files,
            outline=outline,
            expansion_prompt_template=expansion_prompt,
            system_instruction=system_instruction,
            temperature=gemini_config["temperature"],
            retry_delay=summarizer_config["retry_delay"],
            max_retries=gemini_config["max_retries"],
        )

        # 9. 读取原始论文内容（用于图片匹配）
        source_content = read_markdown_content(paper_md)

        # 10. 汇总合并（自动替换图片占位符）
        success = assemble_document(
            outline=outline,
            expanded_sections=expanded_sections,
            output_path=output_path,
            paper_dir=paper_dir,
            source_content=source_content,
        )

        if success:
            logger.info(f"✅ 论文处理完成: {output_path.name}")

        return success

    finally:
        # 11. 清理上传的文件
        cleanup_uploaded_files(client, uploaded_files)


def scan_and_process(client: genai.Client, input_dir: Path, limit: int = None):
    """
    扫描目录并处理所有论文。

    Args:
        client: Gemini API 客户端
        input_dir: 输入目录 (20_Classification)
        limit: 最大处理数量 (None 表示不限制)
    """
    processed = 0
    success = 0

    # 遍历分类目录
    for category_dir in input_dir.iterdir():
        if not category_dir.is_dir():
            continue

        logger.info(f"扫描分类: {category_dir.name}")

        # 遍历论文目录
        for paper_dir in category_dir.iterdir():
            if not paper_dir.is_dir():
                continue

            if limit and processed >= limit:
                logger.info(f"已达到处理数量限制: {limit}")
                break

            try:
                if process_paper(client, paper_dir):
                    success += 1
            except Exception as e:
                logger.error(f"处理失败: {paper_dir.name}, 错误: {e}")

            processed += 1

        if limit and processed >= limit:
            break

    logger.info(f"处理完成: 成功 {success}/{processed}")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="论文总结系统")
    parser.add_argument("--paper", "-p", type=str, help="指定单篇论文目录路径")
    parser.add_argument("--limit", "-l", type=int, default=None, help="最大处理数量")
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        default=None,
        help="输入目录 (默认使用配置中的 input_dir)",
    )

    args = parser.parse_args()

    # 创建 Gemini 客户端
    client = create_client()

    if args.paper:
        # 处理单篇论文
        paper_dir = Path(args.paper)
        if not paper_dir.exists():
            logger.error(f"目录不存在: {args.paper}")
            sys.exit(1)
        process_paper(client, paper_dir)
    else:
        # 扫描并处理所有论文
        input_dir = Path(args.input_dir or CONFIG["summarizer"]["input_dir"])
        if not input_dir.exists():
            logger.error(f"输入目录不存在: {input_dir}")
            sys.exit(1)
        scan_and_process(client, input_dir, args.limit)


if __name__ == "__main__":
    main()
