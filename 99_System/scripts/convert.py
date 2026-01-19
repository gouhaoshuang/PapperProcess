#!/usr/bin/env python3
"""
PDF 转换工具 - 主入口
====================

简洁的命令行接口，用于转换 PDF 到 Markdown

用法示例:
    python convert.py convert --folder "论文仓库/01 - 10 篇"
    python convert.py convert -f "path/to/paper.pdf"
    python convert.py sync
"""
import argparse
import sys
import os

# 确保可以导入本地模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from utils import setup_logging, get_logger
from converter import MarkerConverter, ResultSyncer

# 初始化日志
setup_logging()
logger = get_logger()


def cmd_convert_folder(args):
    """转换文件夹中的 PDF"""
    converter = MarkerConverter(CONFIG["remote"], CONFIG["marker"])
    syncer = ResultSyncer(CONFIG["remote"], CONFIG["local"])

    # 构建远程路径
    if args.folder:
        folder = f'{CONFIG["remote"]["inbox"]}/{args.folder}'
    else:
        folder = CONFIG["remote"]["inbox"]

    logger.info("=" * 60)
    logger.info(f"📁 批量转换: {folder}")
    logger.info("=" * 60)

    # 获取待转换列表
    pending = converter.get_pending_pdfs(folder)

    if not pending:
        logger.info("✅ 所有论文都已转换，无需处理")
        return

    # 显示待转换列表
    logger.info("待转换论文:")
    for i, pdf in enumerate(pending, 1):
        logger.info(f"  {i}. {os.path.basename(pdf)}")

    # 执行转换
    use_multi_gpu = not args.single_gpu
    converter.convert_batch(pending, use_multi_gpu=use_multi_gpu)

    # 同步结果
    logger.info("")
    syncer.sync()

    logger.info("=" * 60)
    logger.info("🎉 完成!")
    logger.info("=" * 60)


def cmd_convert_single(args):
    """转换单个 PDF"""
    converter = MarkerConverter(CONFIG["remote"], CONFIG["marker"])
    syncer = ResultSyncer(CONFIG["remote"], CONFIG["local"])

    # 构建远程路径
    if args.file.startswith("/"):
        pdf_path = args.file
    else:
        pdf_path = f'{CONFIG["remote"]["inbox"]}/{args.file}'

    logger.info("=" * 60)
    logger.info(f"📄 转换: {os.path.basename(pdf_path)}")
    logger.info("=" * 60)

    if converter.convert_single(pdf_path):
        syncer.sync()
        logger.info("🎉 完成!")
    else:
        logger.error("转换失败")


def cmd_sync(args):
    """仅同步结果"""
    syncer = ResultSyncer(CONFIG["remote"], CONFIG["local"])
    syncer.sync()


def main():
    parser = argparse.ArgumentParser(
        description="PDF to Markdown 转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        示例:
        python convert.py convert --folder "11 - 20 篇"
        python convert.py convert -f "paper.pdf"
        python convert.py sync
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # convert 命令
    convert_parser = subparsers.add_parser("convert", help="转换 PDF")
    convert_parser.add_argument("-f", "--file", help="单个 PDF 文件路径")
    convert_parser.add_argument("-d", "--folder", help="文件夹路径 (相对于 Inbox)")
    convert_parser.add_argument("--all", action="store_true", help="转换整个 Inbox")
    convert_parser.add_argument("--single-gpu", action="store_true", help="使用单 GPU")

    # sync 命令
    subparsers.add_parser("sync", help="同步结果到本地")

    args = parser.parse_args()

    if args.command == "convert":
        if args.file:
            cmd_convert_single(args)
        elif args.folder or args.all:
            cmd_convert_folder(args)
        else:
            logger.error("请指定 --file, --folder 或 --all")
            parser.print_help()
    elif args.command == "sync":
        cmd_sync(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
