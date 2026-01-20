"""
图片引用尺寸调整工具 - 主入口

基于图片信息密度智能调整 Markdown 中的图片引用尺寸。

支持三种分析方案：
1. blank_ratio - 白色像素占比法
2. edge_density - 边缘检测法
3. entropy - 熵值法

使用方式:
    # 处理单个文件
    python resize_images.py -i "笔记.md" --dry-run

    # 处理目录下所有笔记
    python resize_images.py -d "D:\\code\\终端推理\\20_Classification" --dry-run

    # 使用熵值法处理目录
    python resize_images.py -d "D:\\code\\终端推理\\20_Classification" -A entropy
"""

import argparse
from pathlib import Path

from analyzers import (
    BlankRatioAnalyzer,
    EdgeDensityAnalyzer,
    EntropyAnalyzer,
)
from processor import process_markdown


# 分析器映射
ANALYZERS = {
    "blank_ratio": BlankRatioAnalyzer,
    "edge_density": EdgeDensityAnalyzer,
    "entropy": EntropyAnalyzer,
}

# 笔记文件后缀
NOTE_SUFFIX = "_笔记.md"


def find_note_files(directory: Path) -> list[Path]:
    """
    递归查找目录下所有笔记文件。

    Args:
        directory: 搜索目录

    Returns:
        笔记文件路径列表
    """
    note_files = []

    # 递归搜索所有 *_笔记.md 文件
    for md_file in directory.rglob("*_笔记.md"):
        if md_file.is_file():
            note_files.append(md_file)

    return sorted(note_files)


def process_directory(
    directory: Path, analyzer, scale: float, min_effective_area: int, dry_run: bool
) -> dict:
    """
    处理目录下所有笔记文件。

    Returns:
        总体处理统计
    """
    note_files = find_note_files(directory)

    if not note_files:
        print(f"❌ 未找到笔记文件 (匹配 *{NOTE_SUFFIX})")
        return {}

    print(f"\n{'#'*60}")
    print(f"📂 扫描目录: {directory}")
    print(f"📄 找到 {len(note_files)} 个笔记文件")
    print(f"{'#'*60}")

    total_stats = {
        "files_processed": 0,
        "files_modified": 0,
        "total_images": 0,
        "total_compressed": 0,
    }

    for idx, note_file in enumerate(note_files, 1):
        print(f"\n[{idx}/{len(note_files)}] {note_file.relative_to(directory)}")

        stats = process_markdown(
            md_path=note_file,
            analyzer=analyzer,
            scale=scale,
            min_effective_area=min_effective_area,
            dry_run=dry_run,
        )

        total_stats["files_processed"] += 1
        total_stats["total_images"] += stats.get("found", 0)
        total_stats["total_compressed"] += stats.get("modified", 0)

        if stats.get("modified", 0) > 0:
            total_stats["files_modified"] += 1

    # 打印总体统计
    print(f"\n{'#'*60}")
    print(f"📊 总体统计:")
    print(f"   处理文件数:   {total_stats['files_processed']}")
    print(f"   修改文件数:   {total_stats['files_modified']}")
    print(f"   处理图片数:   {total_stats['total_images']}")
    print(f"   压缩图片数:   {total_stats['total_compressed']}")
    print(f"{'#'*60}")

    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description="基于信息密度的 Markdown 图片引用尺寸调整工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
分析器说明:
  blank_ratio  - 方案一：统计白色/空白像素占比
  edge_density - 方案二：通过边缘检测评估内容丰富度
  entropy      - 方案三：基于信息熵评估信息量

示例:
  # 处理单个文件 (预览模式)
  python resize_images.py -i "笔记.md" --dry-run
  
  # 处理目录下所有笔记 (预览模式)
  python resize_images.py -d "D:\\code\\终端推理\\20_Classification" --dry-run
  
  # 使用熵值法处理目录
  python resize_images.py -d "D:\\code\\终端推理\\20_Classification" -A entropy
  
  # 实际执行修改
  python resize_images.py -d "D:\\code\\终端推理\\20_Classification" -A entropy --min-area 150000
        """,
    )

    # 输入参数 (二选一)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i", type=str, help="输入的单个 Markdown 笔记文件路径"
    )
    input_group.add_argument(
        "--input-dir", "-d", type=str, help="输入目录，递归处理所有 *_笔记.md 文件"
    )

    parser.add_argument(
        "--analyzer",
        "-A",
        type=str,
        choices=list(ANALYZERS.keys()),
        default="entropy",
        help="分析器类型 (默认: entropy)",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=200000,
        help="最小有效面积阈值，低于此值需要压缩 (默认: 200000)",
    )

    parser.add_argument(
        "--scale",
        "-S",
        type=float,
        default=0.7,
        help="缩小倍率，如 0.7 表示缩小到原来的 70%% (默认: 0.7)",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="分析器最低分数阈值 (可选，默认使用分析器内置值)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="预览模式，仅显示将要进行的操作，不实际修改文件",
    )

    parser.add_argument(
        "--list-analyzers", action="store_true", help="列出所有可用的分析器"
    )

    args = parser.parse_args()

    # 列出分析器
    if args.list_analyzers:
        print("\n可用的分析器:\n")
        for name, cls in ANALYZERS.items():
            analyzer = cls()
            print(f"  {name:15s} - {analyzer.description}")
        print()
        return 0

    # 检查是否提供了输入
    if not args.input and not args.input_dir:
        parser.error("需要提供 --input/-i 或 --input-dir/-d 参数")

    # 参数验证
    if not 0 < args.scale < 1:
        print(f"⚠️ 警告: 缩小倍率应在 0-1 之间，当前值: {args.scale}")

    # 创建分析器
    analyzer_class = ANALYZERS[args.analyzer]
    analyzer_kwargs = {}
    if args.min_score is not None:
        analyzer_kwargs["min_score"] = args.min_score

    analyzer = analyzer_class(**analyzer_kwargs)

    # 处理单个文件
    if args.input:
        md_path = Path(args.input)
        if not md_path.exists():
            print(f"❌ 错误: 文件不存在 - {args.input}")
            return 1

        if not md_path.suffix.lower() == ".md":
            print(f"⚠️ 警告: 文件不是 Markdown 格式 - {args.input}")

        process_markdown(
            md_path=md_path,
            analyzer=analyzer,
            scale=args.scale,
            min_effective_area=args.min_area,
            dry_run=args.dry_run,
        )

    # 处理目录
    elif args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.exists():
            print(f"❌ 错误: 目录不存在 - {args.input_dir}")
            return 1

        if not dir_path.is_dir():
            print(f"❌ 错误: 不是目录 - {args.input_dir}")
            return 1

        process_directory(
            directory=dir_path,
            analyzer=analyzer,
            scale=args.scale,
            min_effective_area=args.min_area,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    exit(main())

"""
# 处理单个文件
python resize_images.py -i "笔记.md" -A entropy --dry-run

# 处理目录
python resize_images.py -d "D:\\code\\终端推理\\20_Classification" -A entropy --dry-run
python resize_images.py -d "D:\code\终端推理\20_Classification" -A entropy --min-area 150000 --scale 0.65
"""
