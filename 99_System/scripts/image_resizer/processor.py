"""
图片处理器

核心处理逻辑：遍历图片、分析、决策、修改。
"""

from pathlib import Path
from typing import Type

from parser import extract_image_lines, create_img_tag, replace_image_reference
from analyzers.base import BaseAnalyzer


def process_markdown(
    md_path: Path,
    analyzer: BaseAnalyzer,
    scale: float = 0.75,
    min_effective_area: int = 200000,
    dry_run: bool = False,
) -> dict:
    """
    处理 Markdown 文件中的图片引用。

    Args:
        md_path: Markdown 文件路径
        analyzer: 图片分析器实例
        scale: 缩小倍率 (如 0.75 表示缩小到原来的 75%)
        min_effective_area: 最小有效面积阈值，低于此值需要压缩
        dry_run: 如果为 True，仅预览不实际修改

    Returns:
        处理结果统计
    """
    # 读取 Markdown 文件
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    base_dir = md_path.parent

    # 提取图片引用
    image_refs = extract_image_lines(content)

    print(f"\n{'='*60}")
    print(f"📄 文件: {md_path.name}")
    print(f"📁 目录: {base_dir}")
    print(f"🔬 分析器: {analyzer.description}")
    print(f"📏 有效面积阈值: {min_effective_area:,} 像素²")
    print(f"🔽 缩小倍率: {scale:.0%}")
    print(f"🔍 图片引用: {len(image_refs)} 个")
    print(f"{'='*60}\n")

    stats = {
        "total": len(image_refs),
        "found": 0,
        "need_compress": 0,
        "modified": 0,
        "skipped": 0,
        "not_found": 0,
    }

    modifications = []  # 记录需要修改的行 [(line_idx, new_line), ...]

    for line_idx, line, image_name, fmt in image_refs:
        image_path = base_dir / image_name

        print(f"[L{line_idx + 1:3d}] {image_name}")

        if not image_path.exists():
            print(f"       ❌ 文件不存在")
            stats["not_found"] += 1
            continue

        stats["found"] += 1

        # 使用分析器分析图片
        result = analyzer.analyze(image_path)

        if result.get("error"):
            stats["skipped"] += 1
            continue

        details = result["details"]
        width = details["width"]
        height = details["height"]
        effective_area = details["effective_area"]
        original_area = details["original_area"]
        score = result["score"]

        # 输出分析结果
        print(f"       尺寸: {width} x {height} = {original_area:,} 像素²")
        print(f"       评分: {score} | 有效面积: {effective_area:,} 像素²", end="")

        # 判断是否需要压缩
        if effective_area < min_effective_area:
            stats["need_compress"] += 1

            # 计算新宽度
            new_width = int(width * scale)

            print(f" → 需要压缩")
            print(f"       📝 新宽度: {new_width}px")

            # 创建新的 <img> 标签
            new_tag = create_img_tag(image_name, new_width)
            new_line = replace_image_reference(line, image_name, new_tag, fmt)

            if new_line != line:
                modifications.append((line_idx, new_line))
                stats["modified"] += 1
        else:
            print(f" ✓ 无需压缩")

    # 打印统计
    print(f"\n{'='*60}")
    print(f"📊 处理统计:")
    print(f"   总计图片引用: {stats['total']}")
    print(f"   找到图片文件: {stats['found']}")
    print(f"   需要压缩:     {stats['need_compress']}")
    print(f"   已修改引用:   {stats['modified']}")
    print(f"   已跳过:       {stats['skipped']}")
    print(f"   未找到文件:   {stats['not_found']}")

    # 应用修改
    if modifications:
        if dry_run:
            print(f"\n💡 [预览模式] 将修改 {len(modifications)} 处图片引用")
            print(f"   移除 --dry-run 参数可执行实际修改")
        else:
            # 应用修改到行
            for line_idx, new_line in modifications:
                lines[line_idx] = new_line

            # 写回文件
            new_content = "\n".join(lines)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            print(f"\n✅ 已修改 {len(modifications)} 处图片引用并保存文件")
    else:
        print(f"\n✓ 无需修改")

    return stats
