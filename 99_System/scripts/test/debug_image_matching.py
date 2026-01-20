"""
图片匹配调试脚本

用于测试 find_image_in_source 函数为何无法正确匹配图片。
"""

import re
from pathlib import Path

# 文件路径
PAPER_DIR = Path(
    r"D:\code\终端推理\20_Classification\存储与IO优化\Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash"
)
SOURCE_FILE = (
    PAPER_DIR
    / "Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash.md"
)
DEBUG_BEFORE = (
    PAPER_DIR
    / "Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash_笔记_debug_before.md"
)
DEBUG_AFTER = (
    PAPER_DIR
    / "Scaling Up On-Device LLMs via Active-Weight Swapping Between DRAM and Flash_笔记_debug_after.md"
)


def find_figure_placeholders(content: str) -> list[str]:
    """查找所有 <Figure X> 占位符"""
    pattern = r"<Figure\s*(\d+)>"
    matches = re.findall(pattern, content)
    return matches


def find_figure_references_in_source(source_content: str) -> dict:
    """
    在原始论文中查找所有 Figure X 的引用及其上下文
    返回: {figure_id: [(line_number, line_content, prev_lines)]}
    """
    lines = source_content.split("\n")
    results = {}

    for i, line in enumerate(lines):
        # 匹配 "Figure X." 或 "Figure X:" 模式
        match = re.search(r"Figure\s+(\d+)[.:]", line)
        if match:
            figure_id = match.group(1)
            if figure_id not in results:
                results[figure_id] = []

            # 获取前面几行的上下文
            prev_lines = []
            for j in range(max(0, i - 5), i):
                prev_lines.append((j + 1, lines[j]))

            results[figure_id].append(
                {"line_num": i + 1, "line": line, "prev_lines": prev_lines}
            )

    return results


def check_image_pattern(line: str) -> str | None:
    """检查行中是否包含图片语法，返回图片路径"""
    match = re.search(r"!\[.*?\]\(([^)]+)\)", line)
    if match:
        return match.group(1)
    return None


def main():
    print("=" * 80)
    print("图片匹配调试分析")
    print("=" * 80)

    # 读取文件
    source_content = SOURCE_FILE.read_text(encoding="utf-8")
    debug_before = DEBUG_BEFORE.read_text(encoding="utf-8")

    # 1. 查找生成笔记中的占位符
    print("\n【1】生成笔记中的 <Figure X> 占位符:")
    placeholders = find_figure_placeholders(debug_before)
    print(f"   找到 {len(placeholders)} 个占位符: {placeholders}")

    # 2. 查找原始论文中的 Figure 引用
    print("\n【2】原始论文中的 Figure 引用格式分析:")
    figure_refs = find_figure_references_in_source(source_content)

    for fig_id, refs in sorted(figure_refs.items(), key=lambda x: int(x[0])):
        print(f"\n   --- Figure {fig_id} ---")
        for ref in refs:
            print(f"   第 {ref['line_num']} 行: {ref['line'][:80]}...")

            # 检查前面的行是否有图片
            found_image = False
            for prev_line_num, prev_line in ref["prev_lines"]:
                img_path = check_image_pattern(prev_line)
                if img_path:
                    print(f"   ✅ 第 {prev_line_num} 行有图片: {img_path}")
                    found_image = True

            if not found_image:
                print(f"   ❌ 前 5 行中没有找到图片语法 '![](...)' ")

    # 3. 问题诊断
    print("\n" + "=" * 80)
    print("【3】问题诊断:")
    print("=" * 80)

    # 检查我们的匹配逻辑
    print("\n   当前匹配逻辑: 搜索 'Figure X.' 然后检查前面第2行是否有 '![]('")
    print("   ")

    # 检查原始论文中的实际格式
    lines = source_content.split("\n")
    sample_figures = ["1", "2", "3"]

    for fig_id in sample_figures:
        search_patterns = [f"Figure {fig_id}.", f"Figure {fig_id}:"]
        for pattern in search_patterns:
            for i, line in enumerate(lines):
                if pattern in line:
                    print(f"\n   测试 Figure {fig_id} (模式: '{pattern}'):")
                    print(f"   - 找到位置: 第 {i+1} 行")
                    print(f"   - 内容: {line[:60]}...")

                    # 检查前面2行
                    if i >= 2:
                        prev_line = lines[i - 2]
                        print(f"   - 前面第2行 (第{i-1}行): {prev_line[:60]}...")
                        if "![](" in prev_line:
                            print(f"   ✅ 包含 '![](' ")
                        else:
                            print(f"   ❌ 不包含 '![](' ")
                    break


if __name__ == "__main__":
    main()
