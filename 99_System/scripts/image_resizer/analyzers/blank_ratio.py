"""
方案一：白色像素占比分析器

通过统计图片中接近白色（空白）的像素占比来评估信息密度。
空白越多 = 信息密度越低 = 越应该压缩
"""

from pathlib import Path
import numpy as np
from base import BaseAnalyzer


class BlankRatioAnalyzer(BaseAnalyzer):
    """白色像素占比分析器"""

    name = "blank_ratio"
    description = "方案一：白色像素占比法 - 统计空白区域占比"

    def __init__(self, blank_threshold: int = 250, min_score: float = 30.0, **kwargs):
        """
        初始化分析器。

        Args:
            blank_threshold: 灰度阈值，大于此值视为空白 (0-255，默认 250)
            min_score: 最低分数阈值，低于此值建议压缩 (默认 30)
        """
        super().__init__(**kwargs)
        self.blank_threshold = blank_threshold
        self.min_score = min_score

    def analyze(self, image_path: Path) -> dict:
        """
        分析图片的空白占比。

        返回:
            score: 非空白占比 * 100 (0-100)
            空白占比越高，分数越低
        """
        img = self.load_image(image_path)
        if img is None:
            return {"score": 0, "details": {}, "should_compress": False, "error": True}

        # 转为灰度
        pixels = self.to_grayscale(img)
        width, height = img.size
        total_pixels = pixels.size

        # 统计空白像素 (灰度值 > 阈值)
        blank_pixels = np.sum(pixels > self.blank_threshold)
        blank_ratio = blank_pixels / total_pixels

        # 非空白占比作为分数
        non_blank_ratio = 1 - blank_ratio
        score = non_blank_ratio * 100

        # 计算有效面积
        effective_area = self.calc_effective_area(width, height, score)

        details = {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "blank_pixels": int(blank_pixels),
            "blank_ratio": round(blank_ratio * 100, 1),
            "non_blank_ratio": round(non_blank_ratio * 100, 1),
            "original_area": width * height,
            "effective_area": int(effective_area),
        }

        return {
            "score": round(score, 1),
            "details": details,
            "should_compress": score < self.min_score,
        }

    def format_result(self, result: dict) -> str:
        """格式化输出结果"""
        d = result["details"]
        return (
            f"空白占比: {d['blank_ratio']}% | "
            f"有效面积: {d['effective_area']:,} / {d['original_area']:,} | "
            f"评分: {result['score']}"
        )
