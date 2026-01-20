"""
方案二：边缘检测分析器

通过边缘检测计算图片的信息密度。
边缘越多 = 信息变化点越多 = 内容越丰富 = 越不应该压缩
"""

from pathlib import Path
import numpy as np
from PIL import ImageFilter
from base import BaseAnalyzer


class EdgeDensityAnalyzer(BaseAnalyzer):
    """边缘密度分析器"""

    name = "edge_density"
    description = "方案二：边缘检测法 - 通过边缘密度评估信息丰富程度"

    def __init__(self, edge_threshold: int = 30, min_score: float = 20.0, **kwargs):
        """
        初始化分析器。

        Args:
            edge_threshold: 边缘检测阈值，大于此值视为边缘像素 (默认 30)
            min_score: 最低分数阈值，低于此值建议压缩 (默认 20)
        """
        super().__init__(**kwargs)
        self.edge_threshold = edge_threshold
        self.min_score = min_score

    def analyze(self, image_path: Path) -> dict:
        """
        分析图片的边缘密度。

        返回:
            score: 边缘密度 * 放大系数 (0-100)
            边缘越少，分数越低
        """
        img = self.load_image(image_path)
        if img is None:
            return {"score": 0, "details": {}, "should_compress": False, "error": True}

        width, height = img.size

        # 转灰度
        gray_img = img.convert("L")

        # 边缘检测
        edges = gray_img.filter(ImageFilter.FIND_EDGES)
        edge_pixels = np.array(edges)

        total_pixels = edge_pixels.size

        # 统计边缘像素 (值 > 阈值)
        edge_count = np.sum(edge_pixels > self.edge_threshold)
        edge_ratio = edge_count / total_pixels

        # 边缘密度通常很低 (1%-10%)，放大 10 倍作为分数
        # 但限制最大值为 100
        score = min(edge_ratio * 1000, 100)

        # 计算有效面积
        effective_area = self.calc_effective_area(width, height, score)

        details = {
            "width": width,
            "height": height,
            "total_pixels": total_pixels,
            "edge_pixels": int(edge_count),
            "edge_ratio": round(edge_ratio * 100, 2),
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
            f"边缘占比: {d['edge_ratio']}% | "
            f"有效面积: {d['effective_area']:,} / {d['original_area']:,} | "
            f"评分: {result['score']}"
        )
