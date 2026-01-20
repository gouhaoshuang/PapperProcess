"""
方案三：图像熵分析器

通过计算图像的信息熵来评估信息密度。
熵越高 = 像素值分布越均匀 = 信息越丰富 = 越不应该压缩
"""

from pathlib import Path
import numpy as np
from base import BaseAnalyzer


class EntropyAnalyzer(BaseAnalyzer):
    """图像熵分析器"""

    name = "entropy"
    description = "方案三：熵值法 - 基于信息论评估图像信息量"

    def __init__(self, min_score: float = 40.0, **kwargs):
        """
        初始化分析器。

        Args:
            min_score: 最低分数阈值，低于此值建议压缩 (默认 40)
        """
        super().__init__(**kwargs)
        self.min_score = min_score

    def calc_entropy(self, pixels: np.ndarray) -> float:
        """
        计算图像熵值。

        图像熵的理论最大值是 8 bits (对于 8-bit 灰度图)
        """
        # 计算直方图
        histogram, _ = np.histogram(pixels.flatten(), bins=256, range=(0, 256))

        # 归一化为概率分布
        prob = histogram / histogram.sum()

        # 过滤掉零概率（避免 log(0)）
        prob = prob[prob > 0]

        # 计算熵: H = -Σ p(x) * log2(p(x))
        entropy = -np.sum(prob * np.log2(prob))

        return entropy

    def analyze(self, image_path: Path) -> dict:
        """
        分析图片的信息熵。

        返回:
            score: 熵值归一化后的分数 (0-100)
            熵值范围: 0-8 bits
            - 纯色/空白图: 0-2
            - 简单图表: 3-5
            - 复杂照片: 6-8
        """
        img = self.load_image(image_path)
        if img is None:
            return {"score": 0, "details": {}, "should_compress": False, "error": True}

        width, height = img.size

        # 转灰度
        pixels = self.to_grayscale(img)

        # 计算熵值
        entropy = self.calc_entropy(pixels)

        # 熵值归一化到 0-100 (最大熵为 8 bits)
        max_entropy = 8.0
        score = (entropy / max_entropy) * 100

        # 计算有效面积
        effective_area = self.calc_effective_area(width, height, score)

        # 判断熵值级别
        if entropy < 2:
            entropy_level = "极低 (纯色/空白)"
        elif entropy < 4:
            entropy_level = "低 (简单图形)"
        elif entropy < 6:
            entropy_level = "中等 (图表)"
        else:
            entropy_level = "高 (复杂图像)"

        details = {
            "width": width,
            "height": height,
            "entropy": round(entropy, 2),
            "entropy_level": entropy_level,
            "max_entropy": max_entropy,
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
            f"熵值: {d['entropy']} bits ({d['entropy_level']}) | "
            f"有效面积: {d['effective_area']:,} / {d['original_area']:,} | "
            f"评分: {result['score']}"
        )
