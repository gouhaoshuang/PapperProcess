"""
分析器基类

定义所有图片分析器的通用接口。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from PIL import Image
import numpy as np


class BaseAnalyzer(ABC):
    """图片分析器基类"""

    name: str = "base"
    description: str = "基础分析器"

    def __init__(self, **kwargs):
        """初始化分析器，子类可覆盖以接受特定参数"""
        pass

    def load_image(self, image_path: Path) -> Image.Image | None:
        """加载图片并转为灰度图"""
        try:
            img = Image.open(image_path)
            return img
        except Exception as e:
            print(f"  ⚠️ 无法加载图片: {image_path.name} - {e}")
            return None

    def to_grayscale(self, img: Image.Image) -> np.ndarray:
        """将图片转为灰度 numpy 数组"""
        return np.array(img.convert("L"))

    @abstractmethod
    def analyze(self, image_path: Path) -> dict:
        """
        分析图片，返回分析结果。

        Args:
            image_path: 图片路径

        Returns:
            dict: 包含以下字段:
                - score: float, 信息密度分数 (0-100)
                - details: dict, 详细分析数据
                - should_compress: bool, 是否建议压缩
        """
        pass

    def get_image_size(self, image_path: Path) -> tuple[int, int] | None:
        """获取图片尺寸"""
        img = self.load_image(image_path)
        if img:
            return img.size
        return None

    def calc_effective_area(self, width: int, height: int, score: float) -> float:
        """计算有效内容面积"""
        return width * height * (score / 100)
