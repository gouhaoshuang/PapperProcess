# GeminiClassifier - AI 分类核心
# ============================================================================
# 调用 Gemini API，输入摘要 + 现有类别，输出分类决策和元数据

import json
import time
import warnings
from typing import List, Optional
from dataclasses import dataclass, field
import logging

# 抑制 google.generativeai 即将废弃的警告
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """分类结果"""

    category: str  # 分类名称
    is_new: bool  # 是否为新类别
    reason: str  # 分类理由
    tags: List[str] = field(default_factory=list)  # 标签列表
    clean_title: str = ""  # 清洗后的标题
    publication_year: Optional[str] = None  # 发表年份
    confidence: float = 0.8  # 置信度


class GeminiClassifier:
    """调用 Gemini 进行论文分类"""

    PROMPT_TEMPLATE = """Role: 你是一位移动端大模型推理与边缘AI领域的资深研究员。
Objective: 根据论文摘要对其进行分类，建立一个**细粒度**的技术索引体系。

当前已有的分类:
{categories}

论文摘要:
\"\"\"
{abstract}
\"\"\"

分类要求:
1. 分析摘要，识别论文使用的**核心具体技术**（不仅仅是应用场景）。
2. 决定该论文的最佳分类:
   - 优先归入"当前已有的分类"中，但前提是该分类能精准描述论文技术。
   - **禁止**使用过于宽泛的通用类别（如 "端侧推理加速", "大模型优化", "深度学习", "移动AI"）。这些是我们要建立的库的名称，不是子分类！
   - 创建新类别时，必须使用**具体的技术手段或研究子领域**。
   - 推荐的细粒度示例: "量化", "剪枝", "投机解码", "稀疏注意力", "KV缓存优化", "NPU算子优化", "端云协同", "异构计算调度", "视觉语言模型", "能效管理"。
3. 提取 3-5 个与技术方法相关的标签 (中文)。
4. 仅返回有效的 JSON 对象，不要使用 markdown 代码块。

JSON Schema:
{{
    "category": "string (中文类别名称，必须是具体技术领域)",
    "is_new": boolean,
    "reason": "string (分类理由，简要说明)",
    "tags": ["string", "string"],
    "clean_title": "string (英文论文标题，清理后的版本)",
    "publication_year": "string or null (从摘要中提取年份，如无则为null)",
    "confidence": float (0.0 to 1.0)
}}"""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        """
        初始化分类器

        Args:
            api_key: Gemini API Key
            model: 模型名称
            temperature: 生成温度
            max_retries: 最大重试次数
        """
        try:
            # 在导入前设置警告过滤，抑制废弃警告
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            self.temperature = temperature
            self.max_retries = max_retries
            logger.info(f"Gemini 分类器初始化成功，模型: {model}")
        except ImportError:
            raise ImportError(
                "请安装 google-generativeai: pip install google-generativeai"
            )

    def classify(
        self, abstract: str, existing_categories: List[str]
    ) -> ClassificationResult:
        """
        对论文摘要进行分类

        Args:
            abstract: 论文摘要
            existing_categories: 现有类别列表

        Returns:
            分类结果
        """
        categories_str = (
            "\n".join(f"- {c}" for c in existing_categories)
            if existing_categories
            else "- (暂无现有类别，请创建合适的新类别)"
        )

        prompt = self.PROMPT_TEMPLATE.format(
            categories=categories_str, abstract=abstract
        )

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt, generation_config={"temperature": self.temperature}
                )
                result = self._parse_response(response.text)
                logger.debug(f"分类成功: {result.category}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (尝试 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    # 返回默认分类
                    return ClassificationResult(
                        category="Uncategorized",
                        is_new=True,
                        reason="API 响应解析失败",
                        confidence=0.0,
                    )
            except Exception as e:
                logger.error(f"API 调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = 2**attempt
                    logger.info(f"等待 {sleep_time} 秒后重试...")
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(
                        f"Gemini API 在 {self.max_retries} 次尝试后仍失败: {e}"
                    )

        # 不应该到达这里
        return ClassificationResult(
            category="Uncategorized",
            is_new=True,
            reason="未知错误",
            confidence=0.0,
        )

    def _parse_response(self, text: str) -> ClassificationResult:
        """
        解析 Gemini JSON 响应

        Args:
            text: API 响应文本

        Returns:
            分类结果对象
        """
        text = text.strip()

        # 清理可能的 markdown 代码块
        if text.startswith("```"):
            # 移除 ```json 或 ``` 开头
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        data = json.loads(text)

        return ClassificationResult(
            category=data.get("category", "Uncategorized"),
            is_new=data.get("is_new", True),
            reason=data.get("reason", ""),
            tags=data.get("tags", []),
            clean_title=data.get("clean_title", ""),
            publication_year=data.get("publication_year"),
            confidence=data.get("confidence", 0.8),
        )
