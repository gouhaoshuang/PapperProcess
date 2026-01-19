# MetadataInjector - 元数据注入器
# ============================================================================
# 将提取的信息按标准 YAML 格式写入文件头部

from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MetadataInjector:
    """注入 YAML Frontmatter 到 Markdown 文件"""

    def __init__(self):
        """初始化注入器"""
        self._frontmatter_available = self._check_frontmatter()

    def _check_frontmatter(self) -> bool:
        """检查 python-frontmatter 是否可用"""
        try:
            import frontmatter

            return True
        except ImportError:
            logger.warning(
                "python-frontmatter 未安装，将使用手动注入方式。"
                "建议安装: pip install python-frontmatter"
            )
            return False

    def inject(
        self,
        file_path: Path,
        title: str,
        category: str,
        tags: List[str],
        authors: Optional[List[str]] = None,
        year: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        """
        注入元数据到文件头部

        Args:
            file_path: Markdown 文件路径
            title: 论文标题
            category: 分类名称
            tags: 标签列表
            authors: 作者列表 (可选)
            year: 发表年份 (可选)
            reason: 分类理由 (可选)
        """
        metadata = self._build_metadata(title, category, tags, authors, year, reason)

        if self._frontmatter_available:
            self._inject_with_frontmatter(file_path, metadata)
        else:
            self._inject_manual(file_path, metadata)

        logger.debug(f"元数据注入完成: {file_path.name}")

    def _build_metadata(
        self,
        title: str,
        category: str,
        tags: List[str],
        authors: Optional[List[str]] = None,
        year: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构建元数据字典"""
        metadata = {
            "title": title,
            "category": category,
            "tags": tags,
            "status": "unread",
            "created": datetime.now().strftime("%Y-%m-%d"),
        }

        if authors:
            metadata["authors"] = authors
        if year:
            metadata["year"] = year
        if reason:
            metadata["classification_reason"] = reason

        return metadata

    def _inject_with_frontmatter(
        self, file_path: Path, metadata: Dict[str, Any]
    ) -> None:
        """使用 python-frontmatter 库注入"""
        import frontmatter

        try:
            post = frontmatter.load(file_path)
        except Exception:
            # 如果解析失败，创建新的
            content = file_path.read_text(encoding="utf-8")
            post = frontmatter.Post(content)

        # 更新元数据 (保留已有的，添加新的)
        post.metadata.update(metadata)

        # 写回文件
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))

    def _inject_manual(self, file_path: Path, metadata: Dict[str, Any]) -> None:
        """手动注入 YAML Frontmatter"""
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk", errors="ignore")

        # 检查是否已有 frontmatter
        if content.startswith("---"):
            # 找到结束的 ---
            end_idx = content.find("---", 3)
            if end_idx != -1:
                # 移除旧的 frontmatter
                content = content[end_idx + 3 :].lstrip()

        # 构建新的 frontmatter
        yaml_lines = ["---"]
        for key, value in metadata.items():
            if isinstance(value, list):
                yaml_lines.append(f"{key}:")
                for item in value:
                    yaml_lines.append(f"  - {item}")
            elif isinstance(value, str) and ("\n" in value or ":" in value):
                yaml_lines.append(f'{key}: "{value}"')
            else:
                yaml_lines.append(f"{key}: {value}")
        yaml_lines.append("---")
        yaml_lines.append("")

        new_content = "\n".join(yaml_lines) + content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
