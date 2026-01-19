#!/usr/bin/env python3
"""
Librarian Agent - 智能论文分类与元数据管理
============================================================================

用法:
    python librarian.py              # 处理所有待分类论文
    python librarian.py --dry-run    # 仅预览，不执行实际操作
    python librarian.py --limit 3    # 仅处理前 3 篇

功能:
    1. 扫描 10_References 目录中的 Markdown 论文
    2. 使用 Gemini API 分析摘要并分类
    3. 注入 YAML Frontmatter 元数据
    4. 将论文移动到 20_Classification 对应分类目录
"""

# 在任何其他导入之前设置警告过滤，抑制 google.generativeai 的 FutureWarning
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import os
import sys
import time
import logging
from pathlib import Path
from argparse import ArgumentParser
from typing import Optional

# 添加模块路径
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from config import CONFIG
from librarian.scanner import Scanner
from librarian.category_manager import CategoryManager
from librarian.content_extractor import ContentExtractor
from librarian.gemini_classifier import GeminiClassifier, ClassificationResult
from librarian.metadata_injector import MetadataInjector
from librarian.archivist import Archivist

# 日志配置
LOG_DIR = SCRIPT_DIR.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "librarian.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class LibrarianAgent:
    """图书管理员智能体 - 论文分类与管理"""

    def __init__(self, dry_run: bool = False):
        """
        初始化 Librarian Agent

        Args:
            dry_run: 是否为预览模式 (不执行实际操作)
        """
        self.dry_run = dry_run
        self.cfg = CONFIG["librarian"]
        self.gemini_cfg = CONFIG["gemini"]

        # 初始化组件
        self.scanner = Scanner(
            Path(self.cfg["staging_dir"]), self.cfg["min_filename_length"]
        )
        self.category_mgr = CategoryManager(Path(self.cfg["target_dir"]))
        self.extractor = ContentExtractor(self.cfg["abstract_max_chars"])
        self.injector = MetadataInjector()
        self.archivist = Archivist(Path(self.cfg["target_dir"]))

        # Gemini 分类器 (延迟初始化)
        self._classifier: Optional[GeminiClassifier] = None

    @property
    def classifier(self) -> GeminiClassifier:
        """延迟初始化 Gemini 分类器"""
        if self._classifier is None:
            api_key = os.environ.get(self.gemini_cfg["api_key_env"])
            if not api_key:
                raise RuntimeError(
                    f"未设置环境变量: {self.gemini_cfg['api_key_env']}\n"
                    f"请设置: set {self.gemini_cfg['api_key_env']}=your_api_key"
                )
            self._classifier = GeminiClassifier(
                api_key=api_key,
                model=self.gemini_cfg["model"],
                temperature=self.gemini_cfg["temperature"],
                max_retries=self.gemini_cfg["max_retries"],
            )
        return self._classifier

    def run(self, limit: Optional[int] = None) -> dict:
        """
        执行分类处理

        Args:
            limit: 处理数量限制 (None 表示全部处理)

        Returns:
            处理结果统计
        """
        logger.info("=" * 60)
        logger.info("Librarian Agent 启动")
        logger.info(f"模式: {'预览模式 (Dry Run)' if self.dry_run else '正常模式'}")
        logger.info("=" * 60)

        # 扫描待处理文件
        papers = self.scanner.scan()
        if limit:
            papers = papers[:limit]

        if not papers:
            logger.info("没有发现待处理的论文")
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

        logger.info(f"发现 {len(papers)} 篇待处理论文")

        # 显示当前类别
        categories = self.category_mgr.get_categories()
        logger.info(f"现有类别 ({len(categories)}): {categories}")

        # 处理每篇论文
        stats = {"total": len(papers), "success": 0, "failed": 0, "skipped": 0}

        for i, paper_path in enumerate(papers, 1):
            paper_dir = paper_path.parent
            logger.info("-" * 40)
            logger.info(f"[{i}/{len(papers)}] 处理: {paper_dir.name}")

            try:
                result = self._process_paper(paper_path)
                if result:
                    stats["success"] += 1
                else:
                    stats["skipped"] += 1
            except Exception as e:
                logger.error(f"处理失败: {e}")
                stats["failed"] += 1

            # API 调用间隔
            if i < len(papers) and not self.dry_run:
                time.sleep(self.cfg["api_delay_seconds"])

        # 输出统计
        logger.info("=" * 60)
        logger.info("处理完成!")
        logger.info(f"  成功: {stats['success']}")
        logger.info(f"  失败: {stats['failed']}")
        logger.info(f"  跳过: {stats['skipped']}")
        logger.info("=" * 60)

        return stats

    def _process_paper(self, paper_path: Path) -> bool:
        """
        处理单篇论文

        Args:
            paper_path: 论文 MD 文件路径

        Returns:
            是否成功处理
        """
        paper_dir = paper_path.parent

        # Step 1: 提取内容
        content = self.extractor.extract(paper_path)
        logger.info(f"  标题: {content.title[:60]}...")
        logger.debug(f"  摘要长度: {len(content.abstract)} 字符")

        # Step 2: 获取当前类别并分类
        categories = self.category_mgr.get_categories()
        result = self.classifier.classify(content.abstract, categories)

        self._log_classification_result(result)

        if self.dry_run:
            logger.info("  [DRY RUN] 跳过实际操作")
            return True

        # Step 3: 注入元数据
        self.injector.inject(
            file_path=paper_path,
            title=result.clean_title or content.title,
            category=result.category,
            tags=result.tags,
            year=result.publication_year,
            reason=result.reason,
        )

        # Step 4: 确保类别目录存在
        if result.is_new:
            self.category_mgr.ensure_category(result.category)

        # Step 5: 移动到分类目录
        archived_path = self.archivist.archive(
            source_dir=paper_dir,
            category=result.category,
            clean_title=result.clean_title,
        )

        logger.info(f"  ✓ 归档至: {result.category}/{archived_path.name}")
        return True

    def _log_classification_result(self, result: ClassificationResult) -> None:
        """记录分类结果"""
        status = "🆕 新建" if result.is_new else "📁 现有"
        logger.info(f"  → 分类: {result.category} ({status})")
        logger.info(f"  → 标签: {result.tags}")
        logger.info(f"  → 置信度: {result.confidence:.0%}")
        logger.info(f"  → 理由: {result.reason}")


def main():
    """主入口"""
    parser = ArgumentParser(description="Librarian Agent - 智能论文分类与元数据管理")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="预览模式，不执行实际操作",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理的论文数量",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试日志",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        agent = LibrarianAgent(dry_run=args.dry_run)
        agent.run(limit=args.limit)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"运行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
