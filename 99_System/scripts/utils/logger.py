# 日志配置模块
import logging
from pathlib import Path
from datetime import datetime

_logger = None


def setup_logging(log_dir: Path = None) -> logging.Logger:
    """配置并返回日志记录器"""
    global _logger

    if _logger is not None:
        return _logger

    # 默认日志目录
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"

    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"converter_{datetime.now().strftime('%Y%m%d')}.log"

    # 创建 logger
    logger = logging.getLogger("pdf_converter")
    logger.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_format)

    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_format)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """获取已配置的 logger，如果未配置则自动配置"""
    global _logger
    if _logger is None:
        return setup_logging()
    return _logger
