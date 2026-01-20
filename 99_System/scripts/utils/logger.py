# 日志配置模块
import logging
from pathlib import Path
from datetime import datetime

_loggers = {}


def setup_logging(name: str = "app", log_dir: Path = None) -> logging.Logger:
    """
    配置并返回日志记录器

    Args:
        name: 日志记录器名称 (如 'pdf_converter', 'summarizer')
        log_dir: 日志目录路径

    Returns:
        配置好的 Logger 实例
    """
    global _loggers

    if name in _loggers:
        return _loggers[name]

    # 默认日志目录
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"

    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"

    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

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

    _loggers[name] = logger
    return logger


def get_logger(name: str = "app") -> logging.Logger:
    """
    获取已配置的 logger，如果未配置则自动配置

    Args:
        name: 日志记录器名称

    Returns:
        Logger 实例
    """
    global _loggers
    if name not in _loggers:
        return setup_logging(name)
    return _loggers[name]
