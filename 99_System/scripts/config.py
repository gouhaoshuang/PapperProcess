# 配置模块 - 所有配置集中管理
# ============================================================================

CONFIG = {
    # 本地路径配置 (Windows 路径)
    "local": {
        "base": r"D:\code\终端推理",
        "inbox": r"D:\code\终端推理\00_Inbox",
        "references": r"D:\code\终端推理\10_References",
        "attachments": r"D:\code\终端推理\30_Attachments",
    },
    # 远程服务器配置
    "remote": {
        "ssh_alias": "L40",  # SSH 别名 (Windows ~/.ssh/config)
        "host": "10.242.6.9",  # 服务器 IP (用于 WSL rsync)
        "username": "ghs",  # SSH 用户名 (用于 WSL rsync)
        # 服务器上的工作目录 (与本地通过 SFTP 同步)
        "base_path": "/data/ghs/终端推理",
        "inbox": "/data/ghs/终端推理/00_Inbox",
        "output": "/data/ghs/终端推理/10_References",
        # Conda 环境
        "conda_env": "pdf",
    },
    # Marker 转换配置
    "marker": {
        "output_format": "markdown",  # markdown, json, html
        "force_ocr": False,  # 强制 OCR (扫描版 PDF)
        "use_llm": False,  # 使用 LLM 提升准确度
        # 多 GPU 配置 (5 张 L40)
        "num_devices": 5,  # GPU 数量
        "num_workers": 15,  # 每 GPU 并行 worker 数
        # 单 GPU 配置
        "workers": 4,
    },
    # Librarian Agent 配置
    "librarian": {
        "staging_dir": r"D:\code\终端推理\10_References",
        "target_dir": r"D:\code\终端推理\20_Classification",
        "min_filename_length": 5,        # 论文文件名最小长度
        "abstract_max_chars": 2000,       # 读取前 N 个字符
        "api_delay_seconds": 1.5,         # API 调用间隔 (防限流)
    },
    # Gemini API 配置
    "gemini": {
        "api_key_env": "GOOGLE_API_KEY",  # 环境变量名
        "model": "gemini-3-pro-preview",
        "temperature": 0.3,
        "max_retries": 3,
    },
}
