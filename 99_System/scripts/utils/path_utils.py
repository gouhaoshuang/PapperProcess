# 路径工具模块
import os


def win_to_wsl_path(win_path: str) -> str:
    """
    将 Windows 路径转换为 WSL 路径

    例如: D:\\code\\终端推理 -> /mnt/d/code/终端推理
    """
    # 处理驱动器号
    if len(win_path) >= 2 and win_path[1] == ":":
        drive = win_path[0].lower()
        rest = win_path[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return win_path.replace("\\", "/")
