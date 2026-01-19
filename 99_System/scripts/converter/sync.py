# 结果同步模块
import subprocess
from utils import get_logger, win_to_wsl_path

logger = get_logger()


class ResultSyncer:
    """使用 rsync 同步转换结果"""

    def __init__(self, remote_config: dict, local_config: dict):
        """
        Args:
            remote_config: 远程服务器配置
            local_config: 本地路径配置
        """
        self.remote = remote_config
        self.local = local_config

    def sync(self) -> bool:
        """
        将远程结果同步到本地

        Returns:
            是否成功
        """
        logger.info("📥 同步结果到本地...")

        # 构建 rsync 命令 (通过 WSL)
        remote_src = (
            f'{self.remote["username"]}@{self.remote["host"]}:{self.remote["output"]}/'
        )
        local_dest = win_to_wsl_path(self.local["references"]) + "/"

        rsync_cmd = f'rsync -avz "{remote_src}" "{local_dest}"'
        wsl_cmd = f"wsl bash -c '{rsync_cmd}'"

        logger.debug(f"执行: {wsl_cmd}")

        try:
            result = subprocess.run(
                wsl_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300,
            )

            stderr = result.stderr or ""
            if result.returncode != 0 and "error" in stderr.lower():
                logger.error(f"❌ 同步失败: {stderr}")
                return False

            logger.info("✅ 同步完成")
            return True

        except subprocess.TimeoutExpired:
            logger.error("❌ 同步超时")
            return False
        except Exception as e:
            logger.error(f"❌ 同步失败: {e}")
            return False
