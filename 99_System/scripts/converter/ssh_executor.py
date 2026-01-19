# SSH 命令执行器
import subprocess
from utils import get_logger

logger = get_logger()


class SSHExecutor:
    """通过 SSH 在远程服务器执行命令"""

    def __init__(self, ssh_alias: str):
        """
        Args:
            ssh_alias: SSH 别名 (在 ~/.ssh/config 中配置)
        """
        self.ssh_alias = ssh_alias

    def run(self, command: str, timeout: int = 600) -> tuple[int, str, str]:
        """
        执行远程 SSH 命令

        Args:
            command: 要执行的命令
            timeout: 超时时间 (秒)

        Returns:
            (return_code, stdout, stderr)
        """
        full_cmd = f'ssh {self.ssh_alias} "{command}"'
        logger.debug(f"SSH 执行: {command[:100]}...")

        try:
            result = subprocess.run(
                full_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"SSH 命令超时 ({timeout}s)")
            return -1, "", "Timeout"
        except Exception as e:
            logger.error(f"SSH 命令失败: {e}")
            return -1, "", str(e)

    def run_script(
        self, script_content: str, timeout: int = 7200
    ) -> tuple[int, str, str]:
        """
        在远程服务器上执行脚本

        使用 base64 编码传输，避免引号问题

        Args:
            script_content: shell 脚本内容
            timeout: 超时时间 (秒)

        Returns:
            (return_code, stdout, stderr)
        """
        import base64

        script_path = "/tmp/remote_script.sh"
        script_b64 = base64.b64encode(script_content.encode()).decode()

        # 写入脚本
        write_cmd = (
            f"echo {script_b64} | base64 -d > {script_path} && chmod +x {script_path}"
        )
        code, _, stderr = self.run(write_cmd)

        if code != 0:
            logger.error(f"写入脚本失败: {stderr}")
            return code, "", stderr

        # 执行脚本
        return self.run(f"bash {script_path}", timeout=timeout)
