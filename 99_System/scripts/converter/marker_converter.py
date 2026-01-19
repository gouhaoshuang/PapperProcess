# Marker PDF 转换器
import os
from .ssh_executor import SSHExecutor
from utils import get_logger

logger = get_logger()


class MarkerConverter:
    """使用 Marker 进行 PDF 到 Markdown 转换"""

    def __init__(self, remote_config: dict, marker_config: dict):
        """
        Args:
            remote_config: 远程服务器配置
            marker_config: Marker 转换配置
        """
        self.remote = remote_config
        self.marker = marker_config
        self.ssh = SSHExecutor(remote_config["ssh_alias"])

    def convert_single(self, pdf_path: str) -> bool:
        """
        转换单个 PDF 文件

        Args:
            pdf_path: 服务器上的 PDF 绝对路径

        Returns:
            是否成功
        """
        filename = os.path.basename(pdf_path)
        output_dir = self.remote["output"]

        # 构建 marker_single 命令
        cmd = (
            f"source ~/.bashrc && "
            f"conda activate {self.remote['conda_env']} && "
            f"marker_single '{pdf_path}' --output_dir '{output_dir}' "
            f"--output_format {self.marker['output_format']}"
        )

        if self.marker.get("force_ocr"):
            cmd += " --force_ocr"
        if self.marker.get("use_llm"):
            cmd += " --use_llm"

        logger.info(f"🔄 转换: {filename}")
        code, stdout, stderr = self.ssh.run(cmd, timeout=600)

        if code != 0:
            logger.error(f"❌ 转换失败: {stderr}")
            return False

        logger.info(f"✅ 转换完成: {filename}")
        return True

    def convert_batch(self, pdf_paths: list[str], use_multi_gpu: bool = True) -> bool:
        """
        批量转换多个 PDF

        Args:
            pdf_paths: PDF 文件路径列表
            use_multi_gpu: 是否使用多 GPU

        Returns:
            是否成功
        """
        if not pdf_paths:
            logger.info("没有需要转换的文件")
            return True

        if use_multi_gpu and len(pdf_paths) > 1:
            return self._batch_multi_gpu(pdf_paths)
        else:
            return self._batch_single_gpu(pdf_paths)

    def _batch_single_gpu(self, pdf_paths: list[str]) -> bool:
        """单 GPU 逐个转换"""
        success = 0
        for i, pdf in enumerate(pdf_paths, 1):
            logger.info(f"[{i}/{len(pdf_paths)}]")
            if self.convert_single(pdf):
                success += 1

        logger.info(f"完成: {success}/{len(pdf_paths)}")
        return success == len(pdf_paths)

    def _batch_multi_gpu(self, pdf_paths: list[str]) -> bool:
        """多 GPU 并行转换"""
        num_devices = self.marker.get("num_devices", 5)
        num_workers = self.marker.get("num_workers", 15)
        output_dir = self.remote["output"]
        temp_dir = "/tmp/marker_batch"

        logger.info(
            f"🚀 多 GPU 并行 ({num_devices} 张 GPU, {num_devices * num_workers} workers)"
        )

        # 构建脚本
        script_lines = [
            "#!/bin/bash",
            "set -e",
            f"rm -rf {temp_dir}",
            f"mkdir -p {temp_dir}",
            "",
            "# 创建软链接",
        ]

        for pdf in pdf_paths:
            escaped = pdf.replace("'", "'\\''")
            script_lines.append(f"ln -sf '{escaped}' {temp_dir}/")

        script_lines.extend(
            [
                "",
                "source ~/.bashrc",
                f"conda activate {self.remote['conda_env']}",
                "",
                f"NUM_DEVICES={num_devices} NUM_WORKERS={num_workers} "
                f"marker_chunk_convert {temp_dir} {output_dir}",
                "",
                f"rm -rf {temp_dir}",
                "echo 'Done!'",
            ]
        )

        script = "\n".join(script_lines)
        code, stdout, stderr = self.ssh.run_script(script, timeout=7200)

        if code != 0:
            logger.error(f"❌ 多 GPU 转换失败: {stderr}")
            return False

        logger.info("✅ 多 GPU 转换完成")
        return True

    def get_converted_papers(self) -> set[str]:
        """获取已转换的论文名称集合"""
        cmd = f"ls -1 '{self.remote['output']}' 2>/dev/null || true"
        code, stdout, _ = self.ssh.run(cmd)

        if code != 0 or not stdout.strip():
            return set()

        return {name.strip() for name in stdout.strip().split("\n") if name.strip()}

    def get_pending_pdfs(self, folder: str) -> list[str]:
        """
        获取待转换的 PDF 列表 (跳过已转换)

        Args:
            folder: 服务器上的文件夹路径

        Returns:
            待转换的 PDF 路径列表
        """
        # 获取已转换
        converted = self.get_converted_papers()
        logger.info(f"📊 已转换: {len(converted)} 篇")

        # 列出所有 PDF
        cmd = f"find '{folder}' -name '*.pdf' -type f"
        code, stdout, _ = self.ssh.run(cmd)

        if code != 0 or not stdout.strip():
            return []

        all_pdfs = [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        logger.info(f"📄 文件夹 PDF 总数: {len(all_pdfs)}")

        # 过滤已转换
        pending = []
        for pdf in all_pdfs:
            name = os.path.splitext(os.path.basename(pdf))[0]
            if name not in converted:
                pending.append(pdf)

        logger.info(f"🔄 待转换: {len(pending)} 篇")
        return pending
