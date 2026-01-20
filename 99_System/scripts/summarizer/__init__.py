"""
论文总结系统 (Paper Summarizer)

基于 Gemini AI 的多阶段论文总结系统。
"""

from .validator import is_valid_paper, find_paper_markdown
from .uploader import (
    collect_paper_resources,
    upload_to_gemini,
    read_markdown_content,
    cleanup_uploaded_files,
)
from .outline_generator import (
    load_prompt_template,
    generate_outline,
    parse_json_response,
    validate_outline,
    save_outline,
)
from .section_expander import (
    expand_section,
    expand_all_sections,
    format_subsections_text,
)
from .assembler import assemble_document, generate_frontmatter, get_output_path

__all__ = [
    # validator
    "is_valid_paper",
    "find_paper_markdown",
    # uploader
    "collect_paper_resources",
    "upload_to_gemini",
    "read_markdown_content",
    "cleanup_uploaded_files",
    # outline_generator
    "load_prompt_template",
    "generate_outline",
    "parse_json_response",
    "validate_outline",
    "save_outline",
    # section_expander
    "expand_section",
    "expand_all_sections",
    "format_subsections_text",
    # assembler
    "assemble_document",
    "generate_frontmatter",
    "get_output_path",
]
