# Markdown 转 PDF 系统设计方案

> **创建日期**: 2026-01-20  
> **版本**: v1.0  
> **状态**: 设计规划阶段
> **技术栈**: Python + markdown2 + WeasyPrint

---

## 1. 项目概述

### 1.1 目标

构建一个将论文笔记 Markdown 文件转换为高质量 PDF 文档的自动化系统，支持：

- 中文内容与 LaTeX 数学公式渲染
- 图片嵌入与自动调整
- 表格与代码块样式美化
- YAML Frontmatter 元数据提取

### 1.2 技术选型

| 组件              | 选择                          | 说明                         |
| ----------------- | ----------------------------- | ---------------------------- |
| **Markdown 解析** | `markdown2`                   | 稳定、功能丰富，支持多种扩展 |
| **HTML → PDF**    | `WeasyPrint`                  | 基于 CSS 的高质量排版引擎    |
| **数学公式**      | `KaTeX` (CSS/JS) 或预渲染 SVG | 高性能 LaTeX 渲染            |
| **字体**          | 思源宋体 / Noto Sans CJK      | 优秀的中文排版支持           |

### 1.3 转换流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                    输入: 论文笔记.md                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Step 1: 解析 YAML Frontmatter                               │
│  • 提取 title, authors, year, tags 等元数据                          │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Step 2: Markdown → HTML                                     │
│  • 使用 markdown2 解析正文                                            │
│  • 启用扩展: tables, fenced-code-blocks, metadata                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Step 3: 数学公式处理                                         │
│  • 检测 $...$ 和 $$...$$ 格式                                         │
│  • 使用 KaTeX 渲染为 HTML/SVG                                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Step 4: 组装完整 HTML                                        │
│  • 注入 CSS 样式表                                                    │
│  • 添加封面页、页眉页脚                                                │
│  • 处理图片路径 (相对路径 → 绝对路径)                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          Step 5: HTML → PDF (WeasyPrint)                             │
│  • 应用打印样式                                                       │
│  • 生成分页 PDF                                                       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│          输出: 论文笔记.pdf                                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Windows 环境配置

### 2.1 安装 MSYS2 及依赖

WeasyPrint 依赖于 GTK+ 库 (Pango, Cairo, GDK-PixBuf)，在 Windows 上需要通过 MSYS2 安装。

**安装步骤**:

```powershell
# 1. 下载并安装 MSYS2
# 访问 https://www.msys2.org/ 下载安装包
# 默认安装到 C:\msys64

# 2. 打开 MSYS2 MINGW64 终端，安装依赖
pacman -S mingw-w64-x86_64-pango

# 3. 安装后关闭 MSYS2 终端
```

### 2.2 配置环境变量

```powershell
# 临时设置 (当前会话)
$env:WEASYPRINT_DLL_DIRECTORIES = "C:\msys64\mingw64\bin"

# 永久设置 (推荐添加到系统环境变量)
[System.Environment]::SetEnvironmentVariable(
    "WEASYPRINT_DLL_DIRECTORIES",
    "C:\msys64\mingw64\bin",
    [System.EnvironmentVariableTarget]::User
)
```

### 2.3 安装 Python 依赖

```powershell
# 进入项目目录
cd D:\code\终端推理

# 激活虚拟环境 (如果有)
# .\venv\Scripts\activate

# 安装核心依赖
pip install weasyprint markdown2 pyyaml

# 验证安装
weasyprint --info
```

### 2.4 安装中文字体

推荐使用 Google Noto 字体系列，支持中英文混排：

1. 下载 [Noto Sans CJK SC](https://fonts.google.com/noto/specimen/Noto+Sans+SC) (思源黑体)
2. 下载 [Noto Serif CJK SC](https://fonts.google.com/noto/specimen/Noto+Serif+SC) (思源宋体)
3. 安装字体 (双击 .ttf 文件安装)

---

## 3. 系统架构

### 3.1 目录结构

```
D:\code\终端推理\
├── 99_System\
│   ├── scripts\
│   │   ├── md2pdf.py              # 主入口脚本
│   │   └── pdf_exporter\          # PDF 导出模块
│   │       ├── __init__.py
│   │       ├── parser.py          # Markdown 解析器
│   │       ├── math_renderer.py   # 数学公式渲染
│   │       ├── html_builder.py    # HTML 组装器
│   │       └── pdf_generator.py   # PDF 生成器
│   ├── templates\
│   │   └── pdf\
│   │       ├── base.html          # HTML 模板
│   │       └── style.css          # PDF 样式表
│   └── fonts\                     # 字体文件 (可选)
└── 20_Classification\
    └── 分类名\
        └── 论文目录\
            ├── 论文_笔记.md        # 输入
            └── 论文_笔记.pdf       # 输出
```

### 3.2 模块职责

| 模块               | 职责                                                       |
| ------------------ | ---------------------------------------------------------- |
| `parser.py`        | 解析 YAML Frontmatter，提取元数据；调用 markdown2 转换正文 |
| `math_renderer.py` | 检测并渲染 LaTeX 公式为 HTML 或 SVG                        |
| `html_builder.py`  | 组装完整 HTML 文档，注入样式和元数据                       |
| `pdf_generator.py` | 调用 WeasyPrint 生成 PDF，处理路径和资源                   |

---

## 4. 核心实现

### 4.1 Markdown 解析示例

```python
import markdown2
import yaml
import re

def parse_markdown(file_path: str) -> tuple[dict, str]:
    """
    解析 Markdown 文件，提取 YAML Frontmatter 和正文。

    Returns:
        (metadata, html_content)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取 YAML Frontmatter
    frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if match:
        metadata = yaml.safe_load(match.group(1))
        body = content[match.end():]
    else:
        metadata = {}
        body = content

    # 转换 Markdown → HTML
    html = markdown2.markdown(
        body,
        extras=[
            'tables',
            'fenced-code-blocks',
            'code-friendly',
            'header-ids',
            'strike',
            'task_list',
        ]
    )

    return metadata, html
```

### 4.2 数学公式处理方案

由于 WeasyPrint 不支持 JavaScript，需要预渲染 LaTeX 公式。推荐方案：

**方案 A: 使用 matplotlib + LaTeX (推荐)**

```python
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re

def render_latex_to_svg(latex: str, inline: bool = True) -> str:
    """使用 matplotlib 将 LaTeX 渲染为 SVG。"""
    fig, ax = plt.subplots(figsize=(0.01, 0.01))
    ax.axis('off')

    fontsize = 12 if inline else 14
    text = ax.text(
        0.5, 0.5, f'${latex}$',
        fontsize=fontsize,
        ha='center', va='center'
    )

    # 调整图片大小以适应公式
    fig.canvas.draw()
    bbox = text.get_window_extent()
    fig.set_size_inches(bbox.width / fig.dpi + 0.1, bbox.height / fig.dpi + 0.1)

    # 导出 SVG
    buffer = BytesIO()
    fig.savefig(buffer, format='svg', transparent=True, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)

    svg_data = buffer.getvalue().decode('utf-8')
    return svg_data


def process_math_in_html(html: str) -> str:
    """替换 HTML 中的 LaTeX 公式为 SVG。"""
    # 处理行间公式 $$...$$
    def replace_display(match):
        latex = match.group(1)
        svg = render_latex_to_svg(latex, inline=False)
        return f'<div class="math-display">{svg}</div>'

    html = re.sub(r'\$\$(.+?)\$\$', replace_display, html, flags=re.DOTALL)

    # 处理行内公式 $...$
    def replace_inline(match):
        latex = match.group(1)
        svg = render_latex_to_svg(latex, inline=True)
        return f'<span class="math-inline">{svg}</span>'

    html = re.sub(r'(?<!\$)\$([^$]+)\$(?!\$)', replace_inline, html)

    return html
```

**方案 B: 使用 KaTeX 预渲染 (可选)**

```bash
# 安装 Node.js 和 KaTeX CLI
npm install -g katex
```

### 4.3 CSS 样式设计

```css
/* 99_System/templates/pdf/style.css */

/* 页面设置 */
@page {
  size: A4;
  margin: 2cm 2.5cm;

  @top-center {
    content: string(title);
    font-size: 10pt;
    color: #666;
  }

  @bottom-center {
    content: counter(page) " / " counter(pages);
    font-size: 10pt;
    color: #666;
  }
}

@page :first {
  @top-center {
    content: none;
  }
}

/* 字体设置 */
body {
  font-family: "Noto Serif SC", "Source Han Serif SC", serif;
  font-size: 11pt;
  line-height: 1.8;
  color: #333;
}

h1,
h2,
h3,
h4,
h5,
h6 {
  font-family: "Noto Sans SC", "Source Han Sans SC", sans-serif;
  font-weight: 600;
  color: #1a1a1a;
  page-break-after: avoid;
}

h1 {
  font-size: 22pt;
  text-align: center;
  margin-bottom: 2em;
  string-set: title content();
}

h2 {
  font-size: 16pt;
  border-bottom: 2px solid #2563eb;
  padding-bottom: 0.3em;
  margin-top: 1.5em;
}

h3 {
  font-size: 13pt;
  color: #374151;
  margin-top: 1.2em;
}

/* 封面页 */
.cover-page {
  page-break-after: always;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 80vh;
  text-align: center;
}

.cover-title {
  font-size: 28pt;
  font-weight: 700;
  color: #1e3a5f;
  margin-bottom: 0.5em;
}

.cover-authors {
  font-size: 12pt;
  color: #4b5563;
  margin-bottom: 0.5em;
}

.cover-date {
  font-size: 11pt;
  color: #6b7280;
}

/* 图片 */
img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 1em auto;
}

figure {
  text-align: center;
  page-break-inside: avoid;
}

figcaption {
  font-size: 10pt;
  color: #666;
  margin-top: 0.5em;
}

/* 表格 */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1em 0;
  font-size: 10pt;
  page-break-inside: avoid;
}

th,
td {
  border: 1px solid #d1d5db;
  padding: 0.5em 0.8em;
  text-align: left;
}

th {
  background-color: #f3f4f6;
  font-weight: 600;
}

tr:nth-child(even) {
  background-color: #f9fafb;
}

/* 代码块 */
pre {
  background-color: #1e293b;
  color: #e2e8f0;
  padding: 1em;
  border-radius: 6px;
  overflow-x: auto;
  font-size: 9pt;
  line-height: 1.5;
  page-break-inside: avoid;
}

code {
  font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
}

/* 行内代码 */
p code,
li code {
  background-color: #f1f5f9;
  padding: 0.1em 0.3em;
  border-radius: 3px;
  font-size: 0.9em;
  color: #dc2626;
}

/* 数学公式 */
.math-display {
  text-align: center;
  margin: 1em 0;
  page-break-inside: avoid;
}

.math-inline svg {
  vertical-align: middle;
}

/* 引用块 */
blockquote {
  border-left: 4px solid #3b82f6;
  padding-left: 1em;
  margin-left: 0;
  color: #4b5563;
  font-style: italic;
}

/* 列表 */
ul,
ol {
  padding-left: 1.5em;
}

li {
  margin-bottom: 0.3em;
}

/* 分隔线 */
hr {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 2em 0;
}
```

### 4.4 HTML 模板

```html
<!-- 99_System/templates/pdf/base.html -->
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <title>{{ title }}</title>
    <style>
      {{ css_content }}
    </style>
  </head>
  <body>
    <!-- 封面页 -->
    <div class="cover-page">
      <h1 class="cover-title">{{ title }}</h1>
      {% if authors %}
      <p class="cover-authors">{{ authors }}</p>
      {% endif %} {% if year %}
      <p class="cover-date">{{ year }}</p>
      {% endif %}
      <p class="cover-date">笔记生成于 {{ created }}</p>
    </div>

    <!-- 正文 -->
    <main>{{ content }}</main>
  </body>
</html>
```

### 4.5 PDF 生成示例

```python
from weasyprint import HTML, CSS
from pathlib import Path

def generate_pdf(html_content: str, output_path: str, base_url: str = None):
    """
    使用 WeasyPrint 生成 PDF。

    Args:
        html_content: 完整的 HTML 字符串
        output_path: PDF 输出路径
        base_url: 资源基础路径 (用于解析相对路径的图片)
    """
    html = HTML(string=html_content, base_url=base_url)
    html.write_pdf(output_path)
    print(f"PDF 已生成: {output_path}")
```

---

## 5. 使用方式

### 5.1 命令行接口

```powershell
# 转换单个文件
python 99_System/scripts/md2pdf.py --input "path/to/笔记.md"

# 转换目录下所有笔记
python 99_System/scripts/md2pdf.py --input-dir "D:\code\终端推理\20_Classification\动态推理"

# 指定输出目录
python 99_System/scripts/md2pdf.py --input "path/to/笔记.md" --output-dir "D:\PDFs"
```

### 5.2 配置选项

在 `config.py` 中添加 PDF 导出配置：

```python
CONFIG = {
    # ... 其他配置 ...

    "pdf_exporter": {
        "template_dir": "D:/code/终端推理/99_System/templates/pdf",
        "default_font": "Noto Serif SC",
        "page_size": "A4",
        "margin": "2cm",
        "render_math": True,  # 是否渲染 LaTeX 公式
    }
}
```

---

## 6. 已知问题与解决方案

### 6.1 常见问题

| 问题         | 原因                 | 解决方案                                   |
| ------------ | -------------------- | ------------------------------------------ |
| 中文乱码     | 缺少中文字体         | 安装 Noto CJK 字体并在 CSS 中指定          |
| 公式不显示   | WeasyPrint 不支持 JS | 使用预渲染方案 (matplotlib 或 KaTeX CLI)   |
| 图片找不到   | 相对路径问题         | 设置正确的 `base_url`                      |
| PDF 空白     | DLL 未加载           | 检查 `WEASYPRINT_DLL_DIRECTORIES` 环境变量 |
| 杀毒软件报警 | WeasyPrint 误报      | 添加白名单排除                             |

### 6.2 调试技巧

```python
# 验证 WeasyPrint 安装
import weasyprint
print(weasyprint.__version__)

# 测试简单渲染
from weasyprint import HTML
HTML(string='<h1>测试中文</h1>').write_pdf('test.pdf')
```

---

## 7. 后续优化

### 7.1 计划功能

- [ ] 支持目录 (TOC) 自动生成
- [ ] 支持交叉引用和脚注
- [ ] 支持代码高亮 (Pygments)
- [ ] 批量转换进度条
- [ ] PDF 书签生成

### 7.2 性能优化

- 缓存已渲染的数学公式 SVG
- 并行处理多个文件
- 图片自动压缩

---

## 8. 参考资源

- [WeasyPrint 官方文档](https://doc.courtbouillon.org/weasyprint/stable/)
- [WeasyPrint Windows 安装指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows)
- [markdown2 PyPI](https://pypi.org/project/markdown2/)
- [CSS Paged Media 规范](https://www.w3.org/TR/css-page-3/)
- [Noto CJK 字体下载](https://github.com/googlefonts/noto-cjk)
