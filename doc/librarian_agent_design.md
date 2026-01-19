# Librarian Agent Design Document: 智能论文分类与元数据管理

## 1. 概述 (Overview)

本设计文档详细描述了 "Librarian Agent" (图书管理员智能体) 的架构与实现方案。该 Agent 旨在自动化处理本地 Markdown 格式的学术论文，利用 Gemini 大模型对论文摘要进行语义分析，提取元数据，并基于动态生长的分类体系对论文进行自动归档。

### 核心目标

1.  **自动化元数据提取**：从 Markdown 内容中提取或生成关键元数据 (Title, Authors, Year, Tags)。
2.  **动态分类 (Dynamic Classification)**：
    - 不预设固定类别列表。
    - 基于 `D:\code\终端推理\20_Classification` 目录下的现状动态感知现有类别。
    - 利用 Gemini 决策是将论文归入现有类别，还是创建新的 "大模型移动端推理" 相关的子类别。
3.  **节省 Token**：仅读取论文摘要 (Abstract) 进行分析，避免全文读取带来的高昂成本。
4.  **标准化归档**：注入 YAML Frontmatter，并将文件移动到分类目录。

---

## 2. 系统架构 (System Architecture)

### 2.1 目录流转状态

- **输入源 (Staging Area)**: `D:\code\终端推理\10_References` (或 `00_Inbox` 中转换好的临时目录)。此处存放刚由 PDF 转换而来、尚未分类的 Markdown 文件。
- **目标库 (Target Repository)**: `D:\code\终端推理\20_Classification`。此处存放分类整理完成的论文库。

### 2.2 核心模块 (Modules)

系统由以下 Python 模块组成：

1.  **Scanner (扫描器)**: 负责遍历输入目录，识别未处理的 `.md` 文件。文件名称字符数大于10 才认定为论文的md文件
2.  **CategoryManager (类别管理器)**: 负责扫描目标库，维护当前的分类树。
3.  **ContentExtractor (内容提取器)**: 负责从 Markdown 文本中精准提取标题和摘要部分。
4.  **GeminiClassifier (分析与推理)**: 调用 Gemini API，输入摘要 + 现有类别，输出分类决策和元数据。
5.  **MetadataInjector (元数据注入器)**: 将提取的信息按标准 YAML 格式写入文件头部。
6.  **Archivist (归档员)**: 负责创建新目录（如需）并将文件移动到最终位置。

---

## 3. 详细设计逻辑 (Detailed Logic)

### 3.1 动态分类逻辑 (Dynamic Classification Logic)

这是本方案的核心。为了保证分类的准确性和动态生长特性，采取以下策略：

1.  **感知**: 程序启动时，`CategoryManager` 扫描 `20_Classification` 下的一级子目录，生成 list `existing_categories` (例如 `["Quantization", "Model_Compression", "NPU_Scheduling"]`)。
2.  **提示词 (Prompting)**: 向 Gemini 发送 prompt，结构如下：
    - **Context**: "你是一个大模型移动端推理领域的专家研究员。"
    - **Task**: "分析以下论文摘要，将其分类到现有的类别中，或者，如果它属于一个全新的子领域，请建议一个新的类别名称。"
    - **Constraints**:
      - 类别必须严格属于 "大模型移动端相关工作" 的子集。
      - 新类别名称请使用英文 (PascalCase)，简练且具有概括性 (如 `SparseAttention`, `DeviceCloudCollaboration`)。
      - 优先匹配现有类别，除非现有类别完全不适用。
    - **Input Data**:
      - `Abstract`: [论文摘要文本]
      - `Existing Categories`: [类别列表]
    - **Output Format**: JSON 结构，包含 `category_name`, `is_new_category`, `confidence`, `reasoning`, `tags`, `suggested_title` (如果原标题乱码)。

### 3.2 文件处理流程 (Workflow)

1.  **初始化**:
    - 加载 Google Gemini API Key。
    - 扫描 `20_Classification` 获取 `current_categories`。

2.  **处理循环 (对每篇论文)**:
    - **Step 1: 读取**: 读取 Markdown 文件前 2000 个字符 (足以涵盖 Header 和 Abstract)。
    - **Step 2: 提取**: 使用正则或简单文本分析提取 `## Abstract` 段落。如果未找到明确 Abstract 标记，截取前 500 个单词。
    - **Step 3: 推理 (API Call)**:
      - 调用 Gemini。
      - 获取 JSON 响应。
    - **Step 4: 元数据构建**:
      - 准备 YAML Frontmatter 数据字典：
        ```python
        metadata = {
            "title": Gemini返回的清洗后的标题,
            "authors": [需从原文提取或Gemini推测],
            "year": [需从原文提取],
            "tags": Gemini生成的tags列表,
            "category": Gemini决定的category_name,
            "status": "unread",
            "created": datetime.now()
        }
        ```
    - **Step 5: 注入**: 使用 `python-frontmatter` 或手动文件操作，在文件最顶部插入 `---` 包裹的 YAML 块。
    - **Step 6: 移动**:
      - 检查 `20_Classification/<category_name>` 是否存在，若不存在且标记为 `is_new_category` 则创建。
      - `shutil.move(src, dst)` 将文件移动到对应子目录。

---

## 4. 数据结构与接口定义

### 4.1 Gemini Prompt 模板

```markdown
Role: Senior Researcher in Mobile LLM & Edge AI.
Objective: Categorize the given research paper based on its abstract.

Current Categories in Repository:
{existing_categories_str}

Paper Abstract:
"""
{paper_abstract}
"""

Instructions:

1. Analyze the abstract focusing on Mobile LLM inference, edge computing, and model optimization.
2. Decide the best category for this paper.
   - PREFER putting it into one of the "Current Categories".
   - ONLY create a new category if the paper represents a significantly different sub-field within Mobile LLM context that doesn't fit existing ones.
   - New category names must be Concise English PascalCase (e.g., "Quantization", "VisionLanguageModels").
3. Extract specific tags (3-5) related to the technical approach.
4. Return a strictly valid JSON object.

JSON Schema:
{
"category": "string (name of the category)",
"is_new": boolean,
"reason": "string (brief explanation)",
"tags": ["string", "string"],
"clean_title": "string (cleaned up paper title)",
"publication_year": "string or null (estimate from text if possible, else null)"
}
```

### 4.2 目标目录结构示例

```text
D:\code\终端推理\20_Classification\
├── KV_Cache_Optimization\
│   ├── paper1.md
│   └── paper2.md
├── NPU_Acceleration\
│   └── text.md
├── OnDevice_Training\
│   └── ...
└── ... (Dynamically created)
```

---

## 5. 实现计划 (Implementation Plan)

### Step 1: 环境准备

需要安装 Google Generative AI SDK。

```bash
pip install google-generativeai python-frontmatter
```

### Step 2: 编写 `librarian.py`

在 `D:\code\终端推理\99_System\scripts` 下创建新脚本 `librarian.py`。

- 实现 `LibraryScanner` 类。
- 实现 `GeminiClassifier` 类 (封装 API 调用)。
- 实现 `main` 函数串联流程。

### Step 3: API Key 配置

将 API Key 放入 `config.py` 或环境变量，确保不硬编码在代码中。

### Step 4: 测试

1.  在 `20_Classification` 手动创建 1-2 个基础文件夹（如 `General`, `Quantization`）作为种子。
2.  选取 3 篇已转换的 MD 文件作为输入进行测试。
3.  验证分类是否合理，YAML 是否正确注入。

---

## 6. 注意事项

1.  **API 限制**: 注意 Gemini 的速率限制 (RPM)，批量处理时建议加入 `time.sleep`。
2.  **错误处理**: 如果 API 调用失败，不应移动文件，而是通过日志记录并跳过，留待下次处理。
3.  **文件名清洗**: 移动文件时，建议同时清洗文件名（去除特殊字符），确保文件系统兼容性。
4.  **摘要提取鲁棒性**: 此时转换的 Markdown 格式可能不完全统一，提取摘要的正则需要有一定的容错能力（如查找 "Abstract", "ABSTRACT", 或取前段文本）。
