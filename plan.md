# Paper-to-Insight System: 本地论文知识库架构设计

> **项目状态**: 🟢 核心功能已完成 (2026-01-18)
>
> 已实现 PDF 到 Markdown 的自动转换，支持 5 张 L40 GPU 并行处理。

---

## 1. 项目目录架构设计 (Obsidian Vault Structure)

符合 Obsidian 最佳实践的标准目录结构：

```text
d:\code\终端推理\
├── 00_Inbox/               # [入口] 收件箱。只需把 PDF 丢进去，工作流自动开始。
│   └── 论文仓库/           #     按需分子文件夹组织待转换论文
│       ├── 01 - 10 篇/     #     ✅ 已转换 (9篇)
│       └── ...
│
├── 10_References/          # [仓库] 存储转换后的 Markdown 论文
│   ├── 论文名/             #     每篇论文一个文件夹
│   │   ├── 论文名.md       #     Markdown 正文
│   │   ├── _meta.json      #     元数据
│   │   └── *.jpeg          #     提取的图片
│   └── ...
│
├── 20_Ideas/               # [生产] Ideas/想法笔记 (待实现)
│   ├── Brainstorming/      #     初期灵感
│   └── Daily_Notes/        #     日常记录
│
├── 30_Attachments/         # [资源] 公共附件资源
│
├── 99_System/              # [核心] 自动化脚本、配置
│   ├── scripts/            #     ✅ Python 转换脚本 (模块化)
│   │   ├── convert.py            # 主入口
│   │   ├── config.py             # 配置
│   │   ├── converter/            # 转换模块
│   │   │   ├── ssh_executor.py   # SSH 执行
│   │   │   ├── marker_converter.py # Marker 转换
│   │   │   └── sync.py           # rsync 同步
│   │   ├── utils/                # 工具模块
│   │   │   ├── logger.py         # 日志
│   │   │   └── path_utils.py     # 路径工具
│   │   └── requirements.txt      # Python 依赖
│   ├── prompts/            #     Agent 指令文件 (待实现)
│   └── logs/               #     运行日志
│
└── .obsidian/              # Obsidian 配置文件夹
```

---

## 2. ✅ 已完成的工作

### 2.1 核心转换脚本 (模块化重构)

代码已重构为模块化结构，易于维护和扩展：

| 模块                            | 职责                |
| ------------------------------- | ------------------- |
| `convert.py`                    | CLI 入口，命令解析  |
| `config.py`                     | 集中配置管理        |
| `converter/ssh_executor.py`     | SSH 命令执行        |
| `converter/marker_converter.py` | Marker 转换核心逻辑 |
| `converter/sync.py`             | rsync 结果同步      |
| `utils/logger.py`               | 日志配置            |
| `utils/path_utils.py`           | 路径转换工具        |

**功能特性**：

- ✅ **远程执行**: 通过 SSH 连接 L40 服务器执行 Marker 转换
- ✅ **多 GPU 并行**: 支持 5 张 L40 并行转换 (75 workers)
- ✅ **智能跳过**: 自动检测已转换论文，避免重复处理
- ✅ **本地同步**: 使用 rsync 将结果同步回本地

**使用命令**：

```powershell
cd D:\code\终端推理\99_System\scripts

# 批量转换指定子目录 (多GPU并行，自动跳过已转换)
python convert.py convert --folder "论文仓库/01 - 10 篇"

# 转换整个 Inbox
python convert.py convert --all

# 转换单个文件
python convert.py convert -f "论文仓库/paper.pdf"

# 仅同步结果 (不转换)
python convert.py sync
```

**配置参数** (`config.py`):

```python
CONFIG = {
    "remote": {
        "ssh_alias": "L40",
        "host": "10.242.6.9",
        "username": "ghs",
        "conda_env": "pdf",
    },
    "marker": {
        "num_devices": 5,
        "num_workers": 15,
    },
}
```

### 2.2 性能数据

| 模式              | 10 篇论文转换时间 | 速度       |
| ----------------- | ----------------- | ---------- |
| 串行 (单GPU)      | ~10 分钟          | 1x         |
| **并行 (5张L40)** | **~2分45秒**      | **~4x** ⚡ |

### 2.3 已转换论文 (19篇)

```
10_References/
├── 01-10 篇目录 (9篇)
└── 11-20 篇目录 (10篇)
```

---

## 3. AI Agent 工作流架构 (规划中)

采用**事件驱动**的流水线（Watcher + Agent）。

### 阶段一：摄入与转换 (Ingestion & Conversion) ✅ 已完成

- **触发器**：监控 `00_Inbox` 文件夹，自动发现新增 `.pdf` 文件
- **执行器**：调用远程服务器 `Marker` 进行格式转换 (5x L40 并行)
- **产出**：生成同名 `.md` 文件和图片资源

### 阶段二：Agent 整理 (The Librarian Agent) 🔜 待实现

- **阅读理解**：读取转换后的 MD 文件头部（标题、摘要）
- **元数据处理**：
  - 提取 Meta 信息：Title, Authors, Year, Keywords
  - **自动分类**：分析内容，将文件移动到 `10_References` 下合适的子类目。切记 ，这里分析内容不能读取文章全部内容，只需要读取摘要信息。
    如果读取全部内容，会非常消耗token。
  - **Frontmatter 注入**：写入 YAML 头方便 Obsidian 检索
  ```yaml
  ---
  tag: #paper #edge-inference
  status: unread
  created: 2026-01-18
  authors: ["Author1", "Author2"]
  year: 2025
  ---
  ```

### 阶段三：Agent 启发 (The Research Fellow Agent) 🔜 待实现

- **连接思维**：基于论文摘要，在 `20_Ideas` 中创建启示笔记
- **内容生成**：
  - **One-Liner**: 一句话总结核心贡献
  - **Critical Questions**: 提出 3 个关键问题
  - **Idea Spark**: 结合你的"生产"目标，生成灵感方向
  - **双链 (Backlinks)**: 自动添加 `[[论文文件名]]`，实现笔记与原文跳转

---

## 4. 🔜 未来开发计划

### Phase 1: 元数据提取与整理 (优先级: 高)

1. **YAML Frontmatter 自动注入**
   - 从 `_meta.json` 或 Markdown 正文提取标题、作者、年份
   - 自动添加 `tag`, `status`, `created` 等字段
   - 使 Obsidian Dataview 插件可检索

2. **自动分类系统**
   - 使用 LLM 分析论文摘要，识别研究领域
   - 自动创建子目录并移动文件
   - 支持自定义分类规则

### Phase 2: AI 启发笔记生成 (优先级: 中)

1. **论文摘要生成**
   - 调用 LLM API 生成一句话总结
   - 提取核心贡献和创新点

2. **启发笔记自动创建**
   - 在 `20_Ideas/` 下生成关联笔记
   - 包含批判性问题和研究灵感
   - 自动创建双向链接

### Phase 3: 增强功能 (优先级: 低)

1. **Web UI 界面**
   - 可视化转换进度
   - 论文管理面板

2. **批注同步**
   - PDF 批注提取
   - 与 Markdown 笔记关联

3. **知识图谱**
   - 论文引用关系可视化
   - 研究主题聚类

---

## 5. 技术栈

| 组件     | 技术选型            | 状态      |
| -------- | ------------------- | --------- |
| PDF 转换 | Marker (远程 GPU)   | ✅ 已实现 |
| 脚本语言 | Python 3.10+        | ✅ 已实现 |
| 远程执行 | SSH + rsync via WSL | ✅ 已实现 |
| 文件同步 | VS Code SFTP 插件   | ✅ 已配置 |
| 笔记阅读 | Obsidian            | ✅ 可用   |
| LLM 接入 | OpenAI SDK (兼容)   | 🔜 待实现 |
| 文件监控 | watchdog (Python)   | ⏸️ 待完善 |

---

## 6. 快速开始

### 环境要求

- Windows 10/11 + WSL
- Python 3.10+
- 配置好的 SSH 连接 (L40 服务器)

### 安装依赖

```powershell
cd D:\code\终端推理\99_System\scripts
pip install -r requirements.txt
```

### 使用流程

1. 将 PDF 论文放入 `00_Inbox/` 下的某个子目录
2. 等待 SFTP 自动同步到服务器
3. 运行转换命令:
   ```powershell
   python convert.py convert --folder "论文仓库/01 - 10 篇"
   ```
4. 在 Obsidian 中打开 `D:\code\终端推理` 作为 Vault
5. 浏览 `10_References/` 阅读转换后的论文

---

_最后更新: 2026-01-18_
