# Gemini Context Caching 技术指南

## 1. 核心概念

**Context Caching (上下文缓存)** 是 Gemini API (特别是 1.5 Pro/Flash 系列) 提供的一项优化功能，旨在解决长文本/大量背景知识场景下的成本与延迟问题。

### 原理

将大量的背景信息（如长篇论文、书籍、代码库、视频等）预先上传并“缓存”在 Gemini 的服务端，生成一个唯一的缓存 ID。后续的 API 调用只需传递这个 ID，而无需重复传输整个内容。

### 核心优势

1.  **降低成本 (Cost Efficiency)**:
    - **普通模式**: 每次提问都需要支付全部 Input Tokens 的费用。
    - **缓存模式**: 背景内容只需支付一次处理费（及低廉的存储费），后续提问仅需支付“Prompt + 缓存 ID”的极少量费用。
2.  **极速响应 (Low Latency)**:
    - 省去了模型通过注意力机制处理海量 Input Tokens 的时间 (First Token Latency 显著降低)。
3.  **状态隔离 (State Isolation)**:
    - 缓存的背景知识是只读且持久的。
    - 对话历史 (Chat Session) 是临时的。
    - _非常适合“维持背景知识，但频繁重置对话”的应用场景。_

---

## 2. 适用场景建议

Context Caching 并非在所有情况下都是最优解，需根据 Token 量级决定：

- **< 32k Tokens (约 5-6 万中文字)**: **不建议使用缓存**。
  - Gemini 1.5 处理几万 Token 的速度极快，且 API 存在最小缓存门槛限制。直接将文本放入 Prompt 即可。
- **> 32k Tokens**: **强烈建议使用缓存**。
  - 例如：分析整个文件夹的 20 篇论文、整本技术书籍、长视频分析。

---

## 3. 技术实现流程 (Python SDK)

以下基于 `google-generativeai` SDK 的标准实现模式。

### A. 创建缓存 (Create Cache)

这一步将数据上传并锁定，通常只需执行一次。

```python
import google.generativeai as genai
from google.generativeai import caching
import datetime

# 假设这是一个非常大的文本内容（需 > 32k tokens）
large_context_content = open("big_doc.md", "r", encoding="utf-8").read()

cache = caching.CachedContent.create(
    model='models/gemini-1.5-flash-001',
    display_name='project_knowledge_base',
    system_instruction='你是一位资深技术专家，请基于提供的知识库回答问题。',
    contents=[large_context_content],
    ttl=datetime.timedelta(minutes=60) # 生存时间，例如 60 分钟
)

print(f"缓存创建成功，ID: {cache.name}")
```

### B. 基于缓存初始化模型

后续通过缓存 ID 实例化模型对象。

```python
# cache 可以是刚创建的对象，也可以是通过 name 获取的引用
model_with_cache = genai.GenerativeModel.from_cached_content(cached_content=cache)
```

### C. 灵活控制会话 (Session Control)

这展示了如何实现“维持窗口”与“手动重置”的逻辑。

**场景 1：持续对话 (维持窗口)**

```python
# 启动一个新的聊天会话
chat_session = model_with_cache.start_chat(history=[])

# 第一轮交互
response = chat_session.send_message("文档中关于架构优化的部分怎么说？")
print(response.text)

# 第二轮交互（模型记得第一轮的内容）
response = chat_session.send_message("你刚才提到的第二点能详细展开吗？")
print(response.text)
```

**场景 2：重置对话 (保留知识，清除历史)**
当用户点击“开启新对话”时，无需重新上传文档，只需创建一个新的 Session。

```python
# 丢弃旧的 chat_session，直接 New 一个
new_chat_session = model_with_cache.start_chat(history=[])

# 此时模型依然拥有缓存的知识背景，但忘记了之前的具体问答，重新开始
response = new_chat_session.send_message("换个话题，请总结一下实验数据。")
```

---

## 4. 关键限制与注意事项

1.  **32,768 Token 门槛**:
    - 目前创建缓存的内容长度必须超过约 3.2 万 Token。如果内容过短，API 调用会失败。
    - _对策_：如果是单篇短论文，直接作为 System Prompt 传入即可，无需缓存技术。
2.  **TTL (Time To Live) 管理**:
    - 缓存有生命周期（默认较短）。如果因为用户长时间未操作导致缓存过期 (404 Error)，程序需要捕获异常并自动重新创建缓存。
    - 支持 `update()` 方法延长 TTL。
3.  **计费模型**:
    - 缓存的计费包含：**创建操作费** (一次性) + **存储费** (按时长/大小) + **Prompt 输入费** (每次提问)。相比纯 Token 输入，存储费通常远低于重复的输入费。