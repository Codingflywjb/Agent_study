
# Agent_study

基于 Python、FastAPI 和 DeepSeek 的 LLM 应用后端与 AI Agent 开发实践项目。

本项目以实际工程能力训练为目标，逐步实现从 Python 后端开发、LLM API 集成、RAG 文档检索增强，到 Function Calling 和 Agent Loop 的完整开发流程。

## 1. 项目简介

Agent_study 是一个面向 AI Agent 开发的 Python 后端学习与实践项目，旨在通过实际编码，逐步掌握大语言模型应用开发所需的核心技术与工程实践。

项目初期实现了基于 FastAPI 和 SQLite 的任务管理系统，涵盖 RESTful API 设计、数据持久化、参数校验和异常处理等基础功能，并保留了基于 JSON 文件存储的任务管理实现。

在此基础上，项目集成 DeepSeek 大语言模型，实现单轮及多轮对话、终端流式输出，以及基于 FastAPI 和 Server-Sent Events（SSE）的 HTTP 流式聊天接口。

随后，项目引入 RAG（Retrieval-Augmented Generation，检索增强生成），使用 Embedding 模型进行文本向量化，构建本地向量索引，并将检索结果与 DeepSeek 生成能力结合，实现基于外部知识的问答流程。

目前，项目已进一步实现基于 Function Calling 的任务管理 Agent，支持工具参数校验、模拟与真实执行模式、Tool Message 回传、多轮 Agent Loop、异常处理，以及单次运行内的重复工具调用防护。

项目使用 pytest、MagicMock、monkeypatch 等工具编写自动化测试，注重模块化设计、可测试性、异常恢复和代码可维护性。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 核心开发语言 |
| FastAPI | 后端 API 开发 |
| Pydantic | 请求参数校验与数据模型定义 |
| SQLite | 任务数据持久化 |
| JSON | 本地任务数据及向量索引存储 |
| DeepSeek API | 大语言模型推理与工具调用 |
| OpenAI Python SDK | 调用兼容 OpenAI API 的 DeepSeek 服务 |
| BGE Embedding | 文本向量化与语义检索 |
| RAG | 检索增强生成 |
| Function Calling | 结构化工具调用 |
| Agent Loop | 多轮工具执行与模型交互 |
| StreamingResponse + SSE | HTTP 流式响应 |
| pytest | 自动化测试 |
| MagicMock / monkeypatch | 模拟外部依赖与异常场景 |
| Git / GitHub | 版本控制与代码管理 |

## 3. 已实现功能

### 3.1 任务管理系统

- 实现任务的创建、查询、更新和删除（CRUD）。
- 使用 SQLite 持久化任务数据。
- 基于 FastAPI 提供 RESTful API。
- 使用 Pydantic 校验请求参数。
- 实现任务 ID 校验、异常处理与日志记录。
- 保留基于 JSON 文件的任务存储模块，为 Agent 工具提供实际执行能力。

### 3.2 DeepSeek 大模型集成

- 使用 OpenAI Python SDK 调用 DeepSeek API。
- 实现用户消息构造与模型响应解析。
- 支持通过历史消息实现多轮对话。
- 支持 API Key 环境变量配置。
- 封装模型客户端初始化逻辑。
- 处理身份认证、请求超时、网络连接异常和请求限流等错误。
- 使用 Mock 隔离外部模型服务，编写自动化测试。

### 3.3 Streaming 流式输出

- 使用生成器 `yield` 逐步处理模型返回的 Chunk。
- 支持终端实时输出模型回答。
- 处理空 Chunk、`None` 内容及无有效文本的响应。
- 保留回答中的空格和换行等原始文本内容。
- 在流式中断时进行异常转换，避免将未完成的回答视为成功结果。

### 3.4 FastAPI AI 聊天接口

**POST /chat**

提供普通 HTTP 聊天接口：

- 接收用户问题和历史消息。
- 使用 Pydantic 校验请求内容。
- 调用 DeepSeek 生成回答。
- 以 JSON 格式返回完整结果。
- 处理参数错误和模型服务异常。

**POST /chat/stream**

提供基于 SSE 的流式聊天接口：

- 通过 `StreamingResponse` 返回增量文本。
- 使用 `token` 事件发送回答片段。
- 使用 `done` 事件表示正常完成。
- 使用 `error` 事件通知客户端流式生成失败。
- 支持在模型流式输出中断时发送错误事件。

### 3.5 RAG 检索增强生成

项目实现了基础的 RAG 问答流程，将外部文本知识检索与大语言模型生成能力相结合。

主要功能：

- 使用 BGE Embedding 模型生成文本向量。
- 构建基于 JSON 文件的本地向量索引。
- 迭代向量索引结构，支持索引元数据管理。
- 根据用户问题进行语义相似度检索。
- 将检索得到的相关内容组织为模型上下文。
- 调用 DeepSeek，根据检索内容生成回答。
- 实践检索模块与生成模块的分离设计。

当前以本地索引和基础检索流程为主，后续计划引入专用向量数据库、检索效果评估以及更完善的知识库管理能力。

### 3.6 Function Calling 与任务管理 Agent

基于 DeepSeek Function Calling 实现任务管理 Agent。

**工具定义与参数校验**

- 使用 JSON Schema 定义 `add_task` 工具。
- 将 SDK Tool Call 对象转换为 Python 字典。
- 校验工具名称和参数。
- 处理非法 JSON、缺失参数及无效工具调用。
- 将工具执行结果转换为标准 Tool Message。

**Agent Loop**

- 支持模型自主判断是否调用工具。
- 支持多轮模型请求与工具调用。
- 支持一次模型响应中包含多个 Tool Call。
- 将工具执行结果回传模型，生成最终回答。
- 使用 `max_rounds` 限制最大请求轮数。
- 对预期的工具参数错误进行处理，避免直接中断 Agent Loop。

**模拟与真实执行模式**

工具执行器支持两种模式：

| 模式 | 行为 |
|---|---|
| `dry_run=True` | 模拟工具执行，不实际保存任务 |
| `dry_run=False` | 调用真实任务管理模块，允许写入 JSON 文件 |

默认使用 `dry_run=True`，降低开发和测试过程中意外修改任务数据的风险。

真实文件写入已通过 pytest 的 `tmp_path` 临时目录进行验证，测试不会依赖日常任务文件。

**重复工具调用保护**

使用缓存记录单次 Agent Loop 中已经处理过的工具调用：

- 相同 Tool Call ID、相同工具签名：复用缓存结果。
- 相同 Tool Call ID、不同工具签名：返回 `tool_call_id_conflict` 错误。
- 避免同一运行过程中因重复 Tool Call ID 导致重复执行。
- 冲突时不覆盖已经缓存的执行结果。

当前使用工具名称与原始参数字符串构造签名，并在单次 Agent Loop 运行期间保存缓存。

该机制尚不支持跨进程、跨请求或不同 Tool Call ID 的完整幂等性保护。

### 3.7 任务管理 Agent 命令行演示

通过独立程序入口运行任务管理 Agent。

已实现：

- 从环境变量读取 DeepSeek API Key。
- 初始化 DeepSeek Client。
- 接收用户自然语言输入。
- 调用 Agent Loop。
- 根据工具执行结果生成最终回答。
- 默认使用模拟模式，不实际修改任务文件。

示例输入：

```text
帮我添加任务：学习 FastAPI
```

示例交互流程：

```text
[第 1 轮] action=tool, tool_calls=1
[第 2 轮] action=final, tool_calls=0
```

随后模型会生成最终回答，并明确说明任务当前处于模拟处理状态、尚未实际保存。

### 3.8 自动化测试

使用 pytest 对主要功能进行自动化测试，包括：

**后端与 LLM 测试**

- 任务管理 CRUD 功能测试。
- API 请求参数校验测试。
- LLM 客户端 Mock 测试。
- 多轮对话历史管理测试。
- Streaming 正常输出与异常中断测试。
- HTTP 200、422、502 等接口响应测试。
- SSE 事件格式、内容及顺序测试。
- 空流、空格 Chunk、`None` Chunk 等边界场景测试。

**Agent 工程测试**

- Tool Call 参数校验测试。
- Tool Message 构造测试。
- Agent Loop 直接回答与工具调用测试。
- 多工具调用和最大轮数限制测试。
- 非法工具参数与异常恢复测试。
- Dry Run 模式执行隔离测试。
- 真实执行模式 Mock 测试。
- 使用 `tmp_path` 的 JSON 实际写入测试。
- 相同 Tool Call ID 重复执行防护测试。
- 相同 ID、不同参数的冲突检测测试。
- 命令行程序入口集成测试。

Agent Loop 相关测试已完成多轮回归验证，命令行入口也已通过离线 Mock 集成测试。

## 4. 项目结构

主要目录及文件如下：

```text
Agent_study/
├── app/
│   ├── api.py                   # FastAPI 接口
│   ├── config.py                # 环境配置
│   ├── sqlite_storage.py        # SQLite 数据操作
│   ├── task_manager.py          # 任务业务逻辑
│   └── storage.py               # JSON 数据存储
│
├── practice/
│   ├── llm_client.py            # DeepSeek 客户端封装
│   ├── llm_messages.py          # LLM 消息构造与解析
│   ├── llm_stream.py            # 流式响应处理
│   ├── chat_cli.py              # 命令行多轮聊天
│   │
│   ├── agent_tools.py           # 工具定义与参数校验
│   ├── agent_tool_messages.py   # Tool Message 构造
│   ├── agent_tool_errors.py     # 工具异常处理
│   ├── agent_tool_executor.py   # 安全工具执行器
│   ├── agent_loop.py            # Agent 多轮执行循环
│   └── agent_demo.py            # Agent 命令行入口
│
├── tests/
│   ├── test_agent_loop.py
│   ├── test_agent_tool_executor.py
│   ├── test_agent_demo.py
│   └── ...                      # 其他自动化测试
│
├── requirements.txt             # 项目依赖
└── README.md
```

项目另外包含 RAG 相关的向量化、索引构建与检索实践代码。

## 5. 快速开始

### 5.1 安装依赖

建议使用 Python 虚拟环境。

在项目根目录执行：

```bash
python -m pip install -r requirements.txt
```

请确保依赖中包含运行相应模块所需的 `openai`、`python-dotenv`、`fastapi`、`pytest` 等包。

### 5.2 配置环境变量

在项目根目录创建 `.env` 文件。

```env
DEEPSEEK_API_KEY=your_api_key_here
LLM_MODEL=your_deepseek_model
```

请根据实际使用的模块配置可用的 DeepSeek 模型名称。

注意：部分独立练习代码可能直接在请求中指定模型名称，不一定读取 `LLM_MODEL`。

**安全要求：**

- 不要将真实 API Key 写入源代码。
- 不要将 `.env` 提交到 GitHub。
- 确认 `.gitignore` 已排除 `.env` 等敏感配置文件。

### 5.3 启动 FastAPI

```bash
python -m uvicorn app.api:app --reload
```

启动后访问：

http://127.0.0.1:8000/docs

即可通过 Swagger UI 测试接口。

### 5.4 启动任务管理 Agent

在项目根目录执行：

```bash
python -m practice.agent_demo
```

输入自然语言，例如：

```text
帮我添加任务：学习 FastAPI
```

程序会调用 DeepSeek，并根据模型生成的 Tool Call 执行模拟工具调用。

**当前命令行演示默认启用 `dry_run=True`，不会实际保存任务。**

不要仅凭模型的文字回复判断任务是否持久化，应以工具返回的 `saved` 状态及实际存储结果为准。

### 5.5 运行原有命令行任务管理系统

```bash
python -m app.main
```

该入口用于原有任务管理功能，与 Agent 命令行演示入口不同。

## 6. API 使用示例

### 6.1 普通聊天接口

请求地址：

`POST /chat`

请求示例：

```json
{
  "message": "它有哪些优点？",
  "history": [
    {
      "role": "user",
      "content": "什么是 FastAPI？"
    },
    {
      "role": "assistant",
      "content": "FastAPI 是一个 Python Web 框架。"
    }
  ]
}
```

响应示例：

```json
{
  "answer": "FastAPI 具有高性能、类型标注支持和自动生成 API 文档等优点。"
}
```

### 6.2 SSE 流式聊天接口

请求地址：

`POST /chat/stream`

请求示例：

```json
{
  "message": "介绍一下 Python",
  "history": []
}
```

响应格式示例：

```text
event: token
data: {"delta": "Python 是"}

event: token
data: {"delta": "一种编程语言。"}

event: done
data: {}
```

如果流式生成中途发生模型服务异常，则通过 `error` 事件返回错误信息。

## 7. 自动化测试

### 7.1 运行所有测试

```bash
python -m pytest -v
```

### 7.2 运行 Agent Loop 测试

```bash
python -m pytest tests/test_agent_loop.py -v
```

### 7.3 运行工具执行器测试

```bash
python -m pytest tests/test_agent_tool_executor.py -v
```

### 7.4 运行命令行入口测试

```bash
python -m pytest tests/test_agent_demo.py -v
```

### 7.5 查看主要 LLM 后端模块覆盖率

```bash
python -m pytest tests/test_api.py tests/test_llm_client.py --cov=app.api --cov=practice.llm_client --cov-report=term-missing
```

此前针对 `app.api` 和 `practice.llm_client` 两个模块的覆盖率统计为 88%，其中 FastAPI 接口模块为 94%，LLM 客户端模块为 80%。

以上数据是此前特定模块、特定测试范围的历史结果，不代表当前整个仓库的测试覆盖率。

项目使用 Mock 隔离真实模型调用，降低自动化测试对外部网络、API Key 和模型调用额度的依赖。

## 8. Agent 执行安全与当前限制

本项目目前主要面向学习与工程实践，尚不属于生产级 Agent 系统。

当前已实现：

- 默认 Dry Run 模式。
- 工具参数校验。
- 工具执行结果回传。
- 预期参数错误处理。
- 单次运行内的工具调用结果缓存。
- 同一 Tool Call ID 的工具签名冲突检测。
- 基于临时目录的真实文件写入测试。

当前尚未实现：

- 跨请求、跨进程的持久化幂等性。
- 不同 Tool Call ID 对应相同操作时的业务级去重。
- 工具执行的事务保障及失败补偿。
- 完整的权限控制与用户确认机制。
- 面向生产环境的持久化执行记录和审计机制。

因此，命令行演示继续默认使用模拟模式，真实写入能力主要通过隔离测试环境进行验证。

## 9. 后续开发计划

- [x] Python 后端基础与任务管理系统
- [x] FastAPI RESTful API
- [x] DeepSeek API 接入
- [x] 多轮对话与 Streaming
- [x] FastAPI SSE 流式聊天
- [x] 基础 RAG 检索增强生成
- [x] BGE Embedding 与本地向量索引
- [x] Function Calling 与工具参数校验
- [x] Agent Loop 多轮工具调用
- [x] Dry Run 与真实工具执行器
- [x] 单次运行内重复工具调用防护
- [x] Agent 命令行演示与自动化测试
- [ ] 引入专用向量数据库
- [ ] 优化 RAG 检索效果与评估流程
- [ ] 完善 Agent 工具执行状态管理
- [ ] 实现跨请求的持久化幂等性
- [ ] 使用 LangGraph 构建更复杂的 Agent 工作流
- [ ] 学习并集成 MCP 工具协议
- [ ] 完善日志、监控、错误恢复和权限控制
- [ ] 增加 Agent API 服务与部署能力
- [ ] 完善项目文档和可复现演示

---

**项目状态：持续开发中**

本项目以实际编码和工程实践为核心，持续迭代 Python 后端、LLM 应用、RAG 和 Agent 开发能力。

当前重点是构建可运行、可测试、具备基本安全控制的 Agent 系统，并逐步完善项目的可靠性与工程化能力。
