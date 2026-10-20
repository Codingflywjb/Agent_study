# Agent_study

基于 Python、FastAPI 和 DeepSeek 的 LLM 应用后端开发实践项目。

## 1. 项目简介

Agent_study 是一个面向 AI Agent 开发的 Python 后端学习与实践项目，旨在通过实际编码，逐步掌握从基础后端开发到大语言模型应用开发的相关技术。

项目初期实现了基于 FastAPI 和 SQLite 的任务管理系统，涵盖 RESTful API 设计、数据持久化、参数校验和异常处理等基础功能。

在此基础上，项目集成 DeepSeek 大语言模型，实现了单轮及多轮对话、终端流式输出，以及基于 FastAPI 和 Server-Sent Events（SSE）的 HTTP 流式聊天接口。

项目使用 pytest、MagicMock 和 monkeypatch 编写自动化测试，覆盖正常请求、参数错误、网络异常及流式响应中断等场景，注重模块化设计、异常处理与代码可维护性。

目前项目已完成基础 LLM 应用后端开发，后续计划逐步引入 RAG、工具调用与 Agent 工作流。

## 2. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 核心开发语言 |
| FastAPI | 后端 API 开发 |
| Pydantic | 请求参数校验与数据模型定义 |
| SQLite | 任务数据持久化 |
| DeepSeek API | 大语言模型推理 |
| OpenAI Python SDK | 调用兼容 OpenAI API 的 DeepSeek 服务 |
| StreamingResponse + SSE | HTTP 流式响应 |
| pytest | 自动化测试 |
| MagicMock / monkeypatch | 模拟外部依赖和异常场景 |
| Git / GitHub | 版本控制与代码管理 |

## 3. 已实现功能

### 3.1 任务管理系统

- 实现任务的创建、查询、更新和删除（CRUD）。
- 使用 SQLite 持久化任务数据。
- 基于 FastAPI 提供 RESTful API。
- 使用 Pydantic 校验请求参数。
- 实现任务 ID 校验、异常处理与日志记录。

### 3.2 DeepSeek 大模型集成

- 使用 OpenAI Python SDK 调用 DeepSeek API。
- 实现用户消息构造与模型响应解析。
- 支持通过历史消息实现多轮对话。
- 支持 API Key 环境变量配置。
- 统一封装模型客户端初始化逻辑。
- 处理身份认证、请求超时、网络连接异常和请求限流等错误。

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

### 3.5 自动化测试

使用 pytest 对主要模块编写自动化测试，包括：

- 任务管理 CRUD 功能测试。
- API 请求参数校验测试。
- LLM 客户端 Mock 测试。
- 多轮对话历史管理测试。
- Streaming 正常输出与异常中断测试。
- HTTP 200、422、502 等接口响应测试。
- SSE 事件格式、内容及顺序测试。
- 空流、空格 Chunk、`None` Chunk 等边界场景测试。

最近一次针对 `app.api` 和 `practice.llm_client` 两个模块的覆盖率统计为 **88%**，其中 FastAPI 接口模块为 **94%**，LLM 客户端模块为 **80%**。

## 4. 项目结构

主要目录及文件：

```text
Agent_study/
├── app/
│   ├── api.py              # FastAPI 接口
│   ├── config.py           # 环境配置
│   ├── sqlite_storage.py   # SQLite 数据操作
│   ├── task_manager.py     # 任务业务逻辑
│   └── storage.py          # JSON 数据存储
├── practice/
│   ├── llm_client.py       # DeepSeek 客户端封装
│   ├── llm_messages.py     # LLM 消息构造与解析
│   ├── llm_stream.py       # 流式响应处理
│   └── chat_cli.py         # 命令行多轮聊天
├── tests/                  # 自动化测试
├── requirements.txt        # 项目依赖
└── README.md
```

## 5. 快速开始

### 安装依赖

```bash
python -m pip install -r requirements.txt
```

### 配置环境变量

在项目根目录创建 `.env` 文件，根据实际情况配置：

```env
DEEPSEEK_API_KEY=your_api_key
LLM_MODEL=your_deepseek_model
```

请将模型名称设置为实际可用的 DeepSeek 模型。

注意：不要将 `.env` 或真实 API Key 提交到 GitHub。

### 启动 FastAPI

```bash
python -m uvicorn app.api:app --reload
```

启动后访问：

http://127.0.0.1:8000/docs

即可使用 FastAPI 自动生成的 Swagger UI 测试接口。

## 6. API 使用示例

### 普通聊天接口

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

### SSE 流式聊天接口

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

运行所有测试：

```bash
python -m pytest -v
```

查看主要 LLM 后端模块的测试覆盖率：

```bash
python -m pytest tests/test_api.py tests/test_llm_client.py --cov=app.api --cov=practice.llm_client --cov-report=term-missing
```

项目使用 Mock 隔离真实模型调用，降低测试对外部网络和 API 额度的依赖。

## 8. 后续开发计划

- [ ] 实现基于文档检索的 RAG 问答
- [ ] 引入向量数据库与 Embedding
- [ ] 实现 LLM Function Calling
- [ ] 使用 LangGraph 构建 Agent 工作流
- [ ] 学习并集成 MCP 工具协议
- [ ] 完善 Agent 应用的测试、日志与部署

---

**项目状态：持续开发中**

这是一个以实际工程能力训练为目标的学习项目，后续会随着 Agent 开发学习进度持续迭代。
