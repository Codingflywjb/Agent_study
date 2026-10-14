# Agent_study

基于 Python、FastAPI 和 SQLite 的任务管理后端项目。

本项目是个人 AI Agent 开发学习计划的一部分，主要用于实践 Python 后端开发、RESTful API 设计、数据库持久化、自动化测试及项目工程化，为后续开发 LLM 应用、RAG 系统和 AI Agent 奠定基础。

## 1. 技术栈

| 技术 | 用途 |
|---|---|
| Python | 后端开发语言 |
| FastAPI | 构建 RESTful API |
| Pydantic | 请求数据校验 |
| SQLite | 任务数据持久化 |
| python-dotenv | 环境变量与配置管理 |
| pytest | 自动化测试 |
| HTTPX / TestClient | HTTP 接口集成测试 |
| Git / GitHub | 版本控制与代码管理 |

## 2. 已实现功能

### 任务管理

- 创建任务
- 查询全部任务
- 根据 ID 查询任务
- 修改任务完成状态
- 删除任务
- 任务数据持久化存储

### 后端工程化

- 使用 FastAPI 提供 HTTP 接口
- 使用 Pydantic 进行请求参数校验
- 使用 SQLite 存储任务数据
- 使用参数化 SQL 操作数据库
- 使用 FastAPI Lifespan 自动初始化数据库
- 使用 `.env` 管理项目配置
- 使用 logging 记录应用运行日志
- 使用 pytest 编写业务和接口自动化测试
- 使用临时数据库实现测试隔离

## 3. 项目结构

```text
Agent_study/
├── app/
│   ├── __init__.py
│   ├── api.py              # FastAPI 接口
│   ├── config.py           # 项目配置
│   ├── sqlite_storage.py   # SQLite 数据访问层
│   ├── task_manager.py     # 早期任务管理逻辑
│   ├── storage.py          # JSON 存储实现
│   └── main.py             # 命令行任务管理程序
├── tests/
│   ├── test_api.py         # HTTP 接口测试
│   └── test_sqlite_storage.py  # SQLite 测试
├── practice/               # 学习练习代码
├── data/                   # 本地数据目录
├── .gitignore
├── requirements.txt
└── README.md
```

项目保留了早期的 JSON 存储实现，用于学习和对比。目前 FastAPI 的核心任务接口已经迁移到 SQLite。

## 4. 安装与运行

### 4.1 克隆项目

```bash
git clone https://github.com/wjiangbo463-ship-explorer/Agent_study.git
cd Agent_study
```

### 4.2 创建虚拟环境

```bash
python -m venv .venv
```

Windows PowerShell 激活虚拟环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

### 4.3 安装依赖

```bash
python -m pip install -r requirements.txt
```

建议使用 Python 3.10 或更高版本。

### 4.4 配置环境变量

项目支持通过根目录下的 `.env` 文件配置运行参数。

示例：

```dotenv
LLM_MODEL=gpt-4.1-mini
```

数据库路径默认指向 `data/tasks.db`，也可以通过 `TASKS_DB_PATH` 环境变量覆盖。

`.env` 文件是可选的，未配置时程序使用默认值。

注意：`.env` 已被 Git 忽略。不要将真实 API Key、密码或其他敏感信息提交到仓库。

目前 `LLM_MODEL` 仅作为后续大模型开发的预留配置，项目尚未实现 LLM API 调用。

### 4.5 启动 FastAPI

在项目根目录执行：

```bash
python -m uvicorn app.api:app --reload
```

启动成功后，访问：

- Swagger API 文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

首次启动时，FastAPI 会通过 Lifespan 自动初始化 SQLite 数据库及 `tasks` 表，不需要手动执行建表脚本。

## 5. REST API 接口

| 方法 | 路径 | 功能 | 成功状态码 |
|---|---|---|---|
| GET | `/health` | 健康检查 | 200 |
| GET | `/tasks` | 查询全部任务 | 200 |
| GET | `/tasks/{task_id}` | 查询指定任务 | 200 |
| POST | `/tasks` | 创建任务 | 201 |
| PATCH | `/tasks/{task_id}` | 修改完成状态 | 200 |
| DELETE | `/tasks/{task_id}` | 删除任务 | 204 |

### 创建任务

请求：

```http
POST /tasks
Content-Type: application/json
```

请求体：

```json
{
  "title": "学习LangGraph"
}
```

成功响应示例：

```json
{
  "id": 1,
  "title": "学习LangGraph",
  "completed": false
}
```

任务 ID 由数据库自动生成。

### 修改任务状态

请求：

```http
PATCH /tasks/1
Content-Type: application/json
```

请求体：

```json
{
  "completed": true
}
```

成功响应示例：

```json
{
  "id": 1,
  "title": "学习LangGraph",
  "completed": true
}
```

### 删除任务

```http
DELETE /tasks/1
```

删除成功返回 HTTP 204，响应体为空。

当目标任务不存在时，接口返回 HTTP 404；请求参数校验不通过时，通常返回 HTTP 422。

## 6. 自动化测试

项目使用 pytest 进行自动化测试，包括：

- 任务创建、查询、修改和删除
- SQLite 数据持久化
- HTTP 接口正常与异常场景
- 请求参数校验
- 数据库自动初始化
- FastAPI Lifespan 启动逻辑
- 数据库初始化异常处理

测试通过 `tmp_path` 和 `monkeypatch` 创建独立的临时数据库，避免修改实际任务数据。

运行完整测试：

```bash
python -m pytest -v
```

单独运行 API 测试：

```bash
python -m pytest tests/test_api.py -v
```

单独运行 SQLite 测试：

```bash
python -m pytest tests/test_sqlite_storage.py -v
```

## 7. 后续开发计划

本项目将逐步从通用任务管理后端扩展为 AI Agent 应用开发练习项目。

计划学习和实现：

- LLM API 调用与响应处理
- 流式输出与异步 API
- Prompt Engineering
- Function Calling / Tool Calling
- RAG 知识库问答
- LangChain / LangGraph
- MCP 工具集成
- AI Agent 工作流
- Agent 服务部署与可观测性

**当前状态：FastAPI + SQLite 后端基础已完成，AI Agent 相关功能尚在规划中。**

## 8. 项目说明

本项目以学习和工程实践为主要目的。

开发过程采用渐进式方式，从 Python 基础、命令行任务管理、JSON 数据存储，逐步过渡到 REST API、SQLite、自动化测试和后端工程化。

后续将持续完善代码结构、测试覆盖率和 AI 应用功能。