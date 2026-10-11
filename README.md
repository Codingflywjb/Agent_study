# Agent_study

这是我的 Python 工程与 AI Agent 开发学习仓库。

## 已实现功能

- 添加任务
- 查看任务
- 完成任务
- 删除任务

## 技术栈

- Python
- JSON
- Git
- pytest
- logging
- pathlib

## 项目结构
- main.py：命令行交互入口.
- task_manager.py：任务管理业务逻辑。
- storage.py：JSON 数据持久化。
- tests/：pytest 自动化测试。

```text
Agent_study/
├── app/
│   ├── main.py
│   ├── task_manager.py
│   └── storage.py
├── tests/
│   ├── test_task_manager.py
│   └── test_task_operations.py
├── .gitignore
└── README.md
```

## 运行方式

- 1.安装Python和项目测试依赖
- 2.在项目根目录启动程序
- 3.执行自动化测试


注意：
- 启动命令
```powershell
python -m app.main
```
- 运行测试命令
```powershell
python -m app.main
```
- 测试依赖pytest，通过以下命令安装
```powershell
python -m app.main
```


## 后续计划

下一步准备学习的 FastAPI、LLM API 等技术，尚未实现