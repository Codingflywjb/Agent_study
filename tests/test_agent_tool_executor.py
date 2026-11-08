from unittest.mock import MagicMock
import json
from app import storage
import practice.agent_tool_executor as executor  ##executor 作为模块别名，是为了方便 monkeypatch 替换函数
from practice.agent_tools import execute_tool_call


def test_run_add_task_tool_dry_run(monkeypatch):
    tool_call = {
        "id" : "call_001",
        "type" : "function",
        "function" : {
            "name" : "add_task",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }
    mock_execute = MagicMock()

    monkeypatch.setattr(executor, "execute_tool_call", mock_execute)
    result = executor.run_add_task_tool(tool_call,dry_run=True)
    assert result["status"] == "simulated"
    assert result["title"] == "学习 FastAPI"
    assert result["saved"] is False
    mock_execute.assert_not_called()

def test_run_add_task_tool_real_mode(monkeypatch):
    tool_call = {
        "id" : "call_001",
        "type" : "function",
        "function" : {
            "name" : "add_task",
            "arguments" : '{"title":"学习 FastAPI"}'
        }
    }
    fake_result = {
        "id": 1,
        "title" : "学习 FastAPI"
    }
    mock_execute = MagicMock(return_value=fake_result)
    monkeypatch.setattr(executor, "execute_tool_call", mock_execute)
    result = executor.run_add_task_tool(tool_call,dry_run=False)

    assert result["status"] == "success"
    assert result["title"] == "学习 FastAPI"
    assert result["saved"] is True
    assert result["result"] == fake_result
    mock_execute.assert_called_once_with(tool_call)

def test_run_add_task_tool_real_write(tmp_path,monkeypatch):
    file_path = tmp_path / "tasks.json"
    monkeypatch.setattr(storage,"FILES_PATH",file_path)
    tool_call = {
        "id" : "call_001",
        "type" : "function",
        "function" : {
            "name" : "add_task",
            "arguments" : '{"title":"学习 FastAPI"}'
        }
    }
    result = executor.run_add_task_tool(tool_call,dry_run=False)
    assert result["status"] == "success"
    assert result["saved"] is True

    file_path = tmp_path / "tasks.json"

    assert file_path.exists()

    tasks = json.loads(file_path.read_text(encoding="utf-8"))
    assert len(tasks) == 1
    assert tasks[0]["title"] == "学习 FastAPI"
