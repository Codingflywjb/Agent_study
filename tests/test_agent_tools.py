import json
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace ##用SimpleNamespace模拟SDK对象
from practice import agent_tools
from practice.agent_tools import parse_tool_call,validate_add_task_arguments,validate_tool_call,sdk_tool_call_to_dict


def test_parse_tool_call_success():
    tool_call = {
        "function": {
            "name": "add_task",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }
    name,args = parse_tool_call(tool_call)
    assert name == "add_task"
    assert args == {"title":"学习 FastAPI"}

def test_parse_tool_call_invalid_json():
    tool_call = {
        "function": {
            "name": "add_task",
            "arguments": '{"title": "学习 FastAPI"'
        }
    }
    with pytest.raises(json.JSONDecodeError):
        parse_tool_call(tool_call)

def test_parse_tool_call_arguments_not_dict():
    tool_call = {
        "function": {
            "name": "add_task",
            "arguments": '["学习 FastAPI"]'
        }
    }
    with pytest.raises(ValueError,match="Tool arguments must be a dict"):
        parse_tool_call(tool_call)

def test_validate_add_task_arguments_success():
    arguments = {
        "title":"学习 FastAPI"
    }
    result=validate_add_task_arguments(arguments)
    assert result == "学习 FastAPI"

def test_validate_add_task_arguments_missing_title():
    with pytest.raises(ValueError, match="title is required"):
        validate_add_task_arguments({})

def test_validate_add_task_arguments_invalid_type():
    arguments = {"title": 123}
    with pytest.raises(ValueError, match="title must be a string"):
        validate_add_task_arguments(arguments)

def test_validate_add_task_arguments_empty_title():
    arguments = {"title": "   "}
    with pytest.raises(ValueError, match="title cannot be empty"):
        validate_add_task_arguments(arguments)

def test_validate_tool_call_success():
    tool_call = {
        "function": {
            "name": "add_task",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }
    tool_name,args = validate_tool_call(tool_call)
    assert tool_name == "add_task"
    assert args == {"title":"学习 FastAPI"}

def test_validate_tool_call_unknown_tool():
    tool_call = {
        "function": {
            "name": "delete_all_tasks",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }
    with pytest.raises(ValueError, match="Unknown tool"):
        validate_tool_call(tool_call)

def test_execute_tool_call_invalid_title(monkeypatch):
    tool_call = {
        "function": {
            "name": "add_task",
            "arguments": '{"title": 123}'
        }
    }

    mock_add = MagicMock()
    monkeypatch.setattr(agent_tools,"add_task",mock_add)
    with pytest.raises(ValueError, match="title must be a string"):
        agent_tools.execute_tool_call(tool_call)
    mock_add.assert_not_called()

def test_execute_tool_call_unknown_tool(monkeypatch):
    tool_call = {
        "function": {
            "name": "delete_all_tasks",
            "arguments": '{}'
        }
    }

    mock_add = MagicMock()
    monkeypatch.setattr(agent_tools,"add_task",mock_add)
    with pytest.raises(ValueError, match="Unknown tool"):
        agent_tools.execute_tool_call(tool_call)
    mock_add.assert_not_called()

def test_sdk_tool_call_to_dict():
    fake_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "学习 FastAPI"}'
        )
    )
    result = sdk_tool_call_to_dict(fake_tool_call)
    assert result["id"] == "call_001"
    assert result["type"] == "function"
    assert result["function"]["name"] == "add_task"
    assert result["function"]["arguments"] == '{"title": "学习 FastAPI"}'
    tool_name,arguments = validate_tool_call(result)
    assert tool_name == "add_task"
    assert arguments == {"title": "学习 FastAPI"}