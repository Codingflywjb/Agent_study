import pytest
from practice.langgraph_state import route_by_intent
from unittest.mock import MagicMock
import practice.langgraph_demo as langgraph_demo
from practice.langgraph_demo import build_graph

def test_graph_add_task():
    graph=build_graph()
    initial_state={
        "user_input":"帮我添加任务：学习 FastAPI",
        "intent":"",
        "dry_run":True
    }
    final_state=graph.invoke(initial_state)
    assert final_state["intent"] == "add_task"
    assert final_state["reply"] == "任务已模拟处理，尚未保存。"
    assert final_state["dry_run"] is True

def test_graph_chat():
    graph = build_graph()
    initial_state = {
        "user_input": "什么是FastAPI？",
        "intent": "",
        "dry_run": True
    }
    final_state = graph.invoke(initial_state)
    assert final_state["intent"] == "chat"
    assert final_state["reply"] == "收到你的问题：什么是FastAPI？"
    assert final_state["user_input"]=="什么是FastAPI？"

def test_graph_task_real_mode_placeholder():
    graph = build_graph()
    initial_state = {
        "user_input": "帮我添加任务：学习 FastAPI",
        "intent": "",
        "dry_run": False
    }
    final_state=graph.invoke(initial_state)
    assert final_state["reply"] == "真实执行暂未接入。"
    assert final_state["intent"] == "add_task"

def test_graph_routes_only_to_task_node(monkeypatch):
    mock_task = MagicMock(return_value={"reply": "任务已模拟处理，尚未保存。"})
    mock_chat = MagicMock(return_value={"reply": "不应该执行"})
    monkeypatch.setattr(langgraph_demo,"task_node",mock_task)
    monkeypatch.setattr(langgraph_demo,"chat_node",mock_chat)
    graph=langgraph_demo.build_graph()
    initial_state={
        "user_input":"帮我添加任务：学习 FastAPI",
        "intent":"",
        "dry_run":True
    }
    final_state=graph.invoke(initial_state)
    assert final_state["intent"] == "add_task"
    assert final_state["reply"] == "任务已模拟处理，尚未保存。"
    mock_task.assert_called_once()
    mock_chat.assert_not_called()

def test_graph_routes_only_to_chat_node(monkeypatch):
    mock_task=MagicMock(return_value={"reply": "不应该执行"})
    mock_chat=MagicMock(return_value={"reply": "收到你的问题：什么是FastAPI？"})
    monkeypatch.setattr(langgraph_demo, "task_node", mock_task)
    monkeypatch.setattr(langgraph_demo, "chat_node", mock_chat)
    graph = langgraph_demo.build_graph()
    initial_state = {
        "user_input": "什么是FastAPI？",
        "intent": "",
        "dry_run": True
    }
    final_state = graph.invoke(initial_state)
    assert final_state["intent"] == "chat"
    assert final_state["reply"] == "收到你的问题：什么是FastAPI？"
    mock_chat.assert_called_once()
    mock_task.assert_not_called()

def test_route_by_intent_unknown():
    state = {
        "user_input": "删除任务",
        "intent": "delete_task",
        "dry_run": True
    }
    with pytest.raises(ValueError,match="unknown intent"):
        route_by_intent(state)