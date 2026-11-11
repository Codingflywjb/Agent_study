import pytest
from practice.langgraph_state import route_by_intent
from practice.langgraph_reducer import build_reducer_graph,build_overwrite_graph

def test_reducer_graph_basic():
    graph=build_reducer_graph()

    initial_state={
        "user_input":"执行任务",
        "steps":[],
        "status":"pending"
    }

    final_state=graph.invoke(initial_state)

    assert final_state["steps"] == ["准备完成", "处理完成"]
    assert final_state["status"] == "finished"
    assert final_state["user_input"] == "执行任务"

def test_reducer_preserves_existing_steps():
    graph = build_reducer_graph()

    initial_state = {
        "user_input": "执行任务",
        "steps": ["用户请求已接收"],
        "status": "pending"
    }

    final_state = graph.invoke(initial_state)

    assert final_state["steps"] == ["用户请求已接收","准备完成", "处理完成"]
    assert final_state["status"] == "finished"
    assert final_state["user_input"] == "执行任务"

def test_reducer_graph_does_not_mutate_input():
    graph = build_reducer_graph()

    initial_state = {
        "user_input": "执行任务",
        "steps": ["历史记录"],
        "status": "pending"
    }

    graph.invoke(initial_state)

    assert initial_state["steps"] == ["历史记录"]
    assert initial_state["status"] == "pending"

def test_reducer_vs_overwrite():
    graph_r=build_reducer_graph()
    graph_o=build_overwrite_graph()

    initial_state = {
        "user_input": "执行任务",
        "steps": ["历史记录"],
        "status": "pending"
    }

    reducer_state = graph_r.invoke(initial_state)
    overwrite_state = graph_o.invoke(initial_state)

    assert reducer_state["steps"] == ["历史记录","准备完成", "处理完成"]
    assert overwrite_state["steps"] == ["处理完成"]
    assert reducer_state["status"] == "finished"
    assert overwrite_state["status"] == "finished"

def test_route_by_intent_unknown():
    state = {
        "user_input": "删除任务",
        "intent": "delete_task",
        "dry_run": True,
        "trace":[]
    }
    with pytest.raises(ValueError,match="unknown intent"):
        route_by_intent(state)