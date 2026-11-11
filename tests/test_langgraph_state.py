from practice.langgraph_state import classify_intent_node,task_node,chat_node

def test_classify_add_task():
    state = {
        "user_input":"帮我添加任务：学习 FastAPI",
        "intent":"",
        "dry_run":True,
        "trace": []
    }
    result = classify_intent_node(state)

    assert result == {"intent":"add_task","trace": ["classify_intent"]}

def test_classify_chat():
    state = {
        "user_input": "什么是FastAPI？",
        "intent": "",
        "dry_run": True,
        "trace": []
    }
    result = classify_intent_node(state)

    assert result == {"intent": "chat","trace": ["classify_intent"]}

def test_node_does_not_mutate_state():
    state = {
        "user_input": "什么是FastAPI？",
        "intent": "",
        "dry_run": True,
        "trace": ["history"]
    }
    result = classify_intent_node(state)

    assert state["intent"] == ""
    assert result["intent"]  == "chat"
    assert state["trace"] == ["history"]
    assert result["trace"] == ["classify_intent"]

def test_task_node_dry_run():
    state = {
        "user_input": "帮我添加任务：学习 FastAPI",
        "intent": "add_task",
        "dry_run": True,
        "trace": []
    }

    result = task_node(state)

    assert result == {
        "reply": "任务已模拟处理，尚未保存。","trace": ["task_node"]
    }

def test_task_node_real_mode():
    state = {
        "user_input": "帮我添加任务：学习 FastAPI",
        "intent": "add_task",
        "dry_run": False,
        "trace": []
    }
    result = task_node(state)
    assert result == {
        "reply": "真实执行暂未接入。","trace": ["task_node"]
    }

def test_chat_node():
    state = {
        "user_input": "什么是FastAPI?",
        "intent": "chat",
        "dry_run": True,
        "trace": []
    }

    result = chat_node(state)
    assert result == {
        "reply": "收到你的问题：什么是FastAPI?","trace": ["chat_node"]
    }