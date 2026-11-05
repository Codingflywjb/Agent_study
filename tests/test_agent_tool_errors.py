from practice.agent_tool_errors import safe_simulate_tool_call


def test_safe_simulate_tool_call_success():
    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }

    result = safe_simulate_tool_call(tool_call)

    assert result["status"] == "simulated"
    assert result["title"] == "学习 FastAPI"
    assert result["saved"] is False


def test_safe_simulate_tool_call_unknown_tool():

    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "unknown_tool",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }

    result = safe_simulate_tool_call(tool_call)

    assert result["status"] == "error"
    assert result["error"] == "invalid_tool_call"
    assert result["saved"] is False

def test_safe_simulate_tool_call_invalid_json():

    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments":'{"title": }'
        }
    }

    result = safe_simulate_tool_call(tool_call)
    assert result["status"] == "error"
    assert result["error"] == "invalid_tool_call"
    assert result["saved"] is False

def test_safe_simulate_tool_call_empty_title():
    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": '{"title": "   "}'
        }
    }

    result = safe_simulate_tool_call(tool_call)

    assert result["status"] == "error"
    assert result["saved"] is False