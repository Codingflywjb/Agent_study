
import json

from practice.agent_tool_messages import build_tool_result_message,build_tool_conversation


def test_build_tool_result_message_success():
    fake_result = {
        "id": 1,
        "title": "学习 FastAPI"
    }

    message = build_tool_result_message(
        tool_call_id="call_001",
        result=fake_result
    )

    assert message["role"] == "tool"

    assert message["tool_call_id"] == "call_001"

    assert isinstance(message["content"],str)

    result = json.loads(message["content"])
    assert result == fake_result

def test_build_tool_result_message_chinese():
    fake_result = {
        "title": "明天学习大模型开发"
    }
    message = build_tool_result_message(
        tool_call_id="call_001",
        result=fake_result
    )
    assert "明天学习大模型开发" in message["content"]

def test_build_tool_conversation():
    tool_call = {
        "id": "call_001",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": '{"title": "学习 FastAPI"}'
        }
    }

    tool_result = {
        "id": 1,
        "title": "学习 FastAPI"
    }

    messages = build_tool_conversation(
        user_text="帮我添加任务：学习 FastAPI",
        tool_call=tool_call,
        tool_result=tool_result
    )

    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert "tool_calls" in messages[1]
    assert messages[2]["tool_call_id"] == "call_001"
    assert messages[2]["role"] == "tool"
    decoded=json.loads(messages[2]["content"])
    assert decoded == tool_result
    assert messages[0]["content"] == "帮我添加任务：学习 FastAPI"

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"] == [tool_call]