
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from practice import agent_tools
from practice.agent_tool_messages import build_tool_conversation


def test_tool_call_full_flow(monkeypatch):

    fake_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "学习 FastAPI"}'
        )
    )


    fake_result = {
        "id": 1,
        "title": "学习 FastAPI"
    }

    mock_add=MagicMock(return_value=fake_result)

    monkeypatch.setattr(agent_tools, "add_task", mock_add)

    fake_tool_call_dict = agent_tools.sdk_tool_call_to_dict(fake_tool_call)

    result = agent_tools.execute_tool_call(fake_tool_call_dict)

    message=build_tool_conversation(
        user_text="帮我添加任务：学习 FastAPI",
        tool_call=fake_tool_call_dict,
        tool_result=result
    )

    mock_add.assert_called_once_with(title="学习 FastAPI")

    assert len(message) == 3

    decoded=json.loads(message[2]["content"])
    assert decoded == fake_result