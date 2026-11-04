from types import SimpleNamespace
from unittest.mock import MagicMock
import json

from practice.agent_function_calling_demo import run_tool_call_demo

def test_run_tool_call_demo():
    fake_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title":"明天学习FastAPI"}'
        )
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[fake_tool_call]
                )
            )
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="任务尚未保存。"
                )
            )
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    result = run_tool_call_demo(mock_client,user_text="请帮我添加一个任务：明天学习FastAPI")

    assert result == "任务尚未保存。"
    assert mock_client.chat.completions.create.call_count == 2
    calls = mock_client.chat.completions.create.call_args_list
    first_message = calls[0].kwargs["messages"]
    second_message = calls[1].kwargs["messages"]
    assert len(first_message)== 2
    assert len(second_message)== 4
    assert second_message[2] is first_response.choices[0].message
    assert second_message[3]["role"]=="tool"
    assert second_message[2].tool_calls[0].id == "call_001"
    decoded = json.loads(second_message[3]["content"])
    assert decoded["saved"] is False
    assert second_message[3]["tool_call_id"] == "call_001"
    assert decoded["status"] == "simulated"