import practice.agent_demo as agent_demo
from unittest.mock import MagicMock
from types import SimpleNamespace


def test_agent_demo_dry_run(monkeypatch,capsys):
    fake_tool_call = SimpleNamespace(
        id = "call_001",
        type = "function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title":"学习 FastAPI"}'
        )
    )
    first_message = SimpleNamespace(
        content = None,
        tool_calls=[fake_tool_call]
    )
    second_message = SimpleNamespace(
        content = "任务尚未保存。",
        tool_calls = None
    )
    first_response = SimpleNamespace(
        choices = [
            SimpleNamespace(message=first_message)
        ]
    )
    second_response = SimpleNamespace(
        choices = [
            SimpleNamespace(
                message=second_message
            )
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response,second_response]
    monkeypatch.setenv("DEEPSEEK_API_KEY","test_key")
    mock_openai = MagicMock(return_value=mock_client)
    monkeypatch.setattr(agent_demo,"OpenAI",mock_openai)
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": "帮我添加任务：学习 FastAPI"
    )
    agent_demo.main()
    captured = capsys.readouterr()
    assert "任务尚未保存。" in captured.out
    assert mock_client.chat.completions.create.call_count ==  2