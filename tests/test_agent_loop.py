import json
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock
from practice.agent_loop import decide_next_action,run_agent_loop

def test_decide_next_action_no_tool():
    message = SimpleNamespace(tool_calls=None)
    result = decide_next_action(message)
    assert result == "final"

def test_decide_next_action_empty_tool():
    message = SimpleNamespace(tool_calls=[])
    result = decide_next_action(message)
    assert result == "final"

def test_decide_next_action_tool_call_success():
    fake_tool_call = MagicMock()
    message = SimpleNamespace(tool_calls=[fake_tool_call])
    result = decide_next_action(message)
    assert result == "tool"

def test_run_agent_loop_direct_answer():
    fake_message = SimpleNamespace(
        content="list 可修改，tuple 不可修改。",
        tool_calls=None
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=fake_message)
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = fake_response
    result = run_agent_loop(
        client=mock_client,
        user_text="Python 中 list 和 tuple 有什么区别？",
        max_rounds=5
    )
    assert result == "list 可修改，tuple 不可修改。"
    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["tool_choice"] == "auto"
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"]=="system"
    assert kwargs["messages"][1]["role"]=="user"

def test_run_agent_loop_one_tool_then_final():
    fake_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "明天学习 FastAPI"}'
        )
    )

    first_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call]
    )

    second_message = SimpleNamespace(
        content="任务尚未保存。",
        tool_calls=None
    )

    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )

    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=second_message)
        ]
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    result = run_agent_loop(
        client=mock_client,
        user_text="帮我添加一个任务：明天学习 FastAPI",
        max_rounds=5
    )
    assert result == "任务尚未保存。"
    assert mock_client.chat.completions.create.call_count == 2
    call_args_list = mock_client.chat.completions.create.call_args_list
    messages_two = call_args_list[1].kwargs["messages"]  ##第二次调用的 关键字中 的 messages
    assert len(messages_two) == 4
    assert messages_two[2] is first_message
    assert messages_two[3]["role"]=="tool"
    assert messages_two[3]["tool_call_id"]=="call_001"

    decoded = json.loads(messages_two[3]["content"])
    assert decoded["status"] == "simulated"
    assert decoded["saved"] is False
    assert decoded["title"] == "明天学习 FastAPI"

def test_run_agent_loop_multiple_tools():
    fake_tool_call_1 = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "明天学习 FastAPI"}'
        )
    )

    fake_tool_call_2 = SimpleNamespace(
        id="call_002",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "后天学习 RAG"}'
        )
    )
    first_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call_1, fake_tool_call_2]
    )
    second_message = SimpleNamespace(
        content="两个任务均已模拟处理，尚未保存。",
        tool_calls=None
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=second_message)
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    result = run_agent_loop(
        client=mock_client,
        user_text="帮我添加两个任务：明天学习 FastAPI，后天学习 RAG。",
        max_rounds=5
    )
    assert result == "两个任务均已模拟处理，尚未保存。"
    assert mock_client.chat.completions.create.call_count == 2
    call_args_list = mock_client.chat.completions.create.call_args_list
    messages_two = call_args_list[1].kwargs["messages"]
    assert len(messages_two) == 5
    assert messages_two[3]["tool_call_id"] == "call_001"
    assert messages_two[4]["tool_call_id"] == "call_002"
    decoded_one = json.loads(messages_two[3]["content"])
    decoded_two = json.loads(messages_two[4]["content"])
    assert decoded_one["title"] == "明天学习 FastAPI"
    assert decoded_two["title"] == "后天学习 RAG"
    assert decoded_two["saved"] is False
    assert decoded_one["saved"] is False

def test_run_agent_loop_max_rounds():
    fake_tool_call_1 = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title":"学习 FastAPI"}'
        )
    )
    fake_tool_call_2 = SimpleNamespace(
        id="call_002",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title":"学习 RAG"}'
        )
    )
    first_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call_1]
    )
    second_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call_2]
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=second_message)
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    result = run_agent_loop(
        client=mock_client,
        user_text="我要学习 FastAPI和RAG",
        max_rounds=2
    )
    assert result == "已达到最大请求轮数，任务执行状态未确认。"
    assert mock_client.chat.completions.create.call_count == 2
    call_args_list = mock_client.chat.completions.create.call_args_list
    messages_two = call_args_list[1].kwargs["messages"]
    assert len(messages_two) == 4
    assert messages_two[3]["tool_call_id"] == "call_001"
    assert messages_two[2] is first_message

def test_run_agent_loop_invalid_max_rounds():
    mock_client = MagicMock()
    with pytest.raises(ValueError,match="max_rounds must be positive"):
        run_agent_loop(
            client=mock_client,
            user_text="学习 FastAPI",
            max_rounds=0
        )
    mock_client.chat.completions.create.assert_not_called()

def test_run_agent_loop_invalid_tool_recovery():
    fake_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": }'
        )
    )
    first_message = SimpleNamespace(
        content=None,
        tool_calls=[fake_tool_call]
    )

    second_message = SimpleNamespace(
        content="任务添加失败，参数格式不正确。",
        tool_calls=None
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=second_message)
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response]
    result = run_agent_loop(
        client=mock_client,
        user_text="帮我添加任务：学习 FastAPI",
        max_rounds=3
    )
    assert result == "任务添加失败，参数格式不正确。"
    assert mock_client.chat.completions.create.call_count == 2
    call_args_list = mock_client.chat.completions.create.call_args_list
    messages_two = call_args_list[1].kwargs["messages"]
    assert len(messages_two) == 4
    assert messages_two[3]["tool_call_id"] == "call_001"
    content=json.loads(messages_two[3]["content"])
    assert content["status"] == "error"
    assert content["error"] == "invalid_tool_call"
    assert content["saved"] is False
    assert messages_two[2] is first_message
    assert messages_two[3]["role"] == "tool"

def test_run_agent_loop_auto_recovery():
    bad_tool_call = SimpleNamespace(
        id="call_001",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": }'
        )
    )

    good_tool_call = SimpleNamespace(
        id="call_002",
        type="function",
        function=SimpleNamespace(
            name="add_task",
            arguments='{"title": "明天学习 FastAPI"}'
        )
    )
    first_message = SimpleNamespace(
        content=None,
        tool_calls=[bad_tool_call]
    )
    second_message = SimpleNamespace(
        content=None,
        tool_calls=[good_tool_call]
    )
    third_message = SimpleNamespace(
        content="任务已模拟处理，但尚未保存。",
        tool_calls=None
    )
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=first_message)
        ]
    )
    second_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=second_message)
        ]
    )
    third_response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=third_message)
        ]
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, second_response, third_response]
    result = run_agent_loop(
        client=mock_client,
        user_text="帮我添加任务：学习 FastAPI",
        max_rounds=5
    )
    assert result=="任务已模拟处理，但尚未保存。"
    assert mock_client.chat.completions.create.call_count == 3
    call_args_list = mock_client.chat.completions.create.call_args_list
    messages_two = call_args_list[1].kwargs["messages"]
    assert len(messages_two) == 4
    messages_three = call_args_list[2].kwargs["messages"]
    assert len(messages_three) == 6
    content_three = json.loads(messages_three[3]["content"])
    assert content_three["status"] == "error"
    content_five = json.loads(messages_three[5]["content"])
    assert content_five["status"] == "simulated"
    assert content_five["title"] == "明天学习 FastAPI"
    assert content_five["saved"] is False
    assert messages_two[3]["tool_call_id"] == "call_001"
    assert messages_three[5]["tool_call_id"] == "call_002"
