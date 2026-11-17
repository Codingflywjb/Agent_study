import json

import pytest
from unittest.mock import MagicMock
from langgraph.errors import GraphRecursionError

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage, ChatMessage
)


from practice.agent_tools import add_task_tool
from practice.langgraph_llm import (
    to_deepseek_messages,
    call_deepseek_once,
    make_llm_node,build_llm_graph,
    to_langchain_tool_call,
    make_tool_call_ai_message,
    sdk_message_to_ai_message,
    tool_message_to_deepseek,
    langchain_tool_call_to_deepseek,
    route_after_llm,
    tool_node,
    build_agent_graph,
    count_tool_messages
)
import practice.langgraph_llm as langgraph_llm

def test_to_deepseek_messages_with_tool_message():
    messages = [
        HumanMessage(content="帮我添加任务：学习 FastAPI"),
        ToolMessage(content='{"success": true}',tool_call_id="call_1")
    ]

    result = to_deepseek_messages(messages)

    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "帮我添加任务：学习 FastAPI"

    assert result[1]["role"] == "tool"
    assert result[1]["content"] =='{"success": true}'
    assert result[1]["tool_call_id"] == "call_1"

def test_unsupported_message_type():
    messages = [ChatMessage(role="custom",content="测试消息")]
    with pytest.raises(ValueError):
        to_deepseek_messages(messages)

def test_non_string_content():
    messages=[HumanMessage(content=[{"type":"text","text":"你好"}])]
    with pytest.raises(ValueError):
        to_deepseek_messages(messages)

def test_call_deepseek_once():
    mock_client=MagicMock()

    mock_client.chat.completions.create.return_value.choices[0].message.tool_calls = None

    mock_client.chat.completions.create.return_value.choices[0].message.content = "FastAPI 是一个 Python Web 框架。"

    messages=[HumanMessage(content="什么是 FastAPI？")]

    result = call_deepseek_once(mock_client,messages)

    assert isinstance(result,AIMessage)

    assert result.content == "FastAPI 是一个 Python Web 框架。"

    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": "什么是 FastAPI？"}
        ]
    )

def test_make_llm_node():
    mock_client = MagicMock()

    mock_client.chat.completions.create.return_value.choices[0].message.tool_calls = None

    mock_client.chat.completions.create.return_value.choices[0].message.content = "FastAPI 是一个 Python Web 框架。"

    node=make_llm_node(mock_client)

    state={
        "messages":[HumanMessage(content="什么是 FastAPI？")]
    }

    result = node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0],AIMessage)
    assert result["messages"][0].content == "FastAPI 是一个 Python Web 框架。"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": "什么是 FastAPI？"}
        ]
    )

def test_build_llm_graph():
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value.choices[0].message.tool_calls = None
    mock_client.chat.completions.create.return_value.choices[0].message.content = "FastAPI 是一个 Python Web 框架。"
    graph=build_llm_graph(mock_client)
    initial_state={
        "messages":[HumanMessage(content="什么是 FastAPI？")]
    }
    mock_client.chat.completions.create.assert_not_called()
    final_state=graph.invoke(initial_state)
    messages=final_state["messages"]
    assert len(messages)==2
    assert isinstance(messages[0],HumanMessage)
    assert messages[0].content == "什么是 FastAPI？"
    assert isinstance(messages[1],AIMessage)
    assert messages[1].content == "FastAPI 是一个 Python Web 框架。"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": "什么是 FastAPI？"}
        ]
    )

def test_to_langchain_tool_call():
    sdk_tool_call ={
        "id":"call_1",
        "type":"function",
        "function":{
            "name":"add_task",
            "arguments":'{"title":"学习 FastAPI"}'
        }
    }

    result = to_langchain_tool_call(sdk_tool_call)

    assert result == {
        "id": "call_1",
        "name": "add_task",
        "args": {"title": "学习 FastAPI"}
    }

def test_tool_call_args_not_dict():
    sdk_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": '["学习 FastAPI"]'
        }
    }
    with pytest.raises(ValueError, match="args is not dict"):
        to_langchain_tool_call(sdk_tool_call)

def test_make_tool_call_ai_message():
    sdk_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": '{"title":"学习 FastAPI"}'
        }
    }
    result = make_tool_call_ai_message(sdk_tool_call)

    assert isinstance(result,AIMessage)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["name"] == "add_task"
    assert result.tool_calls[0]["args"] == {"title": "学习 FastAPI"}

def test_sdk_message_normal_text():
    mock_message = MagicMock()
    mock_message.content = "你好"
    mock_message.tool_calls = None
    result = sdk_message_to_ai_message(mock_message)
    assert isinstance(result,AIMessage)
    assert result.content == "你好"
    assert result.tool_calls == []

def test_sdk_message_with_tool_call():
    mock_message = MagicMock()
    mock_tool_call = MagicMock()
    mock_message.content = None

    mock_tool_call.id = "call_1"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "add_task"
    mock_tool_call.function.arguments = '{"title":"学习 FastAPI"}'
    mock_message.tool_calls = [mock_tool_call]

    result = sdk_message_to_ai_message(mock_message)
    assert isinstance(result, AIMessage)
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["name"] == "add_task"
    assert result.tool_calls[0]["args"] == {"title":"学习 FastAPI"}

def test_sdk_message_with_text_and_tool_call():
    mock_message = MagicMock()
    mock_tool_call = MagicMock()
    mock_message.content = "正在为你添加任务"
    mock_tool_call.id = "call_1"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "add_task"
    mock_tool_call.function.arguments = '{"title":"学习 FastAPI"}'
    mock_message.tool_calls = [mock_tool_call]
    with pytest.raises(ValueError,match="tool call content must be empty"):
        sdk_message_to_ai_message(mock_message)

def test_call_deepseek_once_with_tool_call():
    mock_client = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "add_task"
    mock_tool_call.function.arguments = '{"title":"学习 FastAPI"}'
    message=mock_client.chat.completions.create.return_value.choices[0].message
    message.content = None
    message.tool_calls = [mock_tool_call]
    messages = [
        HumanMessage(content="帮我添加任务：学习 FastAPI")
    ]
    result = call_deepseek_once(mock_client,messages)
    assert isinstance(result,AIMessage)
    assert result.content == ""
    assert len(result.tool_calls)==1
    assert result.tool_calls[0]["id"] == "call_1"
    assert result.tool_calls[0]["name"] == "add_task"
    assert result.tool_calls[0]["args"] == {"title": "学习 FastAPI"}
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": "帮我添加任务：学习 FastAPI"}
        ]
    )

def test_call_deepseek_once_with_tools():
    mock_client = MagicMock()
    mock_message=mock_client.chat.completions.create.return_value.choices[0].message
    mock_message.content = "好的，我可以添加任务。"
    mock_message.tool_calls = None
    message_user = HumanMessage(content="帮我添加任务：学习 FastAPI")
    result = call_deepseek_once(client=mock_client,messages=[message_user],tools=[add_task_tool])
    assert isinstance(result,AIMessage)
    assert result.content == "好的，我可以添加任务。"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "帮我添加任务：学习 FastAPI"}],
        tools=[add_task_tool],
        tool_choice="auto"
    )

def test_make_llm_node_with_tools():
    mock_client = MagicMock()
    mock_message = mock_client.chat.completions.create.return_value.choices[0].message
    mock_message.content = "好的，我可以添加任务。"
    mock_message.tool_calls = None
    node = make_llm_node(mock_client,tools=[add_task_tool])
    state = {
        "messages":[HumanMessage(content="帮我添加任务：学习 FastAPI")]
    }
    result = node(state)
    assert len(result["messages"])==1
    assert isinstance(result["messages"][0],AIMessage)
    assert result["messages"][0].content == "好的，我可以添加任务。"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "帮我添加任务：学习 FastAPI"}],
        tools=[add_task_tool],
        tool_choice="auto"
    )

def test_build_llm_graph_with_tools():
    mock_client = MagicMock()
    mock_message = mock_client.chat.completions.create.return_value.choices[0].message
    mock_message.content = "好的，我可以添加任务。"
    mock_message.tool_calls = None
    graph=build_llm_graph(mock_client,tools=[add_task_tool])
    mock_client.chat.completions.create.assert_not_called()
    initial_state={
        "messages":[HumanMessage(content="帮我添加任务：学习 FastAPI")]
    }
    final_state = graph.invoke(initial_state)
    messages=final_state["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0],HumanMessage)
    assert isinstance(messages[1],AIMessage)
    assert messages[1].content == "好的，我可以添加任务。"
    assert messages[0].content == "帮我添加任务：学习 FastAPI"
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[{"role": "user", "content": "帮我添加任务：学习 FastAPI"}],
        tools=[add_task_tool],
        tool_choice="auto"
    )

def test_tool_message_to_deepseek():
    tool_message=ToolMessage(content= '{"success": true, "dry_run": true}',tool_call_id = "call_1")
    result = tool_message_to_deepseek(tool_message)
    assert result == {
        "role": "tool",
        "content": '{"success": true, "dry_run": true}',
        "tool_call_id": "call_1"
    }

def test_langchain_tool_call_to_deepseek():
    tool_call_langchain = {
        "id":"call_1",
        "name":"add_task",
        "args":{"title": "学习 FastAPI"}
    }
    result = langchain_tool_call_to_deepseek(tool_call_langchain)
    assert result["id"] == "call_1"
    assert result["type"] == "function"
    assert result["function"]["name"] == "add_task"
    assert isinstance(result["function"]["arguments"],str)
    assert json.loads(result["function"]["arguments"]) == {"title": "学习 FastAPI"}

def test_to_deepseek_messages_with_ai_tool_calls():
    message_ai = AIMessage(
        content="",
        tool_calls=[
            {
                "id":"call_1",
                "name":"add_task",
                "args":{"title": "学习 FastAPI"}
            }
        ]
    )
    result=to_deepseek_messages([message_ai])
    assert len(result) == 1
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == ""
    assert len(result[0]["tool_calls"]) == 1
    tool_call = result[0]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "add_task"
    assert json.loads(tool_call["function"]["arguments"]) == {"title": "学习 FastAPI"}

def test_to_deepseek_messages_full_tool_history():
    messages=[
        HumanMessage(content="帮我添加任务：学习 FastAPI"),
        AIMessage(content="",tool_calls=[{
        "id": "call_1",
        "name": "add_task",
        "args": {"title": "学习 FastAPI"}
    }]),
        ToolMessage(content='{"success": true, "dry_run": true}',tool_call_id="call_1")
    ]
    result=to_deepseek_messages(messages)
    assert len(result) == 3
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    assert result[2]["role"] == "tool"
    tool_call = result[1]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["function"]["name"] == "add_task"
    assert json.loads(tool_call["function"]["arguments"]) == {"title": "学习 FastAPI"}
    assert tool_call["id"] == result[2]["tool_call_id"]
    assert json.loads(result[2]["content"]) == {
        "success": True,
        "dry_run": True
    }

def test_route_after_llm_with_tool_calls():
    state = {
        "messages":[
            AIMessage(content="",
                      tool_calls=[{
                          "id": "call_1",
                          "name": "add_task",
                          "args": {"title": "学习 FastAPI"}
                      }]
            )
        ]
    }
    result = route_after_llm(state)
    assert result == "tool"

def test_route_after_llm_without_tool_calls():
    state = {
        "messages": [
            AIMessage(content="你好！", tool_calls=[])
        ]
    }
    result = route_after_llm(state)
    assert result == "end"

def test_route_after_llm_invalid_message():
    state = {
        "messages": [
            HumanMessage(content="你好！")
        ]
    }
    with pytest.raises(ValueError,match="last message is not AImessage"):
        route_after_llm(state)

def test_tool_node_dry_run(monkeypatch):
    expected_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "add_task",
            "arguments": json.dumps(
                {"title": "学习 FastAPI"},
                ensure_ascii=False
            )
        }
    }
    mock_executor = MagicMock(
        return_value={
            "status": "simulated",
            "saved": False
        }
    )
    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )
    state = {
        "messages":[
            AIMessage(
                content="",
                tool_calls=[{
                    "id": "call_1",
                    "name": "add_task",
                    "args": {"title": "学习 FastAPI"}
                }]
            )
        ]
    }
    result = tool_node(state)
    messages=result["messages"]
    assert len(messages) == 1
    assert isinstance(messages[0],ToolMessage)
    assert messages[0].tool_call_id == "call_1"
    assert json.loads(messages[0].content) == {
            "status": "simulated",
            "saved": False
        }
    mock_executor.assert_called_once_with(expected_tool_call,dry_run=True)

def test_tool_node_no_tool_calls():
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[])
        ]
    }
    with pytest.raises(
            ValueError,
            match="expected exactly one tool call"
    ):
        tool_node(state)

def test_tool_node_multiple_tool_calls():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "add_task",
                        "args": {"title": "学习 FastAPI"}
                    },
                    {
                        "id": "call_2",
                        "name": "add_task",
                        "args": {"title": "学习 LangGraph"}
                    }
                ]
            )
        ]
    }
    with pytest.raises(
            ValueError,
            match="expected exactly one tool call"
    ):
        tool_node(state)

def test_build_agent_graph_full_loop(monkeypatch):
    mock_client =MagicMock()
    first_response = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.type = "function"
    mock_tool_call.function.name = "add_task"
    mock_tool_call.function.arguments = json.dumps(
        {"title": "学习 FastAPI"},
        ensure_ascii=False
    )
    first_message = first_response.choices[0].message
    first_message.content = None
    first_message.tool_calls = [mock_tool_call]

    second_response = MagicMock()
    second_message = second_response.choices[0].message
    second_message.content = "已模拟添加任务。"
    second_message.tool_calls = None

    mock_client.chat.completions.create.side_effect = [first_response,second_response]

    mock_executor = MagicMock(
        return_value={
            "status": "simulated",
            "saved": False
        }
    )
    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )
    graph = build_agent_graph(mock_client,tools=[add_task_tool])
    initial_state = {
        "messages":[HumanMessage(content="帮我添加任务：学习 FastAPI")]
    }

    final_state = graph.invoke(initial_state,config={"recursion_limit":10})
    messages = final_state["messages"]
    assert len(messages) == 4
    assert isinstance(messages[0],HumanMessage)
    assert isinstance(messages[1],AIMessage)
    assert isinstance(messages[2],ToolMessage)
    assert isinstance(messages[3],AIMessage)
    assert messages[3].content == "已模拟添加任务。"
    assert messages[2].tool_call_id == "call_1"
    second_call = mock_client.chat.completions.create.call_args_list[1]
    second_messages = second_call.kwargs["messages"]
    assert second_messages[0]["role"] == "user"
    assert second_messages[1]["role"] =="assistant"
    assert second_messages[2]["role"] =="tool"
    assert second_messages[-1]["tool_call_id"] == "call_1"

    assert mock_client.chat.completions.create.call_count == 2
    mock_executor.assert_called_once()
    assert mock_executor.call_args.kwargs["dry_run"] is True
    assert json.loads(second_messages[-1]["content"]) == {
        "status": "simulated",
        "saved": False
    }

def test_build_agent_graph_without_tool_calls(monkeypatch):
    mock_client = MagicMock()

    response = MagicMock()
    response.choices[0].message.content = "你好！"
    response.choices[0].message.tool_calls = None

    mock_client.chat.completions.create.return_value = response
    mock_executor = MagicMock()
    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )
    graph = build_agent_graph(mock_client,tools=[add_task_tool])
    initial_state ={
        "messages": [HumanMessage(content="你好")]
    }
    final_state = graph.invoke(initial_state,config={"recursion_limit":10})

    messages = final_state["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0],HumanMessage)
    assert isinstance(messages[1],AIMessage)

    assert messages[1].content == "你好！"

    mock_client.chat.completions.create.assert_called_once()

    mock_executor.assert_not_called()

def test_build_agent_graph_recursion_limit(monkeypatch):
    mock_client =MagicMock()
    response = MagicMock()
    tool_call = MagicMock()

    tool_call.id = "call_1"
    tool_call.type = "function"
    tool_call.function.name = "add_task"
    tool_call.function.arguments = json.dumps(
        {"title": "学习 LangGraph"},
        ensure_ascii=False
    )

    response.choices[0].message.content = None
    response.choices[0].message.tool_calls = [tool_call]

    mock_client.chat.completions.create.return_value = response

    mock_executor=MagicMock(return_value={"status": "simulated", "saved": False})
    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )
    graph = build_agent_graph(mock_client,tools=[add_task_tool])
    initial_state = {
        "messages":[HumanMessage(content="添加任务：学习 LangGraph")]
    }
    with pytest.raises(GraphRecursionError):
        graph.invoke(initial_state,config={"recursion_limit":4})

    assert mock_client.chat.completions.create.call_count >= 2
    assert mock_executor.call_count >= 1
    assert all(
        call.kwargs["dry_run"] is True
        for call in mock_executor.call_args_list
    )

def test_count_tool_messages_mixed():
    state = {
        "messages":[
            HumanMessage(content="你好"),
            AIMessage(content="你好！"),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_1"
            ),
            AIMessage(content="任务已处理")
        ]
    }
    result = count_tool_messages(state)
    assert result == 1

def test_count_tool_messages_empty():
    state = {
        "messages": []
    }
    assert count_tool_messages(state) == 0

def test_count_tool_messages_multiple():
    state = {
        "messages": [
            HumanMessage(content="添加三个任务"),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_1"
            ),
            AIMessage(content="第一个任务已处理"),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_2"
            ),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_3"
            ),
            AIMessage(content="任务已全部处理完成")
        ]
    }
    assert count_tool_messages(state) == 3

def test_tool_node_limit_exceeded(monkeypatch):
    mock_executor = MagicMock()
    monkeypatch.setattr(langgraph_llm,"run_add_task_tool",mock_executor)

    state = {
        "messages":[
            HumanMessage(content="添加三个任务"),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_1"
            ),
            AIMessage(content="第一个任务已处理"),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_2"
            ),
            ToolMessage(
                content="任务添加成功",
                tool_call_id="call_3"
            ),
            AIMessage(content="任务已全部处理完成",tool_calls=[{"id":"call_4","name":"add_task","args":{"title": "任务"}}])
        ]
    }
    with pytest.raises(ValueError,match="tool call limit exceeded"):
        tool_node(state)

    mock_executor.assert_not_called()

def test_tool_node_limit_allows_last_call(monkeypatch):
    mock_executor = MagicMock(
        return_value={
            "status": "simulated",
            "saved": False
        }
    )

    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )

    state = {
        "messages": [
            HumanMessage(content="添加三个任务"),
            ToolMessage(
                content="任务1已处理",
                tool_call_id="call_1"
            ),
            ToolMessage(
                content="任务2已处理",
                tool_call_id="call_2"
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_3",
                        "name": "add_task",
                        "args": {"title": "学习 LangGraph"}
                    }
                ]
            )
        ]
    }

    result = tool_node(state)

    messages = result["messages"]

    assert len(messages) == 1
    assert isinstance(messages[0], ToolMessage)
    assert messages[0].tool_call_id == "call_3"

    assert json.loads(messages[0].content) == {
        "status": "simulated",
        "saved": False
    }

    mock_executor.assert_called_once()
    assert mock_executor.call_args.kwargs["dry_run"] is True

def test_tool_node_executor_error(monkeypatch):
    mock_executor = MagicMock(side_effect = RuntimeError("database connection failed"))
    monkeypatch.setattr(
        langgraph_llm,
        "run_add_task_tool",
        mock_executor
    )
    state = {
        "messages":[
            AIMessage(content="",tool_calls=[{
                "id":"call_1",
                "name":"add_task",
                "args":{"title": "学习 LangGraph"}
            }])
        ]
    }
    result = tool_node(state)["messages"][0]
    content = json.loads(result.content)
    assert isinstance(result,ToolMessage)
    assert content["status"] == "error"
    assert content["message"] == "tool execution failed"
    assert result.tool_call_id == "call_1"
    mock_executor.assert_called_once()