from practice.langgraph_messages import demo_message_replacement,reply_node,build_messages_graph
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import add_messages

def test_message_replacement():
    result=demo_message_replacement()

    assert len(result)==1
    assert result[0].content == "FastAPI 有什么优点？"
    assert result[0].id == "msg-1"

def test_message_append():
    old_messages=[HumanMessage(content="你好",id="msg-1")]

    new_messages=[AIMessage(content="你好！有什么可以帮助你？",id="msg-2")]

    result=add_messages(old_messages,new_messages)

    assert len(result)==2

    assert result[0].id == "msg-1"
    assert result[0].content == "你好"

    assert result[1].id == "msg-2"
    assert result[1].content == "你好！有什么可以帮助你？"

    assert result[0].type == "human"
    assert result[1].type == "ai"

def test_reply_node():
    test_state = {
        "messages": [
            HumanMessage(content="什么是 FastAPI？")
        ]
    }

    result = reply_node(state=test_state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0],AIMessage)
    assert result["messages"][0].content == "收到你的问题：什么是 FastAPI？"
    assert len(test_state["messages"]) == 1
    assert isinstance(test_state["messages"][0],HumanMessage)
    assert test_state["messages"][0].content == "什么是 FastAPI？"

def test_messages_graph_multi_turn():
    graph=build_messages_graph()

    old_state = {
        "messages":[
            HumanMessage(content="你好")
        ]
    }

    first_state=graph.invoke(old_state)

    new_state = {
        "messages": first_state["messages"] + [HumanMessage(content="什么是 FastAPI？")]
    }

    second_state = graph.invoke(new_state)

    assert len(second_state["messages"]) == 4
    assert second_state["messages"][0].type == "human"
    assert second_state["messages"][1].type == "ai"
    assert second_state["messages"][2].type == "human"
    assert second_state["messages"][3].type == "ai"
    assert second_state["messages"][-1].content == "收到你的问题：什么是 FastAPI？"
    assert len(first_state["messages"]) == 2
    assert first_state["messages"][0].content == "你好"