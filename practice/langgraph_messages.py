from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage
from langgraph.graph import add_messages,StateGraph,START,END
from langgraph.graph import MessagesState

def build_demo_messages() -> list[BaseMessage]:
    system_message=SystemMessage(content="你是一个任务管理助手")

    human_message=HumanMessage(content="什么是 FastAPI？")

    ai_message=AIMessage(content="FastAPI 是一个 Python Web 框架。")

    return [
        system_message,
        human_message,
        ai_message
    ]

def demo_message_replacement():
    old_messages=[HumanMessage(content="什么是 FastAPI？",id="msg-1")]

    new_messages=[HumanMessage(content="FastAPI 有什么优点？",id="msg-1")]

    result = add_messages(old_messages,new_messages)

    return result

def reply_node(state:MessagesState)->dict:
    last_message=state["messages"][-1]

    user_question=last_message.content

    reply=AIMessage(content=f"收到你的问题：{user_question}")
    return {
        "messages":[reply]
    }

def build_messages_graph():
    builder = StateGraph(MessagesState)

    builder.add_node("reply",reply_node)

    builder.add_edge(START,"reply")

    builder.add_edge("reply",END)

    graph=builder.compile()

    return graph

if __name__ == "__main__":
    messages=build_demo_messages()

    for message in messages:
        print(f"{message.type}:{message.content}")