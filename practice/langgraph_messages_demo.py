from langchain_core.messages import HumanMessage
from practice.langgraph_messages import build_messages_graph

def main():
    graph=build_messages_graph()

    initial_state={
        "messages": [
            HumanMessage(content="什么是FastAPI？")
        ]
    }

    final_state=graph.invoke(initial_state)

    messages=final_state["messages"]

    for message in messages:
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()