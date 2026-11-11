from langgraph.graph import StateGraph,START,END

from practice.langgraph_state import (
    AgentState,
    classify_intent_node,
    route_by_intent,
    task_node,
    chat_node,
)

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent",classify_intent_node)

    builder.add_node("task_node",task_node)

    builder.add_node("chat_node",chat_node)

    builder.add_edge(START,"classify_intent")

    builder.add_conditional_edges("classify_intent",route_by_intent)

    builder.add_edge("task_node",END)

    builder.add_edge("chat_node", END)

    graph=builder.compile()

    return graph

if __name__ == "__main__":
    graph=build_graph()
    initial_state= {
        "user_input":"帮我添加任务：学习 FastAPI",
        "intent":"",
        "dry_run":True,
        "trace": []
    }
    final_state=graph.invoke(initial_state)
    print(final_state["intent"])
