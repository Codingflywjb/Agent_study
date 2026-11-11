from typing import TypedDict, Annotated
from langgraph.graph import StateGraph,START,END
import operator

class WorkflowState(TypedDict):
    user_input: str
    steps: Annotated[list[str], operator.add]
    status: str

class OverwriteState(TypedDict):
    user_input:str
    steps:list[str]
    status:str

def prepare_node(state:WorkflowState) ->dict:
    return  {
        "steps":["准备完成"],
        "status": "prepared"
    }

def finish_node(state:WorkflowState) ->dict:
    return {
        "steps":["处理完成"],
        "status":"finished"
    }

def build_reducer_graph():
    builder=StateGraph(WorkflowState)

    builder.add_node("prepare",prepare_node)

    builder.add_node("finish",finish_node)

    builder.add_edge(START,"prepare")

    builder.add_edge("prepare","finish")

    builder.add_edge("finish",END)

    graph = builder.compile()

    return graph

def build_overwrite_graph():
    builder=StateGraph(OverwriteState)

    builder.add_node("prepare", prepare_node,input_schema=OverwriteState)

    builder.add_node("finish", finish_node,input_schema=OverwriteState)

    builder.add_edge(START, "prepare")

    builder.add_edge("prepare", "finish")

    builder.add_edge("finish", END)

    graph = builder.compile()

    return graph

if __name__ == "__main__" :
    graph=build_reducer_graph()
    initial_state = {
        "user_input" : "执行任务",
        "steps":[],
        "status":"pending"
    }
    final_state=graph.invoke(initial_state)

    print(final_state["steps"])
    print(final_state["status"])