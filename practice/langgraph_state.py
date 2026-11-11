from typing import TypedDict, NotRequired,Annotated
import operator

class WorkflowState(TypedDict):
    user_input:str
    steps:Annotated[list[str],operator.add]
    status:str

class AgentState(TypedDict):
    user_input: str
    intent: str
    dry_run: bool
    reply:NotRequired[str]
    trace:Annotated[list[str],operator.add]

def classify_intent_node(state: AgentState) -> dict[str,str | list[str]]:
    user_input=state["user_input"]
    if "添加任务" in user_input:
        return {"intent":"add_task","trace":["classify_intent"]}
    else:
        return {"intent":"chat","trace":["classify_intent"]}

def route_by_intent(state:AgentState)->str:
    intent = state["intent"]
    if intent == "add_task":
        return "task_node"
    elif intent == "chat":
        return "chat_node"
    else:
        raise ValueError("unknown intent")

def task_node(state:AgentState) -> dict[str,str | list[str]]:
    user_input=state["user_input"]
    dry_run=state["dry_run"]
    if dry_run:
        return {"reply":"任务已模拟处理，尚未保存。","trace":["task_node"]}
    else:
        return {"reply":"真实执行暂未接入。","trace":["task_node"]}

def chat_node(state:AgentState) -> dict[str,str | list[str]]:
    user_input=state["user_input"]
    return {"reply":f"收到你的问题：{user_input}","trace":["chat_node"]}