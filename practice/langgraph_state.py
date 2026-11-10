from typing import TypedDict, NotRequired


class AgentState(TypedDict):
    user_input: str
    intent: str
    dry_run: bool
    reply:NotRequired[str]

def classify_intent_node(state: AgentState) -> dict[str,str]:
    user_input=state["user_input"]
    if "添加任务" in user_input:
        return {"intent":"add_task"}
    else:
        return {"intent":"chat"}

def route_by_intent(state:AgentState)->str:
    intent = state["intent"]
    if intent == "add_task":
        return "task_node"
    elif intent == "chat":
        return "chat_node"
    else:
        raise ValueError("unknown intent")

def task_node(state:AgentState) -> dict[str,str]:
    user_input=state["user_input"]
    dry_run=state["dry_run"]
    if dry_run:
        return {"reply":"任务已模拟处理，尚未保存。"}
    else:
        return {"reply":"真实执行暂未接入。"}

def chat_node(state:AgentState) -> dict[str,str]:
    user_input=state["user_input"]
    return {"reply":f"收到你的问题：{user_input}"}