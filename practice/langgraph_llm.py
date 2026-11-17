from langchain_core.messages import BaseMessage,AIMessage,ToolMessage
from langgraph.graph import MessagesState,StateGraph,START,END
from practice.agent_tools import sdk_tool_call_to_dict
from practice.agent_tool_executor import run_add_task_tool
import json
MAX_TOOL_CALLS = 3

def to_deepseek_messages(
        messages:list[BaseMessage]
)-> list[dict]:
    role_map = {
        "human":"user",
        "ai":"assistant",
        "system":"system"
    }

    result = []

    for message in messages:
        if isinstance(message,ToolMessage):
            result.append(tool_message_to_deepseek(message))
            continue
        if isinstance(message,AIMessage) and message.tool_calls:
            if isinstance(message.content,str):
                converted_tool_calls =[]
                for tool_call in message.tool_calls:
                    converted_tool_calls.append(langchain_tool_call_to_deepseek(tool_call))
                result.append({"role":"assistant","content":message.content,"tool_calls":converted_tool_calls})
                continue
            raise ValueError("AImessage content is not str")
        if message.type not in role_map:
            raise ValueError(f"unsupported message type: {message.type}")
        if not isinstance(message.content,str):
            raise ValueError(f"content is not str")
        result.append({"role":role_map[message.type],"content":message.content})

    return result

def call_deepseek_once(
        client,
        messages:list[BaseMessage],
        tools: list[dict] | None = None
) -> AIMessage:
    sdk_messages=to_deepseek_messages(messages)
    request_kwargs = {
        "model" : "deepseek-flash",
        "messages" :sdk_messages
    }
    if tools:
        request_kwargs["tools"] = tools
        request_kwargs["tool_choice"] = "auto"
    response = client.chat.completions.create(**request_kwargs)
    sdk_message = response.choices[0].message
    return sdk_message_to_ai_message(sdk_message)
    # content = sdk_message.content
    # if not isinstance(content,str):
    #     raise ValueError("content is not str")
    # return AIMessage(content=content)

def make_llm_node(client,tools:list[dict] | None=None):
    def llm_node(state:MessagesState) ->dict:
        messages = state["messages"]
        ai_message=call_deepseek_once(client,messages,tools)
        return {"messages":[ai_message]}
    return llm_node

def build_llm_graph(client,tools:list[dict] | None=None):
    builder = StateGraph(MessagesState)

    node = make_llm_node(client,tools=tools)

    builder.add_node("llm",node)

    builder.add_edge(START,"llm")

    builder.add_edge("llm",END)

    graph=builder.compile()

    return graph

def to_langchain_tool_call(tool_call:dict) ->dict:
    tool_call_id = tool_call["id"]
    name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])
    if not isinstance(args,dict):
        raise ValueError("args is not dict")
    return {
        "id":tool_call_id,
        "name":name,
        "args":args
    }

def make_tool_call_ai_message(sdk_tool_call:dict)->AIMessage:
    langchain_tool_call = to_langchain_tool_call(sdk_tool_call)
    message_ai = AIMessage(content="",tool_calls=[langchain_tool_call])
    return message_ai

def sdk_message_to_ai_message(sdk_message)->AIMessage:
    tool_calls = sdk_message.tool_calls
    if not tool_calls:
        content = sdk_message.content
        if isinstance(content,str):
            return AIMessage(content=content)
        raise ValueError("content is not str")
    if len(tool_calls) > 1:
        raise ValueError("tool_calls > 1")
    if sdk_message.content not in (None,""):
        raise ValueError("tool call content must be empty")
    sdk_tool_call = tool_calls[0]
    sdk_tool_call_dict = sdk_tool_call_to_dict(sdk_tool_call)
    result = make_tool_call_ai_message(sdk_tool_call_dict)
    return result

def tool_message_to_deepseek(message:ToolMessage)->dict:
    content = message.content
    if not isinstance(content,str):
        raise ValueError("content is not str")
    tool_call_id = message.tool_call_id

    return {
        "role":"tool",
        "content":content,
        "tool_call_id":tool_call_id
    }

def langchain_tool_call_to_deepseek(tool_call:dict)->dict:
    tool_call_id = tool_call["id"]
    tool_call_name=tool_call["name"]
    tool_call_args=tool_call["args"]
    if not isinstance(tool_call_args,dict):
        raise ValueError("args is not dict")
    tool_call_arguments = json.dumps(tool_call_args,ensure_ascii=False) #Python 字典 args → JSON 字符串 arguments
    return {
        "id":tool_call_id,
        "type":"function",
        "function":{
            "name":tool_call_name,
            "arguments":tool_call_arguments
        }
    }

def route_after_llm(state:MessagesState) ->str:
    last_message = state["messages"][-1]
    if not isinstance(last_message,AIMessage):
        raise ValueError("last message is not AImessage")
    if last_message.tool_calls:
        return "tool"
    else:
        return "end"

def tool_node(state:MessagesState)->dict:
    last_message = state["messages"][-1]
    if not isinstance(last_message,AIMessage):
        raise ValueError("last message is not AImessage")
    if len(last_message.tool_calls) != 1:
        raise ValueError("expected exactly one tool call")
    if count_tool_messages(state) >= MAX_TOOL_CALLS:
        raise ValueError("tool call limit exceeded")
    tool_call_langchain = last_message.tool_calls[0]
    tool_calls_dict = langchain_tool_call_to_deepseek(tool_call_langchain)
    try:
        result=run_add_task_tool(tool_calls_dict,dry_run=True)
    except Exception:
        result = {
        "status": "error",
        "message": "tool execution failed"
    }
    content = json.dumps(result,ensure_ascii=False)
    tool_message = ToolMessage(content=content,tool_call_id=tool_call_langchain["id"])
    return {"messages":[tool_message]}

def build_agent_graph(client,tools:list[dict]):
    builder = StateGraph(MessagesState)
    node = make_llm_node(client, tools=tools)
    builder.add_node("llm",node)
    builder.add_node("tool_node",tool_node)
    builder.add_edge(START,"llm")
    builder.add_conditional_edges("llm",route_after_llm, {"tool":"tool_node","end":END})
    builder.add_edge("tool_node","llm")
    graph=builder.compile()
    return graph

def count_tool_messages(state:MessagesState) -> int:
    count = 0
    for message in state["messages"]:
        if isinstance(message,ToolMessage):
            count += 1
    return count