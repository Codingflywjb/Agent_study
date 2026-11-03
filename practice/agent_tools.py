import json
from app.task_manager import add_task
add_task_tool = {
    "type" : "function",
    "function" : {
        "name" : "add_task",
        "description":"当用户要求创建或添加待办任务时使用此工具",
        "parameters": {
            "type":"object",
            "properties":{
                "title":{
                    "type":"string",
                    "description":"需要添加的任务名称"
                }
            },
            "required":["title"]
        }
    }
}

ALLOWED_TOOLS = {"add_task"}

##负责从模型返回的数据中提取工具名称和参数
def parse_tool_call(tool_call:dict)->tuple[str,dict]:
    tool_name = tool_call["function"]["name"]
    arguments_str = tool_call["function"]["arguments"]
    arguments = json.loads(arguments_str)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dict")
    return tool_name, arguments

##负责检查 title 的合法性
def validate_add_task_arguments(arguments: dict) -> str:
    if "title" not in arguments:
        raise ValueError("title is required")
    title = arguments["title"]
    if not isinstance(title, str):
        raise ValueError("title must be a string")
    if not title.strip():
        raise ValueError("title cannot be empty")
    return title.strip()

##负责检查调用工具是否在白名单
def validate_tool_call(tool_call: dict) -> tuple[str, dict]:
    tool_name, arguments = parse_tool_call(tool_call)
    if tool_name not in ALLOWED_TOOLS:
        raise ValueError("Unknown tool")
    title = validate_add_task_arguments(arguments)
    return tool_name,{"title":title}

##负责调用工具
def execute_tool_call(tool_call: dict):
    tool_name, arguments = validate_tool_call(tool_call)
    if tool_name == "add_task":
        result=add_task(title=arguments["title"])
        return result
    raise ValueError("Unknown tool")

##ds模型返回的是sdk对象而非Python字典,负责转换对象
def sdk_tool_call_to_dict(tool_call) -> dict:
    return {
        "id":tool_call.id,
        "type":tool_call.type,
        "function":{
            "name":tool_call.function.name,
            "arguments":tool_call.function.arguments
        }
    }