from practice.agent_tools import (
    validate_tool_call,
    execute_tool_call,
)


def run_add_task_tool(
        tool_call: dict,
        *,  ## * 表示 dry_run 必须以关键字参数传入
        dry_run: bool = True
) -> dict:
    tool_name,arguments = validate_tool_call(tool_call)

    if dry_run:
        return {
            "status": "simulated",
            "title" : arguments["title"],
            "saved" : False
        }

    result = execute_tool_call(tool_call)

    return {
        "status" : "success",
         "title" : arguments["title"],
         "saved" : True,
        "result" : result
    }