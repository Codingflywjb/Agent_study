from practice.agent_tools import validate_tool_call

def safe_simulate_tool_call(tool_call: dict)->dict:
    try:
        tool_name, arguments = validate_tool_call(tool_call)
        return {
            "status": "simulated",
            "title": arguments["title"],
            "saved": False
        }
    except ValueError:
        return {
            "status": "error",
            "error": "invalid_tool_call",
            "saved": False
        }

