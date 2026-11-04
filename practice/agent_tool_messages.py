import json


def build_tool_result_message(
        tool_call_id:str,
        result:dict
) -> dict:
    content= json.dumps(result,ensure_ascii=False)
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content
    }

def build_tool_conversation(
        user_text:str,
        tool_call:dict,
        tool_result:dict
) -> list[dict]:
    user_message={
        "role": "user",
        "content": user_text
    }
    assistant_message={
        "role": "assistant",
        "content": None,
        "tool_calls":[tool_call]
    }
    tool_result_message=build_tool_result_message(tool_call["id"],tool_result)
    return [user_message,assistant_message,tool_result_message]