import logging

from typing import Optional
from app.storage import load_tasks, save_tasks

logger = logging.getLogger(__name__)

def validate_task_id(task_id:int)->None:
    if task_id <=0 :
        logger.warning("任务id=%d不为正整数", task_id)
        raise ValueError("任务id需要为正整数")

def get_all_tasks() -> list[dict]:
    logger.info("用户发起了查看任务请求")
    return load_tasks()

def find_task_by_id(tasks:list[dict],task_id:int)->Optional[dict]:
    for task in tasks:
        if task["id"]==task_id:
            return task
    return None

def add_task(title:str) -> None:
    if not title.strip():
        raise ValueError("任务名称不能为空")

    tasks = load_tasks()
    ids=[task["id"] for task in tasks]
    new_id=max(ids)+1 if ids else 1
    new_task = {
        "id": new_id,
        "title": title,
        "completed": False
    }
    tasks.append(new_task)
    save_tasks(tasks)
    logger.info("任务创建成功,id=%d",new_id)



def complete_task(task_id:int)->bool:
    validate_task_id(task_id)

    tasks = load_tasks()
    task=find_task_by_id(tasks,task_id)
    if task is None:
        logger.warning("找不到该任务,id=%d", task_id)
        return False

    task["completed"]=True
    save_tasks(tasks)
    logger.info("用户完成任务,id=%d",task_id)
    return True


def delete_task(task_id:int)->bool:
    validate_task_id(task_id)

    tasks = load_tasks()
    task=find_task_by_id(tasks,task_id)
    if task is None:
        logger.warning("找不到该任务,id=%d", task_id)
        return False

    tasks.remove(task)
    save_tasks(tasks)
    logger.info("用户删除任务,id=%d", task_id)
    return True

