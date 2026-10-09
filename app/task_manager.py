import logging

from app.storage import load_tasks, save_tasks

logger = logging.getLogger(__name__)

def get_all_tasks() -> list[dict]:
    logger.info("用户发起了查看任务请求")
    return load_tasks()


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
    tasks = load_tasks()
    for task in tasks:
        if task["id"]==task_id:
            task["completed"]=True
            save_tasks(tasks)
            logger.info("用户完成任务,id=%d",task_id)
            return True
    logger.warning("找不到该任务,id=%d",task_id)
    return False


def delete_task(task_id:int)->bool:
    tasks = load_tasks()
    for task in tasks:
        if task["id"]==task_id:
            tasks.remove(task)
            save_tasks(tasks)
            logger.info("用户删除任务,id=%d", task_id)
            return True
    logger.warning("找不到该任务,id=%d", task_id)
    return False

