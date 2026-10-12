from fastapi import FastAPI,HTTPException,Path
from app.task_manager import get_all_tasks, find_task_by_id, add_task, update_task_completion, delete_task
from pydantic import BaseModel,Field,field_validator

class TaskCreate(BaseModel):
    title: str = Field(min_length=1,max_length=100)

    @field_validator("title")
    @classmethod
    def validate_title(cls,value:str)->str:
        value = value.strip()
        if not value:
            raise ValueError("Title is required")
        return value

class TaskUpdate(BaseModel):
    complete: bool

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Hello FastAPI"}


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service":"Agent_study"
    }


@app.get("/tasks")
def list_tasks():
    return get_all_tasks()


@app.get("/tasks/{task_id}")
def get_task(task_id:int):
    tasks = list_tasks()
    task = find_task_by_id(tasks, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/tasks",status_code=201)
def create_task(task:TaskCreate):
    title=task.title
    new_task=add_task(title=title)
    return new_task


@app.post("/tasks/preview")
def preview_task(task:TaskCreate):
    return {
        "title": task.title,
        "completed": False
    }

@app.patch("/tasks/{task_id}")
def patch_task(update:TaskUpdate,task_id:int=Path(gt=0)):
    complete=update.complete
    task =update_task_completion(task_id,complete)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.delete("/tasks/{task_id}",status_code=204)
def remove_task_api(task_id:int=Path(gt=0)):
    result=delete_task(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")