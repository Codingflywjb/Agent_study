from idlelib import history

from fastapi import FastAPI,HTTPException,Path
from app import sqlite_storage
from app.task_manager import get_all_tasks, find_task_by_id, add_task, update_task_completion, delete_task
from pydantic import BaseModel,Field,field_validator
from contextlib import asynccontextmanager
from typing import Literal
from practice.llm_client import ask_llm,LLMServiceError
import logging

logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("开始初始化数据库")
    try:
        sqlite_storage.init_db()
    except Exception:
        logger.exception("数据库初始化失败")
        raise
    logger.info("数据库初始化成功")
    yield
    logger.info("应用正在关闭")

app = FastAPI(lifespan=lifespan)

class TaskCreate(BaseModel):
    title: str = Field(min_length=1,max_length=100)

    @field_validator("title")
    @classmethod #在class里创建方法
    def validate_title(cls,value:str)->str:
        value = value.strip()
        if not value:
            raise ValueError("Title is required")
        return value

class TaskUpdate(BaseModel):
    completed: bool

class HistoryMessage(BaseModel):
    role:Literal["user","assistant"]
    content:str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def validate_content(cls,content:str)->str:
        content=content.strip()
        if not content:
            raise ValueError("内容为空")
        return content

class ChatRequest(BaseModel):
    message: str = Field(min_length=1,max_length=2000)#规定message的长度范围
    history: list[HistoryMessage] = Field(default_factory=list)

    @field_validator("message") #检验消息是否为空,validator是装饰器，后面得跟一个函数定义def
    @classmethod
    def validate_message(cls,message:str)->str:
        message=message.strip()
        if not message:
            raise ValueError("消息为空")
        return message

class ChatResponse(BaseModel):
    answer:str

@app.post("/chat",response_model=ChatResponse)
def chat(request:ChatRequest):
    history_dicts = [
        msg.model_dump() for msg in request.history
    ] # model_dump()是 Pydantic 提供的方法，可以把一个 Pydantic 模型对象转换成 Python 字典 dict
    #msg不是普通字典，而是一个 HistoryMessage 对象。

    try:
        response= ask_llm(request.message,history_dicts)

        return ChatResponse(answer=response)
    except LLMServiceError as e:
        raise HTTPException(status_code=502,detail="模型服务暂时不可用") #用HTTPException将模型服务错误转换成HTTP 502






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
    return sqlite_storage.get_all_tasks()


@app.get("/tasks/{task_id}")
def get_task(task_id:int = Path(gt=0)):
    row = sqlite_storage.get_task_by_id(task_id)
    task = sqlite_storage.row_to_task(row)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/tasks",status_code=201)
def create_task(task:TaskCreate):
    title=task.title
    new_id=sqlite_storage.create_task(title)
    new_row=sqlite_storage.get_task_by_id(new_id)
    new_task=sqlite_storage.row_to_task(new_row)
    if new_task is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve newly created task")
    return new_task


@app.post("/tasks/preview")
def preview_task(task:TaskCreate):
    return {
        "title": task.title,
        "completed": False
    }

@app.patch("/tasks/{task_id}")
def update_task_api(task:TaskUpdate,task_id:int=Path(gt=0)):
    complete=task.completed
    result=sqlite_storage.update_task_completion(task_id,complete)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    row = sqlite_storage.get_task_by_id(task_id)
    updated_task = sqlite_storage.row_to_task(row)
    if updated_task is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve updated task")
    return updated_task

@app.delete("/tasks/{task_id}",status_code=204)
def remove_task_api(task_id:int=Path(gt=0)):
    result=sqlite_storage.delete_task(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")