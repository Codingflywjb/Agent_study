import sqlite3
from app.config import DB_PATH

def init_db():
    DB_PATH.parent.mkdir(parents=True,exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed INTEGER NOT NULL DEFAULT 0
        )    
        """)
        conn.commit()
    finally:
        conn.close()

def get_task_by_id(task_id:int) ->tuple|None:
    conn=sqlite3.connect(DB_PATH)
    cursor=conn.cursor()
    try:
        cursor.execute("SELECT id,title,completed FROM tasks WHERE id=?",(task_id,))
        return cursor.fetchone()
    finally:
        conn.close()

def create_task(title:str) -> int:
    title=title.strip()
    if len(title)==0:
        raise ValueError("title is empty")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tasks (title) VALUES (?)",(title,))
    new_id=cursor.lastrowid
    conn.commit()
    conn.close()
    return new_id

def update_task_completion(task_id:int, completed:bool)->bool:
    conn=sqlite3.connect(DB_PATH)
    try:
        cursor=conn.cursor()
        cursor.execute("UPDATE tasks SET completed=? WHERE id=?",(int(completed),task_id))
        affected_rows=cursor.rowcount
        conn.commit()
        return affected_rows > 0
    finally:
        conn.close()

def delete_task(task_id:int)->bool:
    conn=sqlite3.connect(DB_PATH)
    try:
        cursor=conn.cursor()
        cursor.execute("DELETE FROM tasks WHERE id=?",(task_id,))
        affected_rows = cursor.rowcount
        conn.commit()
        return affected_rows > 0
    finally:
        conn.close()

def row_to_task(row:tuple|None)->dict|None:
    if row is None:
        return None
    task={
        "id":row[0],
        "title":row[1],
        "completed":bool(row[2])
    }
    return task

def get_all_tasks()->list[dict]:
    conn=sqlite3.connect(DB_PATH)
    try:
        cursor=conn.cursor()
        cursor.execute("SELECT id,title,completed FROM tasks order by id asc")
        rows=cursor.fetchall()
        tasks=[]
        for row in rows:
            tasks.append(row_to_task(row))
        return tasks
    finally:
        conn.close()

