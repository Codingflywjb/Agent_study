import sqlite3

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "tasks.db"

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
