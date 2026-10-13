import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR/"data"/"tasks.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        completed INTEGER NOT NULL DEFAULT 0
    )
""")


cursor.execute("select * from tasks order by id ASC")
rows=cursor.fetchall()
for row in rows:
    print(row)

print("------")

cursor.execute("update tasks set completed=1 where id=?",(1,))
conn.commit()
print("-------")

cursor.execute("select * from tasks where id=?",(1,))
row=cursor.fetchone()
print(row)

print("------")
cursor.execute("delete from tasks where id=?",(2,))
if cursor.rowcount == 0:
    print("任务不存在")
else:
    print("删除成功")
conn.commit()
print("-------")
cursor.execute("select * from tasks order by id ASC")
rows=cursor.fetchall()
for row in rows:
    print(row)


conn.close()