import sqlite3
import pytest

from app import sqlite_storage

@pytest.fixture
def test_db(tmp_path,monkeypatch):
    db_path = tmp_path / "test_tasks.db"

    monkeypatch.setattr(sqlite_storage, "DB_PATH",db_path)

    conn = sqlite3.connect(db_path)

    try:
        conn.execute("""
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                completed INTEGER NOT NULL default 0
            )
        """)
        conn.commit()
    finally:
        conn.close()

def test_get_nonexistent_task(test_db):
    result = sqlite_storage.get_task_by_id(999)
    assert result is None

def test_create_invalid_task(test_db):
    with pytest.raises(ValueError):
        sqlite_storage.create_task("   ")

def test_create_task(test_db):
    task_id = sqlite_storage.create_task("  学习SQLite  ")
    task = sqlite_storage.get_task_by_id(task_id)
    assert isinstance(task_id,int)
    assert task[0]==task_id
    assert task is not None
    assert task[1] == "学习SQLite"
    assert task[2] == 0

def test_update_task(test_db):
    task_id = sqlite_storage.create_task("test")
    result = sqlite_storage.update_task_completion(task_id, True)
    assert result is True
    task=sqlite_storage.get_task_by_id(task_id)
    assert task is not None
    assert task[2]==1

def test_delete_task(test_db):
    task_id = sqlite_storage.create_task("test")
    result = sqlite_storage.delete_task(task_id)
    assert result is True
    result = sqlite_storage.delete_task(task_id)
    assert result is False
    task = sqlite_storage.get_task_by_id(task_id)
    assert task is None

def test_delete_nonexistent_task(test_db):
    result = sqlite_storage.delete_task(999)
    assert result is False