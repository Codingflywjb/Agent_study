import pytest

from app import storage
from app.task_manager import complete_task,delete_task

@pytest.fixture
def sample_tasks(tmp_path,monkeypatch):
    monkeypatch.setattr(
        storage,
        "FILES_PATH",
        tmp_path / "tasks.json"
    )

    tasks = [
        {"id": 1, "title": "学习Python", "completed": False},
        {"id": 2, "title": "学习Agent", "completed": False}
    ]

    storage.save_tasks(tasks)

    return tasks

def test_complete_existing_task(sample_tasks):
    result = complete_task(1)
    assert result is True
    saved_tasks = storage.load_tasks()
    assert saved_tasks[0]["completed"] is True

def test_complete_nonexistent_task(sample_tasks):
    result = complete_task(99)

    assert result is False

    saved_tasks = storage.load_tasks()

    assert sample_tasks == saved_tasks

def test_delete_existing_task(sample_tasks):
    result = delete_task(1)
    assert result is True

    saved_tasks = storage.load_tasks()

    ids = [task["id"] for task in saved_tasks]

    assert len(saved_tasks) == 1

    assert 1 not in ids

    assert 2 in ids


def test_delete_nonexistent_task(sample_tasks):
    result = delete_task(99)
    assert result is False
    saved_tasks = storage.load_tasks()
    assert sample_tasks == saved_tasks