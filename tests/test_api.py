
import pytest
from fastapi.testclient import TestClient
from app.api import app
from app import storage

client = TestClient(app)

@pytest.fixture
def sample_tasks(tmp_path,monkeypatch):
    monkeypatch.setattr(
        storage,
        "FILES_PATH",
        tmp_path / "tasks.json"
    )

    tasks = [
        {"id": 1, "title": "学习Python", "completed": False},
        {"id": 2, "title": "学习Agent", "completed": True}
    ]

    storage.save_tasks(tasks)

    return tasks

def test_health():
    response = client.get("/health")
    data = response.json()
    assert response.status_code == 200
    assert data["status"] == "ok"

def test_get_all_tasks(sample_tasks):
    response = client.get("/tasks")
    data = response.json()
    assert response.status_code == 200
    assert len(data) == 2
    assert data[0]["title"] == "学习Python"

def test_get_existing_task(sample_tasks):
    response = client.get("/tasks/2")
    data = response.json()
    assert response.status_code == 200
    assert data["id"] == 2
    assert data["completed"] is True

def test_create_task(sample_tasks):
    response = client.post("/tasks", json={"title": "学习FastAPI"})
    assert response.status_code == 201
    saved_tasks = storage.load_tasks()
    data = response.json()
    assert data["title"] == "学习FastAPI"
    assert data["completed"] is False
    assert data["id"] == 3
    assert len(saved_tasks) == 3
    assert saved_tasks[2]["id"] == 3
    assert saved_tasks[2]["title"] == "学习FastAPI"
    assert saved_tasks[2]["completed"] is False

def test_update_task(sample_tasks):
    task_two_before=sample_tasks[1]
    response=client.patch("/tasks/1",json={"complete":True})
    assert response.status_code == 200
    saved_tasks = storage.load_tasks()
    task_two_after=saved_tasks[1]
    data = response.json()
    assert data["completed"] is True
    saved_tasks = storage.load_tasks()
    assert saved_tasks[0]["completed"] is True
    assert task_two_before==task_two_after

def test_delete_task(sample_tasks):
    response = client.delete("/tasks/1")
    assert response.status_code == 204
    assert response.content == b""
    saved_tasks = storage.load_tasks()
    ids = [task["id"] for task in saved_tasks]
    assert 1 not in ids
    assert 2 in ids

def test_get_nonexistent_task(sample_tasks):
    response = client.get("/tasks/999")
    assert response.status_code == 404
    saved_tasks = storage.load_tasks()
    assert saved_tasks == sample_tasks

def test_patch_nonexistent_task(sample_tasks):
    response = client.patch("/tasks/999",json={"complete":True})
    assert response.status_code == 404
    saved_tasks = storage.load_tasks()
    assert saved_tasks == sample_tasks

def test_patch_invalid_body(sample_tasks):
    response=client.patch("/tasks/1",json={"title":""})
    assert response.status_code == 422
    saved_tasks = storage.load_tasks()
    assert saved_tasks == sample_tasks

def test_delete_nonexistent_task(sample_tasks):
    response=client.delete("/tasks/999")
    assert response.status_code == 404
    saved_tasks = storage.load_tasks()
    assert saved_tasks == sample_tasks
