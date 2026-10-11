import pytest

from app.task_manager import find_task_by_id,validate_task_id

tasks=[
        {"id": 1,"title": "学习py","completed": False},
        {"id": 2, "title": "学习java", "completed": False}
    ]

def test_find_existing_task():
    result=find_task_by_id(tasks,1)
    assert result["title"] == "学习py" and result["id"] == 1

def test_find_non_existing_task():
    task_id=99
    result=find_task_by_id(tasks,task_id)
    assert result is None

def test_task_id_valid():
    validate_task_id(3)

def test_task_id_zero_isvalid():
    with pytest.raises(ValueError):
        validate_task_id(0)

def test_negative_task_id_raise_value_error():
    task_id=-5
    with pytest.raises(ValueError):
        validate_task_id(task_id)