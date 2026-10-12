import json
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent
FILES_PATH = BASE_PATH / "data" /  "tasks.json"

def save_tasks(tasks):
    FILES_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(FILES_PATH, "w",encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=4)

def load_tasks():
    try:
        with open(FILES_PATH,"r",encoding="utf-8") as f:
            return json.load(f)

    except FileNotFoundError:
        return []