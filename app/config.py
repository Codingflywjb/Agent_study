from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DB_PATH:Path=Path(os.getenv("TASKS_DB_PATH",str(BASE_DIR / "data" / "tasks.db")))
MODEL_NAME:str=os.getenv("LLM_MODEL", "gpt-4.1-mini")

