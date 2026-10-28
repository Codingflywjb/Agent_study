from pathlib import Path
import os
from dotenv import load_dotenv
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class RagSettings:
    embedding_model_name: str
    index_path: Path
    min_score: float

def load_rag_settings() -> RagSettings:
    embedding_model_name = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5"
    )
    raw_path = os.getenv(
        "RAG_INDEX_PATH",
        "data/rag_index.json"
    )
    rag_index_path=Path(raw_path)
    if not rag_index_path.is_absolute():
        rag_index_path=PROJECT_ROOT / rag_index_path
    rag_min_score =float(os.getenv(
        "RAG_MIN_SCORE",
        "0.45"
    ))
    if not -1 <= rag_min_score <= 1:
        raise ValueError("rag_min_score must be between -1 and 1")
    return RagSettings(
        embedding_model_name=embedding_model_name,
        index_path=rag_index_path,
        min_score=rag_min_score
    )

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

DB_PATH:Path=Path(os.getenv("TASKS_DB_PATH",str(BASE_DIR / "data" / "tasks.db")))
MODEL_NAME:str=os.getenv("LLM_MODEL", "gpt-4.1-mini")

