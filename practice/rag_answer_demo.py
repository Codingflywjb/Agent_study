
from pathlib import Path
from sentence_transformers import SentenceTransformer

from practice.rag_index import (
    build_vector_index,
    save_vector_index
)
from practice.rag_answer import ask_rag
# 当前文件位于 Agent_study/practice/ 目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
INDEX_PATH = PROJECT_ROOT / "data" / "rag_index.json"

def main():
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    if not INDEX_PATH.exists():
        indexed_chunks = build_vector_index(
            directory=KNOWLEDGE_DIR,
            model=model
        )

        save_vector_index(
            indexed_chunks=indexed_chunks,
            index_path=INDEX_PATH
        )

    result = ask_rag(
        question="唐朝是哪一年建立的？",
        index_path=INDEX_PATH,
        embedding_model=model,
        k=2,
        min_score=0.45
    )

    print(result["answer"])
    print(result["sources"])


if __name__ == "__main__":
    main()
