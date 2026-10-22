##enumerate() 会在遍历列表时，同时提供索引和元素
from pathlib import Path

from practice.rag_loader import load_documents
from practice.rag_splitter import split_text

def build_chunks(
        directory: str | Path,
        chunk_size: int = 200,
        overlap: int = 30
) -> list[dict]:
    documents = load_documents(directory)

    all_chunks = []

    for document in documents:
        content=document["content"]
        chunks = split_text(content, chunk_size=chunk_size, overlap=overlap)
        for chunk_id,chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": chunk_id,
                    "content": chunk,
                }
            )

    return all_chunks