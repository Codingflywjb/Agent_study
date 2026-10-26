from pathlib import Path
from sentence_transformers import SentenceTransformer

from practice.rag_cached_search import search_cached_knowledge
from practice.rag_prompt import build_rag_prompt
from practice.llm_client import ask_llm
from practice.rag_filter import filter_retrieved_chunks

def ask_rag(
    question: str,
    index_path: str | Path,
    embedding_model: SentenceTransformer,
    k: int = 2,
    min_score: float = 0.45
) -> dict:
    if not question.strip():
        raise ValueError('Empty question')

    if k <= 0:
        raise ValueError('k must be greater than 0')

    retrieved_chunks = search_cached_knowledge(query=question,index_path=index_path,model=embedding_model,k=k)


    if not retrieved_chunks:
        return {
            "answer": "知识库中没有可用的参考资料，暂时无法回答该问题。",
            "sources": []
        }
    retrieved_chunks = filter_retrieved_chunks(
        retrieved_chunks,
        min_score=min_score
    )

    if not retrieved_chunks:
        return {
            "answer": "知识库中没有足够相关的参考资料，暂时无法回答该问题。",
            "sources": []
        }

    prompt = build_rag_prompt(question,retrieved_chunks)

    answer=ask_llm(prompt)

    sources = []
    for chunk in retrieved_chunks:
        source=chunk["source"]
        if source not in sources:
            sources.append(source)

    return {
        "answer": answer,
        "sources": sources
    }