
from sentence_transformers import SentenceTransformer

from practice.rag_embedding import embed_text
from practice.rag_search import search_knowledge

def main():
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    question="FastAPI如何实现流式输出?"

    query_embedding = embed_text(text=question,model=model,is_query=True)
    print(len(query_embedding))

    result=search_knowledge(query=question,directory="knowledge",model=model,k=2)

    for chunk in result:
        print(chunk["source"])
        print(chunk["score"])
        print(chunk["content"])

if __name__=="__main__":
    main()