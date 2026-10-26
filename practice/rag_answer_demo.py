
from pathlib import Path
from sentence_transformers import SentenceTransformer

from practice.rag_index import (
    build_vector_index,
    save_vector_index
)
from practice.rag_answer import ask_rag


def main():
    index_path = Path("data/rag_index.json")

    # TODO 1：加载 BGE 模型
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    # TODO 2：
    # 如果索引文件不存在：
    # 调用 build_vector_index("knowledge", model)
    # 再调用 save_vector_index()
    #
    # 提示：if not index_path.exists():
    if not index_path.exists():
        indexed_chunks=build_vector_index("knowledge",model)
        save_vector_index(indexed_chunks,index_path)

    # TODO 3：
    # 调用 ask_rag()
    # question="FastAPI 如何实现流式输出？"
    # index_path=index_path
    # embedding_model=model
    # k=2
    result=ask_rag(
        question="FastAPI 如何实现流式输出？",
        index_path=index_path,
        embedding_model=model,
        k=2
    )

    # TODO 4：
    # 打印 result["answer"]
    # 遍历 result["sources"] 打印参考来源
    print(result["answer"])
    print(result["sources"])


if __name__ == "__main__":
    main()
