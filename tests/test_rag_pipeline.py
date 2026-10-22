from practice.rag_pipeline import build_chunks

def test_build_chunks_multiple_files(tmp_path):
    file1 = tmp_path / "a.txt"
    file1.write_text("ABCDEFGHIJ", encoding="utf-8")

    file2 = tmp_path / "b.txt"
    file2.write_text("HELLO", encoding="utf-8")

    result = build_chunks(tmp_path, chunk_size=4,overlap=1)

    expected = [
        {"source": "a.txt", "chunk_id": 0, "content": "ABCD"},
        {"source": "a.txt", "chunk_id": 1, "content": "DEFG"},
        {"source": "a.txt", "chunk_id": 2, "content": "GHIJ"},
        {"source": "b.txt", "chunk_id": 0, "content": "HELL"},
        {"source": "b.txt", "chunk_id": 1, "content": "LO"}
    ]
    assert result == expected