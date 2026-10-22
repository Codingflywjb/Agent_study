
def split_text(
        text:str,
        chunk_size:int = 200,
        overlap: int = 30
) -> list[str]: #text为待切分文本，chunk_size单个Chunk的最大字符数,overlap为相邻Chunk重叠字符数 # 注意 chunk_size > 0,overlap >= 0,overlap < chunk_size
    #check环节
    if chunk_size <= 0:
        raise ValueError('chunk_size must be > 0')
    if overlap < 0:
        raise ValueError('overlap must be > 0')
    if overlap >= chunk_size:
        raise ValueError('overlap must be <= chunk_size')
    if not text:
        return []

    chunks = []

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk=text[start:end]
        chunks.append(chunk)

        if end >= len(text):
            break

        start += chunk_size-overlap

    return chunks

