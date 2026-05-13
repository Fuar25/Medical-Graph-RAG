def split_text_chunks(
    content: str, chunk_size: int = 3000, overlap: int = 300
) -> list[str]:
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunks.append(content[start:end])
        start += chunk_size - overlap
    return chunks
