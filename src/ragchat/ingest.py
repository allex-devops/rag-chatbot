from dataclasses import dataclass
from pathlib import Path

from shared.docs import chunk_text, read_pages


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int


def chunks_from_file(path: Path, size: int = 800, overlap: int = 120) -> list[Chunk]:
    out = []
    for page, text in read_pages(path):
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            # stable ids mean re-ingesting the same file replaces its chunks instead of adding duplicates
            out.append(Chunk(id=f"{path.name}:{page}:{i}", text=piece, source=path.name, page=page))
    return out
