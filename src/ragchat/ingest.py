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
            # ids are stable, so ingesting the same file twice replaces its chunks instead of doubling them . up for debate: should we include the page number in the id? it would make it easier to find the original text, but it would also make the ids less stable if we change the chunking algorithm
            out.append(Chunk(id=f"{path.name}:{page}:{i}", text=piece, source=path.name, page=page))
    return out
