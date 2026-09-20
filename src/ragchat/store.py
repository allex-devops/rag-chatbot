import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np

from .ingest import Chunk


WRITE_BATCH = 2000


class EmbeddingMismatch(RuntimeError):
    """The index was built with a different embedding model than the one now in use."""


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # cosine similarity, 1.0 is identical


class Store:
    def __init__(self, path: Path, name: str = "docs", embed_model: str | None = None):
        path = Path(path)
        client = chromadb.PersistentClient(path=str(path))
        self.col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._meta_file = path / f"{name}.embedding.json"
        self._check_model(embed_model)

    def _check_model(self, embed_model: str | None) -> None:
        # Vectors from two different models can have the same size and still mean nothing to each other, so a wrong model gives plausible-looking garbage instead of an error. Record which one built this.
        if embed_model is None:
            return
        if self._meta_file.exists():
            built_with = json.loads(self._meta_file.read_text())["embed_model"]
            if built_with != embed_model:
                raise EmbeddingMismatch(
                    f"index was built with '{built_with}' but this is '{embed_model}'; re-index, or switch back"
                )
        else:
            self._meta_file.write_text(json.dumps({"embed_model": embed_model}))

    def _check_dim(self, vectors: np.ndarray) -> None:
        if self.col.count() == 0:
            return
        stored = len(self.col.get(limit=1, include=["embeddings"])["embeddings"][0])
        if vectors.shape[-1] != stored:
            raise EmbeddingMismatch(f"index holds {stored}-dimensional vectors but got {vectors.shape[-1]}")

    def add(self, chunks: list[Chunk], vectors: np.ndarray, owner: str = "") -> None:
        if not chunks:
            return
        self._check_dim(vectors)
        # Chroma refuses a single write above a few thousand items, so a whole corpus has to go in slices
        for start in range(0, len(chunks), WRITE_BATCH):
            part = chunks[start : start + WRITE_BATCH]
            self.col.upsert(
                ids=[c.id if not owner else f"{owner}/{c.id}" for c in part],
                embeddings=vectors[start : start + WRITE_BATCH].astype(float).tolist(),
                documents=[c.text for c in part],
                metadatas=[{"source": c.source, "page": c.page, "owner": owner} for c in part],
            )

    def query(self, vector: np.ndarray, k: int = 5, owner: str | None = None) -> list[Hit]:
        total = self.col.count()
        if total == 0:
            return []
        self._check_dim(vector)
        res = self.col.query(
            query_embeddings=[vector.astype(float).tolist()],
            n_results=min(k, total),
            where={"owner": owner} if owner is not None else None,
        )
        return [
            Hit(text=doc, source=meta["source"], page=meta["page"], score=1.0 - dist)
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def delete_source(self, source: str, owner: str | None = None) -> None:
        """Drop every chunk that came from one file, for when the file changed or was deleted."""
        where = {"source": source} if owner is None else {"$and": [{"source": source}, {"owner": owner}]}
        self.col.delete(where=where)

    def count(self) -> int:
        return self.col.count()
