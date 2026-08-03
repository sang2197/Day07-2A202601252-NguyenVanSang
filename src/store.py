from __future__ import annotations

import os
import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            persist_dir = os.getenv("CHROMA_PERSIST_DIR")
            if persist_dir:
                client = chromadb.PersistentClient(path=persist_dir)
            else:
                # chromadb's default in-process backend is a singleton shared
                # by (tenant, database) across every Client() instance in the
                # same process, so unrelated EmbeddingStore objects reusing a
                # collection_name would otherwise silently share data. Give
                # each ephemeral store its own database to keep instances
                # isolated, matching what "no persistence" implies.
                database = f"embedding-store-{uuid.uuid4().hex}"
                chromadb.AdminClient().create_database(database)
                client = chromadb.Client(database=database)

            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        metadata = dict(doc.metadata)
        metadata.setdefault("doc_id", doc.id)
        return {
            "id": f"{doc.id}::{self._next_index}",
            "content": doc.content,
            "metadata": metadata,
            "embedding": self._embedding_fn(doc.content),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_vector = self._embedding_fn(query)
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _dot(query_vector, record["embedding"]),
            }
            for record in records
        ]
        return sorted(scored, key=lambda item: item["score"], reverse=True)[:top_k]

    def _search_chroma(self, query: str, top_k: int, where: dict | None = None) -> list[dict[str, Any]]:
        query_vector = self._embedding_fn(query)
        result = self._collection.query(query_embeddings=[query_vector], n_results=top_k, where=where)
        ids = result.get("ids", [[]])[0]
        contents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # collection uses cosine space, so distance = 1 - cosine_similarity.
        return [
            {"id": id_, "content": content, "metadata": metadata, "score": 1 - distance}
            for id_, content, metadata, distance in zip(ids, contents, metadatas, distances)
        ]

    @staticmethod
    def _build_where(metadata_filter: dict | None) -> dict | None:
        if not metadata_filter:
            return None
        if len(metadata_filter) == 1:
            key, value = next(iter(metadata_filter.items()))
            return {key: value}
        return {"$and": [{key: value} for key, value in metadata_filter.items()]}

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records = []
        for doc in docs:
            records.append(self._make_record(doc))
            self._next_index += 1
        if not records:
            return

        if self._use_chroma:
            self._collection.add(
                ids=[r["id"] for r in records],
                documents=[r["content"] for r in records],
                embeddings=[r["embedding"] for r in records],
                metadatas=[r["metadata"] for r in records],
            )
        else:
            self._store.extend(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            return self._search_chroma(query, top_k)
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if self._use_chroma:
            return self._search_chroma(query, top_k, where=self._build_where(metadata_filter))

        if metadata_filter:
            records = [
                record
                for record in self._store
                if all(record["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        else:
            records = self._store
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma:
            existing = self._collection.get(where={"doc_id": doc_id})
            ids = existing.get("ids", [])
            if not ids:
                return False
            self._collection.delete(ids=ids)
            return True

        size_before = len(self._store)
        self._store = [record for record in self._store if record["metadata"].get("doc_id") != doc_id]
        return len(self._store) < size_before
