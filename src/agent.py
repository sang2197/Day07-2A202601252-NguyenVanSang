from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy tài liệu liên quan trong knowledge base để trả lời câu hỏi này."

        context = "\n".join(
            f"[{i}] (source: {r['metadata'].get('doc_id', 'unknown')}) {r['content']}"
            for i, r in enumerate(results, start=1)
        )

        prompt = (
            "Instruction: chỉ dùng context bên dưới để trả lời; nếu context không đủ, "
            "hãy nói rõ là không đủ thông tin.\n"
            f"Context: {context}\n"
            f"Question: {question}\n"
            "Answer:"
        )

        return self.llm_fn(prompt)
