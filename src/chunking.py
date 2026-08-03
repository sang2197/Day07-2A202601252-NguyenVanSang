from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|(?<=\.)\n", text) if s.strip()]

        limit = self.max_sentences_per_chunk
        return [
            " ".join(sentences[index : index + limit])
            for index in range(0, len(sentences), limit)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        pieces = self._split(text, self.separators)
        return [p.strip() for p in pieces if p.strip()]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]

        if not remaining_separators or remaining_separators[0] == "":
            return [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator, rest = remaining_separators[0], remaining_separators[1:]
        if separator not in current_text:
            return self._split(current_text, rest)

        chunks: list[str] = []
        current = ""
        for part in current_text.split(separator):
            candidate = part if not current else current + separator + part
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
            if len(part) > self.chunk_size:
                chunks.extend(self._split(part, rest))
                current = ""
            else:
                current = part

        if current:
            chunks.append(current)
        return chunks


class HeadingChunker:
    """
    Split text into chunks aligned to Markdown ATX headings (#, ##, ...).

    Mỗi section (một dòng heading + nội dung cho tới heading kế tiếp) là một
    đơn vị ngữ nghĩa trọn vẹn -> ưu tiên tách trước mỗi dòng heading. Section
    nào dài quá `chunk_size` mới hạ xuống RecursiveChunker; khi đó tiêu đề
    được gắn lại vào từng mảnh con để không mất ngữ cảnh (nếu không, mảnh
    thứ hai trở đi sẽ không còn biết mình thuộc mục nào).
    """

    HEADING_RE = re.compile(r"^#{1,6}\s+.+$", re.MULTILINE)

    def __init__(self, chunk_size: int = 500) -> None:
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        chunks: list[str] = []
        for heading, body in self._split_by_heading(text):
            piece = f"{heading}\n{body}".strip() if heading else body.strip()
            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                chunks.append(piece)
                continue

            # Section quá dài: hạ xuống RecursiveChunker rồi gắn lại heading
            # vào từng mảnh con để giữ ngữ cảnh.
            for sub in RecursiveChunker(chunk_size=self.chunk_size).chunk(body):
                sub = sub.strip()
                if not sub:
                    continue
                chunks.append(f"{heading}\n{sub}" if heading else sub)

        return chunks

    def _split_by_heading(self, text: str) -> list[tuple[str, str]]:
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            return [("", text)]

        sections: list[tuple[str, str]] = []
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for index, match in enumerate(matches):
            heading = match.group().strip()
            body_start = match.end()
            body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            sections.append((heading, text[body_start:body_end]))

        return sections


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size).chunk(text),
            "by_sentences": SentenceChunker().chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count else 0.0
            result[name] = {"count": count, "avg_length": avg_length, "chunks": chunks}
        return result
