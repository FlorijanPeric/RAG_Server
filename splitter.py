import os
import re
from dataclasses import dataclass, field


def _env_int(name: str, default: int) -> int:
    value = (os.getenv(name) or "").strip()
    return int(value) if value else default


@dataclass
class SplitterConfig:
    # Smaller chunks = more precise retrieval; the LLM gets top_k of them as context.
    # Changing CHUNK_MAX_CHARS only affects new ingests: re-initialize with
    # scripts/init.sh --reset afterwards.
    max_chars: int = field(default_factory=lambda: max(100, _env_int("CHUNK_MAX_CHARS", 300)))
    overlap: int = 100
    min_chunk_size: int = 50
    sentence_split_regex: str = r"(?<=[.!?])\s+"
    paragraph_split_regex: str = r"\n\s*\n"
    overlap_sentences: int = field(default_factory=lambda: max(0, _env_int("CHUNK_OVERLAP_SENTENCES", 1)))


class TextSplitter:
    def __init__(self, config: SplitterConfig):
        self.config = config

    def split_into_paragraphs(self, text):
        normalized = text.replace("\r\n", "\n")
        paragraphs = re.split(self.config.paragraph_split_regex, normalized)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_into_sentences(self, text):
        sentences = re.split(self.config.sentence_split_regex, text)
        units = []
        for sentence in (s.strip() for s in sentences):
            if not sentence:
                continue
            if len(sentence) <= self.config.max_chars:
                units.append(sentence)
            else:
                # Tables and lists often have no sentence punctuation at all; cut such
                # "sentences" into word windows so no chunk grows to thousands of chars.
                units.extend(self._split_words(sentence))
        return units

    def _split_words(self, text):
        pieces = []
        current = []
        current_length = 0
        for word in text.split():
            if current and current_length + 1 + len(word) > self.config.max_chars:
                pieces.append(" ".join(current))
                current, current_length = [], 0
            current.append(word)
            current_length += len(word) + (1 if current_length else 0)
        if current:
            pieces.append(" ".join(current))
        return pieces

    def _chunk_units(self, units):
        chunks = []
        current_units = []
        current_length = 0

        for unit in units:
            unit_length = len(unit)
            additional = unit_length if not current_units else unit_length + 1

            if current_units and current_length + additional > self.config.max_chars:
                chunk_text = " ".join(current_units).strip()
                if len(chunk_text) >= self.config.min_chunk_size:
                    chunks.append(chunk_text)

                overlap_count = max(0, self.config.overlap_sentences)
                if overlap_count > 0:
                    current_units = current_units[-overlap_count:]
                else:
                    current_units = []
                current_length = len(" ".join(current_units))
                # Drop the overlap if it would push the next chunk over the limit.
                if current_units and current_length + 1 + unit_length > self.config.max_chars:
                    current_units = []
                    current_length = 0

            current_units.append(unit)
            current_length = len(" ".join(current_units))

        if current_units:
            chunk_text = " ".join(current_units).strip()
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)

        return chunks

    def chunk_text(self, text):
        paragraphs = self.split_into_paragraphs(text)
        if not paragraphs:
            return []

        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= self.config.max_chars:
                if len(paragraph) >= self.config.min_chunk_size:
                    chunks.append(paragraph)
                continue

            sentences = self.split_into_sentences(paragraph)
            if not sentences:
                continue

            sentence_chunks = self._chunk_units(sentences)
            chunks.extend(sentence_chunks)

        return chunks