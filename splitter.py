import re
from dataclasses import dataclass


@dataclass
class SplitterConfig:
    max_chars: int = 400
    overlap: int = 100
    min_chunk_size: int = 50
    sentence_split_regex: str = r"(?<=[.!?])\s+"
    paragraph_split_regex: str = r"\n\s*\n"
    overlap_sentences: int = 1


class TextSplitter:
    def __init__(self, config: SplitterConfig):
        self.config = config

    def split_into_paragraphs(self, text):
        normalized = text.replace("\r\n", "\n")
        paragraphs = re.split(self.config.paragraph_split_regex, normalized)
        return [p.strip() for p in paragraphs if p.strip()]

    def split_into_sentences(self, text):
        sentences = re.split(self.config.sentence_split_regex, text)
        return [s.strip() for s in sentences if s.strip()]

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