"""Small in-memory BM25 keyword index (no extra dependencies).

Used for hybrid retrieval next to the vector search: BM25 finds chunks that share
the exact (rare) words of the question - names, numbers, technical terms - which
embeddings often miss.
"""
import math
import re
from collections import Counter

# Slovenian is highly inflected (čebela, čebele, čebelam, ...). Cutting every word
# to its first characters is a crude but effective stemmer for keyword matching.
STEM_CHARS = 6
WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [word[:STEM_CHARS] for word in WORD_RE.findall(text.lower())]


class BM25Index:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(document)) for document in documents]
        self.lengths = [sum(counts.values()) for counts in self.term_counts]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        document_frequency = Counter(term for counts in self.term_counts for term in counts)
        total = len(documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        # term -> documents containing it, so a query only scores matching documents.
        self.postings: dict[str, list[int]] = {}
        for index, counts in enumerate(self.term_counts):
            for term in counts:
                self.postings.setdefault(term, []).append(index)

    def __len__(self) -> int:
        return len(self.term_counts)

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Returns [(document index, score)] of the best `limit` matches."""
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index in self.postings[term]:
                frequency = self.term_counts[index][term]
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                scores[index] = scores.get(index, 0.0) + idf * frequency * (self.k1 + 1) / (frequency + norm)
        return sorted(scores.items(), key=lambda item: -item[1])[:limit]
