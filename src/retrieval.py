"""TF-IDF retrieval over the local passage corpus.

For the retrieval-grounded strategy we prepend the single most similar
passage to the prompt. The corpus is tiny (54 passages), so a plain
scikit-learn TF-IDF index is fast, deterministic and needs no downloads.
"""
from __future__ import annotations

import json
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_passages(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Retriever:
    def __init__(self, passages: list[dict]):
        self.passages = passages
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([p["text"] for p in passages])

    def retrieve(self, question: str, top_k: int = 1) -> list[dict]:
        """Return the top-k most similar passages for a question."""
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        order = scores.argsort()[::-1][:top_k]
        return [self.passages[i] for i in order]

    def retrieve_text(self, question: str) -> str:
        return self.retrieve(question, top_k=1)[0]["text"]

    def retrieval_accuracy(self, questions: list[dict]) -> float:
        """Fraction of answerable questions whose top-1 passage is the true one."""
        correct = 0
        total = 0
        for q in questions:
            if not q["answerable"]:
                continue
            total += 1
            top = self.retrieve(q["question"], top_k=1)[0]
            if top.get("question_id") == q["id"]:
                correct += 1
        return correct / total if total else 0.0


def default_retriever(data_dir: str | None = None) -> Retriever:
    base = data_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    return Retriever(load_passages(os.path.join(base, "passages.jsonl")))
