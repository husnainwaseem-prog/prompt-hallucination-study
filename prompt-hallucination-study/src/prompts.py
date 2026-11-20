"""Prompt builders for the five strategies in this study.

Each builder takes a question (and, for retrieval-grounded, a passage) and
returns the user-message text that is sent to the model. The model is called
through its chat template, so these are message contents, not raw strings.
"""
from __future__ import annotations

STRATEGIES = ["zero_shot", "few_shot", "chain_of_thought", "retrieval", "strict_abstention"]

# Fixed few-shot examples. They use questions that are NOT in the test set,
# so the examples teach the answer format without leaking any test answer.
FEW_SHOT_EXAMPLES = [
    ("What is the capital of Italy?", "Rome"),
    ("How many continents are there on Earth?", "Seven"),
    ("Who painted the ceiling of the Sistine Chapel?", "Michelangelo"),
]


def _few_shot_block() -> str:
    lines = []
    for q, a in FEW_SHOT_EXAMPLES:
        lines.append(f"Question: {q}\nAnswer: {a}")
    return "\n\n".join(lines)


def build_prompt(strategy: str, question: str, passage: str | None = None) -> str:
    """Return the user-message text for one strategy and question."""
    if strategy == "zero_shot":
        return (
            "Answer the following question with a short factual answer.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if strategy == "few_shot":
        return (
            "Answer each question with a short factual answer, like the examples.\n\n"
            f"{_few_shot_block()}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if strategy == "chain_of_thought":
        return (
            "Answer the following question. Think step by step first, then give "
            "your final answer on the last line, starting with 'Final answer:'.\n\n"
            f"Question: {question}\n"
            "Let's think step by step:"
        )
    if strategy == "retrieval":
        if passage is None:
            raise ValueError("retrieval strategy needs a passage")
        return (
            "Use the following passage to answer the question. "
            "Answer with a short factual answer.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    if strategy == "strict_abstention":
        return (
            "Answer the following question with a short factual answer. "
            "If you are not sure of the answer, say 'I don't know' instead of guessing. "
            "Never guess.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
    raise ValueError(f"unknown strategy: {strategy}")


def max_new_tokens(strategy: str) -> int:
    """Chain-of-thought needs room to reason; the rest answer in a few words.
    CoT was first run at 96 tokens and two answers were cut off before the
    'Final answer:' line, so the cap is 128 (see REPORT.md)."""
    return 128 if strategy == "chain_of_thought" else 32
