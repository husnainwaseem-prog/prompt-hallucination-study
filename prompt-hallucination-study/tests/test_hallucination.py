"""Tests for the hallucination study. Uses FakeLLM throughout — no model
download and no GPU/CPU generation in the test suite.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generate import FakeLLM
from src.prompts import STRATEGIES, build_prompt, max_new_tokens
from src.retrieval import Retriever, load_passages
from src.run_eval import load_done, load_questions, run_eval
from src.score import (
    contains_answer, is_abstention, is_hedged, score_response, scoring_text, summarize,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
QUESTIONS = load_questions(os.path.join(DATA, "questions.jsonl"))


def q_by_id(qid):
    return next(q for q in QUESTIONS if q["id"] == qid)


# ---------------------------------------------------------------- scoring

def test_gold_inside_longer_word_does_not_count():
    # "Paris" inside "comparison" must not count as correct.
    assert not contains_answer("In comparison, London is bigger.", ["Paris"])
    assert not contains_answer("The comparison was made in August.", ["Au"])


def test_whole_token_match_counts():
    assert contains_answer("The capital is Paris.", ["Paris"])
    assert contains_answer("It boils at 100 degrees Celsius.", ["100"])
    assert contains_answer("The answer is Mount Everest.", ["Mount Everest"])


def test_abstention_phrases_detected():
    assert is_abstention("I don't know.")
    assert is_abstention("Unknown — I cannot answer that.")
    assert is_abstention("I'm not sure about this one.")
    assert is_abstention("The first human landing on Mars is not mentioned in the passage.")
    assert is_abstention("The passage does not say who won that prize.")
    assert not is_abstention("Paris is the capital.")


def test_hedged_wrong_answer_is_other_not_hallucination():
    q = q_by_id("g01")  # capital of France = Paris
    assert score_response("Maybe it is London?", q, "zero_shot") == "other"
    assert score_response("I think it might be London.", q, "zero_shot") == "other"


def test_confident_wrong_answer_is_hallucination():
    q = q_by_id("g01")
    assert score_response("London", q, "zero_shot") == "hallucination"


def test_correct_answer_scores_correct():
    q = q_by_id("g01")
    assert score_response("Paris", q, "zero_shot") == "correct"


def test_trap_question_abstention_is_correct():
    q = q_by_id("t01")  # first human on Mars — nobody has landed
    assert score_response("I don't know.", q, "zero_shot") == "correct"


def test_trap_question_confident_answer_is_hallucination():
    q = q_by_id("t01")
    assert score_response("1969", q, "zero_shot") == "hallucination"


def test_trap_question_premise_denial_is_correct():
    # The model refuses the false premise instead of guessing — that is
    # correct, not a hallucination.
    assert score_response(
        "There is no known human who has walked on the surface of Venus.",
        q_by_id("t03"), "zero_shot") == "correct"
    assert score_response(
        "The Roman Empire did not conquer the Moon.",
        q_by_id("t04"), "few_shot") == "correct"
    assert score_response(
        "There is no exact population for the city of Atlantis as it is a fictional city.",
        q_by_id("t05"), "few_shot") == "correct"
    # But an invented name or year is still a hallucination.
    assert score_response("Valentina Tereshkova", q_by_id("t03"), "zero_shot") == "hallucination"
    assert score_response("Jacques Delors", q_by_id("t06"), "zero_shot") == "hallucination"


def test_cot_scores_only_final_answer_line():
    q = q_by_id("g01")
    resp = ("Paris is a city in France. London is the capital of England. "
            "Final answer: London")
    assert score_response(resp, q, "chain_of_thought") == "hallucination"
    resp2 = ("London is a big city. Paris is the capital of France. "
             "Final answer: Paris")
    assert score_response(resp2, q, "chain_of_thought") == "correct"
    # Without a "Final answer:" line the whole response is scored.
    assert scoring_text("Just Paris.", "zero_shot") == "Just Paris."
    # An empty "Final answer:" line falls back to the whole response.
    resp3 = ("Spiders are arachnids and have eight legs.\nFinal answer:")
    assert scoring_text(resp3, "chain_of_thought") == resp3
    assert score_response(resp3, q_by_id("s05"), "chain_of_thought") == "correct"


# ---------------------------------------------------------------- prompts

def test_all_five_strategies_exist():
    assert STRATEGIES == ["zero_shot", "few_shot", "chain_of_thought",
                          "retrieval", "strict_abstention"]


def test_few_shot_uses_no_test_questions():
    # The examples block must not contain any test question. (The target
    # question itself is of course appended after the examples.)
    from src.prompts import _few_shot_block
    block = _few_shot_block()
    test_questions = {q["question"] for q in QUESTIONS}
    for q in test_questions:
        assert q not in block
    prompt = build_prompt("few_shot", "What is the capital of France?")
    assert "What is the capital of Italy?" in prompt
    assert prompt.count("Question:") == 4  # 3 examples + the target


def test_strict_abstention_contains_the_instruction():
    prompt = build_prompt("strict_abstention", "What is the capital of France?")
    assert "don't know" in prompt.lower()
    assert "instead of guessing" in prompt.lower()


def test_chain_of_thought_asks_for_final_answer_line():
    prompt = build_prompt("chain_of_thought", "What is the capital of France?")
    assert "step by step" in prompt.lower()
    assert "final answer" in prompt.lower()


def test_retrieval_prompt_contains_passage_and_requires_one():
    prompt = build_prompt("retrieval", "What is the capital of France?",
                          passage="Paris is the capital of France.")
    assert "Paris is the capital of France." in prompt
    with pytest.raises(ValueError):
        build_prompt("retrieval", "What is the capital of France?")
    assert max_new_tokens("chain_of_thought") > max_new_tokens("zero_shot")


# ---------------------------------------------------------------- retrieval

def test_retrieval_returns_true_passage():
    r = Retriever(load_passages(os.path.join(DATA, "passages.jsonl")))
    top = r.retrieve("What is the capital of France?", top_k=1)[0]
    assert top["question_id"] == "g01"
    top = r.retrieve("What is the chemical symbol for gold?", top_k=1)[0]
    assert top["question_id"] == "s01"


def test_retrieval_accuracy_is_high():
    r = Retriever(load_passages(os.path.join(DATA, "passages.jsonl")))
    acc = r.retrieval_accuracy(QUESTIONS)
    assert acc >= 0.9, f"retrieval top-1 accuracy only {acc:.2f}"


# ---------------------------------------------------------------- dataset

def test_question_set_has_at_least_40_with_traps():
    answerable = [q for q in QUESTIONS if q["answerable"]]
    traps = [q for q in QUESTIONS if not q["answerable"]]
    assert len(QUESTIONS) >= 40
    assert len(answerable) >= 40
    assert len(traps) >= 6
    assert all(q["answers"] for q in QUESTIONS)


# ---------------------------------------------------------------- resume + fake eval

def test_resume_skips_done_pairs(tmp_path):
    out = tmp_path / "results.jsonl"
    out.write_text(json.dumps({"strategy": "zero_shot", "question_id": "g01"}) + "\n")
    done = load_done(str(out))
    assert ("zero_shot", "g01") in done
    assert ("zero_shot", "g02") not in done


def test_fake_eval_end_to_end(tmp_path):
    fake = FakeLLM({"capital of france": "Paris"})
    out = str(tmp_path / "results.jsonl")
    res = run_eval(
        model_path="unused", questions_path=os.path.join(DATA, "questions.jsonl"),
        passages_path=os.path.join(DATA, "passages.jsonl"), out_path=out,
        strategies=["zero_shot"], generator=fake,
    )
    assert res["done"] == 48
    with open(out, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    assert len(rows) == 48
    assert all("label" in r for r in rows)
    # FakeLLM answers Paris for the France question and "London" otherwise.
    france = next(r for r in rows if r["question_id"] == "g01")
    assert france["label"] == "correct"
    japan = next(r for r in rows if r["question_id"] == "g06")
    assert japan["label"] == "hallucination"
    # Re-running must skip everything (resume).
    res2 = run_eval(
        model_path="unused", questions_path=os.path.join(DATA, "questions.jsonl"),
        passages_path=os.path.join(DATA, "passages.jsonl"), out_path=out,
        strategies=["zero_shot"], generator=fake,
    )
    assert res2["todo"] == 0
    with open(out, encoding="utf-8") as f:
        assert len(f.readlines()) == 48


def test_summarize_counts_rates():
    records = [
        {"strategy": "zero_shot", "label": "correct", "answerable": True},
        {"strategy": "zero_shot", "label": "hallucination", "answerable": True},
        {"strategy": "zero_shot", "label": "abstention", "answerable": True},
        {"strategy": "zero_shot", "label": "correct", "answerable": False},
    ]
    s = summarize(records)["zero_shot"]
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["hallucination_rate"] == pytest.approx(0.25)
    assert s["abstention_rate"] == pytest.approx(0.25)
    assert s["answerable_accuracy"] == pytest.approx(1 / 3)
    assert s["trap_correct_rate"] == pytest.approx(1.0)
