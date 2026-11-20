"""Resumable evaluation runner.

Runs every (strategy, question) pair and appends one JSON line per pair to
results/results.jsonl with flush + fsync after every write, so a VM restart
loses at most the in-flight item. Re-running skips pairs already recorded.
The model is loaded once and shared across all five strategies.
"""
from __future__ import annotations

import json
import os
import time

from src.generate import Generator
from src.prompts import STRATEGIES, build_prompt, max_new_tokens
from src.retrieval import Retriever, load_passages
from src.score import score_response


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_done(path: str) -> set[tuple[str, str]]:
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        done.add((r["strategy"], r["question_id"]))
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def rescore(path: str) -> dict:
    """Recompute every label in results.jsonl from the stored responses.

    Labels are derived data: if a scoring rule is corrected, re-running this
    updates the labels without re-generating anything. Returns a count of
    labels that changed.
    """
    changed = 0
    records = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    for r in records:
        new_label = score_response(r["response"], r, r["strategy"])
        if new_label != r["label"]:
            r["label"] = new_label
            changed += 1
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return {"records": len(records), "changed": changed}


def run_eval(model_path: str, questions_path: str, passages_path: str,
             out_path: str, strategies: list[str] | None = None,
             generator=None) -> dict:
    """Run the eval. Returns {'done': n_done, 'todo': n_remaining}."""
    strategies = strategies or STRATEGIES
    questions = load_questions(questions_path)
    done = load_done(out_path)
    retriever = Retriever(load_passages(passages_path))

    todo = []
    for s in strategies:
        for q in questions:
            if (s, q["id"]) not in done:
                todo.append((s, q))
    print(f"[eval] {len(done)} pairs done, {len(todo)} to do", flush=True)
    if not todo:
        print("[eval] ALL DONE", flush=True)
        return {"done": len(done), "todo": 0}

    gen = generator or Generator(model_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (strategy, q) in enumerate(todo, 1):
            s = time.time()
            passage = None
            if strategy == "retrieval":
                passage = retriever.retrieve_text(q["question"])
            prompt = build_prompt(strategy, q["question"], passage)
            response = gen.generate(prompt, max_new_tokens(strategy))
            label = score_response(response, q, strategy)
            rec = {
                "strategy": strategy,
                "question_id": q["id"],
                "category": q["category"],
                "answerable": q["answerable"],
                "question": q["question"],
                "answers": q["answers"],
                "prompt": prompt,
                "response": response,
                "label": label,
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
            print(f"[eval] {len(done) + i}/{len(done) + len(todo)} "
                  f"{strategy}/{q['id']} -> {label} ({time.time() - s:.0f}s)",
                  flush=True)
    print(f"[eval] finished in {time.time() - t0:.0f}s", flush=True)
    return {"done": len(done) + len(todo), "todo": 0}
