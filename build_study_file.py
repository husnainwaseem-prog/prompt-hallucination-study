#!/usr/bin/env python3
"""Build the single-file study copy: all Python source concatenated into one
copy-pasteable file. Header carries the one-line story + the real results.
Run:  python3 build_study_file.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.expanduser("~/workspace/your_files/prompt-hallucination-study-all-code.py")

FILES = [
    ("src/prompts.py", "1. PROMPTS — the five strategies"),
    ("src/retrieval.py", "2. RETRIEVAL — TF-IDF over the local passage corpus"),
    ("src/score.py", "3. SCORING — the hallucination definition and scoring rules"),
    ("src/generate.py", "4. GENERATION — SmolLM2-1.7B-Instruct, greedy decoding"),
    ("src/run_eval.py", "5. EVAL RUNNER — resumable, one JSON line per pair"),
    ("src/report.py", "6. REPORT BUILDER — results table"),
    ("src/cli.py", "7. CLI"),
    ("data/generate_data.py", "8. DATA BUILDER — regenerates questions + passages"),
]


def _read_results_table():
    import json
    path = os.path.join(HERE, "results", "results.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for s, m in data["strategies"].items():
        rows.append(
            f"#   {s:<18} acc {m['accuracy'] * 100:5.1f}%  "
            f"halluc {m['hallucination_rate'] * 100:5.1f}%  "
            f"abstain {m['abstention_rate'] * 100:5.1f}%"
        )
    return "\n".join(rows)


def main():
    table = _read_results_table()
    header = f'''"""Prompt Engineering and Hallucination Study — all the code in one file.

One-line story: five prompt strategies (zero-shot, few-shot, chain-of-thought,
retrieval-grounded, strict "say unknown if unsure") are tested on 48 factual
questions with SmolLM2-1.7B-Instruct, and every answer is scored as correct,
hallucination, abstention or other-incorrect under a fixed rule-based
definition — so the trade-off between accuracy and confident wrong answers
can be measured, not guessed.

Real results (accuracy / hallucination rate / abstention rate):
{table}

Read-only study copy. Run the project from the zip, not from this file.
"""
'''
    chunks = [header]
    for rel, title in FILES:
        with open(os.path.join(HERE, rel), encoding="utf-8") as f:
            src = f.read()
        # Keep only the first `from __future__ import annotations` (in the header's file).
        src = re.sub(r"^from __future__ import annotations\n", "", src, flags=re.MULTILINE)
        # Strip the module docstring's first line duplication is fine; keep as-is.
        chunks.append(f"\n# {'=' * 70}\n# {title}\n# {'=' * 70}\n")
        chunks.append(src)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    print(f"wrote {OUT} ({os.path.getsize(OUT)} bytes)")


if __name__ == "__main__":
    main()
