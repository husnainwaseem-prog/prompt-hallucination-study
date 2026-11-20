"""Build the results table (results.json + results.md) from results.jsonl."""
from __future__ import annotations

import json
import os

from src.score import summarize


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_report(results_path: str, out_dir: str, retrieval_accuracy: float) -> dict:
    records = load_records(results_path)
    summary = summarize(records)
    order = ["zero_shot", "few_shot", "chain_of_thought", "retrieval", "strict_abstention"]
    labels = {
        "zero_shot": "Zero-shot",
        "few_shot": "Few-shot",
        "chain_of_thought": "Chain-of-thought",
        "retrieval": "Retrieval-grounded",
        "strict_abstention": "Strict abstention",
    }
    table = {s: summary[s] for s in order if s in summary}

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"retrieval_accuracy": retrieval_accuracy, "strategies": table},
                  f, indent=2)

    lines = [
        "# Results — Prompt Engineering and Hallucination Study",
        "",
        f"Model: SmolLM2-1.7B-Instruct (greedy decoding, seed 42). "
        f"Questions: {len({r['question_id'] for r in records})} "
        f"({sum(1 for r in records if r['answerable']) // len(table)} answerable, "
        f"{sum(1 for r in records if not r['answerable']) // len(table)} trap). "
        f"Retrieval top-1 accuracy: {_pct(retrieval_accuracy)}.",
        "",
        "| Strategy | Accuracy | Hallucination rate | Abstention rate | Other | Answerable acc. | Trap correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in order:
        if s not in table:
            continue
        m = table[s]
        lines.append(
            f"| {labels[s]} | {_pct(m['accuracy'])} | {_pct(m['hallucination_rate'])} | "
            f"{_pct(m['abstention_rate'])} | {_pct(m['other_rate'])} | "
            f"{_pct(m['answerable_accuracy'])} | {_pct(m['trap_correct_rate'])} |"
        )
    lines += [
        "",
        "Definitions: accuracy = share of responses labelled correct; hallucination "
        "rate = share that state a confident specific answer that is wrong; "
        "abstention rate = share that say they do not know; trap correct = share "
        "of the 6 trap questions answered by abstaining. See REPORT.md for the "
        "full hallucination definition and the trade-offs.",
        "",
    ]
    with open(os.path.join(out_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return {"retrieval_accuracy": retrieval_accuracy, "strategies": table}
