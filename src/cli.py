#!/usr/bin/env python3
"""CLI for the Prompt Engineering and Hallucination Study.

Usage:
    python -m src.cli run [--model PATH] [--strategies zero_shot few_shot ...]
    python -m src.cli score
    python -m src.cli report
    python -m src.cli retrieval-check
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prompts import STRATEGIES
from src.report import build_report
from src.retrieval import Retriever, load_passages
from src.run_eval import load_questions, run_eval

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUESTIONS = os.path.join(BASE, "data", "questions.jsonl")
PASSAGES = os.path.join(BASE, "data", "passages.jsonl")
RESULTS = os.path.join(BASE, "results", "results.jsonl")
DEFAULT_MODEL = "/home/hatch/workspace/finetune-small-lm/models/base"


def main() -> None:
    ap = argparse.ArgumentParser(prog="prompt-hallucination-study")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run the eval (resumable)")
    p_run.add_argument("--model", default=DEFAULT_MODEL)
    p_run.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=None)
    p_run.add_argument("--questions", default=QUESTIONS)
    p_run.add_argument("--passages", default=PASSAGES)
    p_run.add_argument("--out", default=RESULTS)

    sub.add_parser("score", help="print the per-strategy summary table")
    sub.add_parser("report", help="write results.json and results.md")
    sub.add_parser("retrieval-check", help="measure retrieval top-1 accuracy")
    sub.add_parser("rescore", help="recompute labels from stored responses")

    args = ap.parse_args()

    if args.cmd == "run":
        run_eval(args.model, args.questions, args.passages, args.out,
                 strategies=args.strategies)
    elif args.cmd == "score":
        from src.report import load_records
        from src.score import summarize
        summary = summarize(load_records(RESULTS))
        print(f"{'strategy':<20}{'acc':>7}{'halluc':>8}{'abstain':>9}{'other':>7}")
        for s in STRATEGIES:
            if s in summary:
                m = summary[s]
                print(f"{s:<20}{m['accuracy'] * 100:6.1f}%{m['hallucination_rate'] * 100:7.1f}%"
                      f"{m['abstention_rate'] * 100:8.1f}%{m['other_rate'] * 100:6.1f}%")
    elif args.cmd == "report":
        from src.retrieval import default_retriever
        r = default_retriever(os.path.join(BASE, "data"))
        acc = r.retrieval_accuracy(load_questions(QUESTIONS))
        out = build_report(RESULTS, os.path.join(BASE, "results"), acc)
        print(f"wrote results/results.json and results/results.md "
              f"(retrieval top-1 accuracy {acc:.1%})")
    elif args.cmd == "retrieval-check":
        r = Retriever(load_passages(PASSAGES))
        acc = r.retrieval_accuracy(load_questions(QUESTIONS))
        print(f"retrieval top-1 accuracy: {acc:.1%} "
              f"({int(acc * 42)}/42 answerable questions)")
    elif args.cmd == "rescore":
        from src.run_eval import rescore
        out = rescore(RESULTS)
        print(f"rescored {out['records']} records, {out['changed']} labels changed")


if __name__ == "__main__":
    main()
