# Prompt Engineering and Hallucination Study

Measure how five prompt strategies change a small language model's accuracy
and hallucination rate on factual question answering.

Model: **SmolLM2-1.7B-Instruct** (greedy decoding, seed 42), run locally on CPU.

## The five strategies

| Strategy | Idea |
|---|---|
| Zero-shot | Plain question, short answer. |
| Few-shot | Three worked examples first (questions not in the test set). |
| Chain-of-thought | "Think step by step", then a `Final answer:` line. |
| Retrieval-grounded | The best-matching passage from a local 54-passage corpus is prepended. |
| Strict abstention | "If you are not sure, say 'I don't know' instead of guessing." |

## The question set

48 questions in `data/questions.jsonl`: 42 answerable (7 each in science,
geography, history, arts, sport, general knowledge) and 6 trap questions
with a false premise or an unknowable fact, whose only correct response is
to abstain. `data/passages.jsonl` holds one true passage per answerable
question plus 12 distractors.

## What counts as a hallucination

See `REPORT.md` for the full definition. In short: a response is a
**hallucination** when it states a confident, specific answer that is wrong.
Saying "I don't know" is an **abstention**; a hedged wrong answer
("maybe…", "I think…") is **other-incorrect**. Correctness is judged by
whole-token matching against the accepted answers, so "Paris" never matches
inside "comparison".

## Run it

```bash
python -m venv .venv
# Windows PowerShell:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.cli run        # the eval; resumable, appends to results/results.jsonl
python -m src.cli score      # per-strategy summary table
python -m src.cli report     # writes results/results.json and results/results.md
python -m src.cli retrieval-check
python -m pytest tests\ -q
```

The eval is resumable: each (strategy, question) result is appended with
flush + fsync, and re-running skips pairs already recorded. If you do not
have the model locally, point `--model` at a Hugging Face id
(`HuggingFaceTB/SmolLM2-1.7B-Instruct`) or download `model.safetensors` plus
the tokenizer files into a folder and pass that folder.

## Files

- `src/prompts.py` — the five prompt builders
- `src/retrieval.py` — TF-IDF retrieval over the local passages
- `src/score.py` — the hallucination definition and scoring rules
- `src/generate.py` — model loading and greedy generation (+ FakeLLM for tests)
- `src/run_eval.py` — the resumable eval runner
- `src/report.py` — results table builder
- `data/generate_data.py` — regenerates the question set and passages
- `tests/test_hallucination.py` — pytest suite (uses FakeLLM, no model download)
- `REPORT.md` — the full report: definition, results table, trade-offs
