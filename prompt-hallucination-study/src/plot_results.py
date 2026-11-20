#!/usr/bin/env python3
"""Plot accuracy vs hallucination rate per strategy -> results/plots/strategy_comparison.png"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(HERE, "results", "results.json"), encoding="utf-8") as f:
    data = json.load(f)

order = ["zero_shot", "few_shot", "chain_of_thought", "retrieval", "strict_abstention"]
labels = ["Zero-shot", "Few-shot", "Chain-of-\nthought", "Retrieval-\ngrounded", "Strict\nabstention"]
acc = [data["strategies"][s]["accuracy"] * 100 for s in order]
hal = [data["strategies"][s]["hallucination_rate"] * 100 for s in order]

os.makedirs(os.path.join(HERE, "results", "plots"), exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(order))
w = 0.38
ax.bar([i - w / 2 for i in x], acc, w, label="Accuracy", color="#2e7d32")
ax.bar([i + w / 2 for i in x], hal, w, label="Hallucination rate", color="#c62828")
for i, (a, h) in enumerate(zip(acc, hal)):
    ax.text(i - w / 2, a + 1.5, f"{a:.1f}%", ha="center", fontsize=9)
    ax.text(i + w / 2, h + 1.5, f"{h:.1f}%", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels(labels)
ax.set_ylabel("Share of 48 questions (%)")
ax.set_title("Accuracy vs hallucination rate by prompt strategy\n(SmolLM2-1.7B-Instruct, 48 factual questions)")
ax.set_ylim(0, 110)
ax.legend()
fig.tight_layout()
out = os.path.join(HERE, "results", "plots", "strategy_comparison.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
