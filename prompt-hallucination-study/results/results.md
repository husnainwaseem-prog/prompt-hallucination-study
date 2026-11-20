# Results — Prompt Engineering and Hallucination Study

Model: SmolLM2-1.7B-Instruct (greedy decoding, seed 42). Questions: 48 (42 answerable, 6 trap). Retrieval top-1 accuracy: 100.0%.

| Strategy | Accuracy | Hallucination rate | Abstention rate | Other | Answerable acc. | Trap correct |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 89.6% | 10.4% | 0.0% | 0.0% | 97.6% | 33.3% |
| Few-shot | 85.4% | 14.6% | 0.0% | 0.0% | 90.5% | 50.0% |
| Chain-of-thought | 83.3% | 16.7% | 0.0% | 0.0% | 88.1% | 50.0% |
| Retrieval-grounded | 95.8% | 4.2% | 0.0% | 0.0% | 97.6% | 83.3% |
| Strict abstention | 89.6% | 6.2% | 4.2% | 0.0% | 90.5% | 83.3% |

Definitions: accuracy = share of responses labelled correct; hallucination rate = share that state a confident specific answer that is wrong; abstention rate = share that say they do not know; trap correct = share of the 6 trap questions answered by abstaining. See REPORT.md for the full hallucination definition and the trade-offs.
