## 01 — Pretraining report (baseline vs mgpt2)

### Objective
Demonstrate that mgpt2 tokenizer improves multilingual/translit handling **at the LM level** under a fair comparison.

### Runs
- Baseline run folder: [TODO]
- mgpt2 run folder: [TODO]

### Fairness checklist (must be identical)
- Architecture: GPT‑2 124M (same `model.py` config)
- Padded vocab size: 50304
- Block size: 1024
- Total tokens processed (token budget): [TODO log + verify]
- Optimizer + schedule: [TODO]

### Held-out evaluation
- Held-out text path: [TODO]
- Bucket rules: latin/deva/knda/mixed

### Metrics you must report
- Overall: val loss / perplexity
- Bucketed: perplexity per bucket
- Throughput: tokens/sec (optional but useful)
- Bytes-per-token advantage → connect back to tokenizer report

### Conclusion
[TODO] 10–15 sentences explaining whether the tokenizer materially helped and why.

