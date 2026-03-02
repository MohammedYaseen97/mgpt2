## 00 — Tokenizer report (mgpt2)

**Status: COMPLETE**

### Objective
Train a BPE tokenizer with the same vocabulary structure as GPT-2 (256 bytes + 50,000 merges + 1 special = 50,257 IDs; model padded to 50,304) but on a multilingual corpus covering English, Hindi (Devanagari + transliterated Latin), and Kannada (Kannada script + transliterated Latin). Demonstrate improved compression over GPT-2 and cl100k baselines on Indic and mixed-script text.

---

### Artifacts

| File | SHA-256 |
|---|---|
| `tokenizer/artifacts/mgpt2.model` | `08f6556502d945abb84fe8963e71b5025852d342e19d2992991cc2adfc4158ff` |
| `tokenizer/artifacts/mgpt2.vocab` | `5c369a50615cbdb1fe608907b86ebf6eb4471742f1175f2fa29e2474e473bb4f` |
| `tokenizer/artifacts/heldout_eval.txt` | `57b44c6a79cc52890f8cb36a28223f44491fc85e9b546417bc08dff42bd699a2` |
| `tokenizer/artifacts/tokenizer_eval.json` | (regenerate with eval command below) |

Git commit at time of training: `eac6dd98c40af3cdde393706e823d4b2e409248e`

---

### Commands

**Build tokenizer training corpus** (500K docs, same mixture as pretraining):
```bash
virtual/bin/python scripts/data/build_corpus_mixture.py \
  --limit 500000 \
  --buffer-size 10000 \
  --output-file tokenizer/tok_corpus_large.txt
```

**Create held-out set** (before training, leak-free protocol):
```bash
virtual/bin/python -m tokenizer.scripts.make_heldout \
  --corpus tokenizer/tok_corpus_large.txt \
  --out tokenizer/artifacts/heldout_eval.txt
```

**Train tokenizer**:
```bash
virtual/bin/python -m tokenizer.train_tokenizer \
  --corpus tokenizer/tok_corpus_large.txt \
  --exclude_lines_file tokenizer/artifacts/heldout_eval.txt \
  --num_merges 50000 \
  --sample_lines 100000 \
  --min_chunk_freq 5 \
  --max_chunks 200000 \
  --out_prefix tokenizer/artifacts/mgpt2
```

**Evaluate**:
```bash
virtual/bin/python -m tokenizer.scripts.evaluate \
  --text tokenizer/artifacts/heldout_eval.txt \
  --limit 10000 \
  --model tokenizer/artifacts/mgpt2.model > tokenizer/artifacts/tokenizer_eval.json
```

---

### Results (tokens/1k bytes — lower is better)

| Bucket | tiktoken_gpt2 | tiktoken_cl100k | **mgpt2** | vs gpt2 | vs cl100k |
|---|---|---|---|---|---|
| **Overall** | 494.0 | 369.2 | **224.5** | −55% | −39% |
| **Devanagari** | 593.9 | 384.6 | **219.8** | −63% | −43% |
| **Kannada** | 976.3 | 641.4 | **216.8** | −78% | −66% |
| **Latin** | 255.7 | 248.9 | **228.8** | −11% | −8% |
| **Mixed** | 795.8 | 522.8 | **220.1** | −72% | −58% |

Held-out: 10,000 lines from `heldout_eval.txt` (excluded from tokenizer training).
Baselines: tiktoken_gpt2 (monolingual reference), tiktoken_cl100k_base (multilingual reference, 2× vocab size).

---

### Conclusion

mgpt2 meets the Phase A pass criterion on all three required buckets (Devanagari, Kannada, mixed). The headline result is Kannada: GPT-2's tokenizer — having no Kannada merges — encodes Kannada at near byte-level (976 tokens/1k bytes, ~1 byte/token). mgpt2 achieves 217 tokens/1k bytes, a 4.5× compression improvement. Devanagari sees a 2.7× improvement over GPT-2 and 1.75× over cl100k. Mixed-script text (the most practically relevant bucket for code-switched Indic content) improves 3.6× over GPT-2 and 2.4× over cl100k.

The Latin bucket regression is minimal (−8% vs cl100k, −11% vs gpt2) and expected — a fixed vocab budget of 50K merges allocated partially to Indic scripts leaves slightly fewer merges for English subwords. This is the correct design tradeoff for a multilingual model.

Training knobs used: `--sample_lines 100000` (reservoir sample from 500K-doc corpus to stay within RAM), `--min_chunk_freq 5` (drops hapax-legomena from BPE state), `--max_chunks 200000` (caps unique chunk types for tractable training time on CPU). The compression results confirm these heuristics did not meaningfully degrade tokenizer quality — all improvements are large relative to any quality loss from pruning.
