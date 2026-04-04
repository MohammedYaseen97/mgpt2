## 04 — Final comparison (what you show interviewers)

**Status: COMPLETE**

### Claim

This project demonstrates:
1. A custom multilingual tokenizer (mgpt2) that compresses English, Hindi (Devanagari + Latin translit), and Kannada (Kannada script + Latin translit) significantly more efficiently than tiktoken GPT-2 or tiktoken cl100k_base.
2. A GPT-2 124M model pretrained from scratch with that tokenizer, evaluated against a controlled baseline (same architecture, same data mixture, same token budget, same optimizer).
3. SFT fine-tuning on multilingual instruction data with measurable convergence and a prompt-suite generation assessment.
4. DPO alignment on toxic preference pairs with win-rate and SFT regression checks.
5. Fair, reproducible comparisons: every run captures git hash, tokenizer SHA, config snapshot, and seeds.

---

### Evidence checklist

| Phase | Metric | Report | HF repo |
|---|---|---|---|
| Phase A — Tokenizer | Tokens/1k bytes per bucket; p95 tokens/line | [reports/00_tokenizer_report.md](00_tokenizer_report.md) | [ace-1/mgpt2-tokenizer](https://huggingface.co/ace-1/mgpt2-tokenizer) |
| Phase B — Data pipeline | Corpus build, shard manifest, eval split construction | (no experiment report; pipeline in `scripts/`) | — |
| Phase C — Pretrain | BPB (overall + bucketed); HellaSwag; PPL | [reports/01_pretrain_report.md](01_pretrain_report.md) | [ace-1/mgpt2-pretrain](https://huggingface.co/ace-1/mgpt2-pretrain) |
| Phase D — SFT | Held-out loss + PPL; prompt suite | [reports/02_sft_report.md](02_sft_report.md) | [ace-1/mgpt2-sft](https://huggingface.co/ace-1/mgpt2-sft) |
| Phase E — DPO | Preference win-rate; SFT regression check | [reports/03_dpo_report.md](03_dpo_report.md) | [ace-1/mgpt2-dpo](https://huggingface.co/ace-1/mgpt2-dpo) |

---

### Architecture (both models)

| Parameter | Value |
|---|---|
| Parameters | 124M |
| Layers | 12 |
| Heads | 12 |
| d_model | 768 |
| Block size | 1024 tokens |
| Total training tokens | ~14.4B (27,537 steps × 524,288 tokens/step) |
| Optimizer | AdamW, seed 1337 |

---

### Phase A — Tokenizer (tokens/1k bytes, lower is better)

All three tokenizers have similar vocabulary sizes: tiktoken_gpt2: 50,257 tokens; **mgpt2: 50,257 tokens** (256 bytes + 1 EOT + 50,000 BPE merges); tiktoken_cl100k: 100,256 tokens. mgpt2 achieves its compression gains entirely through better merge priorities for Indic scripts — not by inflating vocabulary size.

| Bucket | tiktoken_gpt2 | tiktoken_cl100k | **mgpt2** | vs gpt2 | vs cl100k | % of eval bytes |
|---|---:|---:|---:|---:|---:|---:|
| **Overall** | 480.0 | 356.7 | **222.7** | −54% | −38% | 100% |
| **Latin** | 256.6 | 249.5 | **229.8** | −10% | −8% | 53.6% |
| **Devanagari** | 591.8 | 384.1 | **215.0** | −64% | −44% | 12.3% |
| **Kannada** | 980.9 | 639.7 | **213.3** | −78% | −67% | 8.2% |
| **Mixed** | 730.7 | 476.1 | **214.7** | −71% | −55% | 25.9% |

Latin accounts for 53.6% of eval bytes and shows the smallest compression gain (−10% vs gpt2). The overall −54% is driven by Indic scripts and Mixed buckets. For a predominantly Latin/English corpus, the tokenizer advantage narrows to single digits — this is expected and honest.

p95 tokens/line (overall): tiktoken_gpt2: 6,645 — tiktoken_cl100k: 4,647 — **mgpt2: 2,497**

---

### Phase C — Pretraining (controlled comparison)

Both runs: identical GPT-2 124M architecture, seed 1337, 14.4B tokens, same data mixture.

#### Raw metrics (invalid for cross-tokenizer PPL comparison)

| Metric | Baseline (gpt2 tok) | mgpt2 |
|---|---:|---:|
| Final train loss | 1.431 | 2.733 |
| Final val loss | 1.296 | 2.500 |
| HellaSwag acc | 0.2768 | **0.2869** |
| PPL (overall) | **3.563** | 12.407 |

The baseline PPL appears lower because GPT-2's vocabulary is sparse over Indic scripts — predicting the next byte among 256 possible values yields low perplexity regardless of language understanding. PPL without bits-per-byte normalisation is not a valid cross-tokenizer comparison metric.

#### Bits-per-byte (BPB) — fair cross-tokenizer comparison

`BPB = log₂(PPL) × (tokens/1k bytes ÷ 1000)`

| Bucket | PPL (baseline) | PPL (mgpt2) | tok/1kB (baseline) | tok/1kB (mgpt2) | BPB (baseline) | BPB (mgpt2) | Δ BPB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Overall** | 3.563 | 12.407 | 480.0 | 222.7 | 0.880 | **0.809** | −0.071 |
| **Latin** | 20.679 | 25.604 | 256.6 | 229.8 | 1.121 | **1.075** | −0.046 |
| **Devanagari** | 1.830 | 4.531 | 591.8 | 215.0 | 0.516 | **0.469** | −0.047 |
| **Kannada** | 1.445 | 4.470 | 980.9 | 213.3 | 0.521 | **0.461** | −0.060 |
| **Mixed** | 1.710 | 4.576 | 730.7 | 214.7 | 0.566 | **0.471** | −0.095 |

**mgpt2 is better on every single bucket, including English (Latin), when measured correctly.**

HellaSwag is consistent with the BPB direction: mgpt2 scores +1.01 pp higher (0.2869 vs 0.2768) on an English-only cloze benchmark despite using a multilingual tokenizer. **Note on significance**: with n=10,042 HellaSwag examples, the two-proportion z-score is 1.59 (95% CI: −0.23pp to +2.25pp), which does not reach the conventional 95% significance threshold (z=1.96). The result is directionally consistent with BPB but should not be interpreted as a statistically confirmed finding in isolation. BPB is the primary and pre-specified comparison metric; HellaSwag is corroborating evidence. The most likely mechanism behind the HellaSwag gap is shared byte embedding interference in the baseline — GPT-2's byte-level Indic encoding causes the same token IDs to carry contradictory gradient signal across scripts (Latin-extended bytes vs. Devanagari/Kannada UTF-8 continuation bytes), degrading all representations including English.

Relevant git hashes: baseline `6c118bd`, mgpt2 `8aa064a`.

---

### Phase D — SFT

Fine-tuned the Phase C mgpt2 checkpoint on 30K multilingual instruction–response pairs (ai4bharat/indic-align + IndicTrans2 romanisation), 3 epochs, masked CE loss over response tokens only.

| Metric | Value |
|---|---|
| Final train loss | 0.9113 |
| Final val loss | 1.2404 |
| Val PPL (SFT set) | 3.457 ¹ |
| Training steps | 1,262 |
| Eval interval | 50 steps |

¹ SFT val PPL is not comparable to pretrain LM eval PPL (12.4). The SFT eval uses masked CE over response tokens only, measured on the held-out SFT set — a much narrower domain. The lower PPL reflects the easier in-distribution domain and masked formulation, not a general LM improvement.

Prompt suite highlights: Devanagari and Kannada native scripts respond coherently. Transliterated Latin (`hin_Latn`, `kan_Latn`) is prone to mid-generation script drift — caused by the absence of a Unicode anchor (transliterated text shares ASCII with English), low training data volume (2,100 examples per variant), and irreducible ASCII space ambiguity under any BPE tokenizer. This is documented and expected at this scale.

Git hash: `d072240`.

---

### Phase E — DPO

Applied DPO (β=0.1) on top of the SFT checkpoint using 13,493 preference pairs. Chosen: Llama2-70B-Chat safety refusals (coherent, formatted). Rejected: pretrained model continuations (incoherent word-salad).

| Metric | Value |
|---|---|
| Preference win-rate (held-out, n=1,496) | **1.000 (100%)** |
| DPO val loss (final) | 3.8e-10 (≈ 0) |
| SFT val loss regression | +0.0153 (+1.2%) |
| Regression check | ✅ passed |
| Total steps | 420 |

The 100% win-rate reflects easy discrimination between coherent chosen responses and word-salad rejected responses. More meaningfully: DPO increased the absolute log-probability of chosen (refusal) responses by **+6.1%** and decreased the log-probability of rejected responses by **−16.8%** relative to the SFT checkpoint (n=1,496 val pairs). Generation comparisons confirm the behavioural shift — the SFT model tries to helpfully comply with toxic prompts (it was only trained on benign instruction data); the DPO model redirects or reframes. The deflections are not clean structured refusals, but the directional shift is real and measurable. This is format-preference alignment rather than genuine safety alignment at production scale; the pipeline is fully functional and the result is honest about its scope.

Git hash: `e463752`.

---

### Reproducibility

Every run records:

| Field | Where |
|---|---|
| Git commit hash | `runs/*/run_info.json` |
| Tokenizer SHA-256 | `runs/*/run_info.json` |
| Config snapshot | `configs/*.yaml` + run_info |
| Seed | All runs: seed 1337 |
| Eval data | `data/eval/` (fixed, versioned) |

Tokenizer artifacts: `f2911100` (mgpt2.model SHA-256 prefix).

---

### Narrative

This project built a complete multilingual language model pipeline from the tokenizer up to preference alignment. The central contribution is a custom regex + BPE tokenizer targeting English, Hindi, and Kannada (both native and romanised scripts), which achieves the same 50,257-token vocabulary size as tiktoken GPT-2 but compresses the target corpus 54% more efficiently overall — driven by Indic and Mixed buckets (−64% to −78%); the gain on Latin-only text is modest (−10%). When evaluated fairly using bits-per-byte, the mgpt2 model is superior on every language bucket in the held-out eval set, reversing the naïve PPL ranking entirely. HellaSwag (+1.01 pp on English) is directionally consistent but falls short of statistical significance at 95% (z=1.59); BPB is the primary pre-specified metric. SFT converged cleanly to val loss 1.24 on 30K multilingual instruction pairs. DPO ran to 100% win-rate with a 1.2% SFT regression (well within the 5% threshold), confirming the alignment pipeline is end-to-end operational. The primary known limitation is that format alignment — rather than safety alignment — is the realistic output of small-scale DPO with self-generated rejected pairs; addressing this requires either a larger base model for coherent harmful generation, or an external unsafe-model rejected source. Every claim in the reports is tied to a specific metric in a specific run file at a specific git hash. Nothing is asserted that cannot be reproduced from the config files and data pipeline scripts in this repository.

---

### Note on comparisons

- **Controlled ("equal terms") results** in this repo come from running baseline vs mgpt2 with the same architecture, same data mixture, and same token budget. These are the primary results.
- Comparisons to external multilingual models (mGPT, BLOOM, Llama, etc.) are **contextual only**; training conditions, data, and scale differ. No such comparisons are made in the body reports.
- HellaSwag comparison to the reference HF GPT-2 (`acc: 0.2955`) is contextual: that model used 40B tokens (3× our budget). Our baseline at 14.4B tokens reaches 0.2768, which is consistent with the training budget difference.
