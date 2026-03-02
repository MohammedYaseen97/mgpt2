## mgpt2 — research assignment scaffold

This repo is structured as a phased research assignment:
- you **own** the core work (TODOs)
- you don't get derailed (guardrails, reproducibility, checks)
- you end with a publishable story (metrics + fair comparisons + HF release)

### Already done (do NOT redo)
- **Model implementation**: `model.py` (Karpathy-style GPT-2)
- **Baseline training loop**: `train.py` (Karpathy walkthrough follow-through)
  - A frozen reference copy also exists at `baselines/karpathy_train_reference.py`

---

## Global rules (grading rubric)

### Reproducibility (required)
Every phase must produce a run folder containing:
- **config** file copied verbatim
- **git commit hash**
- RNG seeds
- exact dataset mixture identifiers / sampling seeds
- tokenizer artifact identity (`mgpt2.model` path + hash)
- metrics JSON outputs

### Fair comparisons (required)
When comparing baseline vs mgpt2:
- same model architecture
- same optimizer + schedule
- same token-level compute budget (tokens processed), not "epochs"
- same held-out eval sets (exact text lines)

**Note on "equal terms" comparisons**
- **Controlled (equal terms)**: comparisons you run *inside this repo* where you hold architecture, data mixture, and token-budget constant and change only the tokenizer (baseline GPT-2 tokens vs mgpt2 tokens).
- **Contextual (not equal terms)**: comparisons to external models (different training data, corpora, compute, objectives). You may report these for context, but you must label them explicitly as non-controlled.

**Per-phase comparison map**

| Phase | Controlled baseline | Contextual reference (optional) |
|---|---|---|
| A — Tokenizer | — | tiktoken_cl100k (primary), tiktoken_gpt2 (floor) |
| C — Pretrain | Your GPT-2-tokenized model (same corpus, same budget) | HF GPT-2 model (different data) — label explicitly |
| D — SFT | Your Phase C pretrained mgpt2 model (untuned) | — |
| E — DPO | Your Phase D SFT model (pre-DPO) | — |

### Metrics (required)
You must provide:
- **tokenizer-only**: tokens/1k bytes + p95 tokens/line (bucketed by latin/deva/knda/mixed)
- **LM pretrain**: perplexity overall + bucketed (latin/deva/knda/mixed); HellaSwag score for both controlled models + vs HF GPT-2 (contextual, labelled)
- **SFT**: held-out SFT loss + fixed prompt suite (saved generations)
- **DPO**: preference win-rate/accuracy on held-out pairs + regression check (SFT loss must not regress)

---

## Phase A — Tokenizer (mgpt2) final ✓ COMPLETE

### Goal
Train a tokenizer with:
- **GPT-2-exact tokenizer terms**: 256 bytes + 50,000 merges + 1 `<|endoftext|>` = 50,257 IDs
- model padded vocab = 50304 (apples-to-apples with baseline GPT-2)

### Deliverables
- `tokenizer/artifacts/mgpt2.model` and `tokenizer/artifacts/mgpt2.vocab`
- `tokenizer/artifacts/heldout_eval.txt` (fixed seed sample from corpus mixture)
- `tokenizer/artifacts/tokenizer_eval.json` (from evaluator)
- `reports/00_tokenizer_report.md`

### Pass criteria
- mgpt2 improves tokens/1k-bytes on devanagari/kannada/mixed buckets vs tiktoken_cl100k (primary baseline)
- tiktoken_gpt2 included as contextual floor; Latin regression vs cl100k must be justified

---

## Phase B — Data preparation (all phases)

### Goal
Build all raw corpora and tokenized shards needed across pretraining, SFT, and DPO.
All data work lives here; downstream training phases consume these outputs and add nothing to `scripts/data/`.

### Pretraining corpus
- [TODO] implement `scripts/data/build_corpus_mixture.py`
  - outputs a line-based corpus file (and/or doc-id mapping) with deterministic sampling
- [TODO] implement `scripts/data/tokenize_shards.py`
  - write `data/shards_gpt2/*.npy` and `data/shards_mgpt2/*.npy` (int32)
- [TODO] implement `scripts/data/make_lm_eval_sets.py`
  - heldout text sets and bucket splits

### SFT corpus
- [TODO] implement `scripts/data/build_sft_data.py`
  - downloads IndicAlign instruct split; produces train/val splits with prompt boundaries preserved
- [TODO] implement `scripts/data/tokenize_sft_shards.py`
  - tokenizes into shards; prompt/response boundary must be encoded per example (required for selective loss masking during training)

### DPO corpus
- [TODO] implement `scripts/data/build_dpo_data.py`
  - downloads IndicAlign toxic split; produces aligned chosen/rejected pair splits with train/val held-out
- [TODO] implement `scripts/data/tokenize_dpo_shards.py`
  - tokenizes chosen/rejected pairs; chosen/rejected alignment must be maintained across shards

### Checks
- [TODO] implement `scripts/checks/check_shards.py`
  - verifies dtype, token ranges, shard sizes, document parity across tokenizers

---

## Phase C — Pretraining (baseline vs mgpt2)

### Goal
Train two models under controlled conditions:
- **baseline**: GPT-2 tokenizer shards (your own model, same corpus, same budget)
- **mgpt2**: mgpt2 tokenizer shards

The HF GPT-2 model is a contextual reference only (different training data); it must be labelled as non-controlled if reported.

### Your tasks (TODO)
- [TODO] implement `scripts/run_pretrain.py`
  - uses `train.py` as the baseline engine, parameterized via config; `--tokenizer [gpt2|mgpt2]`
- [TODO] implement `eval/lm_eval.py`
  - perplexity overall + bucketed perplexity on fixed heldout sets
- [TODO] implement `eval/hellaswag_eval.py`
  - HellaSwag score for both models; `hellaswag.py` is already scaffolded
- [TODO] write `reports/01_pretrain_report.md`

### Checks
- [TODO] `scripts/checks/check_pretrain_run.py` verifies:
  - token budget equality across both runs
  - comparable hyperparams
  - metrics exist and are bucketed
  - HellaSwag scores present for both models

---

## Phase D — SFT (IndicAlign Instruct)

### Your tasks (TODO)
- [TODO] implement `scripts/run_sft.py`
- [TODO] implement `eval/sft_eval.py` (heldout loss + fixed prompt suite)
- [TODO] write `reports/02_sft_report.md`

---

## Phase E — DPO (alignment)

### Your tasks (TODO)
- [TODO] implement `scripts/run_dpo.py`
- [TODO] implement `eval/dpo_eval.py` (win-rate + regression checks)
- [TODO] write `reports/03_dpo_report.md`

---

## Phase F — Publish to HF

### Tokenizer (already scaffolded)
- `tokenizer/scripts/publish_hf.py` can publish trained + evaluated tokenizer.

### Model (you will implement)
- [TODO] implement `scripts/publish_model_hf.py`
  - either a minimal Transformers wrapper (`trust_remote_code=True`) or a documented state_dict release

---

## What to do next (recommended order)
1) ~~Finish Phase A (final mgpt2 tokenizer + tokenizer_eval.json)~~ ✓ done
2) Implement Phase B data pipeline — pretraining corpus (build + tokenize shards)
3) Implement Phase C eval (bucketed perplexity + HellaSwag) and run a small-scale baseline comparison
