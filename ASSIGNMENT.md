## mgpt2 — research assignment scaffold

This repo is structured as a phased research assignment:
- you **own** the core work (TODOs)
- you don’t get derailed (guardrails, reproducibility, checks)
- you end with a publishable story (metrics + fair comparisons + HF release)

### Already done (do NOT redo)
- **Model implementation**: `model.py` (Karpathy-style GPT‑2)
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
- same token-level compute budget (tokens processed), not “epochs”
- same held-out eval sets (exact text lines)

**Note on “equal terms” comparisons**
- **Controlled (equal terms)**: comparisons you run *inside this repo* where you hold architecture, data mixture, and token-budget constant and change only the tokenizer (baseline GPT‑2 tokens vs mgpt2 tokens).
- **Contextual (not equal terms)**: comparisons to external multilingual models (different vocab sizes, corpora, compute, objectives). You may report these for context, but you must label them explicitly as non-controlled.

### Metrics (required)
You must provide:
- **tokenizer-only**: tokens/1k bytes + p95 tokens/line (bucketed)
- **LM pretrain**: perplexity overall + perplexity bucketed (latin/deva/knda/mixed)
- **SFT**: held-out SFT loss + fixed prompt suite (saved generations)
- **DPO**: preference win-rate/accuracy on held-out pairs + regression checks

---

## Phase A — Tokenizer (mgpt2) final

### Goal
Train a tokenizer with:
- **GPT-2-exact tokenizer terms**: 256 bytes + 50,000 merges + 1 `<|endoftext|>` = 50,257 IDs
- model padded vocab = 50304 (apples-to-apples with baseline GPT‑2)

### Deliverables
- `tokenizer/artifacts/mgpt2.model` and `tokenizer/artifacts/mgpt2.vocab`
- `tokenizer/artifacts/heldout_eval.txt` (fixed seed sample from corpus mixture)
- `tokenizer/artifacts/tokenizer_eval.json` (from evaluator)

### Check
Run:
```bash
./virtual/bin/python -m tokenizer.scripts.evaluate \
  --text tokenizer/artifacts/heldout_eval.txt \
  --limit 10000 \
  --model tokenizer/artifacts/mgpt2.model > tokenizer/artifacts/tokenizer_eval.json
```

Leak-free protocol (recommended):
1) Create held-out first
2) Train tokenizer while excluding held-out lines:

```bash
./virtual/bin/python -m tokenizer.train_tokenizer \
  --corpus tokenizer/tok_corpus_large.txt \
  --exclude_lines_file tokenizer/artifacts/heldout_eval.txt \
  --num_merges 50000 \
  --out_prefix tokenizer/artifacts/mgpt2
```

Pass criteria (you must justify):
- mgpt2 improves tokens/1k-bytes on devanagari/kannada/mixed buckets vs baseline tokenizer

---

## Phase B — Data mixture + sharding (baseline + mgpt2)

### Goal
Create a single underlying multilingual+translit corpus mixture, then tokenize it into:
- baseline GPT‑2 tokens (tiktoken gpt2)
- mgpt2 tokens (your tokenizer)

### Your tasks (TODO)
- [TODO] implement `scripts/data/build_corpus_mixture.py`
  - outputs a line-based corpus file (and/or doc-id mapping) with deterministic sampling
- [TODO] implement `scripts/data/tokenize_shards.py`
  - write `data/shards_gpt2/*.npy` and `data/shards_mgpt2/*.npy` (int32)
- [TODO] implement `scripts/data/make_lm_eval_sets.py`
  - heldout text sets and bucket splits

### Checks
- [TODO] implement `scripts/checks/check_shards.py`
  - verifies dtype, token ranges, shard sizes, document parity across tokenizers

---

## Phase C — Pretraining (baseline vs mgpt2)

### Goal
Train two models fairly:
- baseline: GPT‑2 tokenizer shards
- mgpt2: mgpt2 tokenizer shards

### Your tasks (TODO)
- [TODO] implement `scripts/run_pretrain.py`
  - uses `train.py` as the baseline engine, but parameterized via config
- [TODO] implement `eval/lm_eval.py`
  - perplexity overall + bucketed perplexity on fixed heldout sets
- [TODO] write `reports/pretrain_report.md`

### Checks
- [TODO] `scripts/checks/check_pretrain_run.py` verifies:
  - token budget equality
  - comparable hyperparams
  - metrics exist and are bucketed

---

## Phase D — SFT (IndicAlign Instruct)

### Data preparation (prerequisite)
- [TODO] implement `scripts/data/build_sft_data.py`
  - downloads IndicAlign instruct split; produces train/val splits with prompt boundaries preserved
- [TODO] implement `scripts/data/tokenize_sft_shards.py`
  - tokenizes into shards; prompt/response boundary must be encoded per example (required for selective loss masking during training)

### Your tasks (TODO)
- [TODO] implement `scripts/run_sft.py`
- [TODO] implement `eval/sft_eval.py` (heldout loss + fixed prompt suite)
- [TODO] write `reports/sft_report.md`

---

## Phase E — DPO (alignment)

### Data preparation (prerequisite)
- [TODO] implement `scripts/data/build_dpo_data.py`
  - downloads IndicAlign toxic split; produces aligned chosen/rejected pair splits with train/val held-out
- [TODO] implement `scripts/data/tokenize_dpo_shards.py`
  - tokenizes chosen/rejected pairs; chosen/rejected alignment must be maintained across shards

### Your tasks (TODO)
- [TODO] implement `scripts/run_dpo.py`
- [TODO] implement `eval/dpo_eval.py` (win-rate + regression checks)
- [TODO] write `reports/dpo_report.md`

---

## Phase F — Publish to HF

### Tokenizer (already scaffolded)
- `tokenizer/scripts/publish_hf.py` can publish trained + evaluated tokenizer.

### Model (you will implement)
- [TODO] implement `scripts/publish_model_hf.py`
  - either a minimal Transformers wrapper (`trust_remote_code=True`) or a documented state_dict release

---

## What to do next (recommended order)
1) Finish Phase A (final mgpt2 tokenizer + tokenizer_eval.json)
2) Implement Phase B sharding (baseline + mgpt2) with deterministic mixture
3) Implement Phase C eval (bucketed perplexity) and run a small-scale baseline comparison

