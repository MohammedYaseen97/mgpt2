## Data scripts (Phase B — all phases)

All raw corpus building and shard tokenization lives here.
Downstream training phases (C, D, E) consume outputs from this directory and add nothing to it.

---

### Pretraining corpus

- `build_corpus_mixture.py`
  - reads FineWeb + Sangraha subsets (including translit); writes line-based corpus under `data/raw/` with deterministic sampling

- `tokenize_shards.py`
  - reads `data/raw/*.txt` and a tokenizer choice (baseline gpt2 or mgpt2); writes `data/shards_gpt2/*.npy` and `data/shards_mgpt2/*.npy` (int32)

- `make_lm_eval_sets.py`
  - writes held-out text files and script/bucket splits under `data/eval/`

### SFT corpus

- `build_sft_data.py`
  - reads IndicAlign instruct split; writes train/val splits with prompt boundaries preserved under `data/sft/`

- `tokenize_sft_shards.py`
  - reads `data/sft/`; writes shards under `data/shards_sft/`; prompt/response boundary must be encoded per example

### DPO corpus

- `build_dpo_data.py`
  - reads IndicAlign toxic split; writes aligned chosen/rejected pair splits under `data/dpo/`

- `tokenize_dpo_shards.py`
  - reads `data/dpo/`; writes shards under `data/shards_dpo/`; chosen/rejected alignment must be maintained across shards

---

### Reproducibility (required for every script)

- fixed seeds; all sampling parameters passed explicitly, never hardcoded
- write a `manifest.json` alongside every output directory: dataset IDs, splits, seeds, sampling weights, git commit hash
