## Data scripts (Phase B — all phases)

All raw corpus building and shard tokenization lives here.
Downstream training phases (C, D, E) consume outputs from this directory and add nothing to it.

---

### Pretraining corpus

- `build_corpus_mixture.py` ✓
  - streams FineWeb + Sangraha subsets (including translit variants); writes a **globally shuffled** line-based corpus to `data/raw/corpus_mixture.txt`
  - the global shuffle (awk | GNU-sort | cut, seeded) runs as the final step of this script — not inside `tokenize_shards.py` — so that `make_lm_eval_sets.py` and `tokenize_shards.py` are independent consumers of the same file with no ordering dependency between them

- `make_lm_eval_sets.py`
  - cuts a **fixed positional slice** from the top of `data/raw/corpus_mixture.txt` (e.g. first 50K lines) as the held-out eval set
  - writes held-out text files and bucket splits under `data/eval/`
  - must record the exact line range (start, end) in its manifest so `tokenize_shards.py` can skip those lines

- `tokenize_shards.py`
  - reads `data/raw/corpus_mixture.txt` **starting after the eval slice** (line offset from `make_lm_eval_sets.py` manifest)
  - takes `--tokenizer [gpt2|mgpt2]`; writes `data/shards_gpt2/*.npy` and `data/shards_mgpt2/*.npy` (int32)

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
