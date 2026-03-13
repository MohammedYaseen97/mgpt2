## Data scripts (Phase B — all phases)

All raw corpus building and shard tokenization lives here.
Downstream training phases (C, D, E) consume outputs from this directory and add nothing to it.

---

### Pretraining corpus ✓ COMPLETE

- `build_corpus_mixture.py` ✓
  - streams FineWeb + Sangraha subsets (including translit variants); writes a **globally shuffled** line-based corpus to `data/raw/corpus_mixture.txt` (15M docs)
  - the global shuffle (awk | GNU-sort | cut, seeded) runs as the final step — not inside `tokenize_shards.py` — so that `make_lm_eval_sets.py` and `tokenize_shards.py` are independent consumers with no ordering dependency between them

- `make_lm_eval_sets.py` ✓
  - cuts lines 14,850,000–15,000,000 (150K docs) as the held-out eval set; writes per-bucket text files under `data/eval/`
  - actual bucket counts: latin 113,882 / deva 16,803 / knda 17,493 / mixed 1,822
  - large `--eval-start` offsets are skipped at OS level via `tail -n +N` (no Python heap cost)

- `tokenize_shards.py` ✓
  - reads manifest offsets from `data/eval/manifest.json` to skip the eval slice; splits remainder ~98/2 train/val
  - **gpt2** (`tiktoken.get_encoding("gpt2")`): 313 train + 7 val shards, **31.86B tokens** → `data/shards_gpt2/`
  - **mgpt2** (`RegexTokenizer.load("tokenizer/artifacts/mgpt2.model")`): 142 train + 3 val shards, **14.44B tokens** → `data/shards_mgpt2/`
  - mgpt2 encodes the same corpus in 2.2× fewer tokens — direct evidence of Phase A tokenizer efficiency on the actual training data
  - multiprocessing (`Pool.imap`, 14 workers) used for throughput; output is bit-for-bit identical to single-threaded

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

### HuggingFace dataset releases

Every corpus is published to a private HF dataset repo as a first-class Phase B deliverable.
This serves two purposes: (1) each dataset is pull-ready on any cloud training machine via
`datasets.load_dataset(...)`, and (2) it documents the dataset publicly as part of the project.
No data files need to be manually transferred — `git clone` + script re-run is always sufficient.

- `publish_pretraining_dataset.py` → `ace-1/mgpt2-pretrain-corpus`
- `publish_sft_dataset.py`         → `ace-1/mgpt2-sft-data`
- `publish_dpo_dataset.py`         → `ace-1/mgpt2-dpo-data`

Each script attaches a dataset card with sources, mixture weights, shuffle seed, eval slice, and
the exact CLI command to reproduce the corpus locally from scratch.

---

### Reproducibility (required for every script)

- fixed seeds; all sampling parameters passed explicitly, never hardcoded
- write a `manifest.json` alongside every output directory: dataset IDs, splits, seeds, sampling weights, git commit hash
- **nothing here is irreplaceable** — every artifact is derivable from `git clone` + re-run
