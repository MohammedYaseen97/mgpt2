## Data scripts (TODO)

You will implement these scripts as part of the assignment:

### Pretraining corpus (Phase B)

- `build_corpus_mixture.py`
  - downloads/loads FineWeb + Sangraha subsets (including translit)
  - creates deterministic train/val splits
  - writes line-based corpora under `data/raw/`

- `tokenize_shards.py`
  - takes `data/raw/*.txt` and a tokenizer choice (baseline vs mgpt2)
  - writes `data/shards_*/*.npy` shards

- `make_lm_eval_sets.py`
  - creates held-out text files and bucket splits under `data/eval/`

### SFT data (Phase D prerequisite)

- `build_sft_data.py`
  - downloads IndicAlign instruct split; produces train/val splits with prompt boundaries preserved

- `tokenize_sft_shards.py`
  - tokenizes into shards; prompt/response boundary must be encoded per example

### DPO data (Phase E prerequisite)

- `build_dpo_data.py`
  - downloads IndicAlign toxic split; produces aligned chosen/rejected pair splits

- `tokenize_dpo_shards.py`
  - tokenizes chosen/rejected pairs; alignment must be maintained across shards

Keep outputs reproducible:
- fixed seeds
- write a `manifest.json` with dataset IDs and sampling parameters

