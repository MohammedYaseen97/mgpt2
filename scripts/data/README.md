## Data scripts (TODO)

You will implement these scripts as part of the assignment:

- `build_corpus_mixture.py`
  - downloads/loads FineWeb + Sangraha subsets (and optionally translit)
  - creates deterministic train/val splits
  - writes line-based corpora under `data/raw/`

- `tokenize_shards.py`
  - takes `data/raw/*.txt` and a tokenizer choice (baseline vs mgpt2)
  - writes `data/shards_*/*.npy` shards

- `make_lm_eval_sets.py`
  - creates held-out text files and bucket splits under `data/eval/`

Keep outputs reproducible:
- fixed seeds
- write a `manifest.json` with dataset IDs and sampling parameters

