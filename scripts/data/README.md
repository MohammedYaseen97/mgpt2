## Data scripts (Phase B — all phases)

All raw corpus building and shard tokenization lives here.
Downstream training phases (C, D, E) consume outputs from this directory and add nothing to it.

---

### Pretraining corpus ✓ COMPLETE

- `build_corpus_mixture.py` ✓
  - streams FineWeb + Sangraha subsets; writes a **globally shuffled** line-based corpus to `data/raw/corpus_mixture.txt`
  - native-script splits use Sangraha `verified/` (scraped websites + OCR + transcriptions); Latin splits use `synthetic/` (no verified alternative)
  - weights: 55% FineWeb / 18% `verified/hin` / 7% `synthetic/hin_Latn` / 13% `verified/kan` / 7% `synthetic/kan_Latn`
  - global shuffle (awk | GNU-sort | cut, seeded) runs as the final step so `make_lm_eval_sets.py` and `tokenize_shards.py` are independent consumers

- `make_lm_eval_sets.py` ✓
  - cuts lines 14,850,000–15,000,000 (150K docs) as the held-out eval set; writes per-bucket text files under `data/eval/`
  - actual bucket counts: latin 113,882 / deva 16,803 / knda 17,493 / mixed 1,822
  - large `--eval-start` offsets are skipped at OS level via `tail -n +N` (no Python heap cost)

- `tokenize_shards.py` ✓
  - reads manifest offsets from `data/eval/manifest.json` to skip the eval slice; splits remainder ~98/2 train/val
  - **gpt2** (`tiktoken.get_encoding("gpt2")`): 100M-token int32 shards → `data/shards_gpt2/`
  - **mgpt2** (`RegexTokenizer.load("tokenizer/artifacts/mgpt2.model")`): 100M-token int32 shards → `data/shards_mgpt2/`
  - multiprocessing (`Pool.imap`, 14 workers) used for throughput; output is bit-for-bit identical to single-threaded

### SFT corpus ✓ COMPLETE

- `build_sft_data.py` ✓
  - streams `ai4bharat/indic-align` configs `Dolly_T`, `OpenAssistant_T`, and `Anudesh` from HF Hub
  - language distribution mirrors pretraining weights: 55% `eng_Latn` / 18% `hin_Deva` / 7% `hin_Latn` / 13% `kan_Knda` / 7% `kan_Latn`
  - **disjoint row partitioning**: each source row is assigned to exactly one language variant; translated copies of the same row are never used more than once across variants
  - swap-correction heuristic detects and fixes prompt/response inversions in Latin-script columns (Dolly_T) using the `eng_Latn` column as length-ratio reference
  - global shuffle (seed=42) + 90/10 positional train/val cut → `data/sft/train.jsonl`, `data/sft/val.jsonl`, `data/sft/manifest.json`
  - output fields per example: `{"prompt": "…", "response": "…", "lang": "hin_Deva"}`

- `tokenize_sft_shards.py` ✓
  - **mgpt2 only** — gpt2-tokenized model is retired after Phase C
  - reads pre-split `data/sft/train.jsonl` and `data/sft/val.jsonl`; no eval-slice logic needed
  - each example padded to `seq_len=1024`; response trimmed from right if over budget; example skipped if prompt alone exceeds budget
  - padding token: EOT = 50256 (token 0 is a real vocab token, must not be used)
  - output per shard: paired `{split}_{idx:06d}_tokens.npy` + `{split}_{idx:06d}_mask.npy`, both shape `(N, 1024)` int32
  - mask: 0 for prompt tokens, 1 for response tokens + EOT, 0 for padding
  - default 1000 examples/shard → `data/shards_sft/`

### DPO corpus

- `build_dpo_data.py`
  - reads IndicAlign toxic split; writes aligned chosen/rejected pair splits under `data/dpo/`

- `tokenize_dpo_shards.py`
  - reads `data/dpo/`; writes shards under `data/shards_dpo/`; chosen/rejected alignment must be maintained across shards

---

### Reproducibility (required for every script)

- fixed seeds; all sampling parameters passed explicitly, never hardcoded
- write a `manifest.json` alongside every output directory: dataset IDs, splits, seeds, sampling weights, git commit hash
- **nothing here is irreplaceable** — every artifact is derivable from `git clone` + re-run

---

### Cloud bootstrap

The only non-reproducible artifact is the trained tokenizer model (`tokenizer/artifacts/mgpt2.model`).
It is fetched from the published HF tokenizer repo using:

```bash
python scripts/download_tokenizer_artifacts.py --repo_id ace-1/mgpt2-tokenizer
```

This maps HF filenames → local `tokenizer/artifacts/` names and is idempotent.
After it runs, all data scripts above can be executed as normal.
