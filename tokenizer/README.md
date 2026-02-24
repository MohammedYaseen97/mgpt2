## Tokenizer (mgpt2)

This folder contains the **pure-Python** tokenizer implementations and the utilities used to:
- build multilingual corpora for tokenizer training (English/Hindi/Kannada + transliteration)
- iterate on regex split patterns
- train a BPE tokenizer and save a reusable tokenizer artifact (`.model` + `.vocab`)
- evaluate tokenization efficiency vs baselines (e.g. GPT-4 `cl100k_base`)

### Key modules
- `base.py`: BPE primitives + `Tokenizer.save()`/`Tokenizer.load()`
- `basic.py`: `BasicTokenizer` (byte-level BPE)
- `regex_tokenizer.py`: `RegexTokenizer` (regex chunking + byte-level BPE merges)
- `gpt4.py`: `GPT4Tokenizer` (template/reference: matches `tiktoken` `cl100k_base`)
- `patterns.py`: shared regex patterns (`GPT4_SPLIT_PATTERN`, `INDIC_SPLIT_PATTERN`)

### Training (recommended: run from repo root)
Train an Indic-focused tokenizer over `tokenizer/tok_corpus.txt` and save artifacts into `tokenizer/artifacts/`:

```bash
./virtual/bin/python -m tokenizer.train_tokenizer --num_merges 50000
```

### Publish to Hugging Face (tokenizer-only)
1) Make a held-out eval set (sampled from your training corpus):

```bash
./virtual/bin/python -m tokenizer.scripts.make_heldout \
  --corpus tokenizer/tok_corpus_large.txt \
  --out tokenizer/artifacts/heldout_eval.txt \
  --n 10000 --seed 1337
```

2) Publish the trained tokenizer + evaluation metrics:

```bash
export HF_TOKEN="your_hf_token"

./virtual/bin/python -m tokenizer.scripts.publish_hf \
  --repo_id "YOUR_USERNAME/mgpt2-tokenizer" \
  --model tokenizer/artifacts/mgpt2.model \
  --eval_text tokenizer/artifacts/heldout_eval.txt \
  --eval_limit 10000
```

Notes:
- Set `HF_TOKEN` in your environment (write access to the target repo).
- The published repo includes the python implementation under `tokenizer/` and is meant to be loaded with `trust_remote_code=True`.

### Notes
- The training scripts assume large corpora; for quick experiments, use a smaller `vocab_size` and a smaller corpus.
- Notebooks in this folder should import from `tokenizer.*` without `sys.path` hacks.

