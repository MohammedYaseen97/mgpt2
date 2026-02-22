## Tokenizer training speedup (what changed)

This repo originally used a **naive BPE training loop** in `RegexTokenizer.train()`, which was too slow on large corpora.
We fixed it while keeping everything **pure Python** and reasonably readable.

### The original bottleneck
The first implementation effectively did this for each merge:

- scan the entire corpus (all chunks) to recompute pair stats
- scan the entire corpus again to apply the merge

With ~50k merges at GPT‑2 scale, the runtime was roughly:

\[
O(\text{num\_merges} \times \text{total\_byte\_tokens})
\]

On `tok_corpus_large.txt` this is not practical.

---

## Fix 1: incremental BPE trainer (algorithmic)

`tokenizer/regex_tokenizer.py` → `RegexTokenizer.train()` was rewritten into a more classic, efficient BPE approach:

- **Chunk frequency counting**: treat regex chunks as “words” with counts, so repeated chunks don’t cost repeated work.
  - `chunk_counts: Counter[bytes]`
  - `words: dict[tuple[int, ...], freq]`

- Maintain global state incrementally:
  - `pair_counts[pair]` = weighted (frequency-aware) global pair count
  - `pair_to_words[pair]` = set of words that contain the pair

- Select the next merge efficiently:
  - max-heap of `(count, pair)` with lazy stale-entry cleanup (no full rescans)

- Apply merges locally:
  - only touch `affected = pair_to_words[best_pair]`
  - remove old word’s pair contributions, add new word’s contributions

This removes the “scan everything twice per merge” behavior.

---

## Fix 2: practical “speed knobs” to reduce training state

Large web corpora create tons of unique chunk types. Even with an incremental trainer, this can dominate memory/time.
We added simple approximations that are common in practice:

### In `RegexTokenizer.train(...)`
- **`min_chunk_freq`**: ignore regex chunk types that occur fewer than N times
- **`max_chunks`**: keep only the top‑N most frequent chunk types

These reduce:
- number of unique “words”
- size of `pair_to_words` / `pair_counts`
- heap churn + memory pressure

### In `tokenizer/train_tokenizer.py`
We added corpus limiting/sampling so you can iterate fast:

- **`--max_lines` / `--max_chars`**: quick “first N” truncation (fast but can be biased)
- **`--sample_lines N --seed S`**: reservoir-sample N lines from the whole corpus (representative + fast)

Also exposes:
- `--min_chunk_freq`
- `--max_chunks`

---

## Why GPU is not the right tool here

This training loop is dominated by **Python dict/set/heap operations** over variable-length sequences
(irregular, sparse, branchy). GPUs accelerate dense numeric kernels; a real GPU BPE trainer would require
a major redesign (tensorized histograms + custom kernels + transfer overhead), which is not a “simple readable” change.

---

## Recommended commands

### Fast dev run (representative, finishes in minutes)

```bash
./virtual/bin/python -m tokenizer.train_tokenizer \
  --corpus tokenizer/tok_corpus_large.txt \
  --sample_lines 20000 --seed 1337 \
  --vocab_size 8000 \
  --out_prefix tokenizer/artifacts/mgpt2_dev \
  --min_chunk_freq 2 \
  --max_chunks 50000
```

### Smoke test (tiny)

```bash
./virtual/bin/python -m tokenizer.train_tokenizer \
  --corpus tokenizer/tok_corpus.txt \
  --max_lines 200 \
  --vocab_size 300 \
  --out_prefix tokenizer/artifacts/smoke
```

### Final run (quality-first)

- increase `--sample_lines` a lot (or remove it and accept the runtime)
- set `--min_chunk_freq 1`
- set `--max_chunks` to a large number or remove it
- set `--vocab_size 50257` (includes the 5 default special tokens)

