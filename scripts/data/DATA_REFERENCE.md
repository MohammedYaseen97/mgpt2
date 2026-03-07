# Data directory — formats, splits, and manifests

This document is the canonical reference for every data format decision made
during Phase B.  It covers pretraining, SFT, and DPO in full detail: how the
corpus is split, what the on-disk tensor layout looks like, what the file names
are, and what every manifest.json must contain.

---

## Table of contents

1. [Directory layout](#directory-layout)
2. [Pretraining](#pretraining)
3. [SFT (IndicAlign Instruct)](#sft-indicalign-instruct)
4. [DPO (IndicAlign Toxic)](#dpo-indicalign-toxic)
5. [Cross-cutting rules](#cross-cutting-rules)

---

## Directory layout

```
data/
├── raw/
│   ├── corpus_mixture.txt      # 15 M shuffled documents, one per line
│   └── manifest.json           # build parameters for corpus_mixture.txt
├── eval/
│   ├── eval_latin.txt          # held-out eval — Latin-script docs
│   ├── eval_deva.txt           # held-out eval — Devanagari docs
│   ├── eval_knda.txt           # held-out eval — Kannada-script docs
│   ├── eval_mixed.txt          # held-out eval — mixed-script docs
│   └── manifest.json           # exact line range + bucket counts
├── shards_gpt2/
│   ├── train_000000.npy  …     # pretraining shards, GPT-2 tokenizer
│   ├── val_000000.npy
│   └── manifest.json
├── shards_mgpt2/
│   ├── train_000000.npy  …     # pretraining shards, mgpt2 tokenizer
│   ├── val_000000.npy
│   └── manifest.json
├── sft/
│   ├── train.jsonl             # raw IndicAlign instruct examples (train split)
│   ├── val.jsonl               # raw IndicAlign instruct examples (val split)
│   └── manifest.json
├── shards_sft/
│   ├── train_000000_tokens.npy
│   ├── train_000000_mask.npy
│   ├── val_000000_tokens.npy
│   ├── val_000000_mask.npy
│   └── manifest.json
├── dpo/
│   ├── train.jsonl             # raw IndicAlign toxic examples (train split)
│   ├── val.jsonl               # raw IndicAlign toxic examples (val split)
│   └── manifest.json
└── shards_dpo/
    ├── train_000000_chosen.npy
    ├── train_000000_rejected.npy
    ├── train_000000_prompt_lens.npy
    ├── val_000000_chosen.npy
    ├── val_000000_rejected.npy
    ├── val_000000_prompt_lens.npy
    └── manifest.json
```

---

## Pretraining

### Source and corpus build

`corpus_mixture.txt` is built by `scripts/data/build_corpus_mixture.py`.  It
streams five HF datasets (FineWeb + four Sangraha subsets), interleaves them at
the configured probability weights, writes all documents to disk, then
**globally shuffles the file in-place** using `awk | GNU sort | cut` (seeded,
disk-safe at any file size).  The shuffle runs here — not inside
`tokenize_shards.py` — so `make_lm_eval_sets.py` and `tokenize_shards.py` are
independent consumers of the same already-shuffled file with no ordering
dependency between them.

Actual run parameters (from `data/raw/manifest.json`):

| Parameter | Value |
|---|---|
| lines_written | 15,000,000 |
| seed | 42 |
| fineweb weight | 0.60 |
| sangraha hin_Deva weight | 0.12 |
| sangraha hin_Latn weight | 0.08 |
| sangraha kan_Knda weight | 0.12 |
| sangraha kan_Latn weight | 0.08 |

### Train / val / eval split

The three sets are positional slices of the shuffled `corpus_mixture.txt`.
Because the file is pre-shuffled, any contiguous slice is statistically
representative.

```
corpus_mixture.txt  (15,000,000 lines total)
│
├── lines 0 … 14,849,999          → training + validation   (14,850,000 lines)
│       │
│       ├── first ~98 %           → train shards
│       └── last  ~2  %           → val shard(s)
│
└── lines 14,850,000 … 14,999,999 → EVAL (150,000 lines, never used in training)
```

The exact eval slice is recorded in `data/eval/manifest.json`:

```json
{
  "source_file":     "data/raw/corpus_mixture.txt",
  "eval_line_start": 14850000,
  "eval_line_end":   15000000,
  "eval_line_count": 150000,
  "buckets": {
    "latin": { "file": "data/eval/eval_latin.txt", "line_count": 113882 },
    "deva":  { "file": "data/eval/eval_deva.txt",  "line_count": 16803  },
    "knda":  { "file": "data/eval/eval_knda.txt",  "line_count": 17493  },
    "mixed": { "file": "data/eval/eval_mixed.txt", "line_count": 1822   }
  }
}
```

`tokenize_shards.py` reads `eval_line_start` and `eval_line_count` from this
manifest and skips those lines.  **This is the only coupling point between
`make_lm_eval_sets.py` and `tokenize_shards.py`.**

### Script buckets

Documents are assigned to one of four buckets based on dominant Unicode script
(85 % threshold on non-whitespace characters):

| Bucket | Unicode range | Contents |
|---|---|---|
| `latin` | U+0000–U+007F | English + transliterated Hindi/Kannada |
| `deva`  | U+0900–U+097F | Hindi in Devanagari script |
| `knda`  | U+0C80–U+0CFF | Kannada in Kannada script |
| `mixed` | — | No single script ≥ 85 % |

The same bucketing logic (`tokenizer/scripts/evaluate.py`) is used for
tokenizer evaluation (Phase A), LM perplexity reporting (Phase C), and eval set
creation (Phase B).  All three must use identical bucket boundaries so
phase-over-phase comparisons are measuring the same text populations.

### Shard format — pretraining

Each shard is a **flat 1-D `int32` NumPy array** saved with `np.save()`.

Documents are packed sequentially.  Every document is **prepended** with the
EOT token so the boundary is unambiguous when the array is sliced arbitrarily
at training time:

```
shard contents (conceptual):
[ EOT | doc_0_tok_0 … doc_0_tok_N | EOT | doc_1_tok_0 … doc_1_tok_M | EOT | … ]
```

Toy example (token IDs are illustrative):

```python
# doc 0: "Hello world"  → [15496, 995]
# doc 1: "नमस्ते"       → [30501, 30502]
# EOT token ID for gpt2 = 50256

shard = np.array([50256, 15496, 995,
                  50256, 30501, 30502], dtype=np.int32)
```

| Property | Value |
|---|---|
| dtype | `int32` |
| shape | `(shard_size_tokens,)` — last shard may be shorter |
| shard size | 100,000,000 tokens (100 M) → ~100 shards for 10 B tokens |
| EOT position | prepended to every document |
| padding | none — shards are packed tight |

### File naming — pretraining

```
data/shards_gpt2/
    train_000000.npy
    train_000001.npy
    …
    val_000000.npy       ← one val shard (last ~2 % of non-eval lines)

data/shards_mgpt2/
    train_000000.npy
    …
    val_000000.npy
```

Format: `{split}_{index:06d}.npy`

### Tokenizer identity

| Tokenizer | Load call | Justification |
|---|---|---|
| gpt2 baseline | `tiktoken.get_encoding("gpt2")` | Identical to `tokenizer/scripts/evaluate.py` line 128 and to Karpathy's `fineweb.py` |
| mgpt2 | `RegexTokenizer(); tok.load("tokenizer/artifacts/mgpt2.model")` | Identical to `tokenizer/scripts/evaluate.py` lines 140–142; local artifact, no network |

### Shard manifest — pretraining

`data/shards_gpt2/manifest.json` (and identically `shards_mgpt2/`):

```json
{
  "tokenizer":           "gpt2",
  "tokenizer_artifact":  { "path": "tokenizer/artifacts/mgpt2.model", "sha256": "…" },
  "source_file":         "data/raw/corpus_mixture.txt",
  "eval_line_start":     14850000,
  "eval_line_count":     150000,
  "val_line_count":      "<N — recorded at write time>",
  "shard_size_tokens":   100000000,
  "dtype":               "int32",
  "n_train_shards":      "<N>",
  "n_val_shards":        1,
  "total_tokens":        "<N>",
  "seed":                42
}
```

---

## SFT (IndicAlign Instruct)

### Source

HuggingFace dataset: `ai4bharat/IndicAlign`, instruct split.

`build_sft_data.py` downloads this split, preserves prompt/response boundaries,
and writes:
- `data/sft/train.jsonl` — training examples
- `data/sft/val.jsonl`   — validation examples

Each JSONL line: `{"prompt": "…", "response": "…"}`.

### Train / val split

Deterministic positional cut after shuffling with a fixed seed (no stratified
sampling needed — the dataset is already multilingual).  The exact cut point is
recorded in the manifest.

### Shard format — SFT

Each shard is **two parallel `int32` NumPy arrays** of the same length:

```
{split}_{index:06d}_tokens.npy   — full token sequence
{split}_{index:06d}_mask.npy     — loss mask (0 = ignore, 1 = compute loss)
```

The `tokens` array contains the complete sequence: prompt tokens followed by
response tokens followed by EOT.  The `mask` array is `0` for every prompt
token and `1` for every response token and the final EOT.

Toy example:

```
Prompt:   "Translate to Hindi: Hello"   → token IDs [91, 2604, 311, 39452, 25, 18435]
Response: "नमस्ते"                       → token IDs [30501, 30502]
EOT:                                    → token ID  50256

tokens: [ 91, 2604, 311, 39452, 25, 18435, 30501, 30502, 50256 ]
mask:   [  0,    0,   0,     0,  0,     0,     1,     1,     1 ]
         ←————————— prompt —————————————→ ←— response + EOT ——→
```

During the forward pass the model sees `tokens[:-1]` as input and predicts
`tokens[1:]`; loss is computed only at positions where `mask[1:] == 1`.

| Property | Value |
|---|---|
| dtype | `int32` for both arrays |
| alignment | `tokens[i]` and `mask[i]` always correspond to the same position |
| padding | zero-padded to a fixed sequence length within each shard |
| EOT | appended at end of response; mask = 1 |
| prompt tokens | mask = 0 (loss ignored) |

### File naming — SFT

```
data/shards_sft/
    train_000000_tokens.npy
    train_000000_mask.npy
    train_000001_tokens.npy
    train_000001_mask.npy
    …
    val_000000_tokens.npy
    val_000000_mask.npy
```

Format: `{split}_{index:06d}_{array}.npy`

### SFT manifest

`data/shards_sft/manifest.json`:

```json
{
  "source":              "ai4bharat/IndicAlign (instruct split)",
  "tokenizer":           "mgpt2",
  "tokenizer_artifact":  { "path": "tokenizer/artifacts/mgpt2.model", "sha256": "…" },
  "seed":                42,
  "n_train_examples":    "<N>",
  "n_val_examples":      "<N>",
  "max_seq_len":         "<N>",
  "n_train_shards":      "<N>",
  "n_val_shards":        "<N>",
  "dtype":               "int32"
}
```

---

## DPO (IndicAlign Toxic)

### Source

HuggingFace dataset: `ai4bharat/IndicAlign`, toxic split.

`build_dpo_data.py` downloads this split, verifies chosen/rejected alignment
per prompt, and writes:
- `data/dpo/train.jsonl` — training pairs
- `data/dpo/val.jsonl`   — validation pairs

Each JSONL line: `{"prompt": "…", "chosen": "…", "rejected": "…"}`.

### Train / val split

Deterministic positional cut with a fixed seed, same approach as SFT.
Chosen/rejected alignment is maintained — `train.jsonl[i].chosen` and
`train.jsonl[i].rejected` are always responses to the same prompt.

### Shard format — DPO

Each shard is **three parallel `int32` NumPy arrays** where index `i` across
all three arrays always refers to the same prompt:

```
{split}_{index:06d}_chosen.npy       — prompt + chosen response + EOT
{split}_{index:06d}_rejected.npy     — prompt + rejected response + EOT
{split}_{index:06d}_prompt_lens.npy  — length of the prompt portion (scalar per example)
```

`prompt_lens[i]` tells the model exactly where the prompt ends in both
`chosen[i]` and `rejected[i]`, which is needed to split the log-probability
into prompt and response parts during the DPO loss calculation.

Toy example:

```
Prompt:   "Is this toxic?"             → [91, 318, 428, 11,  30945, 30, EOT]  (len = 7)
Chosen:   "No, it is respectful."      → [2949, 11, 340, 318, 46512, 13, EOT]
Rejected: "Yes, kill them all."        → [3363, 11, 1494, 606, 477, 13, EOT]

chosen   = [ 91, 318, 428, 11, 30945, 30,  2949, 11, 340, 318, 46512, 13, 50256 ]
rejected = [ 91, 318, 428, 11, 30945, 30,  3363, 11, 1494, 606, 477,  13, 50256 ]
            ←————————— prompt (len=6) ————————→ ←——— response + EOT ———————————→

prompt_lens = [ 6 ]    ← same value applies to both chosen and rejected for this example
```

Note: the prompt is **not** followed by EOT in the sequence — it flows directly
into the response.  EOT is appended only at the end of the response.  The
`prompt_lens` value is the number of prompt tokens before the response starts.

| Property | Value |
|---|---|
| dtype | `int32` for all three arrays |
| alignment | `chosen[i]`, `rejected[i]`, `prompt_lens[i]` always the same prompt |
| EOT | appended at end of each response |
| padding | zero-padded to max length within shard; pad tokens excluded from loss |
| prompt in sequence | prompt tokens are identical prefix in chosen and rejected |

### File naming — DPO

```
data/shards_dpo/
    train_000000_chosen.npy
    train_000000_rejected.npy
    train_000000_prompt_lens.npy
    train_000001_chosen.npy
    …
    val_000000_chosen.npy
    val_000000_rejected.npy
    val_000000_prompt_lens.npy
```

Format: `{split}_{index:06d}_{array}.npy`

### DPO manifest

`data/shards_dpo/manifest.json`:

```json
{
  "source":              "ai4bharat/IndicAlign (toxic split)",
  "tokenizer":           "mgpt2",
  "tokenizer_artifact":  { "path": "tokenizer/artifacts/mgpt2.model", "sha256": "…" },
  "seed":                42,
  "n_train_pairs":       "<N>",
  "n_val_pairs":         "<N>",
  "max_seq_len":         "<N>",
  "n_train_shards":      "<N>",
  "n_val_shards":        "<N>",
  "dtype":               "int32"
}
```

---

## Cross-cutting rules

1. **dtype is always `int32`** across all three pipelines.  Both tokenizers
   produce IDs in `[0, 50256]` (50,257 valid values; max ID = 50,256), so
   `uint16` (max 65,535) would technically hold them.  The 50,304 figure is
   the model embedding matrix dimension padded for GPU tensor-core alignment —
   it is a model architecture constant, not a tokenizer vocab size, and the
   tokenizer never emits an ID ≥ 50,257.  `int32` is used because it is the
   dtype PyTorch embedding layers natively expect, it allows `-1` as an
   explicit pad/ignore sentinel, and it is future-proof if the vocab ever
   expands beyond 65,535.

2. **Every shard directory must have a `manifest.json`** before any training
   script reads from it.  Training scripts should assert the manifest exists and
   read `tokenizer_artifact.sha256` to confirm they are loading the correct
   weights.

3. **Eval sets are never tokenized**.  `data/eval/*.txt` are raw text files read
   directly by `eval/lm_eval.py` at evaluation time.  This keeps them reusable
   regardless of which tokenizer is under test.

4. **Seeds are fixed at 42** everywhere unless explicitly overridden.  Any
   deviation must be recorded in the relevant manifest.

5. **Shard arrays within a pipeline must be aligned**.  For SFT, `tokens[i]`
   and `mask[i]` must be the same position.  For DPO, `chosen[i]`,
   `rejected[i]`, and `prompt_lens[i]` must be the same example.  A
   misalignment here is silent and will corrupt training.
