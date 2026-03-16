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
| fineweb (`sample-10BT`) weight | 0.55 |
| sangraha `verified/hin` weight | 0.18 |
| sangraha `synthetic/hin_Latn` weight | 0.07 |
| sangraha `verified/kan` weight | 0.13 |
| sangraha `synthetic/kan_Latn` weight | 0.07 |

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

### Source and composition

HuggingFace dataset: `ai4bharat/indic-align` (note lowercase, hyphenated).

**Target: 30,000 examples total, language distribution mirrors pretraining weights.**

| Language | Count | Ratio | Source config(s) |
|---|---|---|---|
| `eng_Latn` | 16,500 | 55% | Anudesh (crowd-sourced native English) |
| `hin_Deva` | 5,400 | 18% | Dolly_T + OpenAssistant_T pool, partition A |
| `kan_Knda` | 3,900 | 13% | Dolly_T + OpenAssistant_T pool, partition B |
| `hin_Latn` | 2,100 | 7% | Dolly_T + OpenAssistant_T pool, partition C |
| `kan_Latn` | 2,100 | 7% | Dolly_T + OpenAssistant_T pool, partition D |

**Disjoint row partitioning:** Dolly_T (15K rows) and OpenAssistant_T (19.9K rows)
are pooled (34.9K rows), shuffled with a fixed seed, then sliced into 4 non-overlapping
partitions.  A given source row contributes exactly ONE language column to the final
dataset — the same English content never appears in multiple scripts.

**Multi-turn:** only the first turn of each conversation is used.  Subsequent turns
are context-dependent and unsuitable for single-turn SFT at this model scale.

**Known quality note:** ~10% of Dolly_T Latin-script rows have prompt/response
swapped (dataset-level issue).  The script detects and corrects these by aligning
against the `eng_Latn` column of the same row before partitioning.

`build_sft_data.py` writes:
- `data/sft/train.jsonl` — 90% of examples
- `data/sft/val.jsonl`   — 10% of examples

Each JSONL line: `{"prompt": "…", "response": "…", "lang": "hin_Deva"}`.
The `lang` field is included for per-language loss monitoring and eval bucketing.

### Train / val split

90/10 positional cut after a global shuffle with a fixed seed.  The exact counts
are recorded in the manifest.  There is no separate test set — SFT quality is
evaluated through the fixed prompt suite in `eval/sft_eval.py` (generative eval)
and a perplexity regression check on the pretraining eval buckets.

### Shard format — SFT

Each shard is **two parallel `int32` NumPy arrays**, both of shape
`(N_examples_in_shard, seq_len)`:

```
{split}_{index:06d}_tokens.npy   — shape (N, 1024)  int32  full token sequence
{split}_{index:06d}_mask.npy     — shape (N, 1024)  int32  loss mask (0=ignore, 1=compute)
```

Each **row** is one complete padded example.  The DataLoader indexes directly by
example: `tokens[i]`, `mask[i]`.  Examples must never be stitched across row
boundaries in the loss window.

The `tokens` array contains the complete sequence: prompt tokens followed by
response tokens followed by EOT.  The `mask` array is `0` for every prompt
token and `1` for every response token and the final EOT.

Toy example (seq_len = 1024; only first and last positions shown):

```
Prompt:   "Translate to Hindi: Hello"   → token IDs [91, 2604, 311, 39452, 25, 18435]
Response: "नमस्ते"                       → token IDs [30501, 30502]
EOT:      (end of response)             → token ID  50256
Padding:  (fill to 1024)                → token ID  50256  ×  1015 times

tokens[i]: [ 91, 2604, 311, 39452, 25, 18435, 30501, 30502, 50256, 50256, …, 50256 ]
mask[i]:   [  0,    0,   0,     0,  0,     0,     1,     1,     1,     0, …,     0 ]
            ←————————— prompt —————————————→ ←— response + EOT ——→ ←——— padding ——→
```

The response-EOT position (mask=1) and the padding positions (mask=0) both use token
ID 50256, but are distinguished solely by the mask.  Padding never contributes to loss.

Training loop per example:

```python
tokens = np.load("train_000000_tokens.npy")  # (N, 1024)
mask   = np.load("train_000000_mask.npy")    # (N, 1024)
# input / target / loss mask for one example i:
x    = tokens[i, :-1]   # (1023,)
y    = tokens[i, 1:]    # (1023,)
m    = mask[i, 1:]      # (1023,)  — loss computed only where m == 1
```

| Property | Value |
|---|---|
| dtype | `int32` for both arrays |
| shape | `(N_examples_in_shard, 1024)` — 2D, one row per example |
| alignment | `tokens[i, j]` and `mask[i, j]` always correspond to the same position |
| seq_len | fixed at `model_max_len = 1024`; shorter sequences padded, longer truncated (response trimmed, prompt kept) |
| pad token | EOT = 50256 — token 0 is a real vocabulary token and must not be used for padding |
| padding mask | mask = 0 at all padding positions; padding never contributes to loss |
| EOT | appended at end of response; mask = 1 |
| prompt tokens | mask = 0 (loss ignored) |

### Tokenizer — SFT

**mgpt2 only.** The gpt2-tokenized model is retired after Phase C. Phase D's
controlled baseline is the same mgpt2 pretrained model *without* SFT — not a
gpt2 model. No gpt2 SFT shard is needed.

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
  "source":              "ai4bharat/indic-align (Dolly_T + OpenAssistant_T + Anudesh)",
  "tokenizer":           "mgpt2",
  "tokenizer_artifact":  { "path": "tokenizer/artifacts/mgpt2.model", "sha256": "…" },
  "seed":                42,
  "n_train_examples":    "<N>",
  "n_val_examples":      "<N>",
  "max_seq_len":         1024,
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

chosen   = [ 91, 318, 428, 11, 30945, 30,  2949, 11, 340, 318, 46512, 13, 50256, 50256, …, 50256 ]
rejected = [ 91, 318, 428, 11, 30945, 30,  3363, 11, 1494, 606, 477,  13, 50256, 50256, …, 50256 ]
            ←————————— prompt (len=6) ————————→ ←——— response + EOT ———————————→ ←— padding ——→

prompt_lens = [ 6 ]    ← same value applies to both chosen and rejected for this example
```

Note: the prompt is **not** followed by EOT in the sequence — it flows directly
into the response.  EOT is appended only at the end of the response.  The
`prompt_lens` value is the number of prompt tokens before the response starts.
Padding positions (after the response EOT) use token ID 50256 and are excluded
from loss by the DPO trainer using `prompt_lens` + actual sequence length.

| Property | Value |
|---|---|
| dtype | `int32` for all three arrays |
| alignment | `chosen[i]`, `rejected[i]`, `prompt_lens[i]` always the same prompt |
| seq_len | fixed at `model_max_len = 1024`; shorter sequences padded, longer truncated (response trimmed, prompt kept) |
| pad token | EOT = 50256 — token 0 is a real vocabulary token and must not be used for padding |
| EOT | appended at end of each response; padding positions after also use EOT but are excluded from loss |
| prompt in sequence | prompt tokens are identical prefix in chosen and rejected |

### Tokenizer — DPO

**mgpt2 only.** Same reasoning as SFT. Phase E's controlled baseline is the
Phase D SFT model pre-DPO — not a gpt2 model.

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
  "max_seq_len":         1024,
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
