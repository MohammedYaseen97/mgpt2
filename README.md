# mgpt2 — Multilingual GPT-2

A complete, reproducible pipeline for building a multilingual language model from scratch, targeting **English, Hindi (Devanagari + Latin translit), and Kannada (Kannada script + Latin translit)**. Every component — tokenizer, pretraining, SFT, and DPO — is implemented from scratch and evaluated with fair, documented comparisons.

---

## What this is

This repo reproduces GPT-2 (124M) with a custom multilingual tokenizer and applies the full modern LLM training pipeline: pretraining → supervised fine-tuning → direct preference optimisation. It is structured as a phased research assignment with strict reproducibility requirements: every run records its git hash, tokenizer SHA-256, config, and seeds.

**Languages:** English (eng_Latn), Hindi (hin_Deva, hin_Latn), Kannada (kan_Knda, kan_Latn)  
**Model size:** 124M parameters (GPT-2 exact architecture)  
**Training compute:** ~14.4B tokens for pretraining; H100 80GB

## Hugging Face

| Artifact | Hub repo |
|---|---|
| Tokenizer | [ace-1/mgpt2-tokenizer](https://huggingface.co/ace-1/mgpt2-tokenizer) |
| Pretrained model | [ace-1/mgpt2-pretrain](https://huggingface.co/ace-1/mgpt2-pretrain) |
| SFT model | [ace-1/mgpt2-sft](https://huggingface.co/ace-1/mgpt2-sft) |
| DPO model ⭐ recommended | [ace-1/mgpt2-dpo](https://huggingface.co/ace-1/mgpt2-dpo) |

Each HF repo is self-contained: it bundles `model.py`, `pytorch_model.pt`, the tokenizer code, and a copy-paste quickstart. See any repo's README for the one-command load snippet.

---

## Results at a glance

### Phase A — Tokenizer efficiency (tokens/1k bytes, lower is better)

| Bucket | tiktoken_gpt2 | tiktoken_cl100k | **mgpt2** | vs gpt2 | vs cl100k |
|---|---:|---:|---:|---:|---:|
| **Overall** | 480.0 | 356.7 | **222.7** | −54% | −38% |
| **Devanagari** | 591.8 | 384.1 | **215.0** | −64% | −44% |
| **Kannada** | 980.9 | 639.7 | **213.3** | −78% | −67% |
| **Latin** | 256.6 | 249.5 | **229.8** | −10% | −8% |
| **Mixed** | 730.7 | 476.1 | **214.7** | −71% | −55% |

mgpt2 achieves its gains using the **same 50,257-token vocabulary size** as tiktoken_gpt2 — the compression comes entirely from better Indic-aware merge priorities, not from a larger vocabulary.

### Phase C — Pretraining (controlled comparison, same architecture + data + token budget)

| Metric | Baseline (gpt2 tok) | **mgpt2** |
|---|---:|---:|
| HellaSwag (English cloze) | 0.2768 | **0.2869** |
| BPB overall | 0.880 | **0.809** |
| BPB Devanagari | 0.516 | **0.469** |
| BPB Kannada | 0.521 | **0.461** |
| BPB Mixed | 0.566 | **0.471** |
| BPB Latin | 1.121 | **1.075** |

Raw perplexity favours the baseline (3.56 vs 12.4); **this comparison is invalid across tokenizers**. Bits-per-byte (BPB = log₂(PPL) × tokens/byte) normalises for tokenizer density and reverses the result on every bucket.

### Phase D — SFT

Fine-tuned on 30K multilingual instruction pairs (Anudesh / Dolly / OpenAssistant, 5 language variants). Final val loss: **1.2404**, val PPL: **3.457** (masked CE over response tokens). Native-script variants (Devanagari, Kannada) follow instructions coherently; transliterated Latin variants show expected script-drift due to ASCII ambiguity and low training data volume.

### Phase E — DPO

Applied DPO (β=0.1) on 13.5K toxic preference pairs. Win-rate: **100%** on held-out pairs. More meaningfully: DPO increased the log-probability of safety-refusal responses by **+6.1%** and decreased the log-probability of rejected (word-salad) responses by **−16.8%** relative to the SFT checkpoint. Generation comparisons confirm the shift: the SFT model (trained only on benign data) tries to comply with toxic prompts; the DPO model redirects. SFT regression: **+1.2%** (well within the 5% threshold).

---

## Repository layout

```
mgpt2/
├── model.py                  # GPT-2 124M (Karpathy-style)
├── train.py                  # pretraining loop
├── train_sft.py              # SFT loop (masked CE)
├── train_dpo.py              # DPO loop (Bradley-Terry)
│
├── tokenizer/                # pure-Python regex + BPE tokenizer
│   ├── regex_tokenizer.py
│   ├── base.py, basic.py, gpt4.py, patterns.py
│   ├── artifacts/            # mgpt2.model, mgpt2.vocab, heldout_eval.txt
│   └── README.md
│
├── configs/                  # YAML training configs (pretrain / sft / dpo, prod + smoke)
│
├── scripts/
│   ├── run_pretrain.py       # Phase C orchestrator
│   ├── run_sft.py            # Phase D orchestrator
│   ├── run_dpo.py            # Phase E orchestrator
│   ├── generate_dpo_rejected.py
│   ├── publish_model_hf.py
│   ├── download_tokenizer_artifacts.py
│   ├── checks/               # sanity-check scripts per phase
│   └── data/                 # Phase B data pipeline (see scripts/data/README.md)
│
├── eval/
│   ├── lm_eval.py            # streaming perplexity, bucketed
│   ├── hellaswag_eval.py     # full HellaSwag (10,042 examples)
│   ├── sft_eval.py           # masked val loss + prompt suite
│   ├── dpo_eval.py           # preference win-rate + SFT regression check
│   └── README.md
│
├── data/
│   ├── raw/                  # corpus_mixture.txt (15M docs)
│   ├── eval/                 # held-out eval sets (per bucket)
│   ├── sft/                  # SFT train/val JSONL
│   ├── dpo/                  # DPO train/val JSONL (chosen + rejected)
│   ├── shards_gpt2/          # pretraining shards, gpt2 tokenizer
│   ├── shards_mgpt2/         # pretraining shards, mgpt2 tokenizer
│   ├── shards_sft/           # SFT shards (tokens + mask)
│   └── shards_dpo/           # DPO shards (chosen + rejected + prompt_lens)
│
├── runs/                     # one folder per training run (auto-created)
│   ├── baseline_gpt2_tokenizer_20260401_092655/
│   ├── mgpt2_custom_tokenizer_20260402_162253/
│   ├── sft_mgpt2_20260403_074142/
│   └── dpo_dpo_mgpt2_custom_tokenizer_20260404_134347/
│
└── reports/
    ├── 00_tokenizer_report.md
    ├── 01_pretrain_report.md
    ├── 02_sft_report.md
    ├── 03_dpo_report.md
    └── 04_final_comparison.md
```

---

## Reproducing from scratch

### Prerequisites

```bash
git clone <this-repo>
cd mgpt2
python -m venv virtual
source virtual/bin/activate
pip install -r requirements.txt
```

### Step 1 — Download the tokenizer artifact

The tokenizer is the only non-derivable artifact (BPE training is non-deterministic at large scale). Fetch it from HF:

```bash
python scripts/download_tokenizer_artifacts.py --repo_id ace-1/mgpt2-tokenizer
```

This populates `tokenizer/artifacts/` with `mgpt2.model`, `mgpt2.vocab`, and `heldout_eval.txt`.

### Step 2 — Build data (Phase B)

```bash
# Pretraining corpus (~15M docs, ~40GB)
python -m scripts.data.build_corpus_mixture
python -m scripts.data.make_lm_eval_sets
python -m scripts.data.tokenize_shards          # writes shards_gpt2/ and shards_mgpt2/

# SFT corpus
python -m scripts.data.build_sft_data
python -m scripts.data.tokenize_sft_shards      # writes shards_sft/

# DPO corpus (needs Phase C checkpoint first — see Step 4b)
python -m scripts.data.build_dpo_data
# ... run after Step 4a:
python scripts/generate_dpo_rejected.py         # fills rejected field
python -m scripts.data.tokenize_dpo_shards      # writes shards_dpo/
```

### Step 3 — Pretrain (Phase C)

```bash
# Baseline (GPT-2 tokenizer)
python scripts/run_pretrain.py --config configs/pretrain_baseline.yaml

# mgpt2 (custom tokenizer)
python scripts/run_pretrain.py --config configs/pretrain_mgpt2.yaml
```

Each run creates `runs/<name>/` containing `run_info.json`, `log.txt`, `metrics.jsonl`, `final_metrics.json`, `lm_eval.json`, and the final checkpoint.

### Step 4 — SFT (Phase D)

```bash
python scripts/run_sft.py --config configs/sft_mgpt2.yaml
```

Auto-discovers the latest Phase C mgpt2 checkpoint. Creates `runs/sft_mgpt2_<timestamp>/`.

### Step 5 — DPO (Phase E)

```bash
python scripts/run_dpo.py --config configs/dpo_mgpt2.yaml
```

Auto-discovers the latest Phase D SFT checkpoint. Creates `runs/dpo_<timestamp>/` including `dpo_eval.json`.

### Step 6 — Publish to HF (Phase F)

```bash
export HF_TOKEN="your_hf_token"

# Publish any stage: pretrain | sft | dpo
python scripts/publish_model_hf.py --stage sft --repo_id YOUR_USERNAME/mgpt2
```

---

## Training configs (production, H100 80GB)

| Phase | Config | Batch | Steps/Epochs | Max LR |
|---|---|---|---|---|
| Pretrain baseline | `configs/pretrain_baseline.yaml` | 512K tokens | 27,537 steps | 3e-3 |
| Pretrain mgpt2 | `configs/pretrain_mgpt2.yaml` | 512K tokens | 27,537 steps | 3e-3 |
| SFT | `configs/sft_mgpt2.yaml` | 64 seqs | 3 epochs (1,262 steps) | 3e-4 |
| DPO | `configs/dpo_mgpt2.yaml` | 32 pairs | 1 epoch (420 steps) | 1e-6 |

Smoke configs (`*_smoke.yaml`) run in under 2 minutes for end-to-end pipeline validation.

---

## Reproducibility

Every run records the following in `runs/<name>/run_info.json`:

| Field | Value |
|---|---|
| `git_hash` | exact commit at launch |
| `tokenizer.sha256` | SHA-256 of `mgpt2.model` |
| `config_file` | YAML path |
| `train.seed` | 1337 (all runs) |

Eval sets are fixed lines from the corpus (`data/eval/`); they never appear in any training shard.

---

## Design choices and known limitations

**Tokenizer:** mgpt2 uses a multilingual regex split pattern with explicit Unicode ranges for Devanagari (U+0900–U+097F) and Kannada (U+0C80–U+0CFF) before BPE training. This gives dedicated merge tokens for Indic scripts, preventing the byte-level embedding interference that degrades GPT-2's English representations when trained on multilingual data.

**Transliterated Latin (hin_Latn, kan_Latn):** Both native-script variants work well; transliterated Latin is prone to script drift at generation time. The cause is irreducible: transliterated text shares the ASCII range with English, so there is no token-level anchor. 2,100 training examples per variant is insufficient to fully stabilise generation. Documented in `reports/02_sft_report.md`.

**DPO rejected quality:** At 124M parameters, the pretrained model generates incoherent word-salad for toxic prompts rather than coherent harmful responses. DPO therefore trains format-preference alignment (coherent refusals vs. noise) rather than genuine safety alignment. The behavioural shift is real (SFT complies; DPO redirects) but not equivalent to production safety alignment. Documented in `reports/03_dpo_report.md`.

**HellaSwag:** The +1.01 pp mgpt2 advantage on HellaSwag is directionally consistent with the BPB results but does not reach statistical significance at 95% (z=1.59). BPB is the primary pre-specified comparison metric.

---

## Reports

All experimental reports are in `reports/`:

| Report | Phase | Key finding |
|---|---|---|
| [00_tokenizer_report.md](reports/00_tokenizer_report.md) | A — Tokenizer | mgpt2 −54% vs gpt2, −38% vs cl100k overall |
| [01_pretrain_report.md](reports/01_pretrain_report.md) | C — Pretrain | mgpt2 wins on every BPB bucket including English |
| [02_sft_report.md](reports/02_sft_report.md) | D — SFT | Val loss 1.24; native scripts coherent; Latin-translit drifts |
| [03_dpo_report.md](reports/03_dpo_report.md) | E — DPO | 100% win-rate; +6.1% chosen log-p; SFT regression +1.2% |
| [04_final_comparison.md](reports/04_final_comparison.md) | All phases | Consolidated evidence and project narrative |

---

## References

- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI.
- Rafailov et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* NeurIPS 2023. https://arxiv.org/abs/2305.18290
- Petrov et al. (2023). *Language Model Tokenizers Introduce Unfairness Between Languages.* https://arxiv.org/abs/2305.15425
- Mielke et al. (2021). *Between words and characters: A brief history of open-vocabulary modeling.* https://arxiv.org/abs/2112.10508
- Rust et al. (2021). *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models.* ACL 2021. https://arxiv.org/abs/2012.15613
- Karpathy, A. [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (YouTube)
- Karpathy, A. [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (YouTube)
- AI4Bharat. [indic-align](https://huggingface.co/datasets/ai4bharat/indic-align) dataset (HuggingFace)
