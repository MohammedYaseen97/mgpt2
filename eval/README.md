## Evaluation scripts

All evaluation scripts live here. Each phase has its own eval script; all write
results as JSON to the corresponding run folder for reproducibility.

---

### Comparison methodology

The project distinguishes two types of comparisons. Every report must label each
result clearly.

**Controlled** — variables held constant except the one being tested:

| Phase | What varies | Baseline |
|---|---|---|
| C — Pretrain | Tokenizer only | Your GPT-2-tokenized model (same corpus, same token budget) |
| D — SFT | Fine-tuning (SFT) | Your Phase C pretrained mgpt2 model (untuned) |
| E — DPO | Alignment (DPO) | Your Phase D SFT model (pre-DPO) |

**Contextual** — external reference points, different training conditions. Must be
labelled explicitly as non-controlled in any report:

| Phase | Contextual reference | Why it differs |
|---|---|---|
| A — Tokenizer | tiktoken_cl100k, tiktoken_gpt2 | Different vocab size and/or training data |
| C — Pretrain | HF GPT-2 model | Trained on WebText (English-only), not your multilingual corpus |

---

### Scripts

- `buckets.py`
  - shared script-detection logic; classifies lines into latin/devanagari/kannada/mixed buckets

- `lm_eval.py` ✓
  - streaming perplexity overall + bucketed (latin/deva/knda/mixed) on fixed heldout sets
  - memory-efficient rolling-buffer tokenization; reads `data/eval/manifest.json`

- `hellaswag_eval.py` ✓
  - full 10,042-example HellaSwag for a trained model checkpoint using its own tokenizer
  - run for both controlled models (GPT-2-tokenized and mgpt2-tokenized); optional `--hf-reference` flag for contextual HF GPT-2 comparison (must be labelled non-controlled)

- `sft_eval.py` ✓
  - full val masked loss (all val examples, properly weighted per response token)
  - fixed 10-prompt multilingual generation suite (2 prompts × 5 language variants: eng_Latn, hin_Deva, hin_Latn, kan_Knda, kan_Latn)

- `dpo_eval.py`
  - [TODO] preference win-rate/accuracy on held-out pairs + SFT loss regression check

---

### Output contract (all scripts)
- write results to `runs/<run_name>/metrics.json`
- include tokenizer identity (model path + SHA-256), git commit hash, and RNG seed in every output
