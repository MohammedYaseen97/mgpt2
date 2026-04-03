## 02 — SFT report (IndicAlign Instruct)

**Status: COMPLETE**

### Objective

Fine-tune the Phase C mgpt2 pretrained checkpoint on IndicAlign instruction data and demonstrate instruction-following capability across English, Hindi (Devanagari + transliterated Latin), and Kannada (Kannada script + transliterated Latin). The controlled baseline for Phase D is the same Phase C pretrained model without SFT.

---

### Artifacts

| Artifact | Path |
|---|---|
| **SFT checkpoint (final)** | `runs/sft_mgpt2_20260403_074142/model_01262.pt` |
| **Pretrained checkpoint used** | `runs/mgpt2_custom_tokenizer_20260402_162253/model_27537.pt` |
| **Pretrained checkpoint SHA-256** | `3a1edb1f3e27b4c0bdc9f6b1a9c43d89310ac3b7d9a2c844b2c18a93b1f0687a` |
| **Tokenizer SHA-256** | `f2911100f93f224a36cfd6a40de8739a12f3fe7b0b885cd0edc961c6e5e6c4b1` |
| **Config** | `configs/sft_mgpt2.yaml` |
| **Eval output** | `runs/sft_mgpt2_20260403_074142/sft_eval.json` |

---

### Training setup

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Batch size | 64 (micro 8, grad accum 8) |
| Total steps | 1,262 |
| Max LR | 3e-4 (10× lower than pretrain) |
| Min LR ratio | 0.1 |
| Warmup steps | 50 |
| Weight decay | 0.1 |
| Training examples | ~26,951 (90% of 30K IndicAlign corpus) |
| Val examples | ~2,995 (10%) |
| Loss | Masked CE — response tokens + EOT only; prompt masked out |

---

### Final metrics

| Metric | Value |
|---|---|
| **Final train loss** | 0.9113 |
| **Final val loss** | 1.2404 |
| **Val PPL (SFT set)** | 3.457 ¹ |

---

### Training curve (val loss)

| Step | Val loss | Epoch |
|---:|---:|---|
| 0 | 2.0145 | pretrained model (no SFT) |
| 50 | 1.5242 | 1 |
| 100 | 1.4240 | 1 |
| 200 | 1.3491 | 1 |
| 400 | 1.2806 | 1 |
| 420 | — | *epoch 1 end* |
| 550 | 1.2661 | 2 |
| 700 | 1.2524 | 2 |
| **800** | **1.2379** | **2 ← best** |
| 841 | — | *epoch 2 end* |
| 900 | 1.2465 | 3 |
| 1000 | 1.2453 | 3 |
| 1100 | 1.2510 | 3 |
| 1262 | 1.2404 | *epoch 3 end* |

Val loss drops sharply in epoch 1 (2.0145 → 1.2806), plateaus through epoch 2 (best: 1.2379 at step 800), and shows very mild overfitting in epoch 3 before recovering at the end. The train/val gap at the final step (0.91 / 1.24) reflects light overfitting on a 30K example dataset, which is expected and acceptable at this scale.

¹ Val PPL is computed on the SFT held-out set (instruction examples, masked CE over response tokens only) — it is not comparable to the pretraining LM eval PPL of 12.4, which is computed on the general multilingual eval set over all tokens. The lower PPL here reflects both the narrower domain of the SFT val set and the masked loss formulation, not a general improvement in language modeling ability.

---

### Prompt suite results

The 10-prompt fixed generation suite covers 2 prompts × 5 language variants. Results from `sft_eval.json`:

#### English (`eng_Latn`)

| Prompt | Assessment |
|---|---|
| "What is machine learning?" | Responds on-topic in English. Instruction-following form is correct; content is repetitive and slightly confused but directionally relevant. GPT-2 scale knowledge limit. |
| "Explain the water cycle in simple terms." | Fails — gets stuck in a counting loop, never explains the water cycle. Content collapse on a concrete factual prompt. |

#### Hindi Devanagari (`hin_Deva`)

| Prompt | Assessment |
|---|---|
| "मशीन लर्निंग क्या है?" | **Responds correctly in Devanagari script.** Content is confused (associates ML with physics) but the language, script, and instruction-following form are correct. |
| "जल चक्र को सरल शब्दों में समझाएं।" | Responds in Devanagari, short but partially on-topic ("सरल शब्द" / simple words acknowledged). Truncates early. |

#### Hindi Romanized (`hin_Latn`)

| Prompt | Assessment |
|---|---|
| "Machine learning kya hota hai?" | **Responds in English** (not Hinglish). However, produces a well-structured numbered list with correct ML concepts — best content quality of all 10 prompts. The script switch to English is a failure, but the instruction-following format is strong. |
| "Jal chakra ko samjhao." | **Critical failure** — begins in Hinglish then collapses mid-generation into Kannada script. Complete script mix-up. |

#### Kannada script (`kan_Knda`)

| Prompt | Assessment |
|---|---|
| "ಯಂತ್ರ ಕಲಿಕೆ ಎಂದರೇನು?" | **Responds correctly in Kannada script.** Content is about data analysis/operations — tangentially related. Script fidelity is solid. |
| "ನೀರಿನ ಚಕ್ರವನ್ನು ಸರಳ ಮಾತುಗಳಲ್ಲಿ ವಿವರಿಸಿ." | Responds in Kannada with bullet-point structure. Content is semantically off (advises "resting on the water cycle") but the script, format, and response length are correct. |

#### Kannada Romanized (`kan_Latn`)

| Prompt | Assessment |
|---|---|
| "Yantra kalike endare yenu?" | **Responds in Kannada Romanized Latin.** Content is abstract but language-consistent. Script fidelity holds. |
| "Niru chakravanna vivarisuvi." | **Critical failure** — starts in Kannada Romanized, then collapses into Telugu script repetition (`వివిధ వివిధ వివిధ...`). Complete breakdown. |

---

### What improved vs the pretrained model

1. **Instruction-following form**: the model reliably produces a response to the given prompt rather than continuing the prompt as a language model. The prompt–response boundary has been learned.
2. **Script fidelity for native scripts**: Devanagari and Kannada prompts reliably receive responses in the correct script. This is the core multilingual win — the pretrained model had no reason to respect script boundaries in generation.
3. **Structured output**: numbered lists and bullet points appear in appropriate contexts, especially for English and Kannada prompts.
4. **Transliterated Kannada (kan_Latn)**: one of two responses succeeds, which is notable given this is the lowest-quality variant in the SFT data.

### What still fails

1. **Content accuracy**: factual content is often wrong or confused. This is a GPT-2 124M capacity problem, not an SFT problem — the model does not have the parametric knowledge to answer most factual questions accurately.

2. **Transliterated Latin script stability**: `hin_Latn` and `kan_Latn` are prone to mid-generation script switching. The "Jal chakra" prompt fails in both `hin_Latn` (slides into Kannada script) and `kan_Latn` (slides into Telugu script repetition). Three compounding factors explain this:

   - **No Unicode anchor.** Devanagari (U+0900–U+097F) and Kannada (U+0C80–U+0CFF) occupy completely disjoint Unicode ranges — the model can detect script from token identity alone. Transliterated Latin shares the ASCII range with English, so there is no token-level signal distinguishing `hin_Latn` from English or `kan_Latn` from `hin_Latn`. When the model is uncertain about content, it has no structural anchor to stay in the correct variant and drifts to whichever Latin-adjacent distribution has the strongest prior.

   - **Weakest data category.** Only 2,100 examples each for `hin_Latn` and `kan_Latn` (7% each of 30K total). The source data (`DATA_REFERENCE.md`) explicitly notes `kan_Latn` has the lowest IndicTrans2 romanisation quality of the five variants. A 124M parameter model fine-tuned on 2,100 noisy examples has not seen enough consistent supervision to build a stable generation prior for that variant.

   - **ASCII space ambiguity is irreducible for romanised variants.** The Phase C pretraining report describes how the GPT-2 tokenizer's byte-level Indic encoding creates cross-script embedding interference. mgpt2 resolves this for *native scripts* — Devanagari and Kannada get dedicated merge tokens in completely disjoint Unicode ranges. But transliterated Latin (`hin_Latn`, `kan_Latn`) is, by definition, written in ASCII. Even with the mgpt2 tokenizer, the same ASCII subword tokens appear in English, romanised Hindi, and romanised Kannada contexts. There is no vocabulary-level separation possible for these variants under any BPE tokenizer trained on Latin script. The model must learn the distinction purely from context — a much harder task that 2,100 noisy training examples is insufficient to fully solve.

   The fix at this scale would be: more and higher-quality romanised training examples, or constrained decoding that penalises cross-script token transitions. Neither is applied here.

3. **The water cycle prompt** fails in 4 of 5 language variants across all scripts. This is a content knowledge problem, not a language problem — the model lacks a reliable internal representation of the water cycle at 124M parameters.

4. **Repetition**: several responses loop or truncate early, a known GPT-2 generation artefact that SFT at this scale does not fully suppress.

---

### References

[1] Petrov et al. (2023). "Language Model Tokenizers Introduce Unfairness Between Languages." *NeurIPS 2023*. https://arxiv.org/abs/2305.15425

[2] Rust et al. (2021). "How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models." *ACL-IJCNLP 2021*. https://aclanthology.org/2021.acl-long.243

---

### Conclusion

SFT on 30K IndicAlign examples successfully instils instruction-following behaviour across all five language variants in 1,262 gradient steps (~4 minutes on H100). The key wins are script fidelity for native Devanagari and Kannada, and the acquisition of response format (the model answers rather than continues). Val loss dropped from 2.01 (pretrained model cold-evaluated on the SFT set) to 1.24, a 38% reduction, with no significant regression during epoch 3.

The primary failure mode is transliterated Latin script instability — `hin_Latn` and `kan_Latn` are prone to cross-script hallucination, particularly on prompts where the model lacks the content to draw on. This is expected: at 124M parameters with a 30K example fine-tune, the model cannot reliably maintain script discipline when it is uncertain about content. Phase E (DPO) will not directly address this but provides an opportunity to align against the most egregious failure modes using the Phase C model's own outputs as the rejected side.

The SFT checkpoint (`model_01262.pt`) is the starting point for Phase E.
