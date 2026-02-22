from __future__ import annotations

import os
from typing import Any, Optional

from transformers import PreTrainedTokenizer

from tokenizer.regex_tokenizer import RegexTokenizer


class MGPT2Tokenizer(PreTrainedTokenizer):
    """
    Hugging Face-compatible (slow) tokenizer wrapper around `RegexTokenizer`.

    This is intended for publishing alongside the model using `trust_remote_code=True`.
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_file: str, **kwargs: Any):
        if not model_file.endswith(".model"):
            raise ValueError(f"model_file must end with .model, got: {model_file}")

        self._tok = RegexTokenizer()
        self._tok.load(model_file)

        # Bind common special tokens if present in the trained tokenizer.
        special = self._tok.special_tokens
        kwargs.setdefault("eos_token", "<|endoftext|>" if "<|endoftext|>" in special else None)
        kwargs.setdefault("unk_token", None)
        kwargs.setdefault("pad_token", None)
        kwargs.setdefault("bos_token", None)

        super().__init__(**kwargs)

        self.model_file = model_file

    @property
    def vocab_size(self) -> int:
        # vocab is sparse only if merges are incomplete; generally size is max_id+1
        return max(self._tok.vocab.keys()) + 1

    def get_vocab(self) -> dict[str, int]:
        # Provide a stable token-string mapping for HF internals.
        inv_special = self._tok.inverse_special_tokens
        vocab: dict[str, int] = {}
        for i in range(self.vocab_size):
            if i in inv_special:
                vocab[inv_special[i]] = i
            else:
                vocab[f"<|bytebpe_{i}|>"] = i
        return vocab

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        ids = self._tok.encode(text, allowed_special="all")
        inv_special = self._tok.inverse_special_tokens
        out: list[str] = []
        for i in ids:
            out.append(inv_special.get(i, f"<|bytebpe_{i}|>"))
        return out

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._tok.special_tokens:
            return self._tok.special_tokens[token]
        if token.startswith("<|bytebpe_") and token.endswith("|>"):
            inner = token[len("<|bytebpe_") : -len("|>")]
            return int(inner)
        raise KeyError(f"Unknown token string: {token!r}")

    def _convert_id_to_token(self, index: int) -> str:
        return self._tok.inverse_special_tokens.get(index, f"<|bytebpe_{index}|>")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        ids = [self._convert_token_to_id(t) for t in tokens]
        return self._tok.decode(ids)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("This tokenizer does not support pair inputs.")
        return token_ids_0

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = filename_prefix or "tokenizer"
        out_prefix = os.path.join(save_directory, prefix)
        # Save in the native `.model`/`.vocab` format (human + machine readable for this repo).
        self._tok.save(out_prefix)
        return (out_prefix + ".model",)

