from tokenizer.base import get_stats, merge, visualise_tokens
from tokenizer.basic import BasicTokenizer
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = regex
        self.regex = re.compile(self.pattern)
    
    def register_special_tokens(self, special_tokens: dict[str, int]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def train(self, text: str, vocab_size: int = 50_257, verbose: bool = False):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        
        text_chunks = re.findall(self.regex, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and i % 100 == 0:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab
        
    def decode(self, ids) -> str:
        part_bytes = []
        for id in ids:
            if id in self.vocab:
                part_bytes.append(self.vocab[id]) # id can be > 256 after merging
            elif id in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[id])
            else:
                raise ValueError(f"id={id} not in vocab or special_tokens")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode(encoding="utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, chunk_bytes: bytes, verbose=False) -> list[int]:
        tokens = list(chunk_bytes)
        while len(tokens) >= 2:
            if verbose:
                visualise_tokens([self.vocab[token] for token in tokens]) # token can be > 256 after merging
            stats = {}
            get_stats(tokens, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if not pair in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
    
    def encode_ordinary(self, text, verbose=False) -> list[int]:
        chunk_texts = re.findall(self.regex, text)
        ids_list = []
        for i, text in enumerate(chunk_texts):
            if verbose:
                print()
                print(f"encoding chunk {i+1}/{len(chunk_texts)}: {text}")
            chunk_bytes = text.encode("utf-8") # raw bytes
            ids = self._encode_chunk(chunk_bytes, verbose)
            ids_list.extend(ids)
        return ids_list
    
    def encode(self, text, verbose=False, allowed_special="none") -> list[int]:
        special = {}
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens), "Text contains special tokens that are not allowed"
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood.")
        if not special:
            return self.encode_ordinary(text, verbose)
        special_pattern = "(" + "|".join(re.escape(token) for token in special) + ")"
        parts = re.split(special_pattern, text)
        ids = []
        for part in parts:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part, verbose))
        return ids

