from tokenizer.base import get_stats, merge, visualise_tokens
from tokenizer.basic import BasicTokenizer
import regex as re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.regex = re.compile(regex)

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
        text = b"".join([self.vocab[id] for id in ids])
        text = text.decode(encoding="utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, chunk_bytes, verbose=False) -> list[int]:
        tokens = chunk_bytes.copy()
        while len(tokens) >= 2:
            if verbose:
                visualise_tokens([self.vocab[token] for token in tokens])
            stats = {}
            get_stats(tokens, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if not pair in self.merges:
                break
            idx = self.merges[pair]
            tokens = merge(tokens, pair, idx)
        return tokens
    
    def encode(self, text, verbose=False) -> list[int]:
        chunk_texts = re.findall(self.regex, text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunk_texts]
        if verbose:
            tokens = []
            for chunk in chunk_bytes:
                tokens += [self.vocab[byte] for byte in chunk]
            visualise_tokens(tokens)
        ids_list = []
        for i, chunk_byte in enumerate(chunk_bytes):
            if verbose:
                print()
                print(f"encoding chunk {i+1}/{len(chunk_bytes)}: {chunk_texts[i]}")
            ids = self._encode_chunk(chunk_byte, verbose)
            ids_list.extend(ids)
        return ids_list
