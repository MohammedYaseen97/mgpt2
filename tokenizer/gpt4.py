from regex_tokenizer import RegexTokenizer
from base import visualise_tokens, get_stats, merge
from typing import Optional
import regex as re
import tiktoken

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

def bpe(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: Optional[int] = None) -> list[bytes]:
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks: dict[bytes, int]) -> dict[bytes, tuple[bytes, bytes]]:
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, idx in self.merges.items():
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.vocab = vocab
        # for some reason, the tokens corresponding to individual bytes
        # are permuted in a different order. This is completely non-sensical
        # and probably historical, but therefore we have to deal with it here
        self.byte_shuffle = {idx: mergeable_ranks[bytes([idx])] for idx in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}
        self.register_special_tokens(GPT4_SPECIAL_TOKENS)
    
    def train(self, text: str, vocab_size: int = 50_257, verbose: bool = False):
        raise NotImplementedError
    
    def _encode_chunk(self, chunk_bytes: bytes, verbose: bool = False) -> list[int]:
        chunk_bytes = bytes(self.byte_shuffle[b] for b in chunk_bytes)
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            if verbose:
                decodable_ids = [] # each id can be multiple bytes i.e. any utf-8 character
                for id in ids:
                    char = self.vocab[id] # id can be > 256 after merging
                    decodable_ids.append(bytes(self.inverse_byte_shuffle[b] for b in char))
                visualise_tokens(decodable_ids)
            stats = {}
            get_stats(ids, stats)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if not pair in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def decode(self, ids) -> str:
        part_bytes = []
        for id in ids:
            if id in self.vocab:
                char = self.vocab[id] # id can be > 256 after merging
                part_bytes.extend(self.inverse_byte_shuffle[b] for b in char)
            elif id in self.inverse_special_tokens:
                part_bytes.extend(self.inverse_special_tokens[id].encode("utf-8"))
            else:
                raise ValueError(f"id={id} not in vocab or special_tokens")
        text_bytes = bytes(part_bytes)
        text = text_bytes.decode(encoding="utf-8", errors="replace")
        return text
    
    def save(self, path: str):
        raise NotImplementedError("GPT4Tokenizer not meant to be saved")
    
    def load(self, path: str):
        raise NotImplementedError("GPT4Tokenizer not meant to be loaded")