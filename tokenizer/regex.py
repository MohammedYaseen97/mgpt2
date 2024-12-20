from .base import get_stats, merge
from .basic import BasicTokenizer
import re

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(BasicTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.regex = re.compile(regex)

    def train(self, text: str, vocab_size: int = 50_257, verbose: bool = False):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        
        text_chunks = re.findall(self.regex, text)
        self.ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        for i in range(num_merges):
            stats = {}
            for chunk_ids in self.ids:
                get_stats(chunk_ids, stats)
            pair = min(stats, key=lambda p: stats.get(p, float("inf")))
            idx = 256 + i
            self.ids = [merge(chunk_ids, pair, idx) for chunk_ids in self.ids]
            self.merges[pair] = idx
        
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        