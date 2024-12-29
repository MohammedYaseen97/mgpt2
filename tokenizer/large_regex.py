from regex_tokenizer import RegexTokenizer
from tokenizer import get_stats_parallel, merge_parallel
import regex as re
from tqdm import tqdm
import time

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class LargeRegexTokenizer(RegexTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__(regex)

    def train(self, text: str, vocab_size: int = 50_257, verbose: bool = False):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        
        text_chunks = re.findall(self.regex, text)
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Output will look like:
        # Training tokenizer: 100%|██████████| 50000/50000 [12:34<00:00, 66.23it/s]
        # merge 1/50000: (97, 101) -> 256 (b'ae') had 2945 occurrences (took 0.15s)
        # merge 2/50000: (104, 108) -> 257 (b'hl') had 2842 occurrences (took 0.14s)
        # ...
        
        for i in tqdm(range(num_merges), desc="Training tokenizer"):
            start_time = time.time()
            
            # Compute stats in parallel using Rust
            stats = get_stats_parallel(ids)
            
            pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Perform merging in parallel using Rust
            ids = merge_parallel(ids, pair, idx)
            
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and i % 10 == 0:
                time_taken = time.time() - start_time
                tqdm.write(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences (took {time_taken:.2f}s)")
        
        self.merges = merges
        self.vocab = vocab

