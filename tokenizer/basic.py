from base import Tokenizer, get_stats, merge, visualise_tokens

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        # 'ids' is a list of integers, each representing a byte from the UTF-8 encoded string
        ids = list(text.encode("utf-8")) # list[int]
        if verbose:
            print(f"len(text) = {len(text)}")
            print(f"len(tokens) = {len(ids)}")
            
        num_merges = vocab_size - 256
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            get_stats(ids, stats)
            pair = max(stats, key=stats.get) # (int, int)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and i % 100 == 0:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
        
        self.vocab = vocab
        self.merges = merges
    
    def decode(self, ids) -> str:
        text = b"".join([self.vocab[id] for id in ids])
        text = text.decode(encoding="utf-8", errors="replace")
        return text
    
    def encode(self, text, verbose=False) -> list[int]:
        tokens = list(text.encode("utf-8"))
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
