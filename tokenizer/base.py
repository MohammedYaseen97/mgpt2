"""
A minimal implementation of Byte-Pair Encoding (BPE) tokenization.

BPE is a subword tokenization algorithm that iteratively merges the most frequent pairs of bytes or characters 
to build a vocabulary of subword tokens. This implementation is inspired by Andrej Karpathy's minbpe 
(https://github.com/karpathy/minbpe).
"""

def get_stats(ids, freq):
    for pair in zip(ids[:-1], ids[1:]):
        freq[pair] = freq.get(pair, 0) + 1

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

def visualise_tokens(token_values: list[bytes]) -> None:
    background = [f"\u001b[48;5;{i}m" for i in [167, 179, 185, 77, 80, 68, 134]]
    # If token boundaries do not occur at unicode character boundaries, it's unclear how best to
    # visualise the token. Here, we'll just use the unicode replacement character to represent some
    # fraction of a character.
    unicode_token_values = [x.decode("utf-8", errors="replace") for x in token_values]

    running_length = 0
    last_color = None
    for token in unicode_token_values:
        color = background[running_length % len(background)]
        if color == last_color:
            color = background[(running_length + 1) % len(background)]
            assert color != last_color
        last_color = color
        running_length += len(token)
        print(color + token, end="")
    print("\u001b[0m")

#--------------------------------------------------------------------------------------------------
class Tokenizer:
    def __init__(self):
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int e.g {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes
    
    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def decode(self, ids) -> str:
        raise NotImplementedError
    
    def encode(self, text, verbose=False) -> list[int]:
        raise NotImplementedError
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
