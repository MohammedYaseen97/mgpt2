from regex_tokenizer import RegexTokenizer
import regex as re
from tqdm import tqdm
import time
import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libtokenizer.so')

# Define the Pair struct
class Pair(ctypes.Structure):
    _fields_ = [("first", ctypes.c_uint32),
                ("second", ctypes.c_uint32)]

# Define the PairCount struct
class PairCount(ctypes.Structure):
    _fields_ = [("pair", Pair),
                ("count", ctypes.c_uint32)]

# Define the argument and return types for the C functions
lib.get_stats_parallel.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), 
                                   ctypes.POINTER(ctypes.c_size_t), 
                                   ctypes.c_size_t, 
                                   ctypes.POINTER(ctypes.c_size_t)]
lib.get_stats_parallel.restype = ctypes.POINTER(PairCount)

lib.merge_parallel.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)), 
                               ctypes.POINTER(ctypes.c_size_t), 
                               ctypes.c_size_t, 
                               Pair, 
                               ctypes.c_uint32]

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class LargeRegexTokenizer(RegexTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__(regex)

    def train(self, text: str, vocab_size: int = 50_257, verbose: bool = False, batch_size: int = 1000):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256
        
        # Split text into batches
        text_chunks = re.findall(self.regex, text)
        num_batches = len(text_chunks) // batch_size + (1 if len(text_chunks) % batch_size != 0 else 0)
        
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        
        for batch_index in range(num_batches):
            batch_start = batch_index * batch_size
            batch_end = min((batch_index + 1) * batch_size, len(text_chunks))
            batch = text_chunks[batch_start:batch_end]
            
            ids = []
            for ch in batch:
                encoded = ch.encode("utf-8")
                padding_length = (4 - len(encoded) % 4) % 4
                padded_encoded = encoded + b'\0' * padding_length
                ids.append(np.frombuffer(padded_encoded, dtype=np.uint32))
            
            chunk_sizes = np.array([len(chunk) for chunk in ids], dtype=np.intp)
            chunk_arrays = (ctypes.POINTER(ctypes.c_uint32) * len(ids))(
                *[chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)) for chunk in ids]
            )
            
            for i in tqdm(range(num_merges), desc=f"Training tokenizer (batch {batch_index+1}/{num_batches})"):
                start_time = time.time()
                
                num_pairs = ctypes.c_size_t()
                pair_counts = lib.get_stats_parallel(chunk_arrays, chunk_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)), len(ids), ctypes.byref(num_pairs))
                
                max_pair = max((pair_counts[j] for j in range(num_pairs.value)), key=lambda pc: pc.count)
                pair = (max_pair.pair.first, max_pair.pair.second)
                idx = 256 + i
                
                lib.merge_parallel(chunk_arrays, chunk_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)), len(ids), Pair(*pair), idx)
                
                merges[pair] = idx
                vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
                if verbose and i % 10 == 0:
                    time_taken = time.time() - start_time
                    tqdm.write(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {max_pair.count} occurrences (took {time_taken:.2f}s)")
            
            # Free memory used by the current batch
            del ids
            del chunk_sizes
            del chunk_arrays
        
        self.merges = merges
        self.vocab = vocab

