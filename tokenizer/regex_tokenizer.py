try:
    from .base import get_stats, merge, visualise_tokens
    from .basic import BasicTokenizer
    from .patterns import GPT4_SPLIT_PATTERN
except ImportError:  # allow running as a script from inside `tokenizer/`
    from base import get_stats, merge, visualise_tokens
    from basic import BasicTokenizer
    from patterns import GPT4_SPLIT_PATTERN
from collections import Counter, defaultdict
import heapq
import regex as re
from tqdm import tqdm
import time

class RegexTokenizer(BasicTokenizer):
    def __init__(self, regex: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = regex
        self.regex = re.compile(self.pattern)
    
    def register_special_tokens(self, special_tokens: dict[str, int]):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    @staticmethod
    def _merge_word(word: tuple[int, ...], pair: tuple[int, int], new_id: int) -> tuple[int, ...]:
        """Merge all non-overlapping occurrences of `pair` in `word`."""
        out: list[int] = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(word[i])
                i += 1
        return tuple(out)

    @staticmethod
    def _pair_occurrences(word: tuple[int, ...]) -> dict[tuple[int, int], int]:
        """Return unweighted pair -> count for a single word/chunk."""
        if len(word) < 2:
            return {}
        counts: dict[tuple[int, int], int] = {}
        a = word[0]
        for b in word[1:]:
            p = (a, b)
            counts[p] = counts.get(p, 0) + 1
            a = b
        return counts

    def train(
        self,
        text: str,
        vocab_size: int = 50_257,
        verbose: bool = False,
        *,
        min_chunk_freq: int = 1,
        max_chunks: int | None = None,
    ):
        assert vocab_size >= 256, "Vocab size must be at least 256"
        num_merges = vocab_size - 256

        # Count chunk frequencies without storing a giant list of chunks.
        # Each unique chunk becomes a "word" in classic BPE training.
        chunk_counts: Counter[bytes] = Counter()
        for m in self.regex.finditer(text):
            s = m.group(0)
            if s:
                chunk_counts[s.encode("utf-8")] += 1

        # Heuristic speed knobs: ignore rare chunks and/or cap unique chunk types.
        # This massively reduces training state on web-scale corpora and keeps code simple.
        if min_chunk_freq > 1:
            chunk_counts = Counter({b: f for b, f in chunk_counts.items() if f >= min_chunk_freq})
        if max_chunks is not None and len(chunk_counts) > max_chunks:
            chunk_counts = Counter(dict(chunk_counts.most_common(max_chunks)))

        # words: tuple(symbol_ids) -> frequency
        words: dict[tuple[int, ...], int] = {}
        for b, freq in chunk_counts.items():
            words[tuple(b)] = freq

        # Global pair stats and a reverse index pair -> set(words containing it)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
        for w, freq in words.items():
            local = self._pair_occurrences(w)
            for p, occ in local.items():
                pair_counts[p] += freq * occ
                pair_to_words[p].add(w)

        # Max-heap for fast "most frequent pair" selection (lazy updates).
        heap: list[tuple[int, tuple[int, int]]] = [(-c, p) for p, c in pair_counts.items()]
        heapq.heapify(heap)

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        def bump_pair(p: tuple[int, int], delta: int) -> None:
            if delta == 0:
                return
            new = pair_counts.get(p, 0) + delta
            if new <= 0:
                pair_counts.pop(p, None)
                pair_to_words.pop(p, None)
                return
            pair_counts[p] = new
            heapq.heappush(heap, (-new, p))

        for i in tqdm(range(num_merges), desc="Training tokenizer"):
            start_time = time.time()

            # Pop stale heap entries until the top matches current counts.
            while heap:
                negc, p = heap[0]
                c = pair_counts.get(p, 0)
                if c > 0 and -negc == c:
                    break
                heapq.heappop(heap)
            if not heap:
                break

            pair = heap[0][1]
            count = pair_counts.get(pair, 0)
            if count <= 0:
                break

            idx = 256 + i
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

            affected = list(pair_to_words.get(pair, ()))
            if not affected:
                pair_counts.pop(pair, None)
                pair_to_words.pop(pair, None)
                continue

            # Apply merge to all words that contain the best pair.
            for w in affected:
                freq = words.get(w)
                if not freq:
                    continue

                new_w = self._merge_word(w, pair, idx)
                if new_w == w:
                    continue

                # Remove old word contributions
                old_local = self._pair_occurrences(w)
                for p, occ in old_local.items():
                    bump_pair(p, -freq * occ)
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(w)
                        if not s:
                            pair_to_words.pop(p, None)

                # Update words dict (merge words that collapse to the same new tuple)
                del words[w]
                words[new_w] = words.get(new_w, 0) + freq

                # Add new word contributions
                new_local = self._pair_occurrences(new_w)
                for p, occ in new_local.items():
                    bump_pair(p, freq * occ)
                    pair_to_words[p].add(new_w)

            # This pair should be fully merged away.
            pair_counts.pop(pair, None)
            pair_to_words.pop(pair, None)

            if verbose and i % 10 == 0:
                time_taken = time.time() - start_time
                tqdm.write(
                    f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) "
                    f"had {count} occurrences (took {time_taken:.2f}s)"
                )
        
        self.merges = merges
        self.vocab = vocab
        
    def decode(self, ids) -> str:
        part_bytes = []
        for id in ids:
            if id in self.vocab:
                part_bytes.append(self.vocab[id]) # id can be > 256 after merging
            elif id in getattr(self, "inverse_special_tokens", {}):
                part_bytes.append(self.inverse_special_tokens[id].encode("utf-8"))
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

