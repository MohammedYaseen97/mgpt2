"""
Regex patterns used by tokenizers in this package.

Keep patterns centralized so experiments + training scripts + notebooks
stay in sync.
"""

# Default GPT-4-ish split pattern (as used in `RegexTokenizer` and `GPT4Tokenizer`)
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Indic-focused experimental pattern (Hindi Devanagari + Kannada ranges and punctuation)
INDIC_SPLIT_PATTERN = r"""(?i) 's|'t|'re|'ve|'m|'ll|'d| ?\b[\p{L}\u0900-\u0963|\u0966-\u097F]+\b| ?\b[\p{L}\u0C80-\u0C9E|\u0CA0-\u0CFF]+\b| ?[\p{N}]+| ?[.,!?;:'\"-]| ?[\u0964-\u0965]| ?[\u0C9E-\u0C9F]| ?[^\s\p{L}\p{N}\u0900-\u097F\u0C80-\u0CFF]+| \s+(?!\S)| \s+"""

