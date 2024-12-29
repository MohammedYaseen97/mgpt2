from regex_tokenizer import RegexTokenizer
import regex as re

# read from corpus.txt
with open("tok_corpus.txt", "r", encoding="utf-8") as file:
    corpus = file.read()

regex = re.compile(
    r"""(?i) 's|'t|'re|'ve|'m|'ll|'d| ?\b[\p{L}\u0900-\u0963|\u0966-\u097F]+\b| ?\b[\p{L}\u0C80-\u0C9E|\u0CA0-\u0CFF]+\b| ?[\p{N}]+| ?[.,!?;:'\"-]| ?[\u0964-\u0965]| ?[\u0C9E-\u0C9F]| ?[^\s\p{L}\p{N}\u0900-\u097F\u0C80-\u0CFF]+| \s+(?!\S)| \s+"""
)

tokenizer = RegexTokenizer(
    regex=regex
)

tokenizer.train(corpus, 50256, verbose=True)
tokenizer.register_special_tokens(
    {
        '<|endoftext|>': 50256
    }
)
tokenizer.save("mgpt2")