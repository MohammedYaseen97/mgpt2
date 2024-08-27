from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("./tokenizer/mgpt2-tokenizer")

# print(tokenizer.tokenize("Tokenizer Test ... "))
# print(tokenizer.decode(tokenizer.encode("Tokenizer Test ... ")))