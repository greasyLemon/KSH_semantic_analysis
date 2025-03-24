from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

def get_token_num(text: str):
    return len(tokenizer.tokenize(text))