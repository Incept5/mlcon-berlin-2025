import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def tokenize_text(text):
    inputs = tokenizer(text, return_tensors="pt")
    return inputs['input_ids'].tolist()[0]

def detokenize_tokens(token_ids):
    return tokenizer.decode(token_ids)

def main():
    text = "Attention Is All You Need"
    token_ids = tokenize_text(text)

    for token_id in token_ids:
        print(f"{tokenizer.decode(token_id)} = {token_id}")

    detokenized_text = detokenize_tokens(token_ids)
    print(detokenized_text)

if __name__ == "__main__":
    main()