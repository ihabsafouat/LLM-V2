from llm.tokenizer import CharTokenizer

texts = [
    "Hello world!\n",
    "Medical QA: fever, cough, headache.\n"
]

tok = CharTokenizer.from_texts(texts)
print("Vocab size:", tok.vocab_size)

s = "Hello fever!"
ids = tok.encode(s, add_bos=True, add_eos=True)
print("Encoded:", ids)
print("Decoded:", tok.decode(ids))

tok.save("artifacts/char_tokenizer.json")
tok2 = CharTokenizer.load("artifacts/char_tokenizer.json")
assert tok2.decode(tok2.encode(s)) == s
print("Save/load OK")
