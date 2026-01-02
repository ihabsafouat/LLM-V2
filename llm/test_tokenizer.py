from llm.tokenizer import CharTokenizer, normalize_text

MED_EXTRA_CHARS = list("°±≥≤→←αβμΩΔ")

paths = [
    "data/pretrain.txt",
    "data/medical_corpus.txt",
]

def read_all(paths):
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            texts.append(normalize_text(f.read()))
    return texts

texts = read_all(paths)

# Start with forced extras
extra = set(MED_EXTRA_CHARS)

for iteration in range(10):  # safety cap
    tok = CharTokenizer.from_texts(texts, extra_chars=sorted(extra), keep_whitespace=True)

    # Find unknown characters across all texts
    unknown = set()
    for t in texts:
        unknown.update(ch for ch in t if ch not in tok.stoi)

    if not unknown:
        print(f"✅ No unknown chars after {iteration} iteration(s).")
        break

    print(f"Iteration {iteration}: found {len(unknown)} unknown chars. Adding them.")
    extra.update(unknown)

tok.save("artifacts/char_tokenizer.json")
print("Saved tokenizer with vocab size:", tok.vocab_size)
