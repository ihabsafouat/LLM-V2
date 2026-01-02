import torch
from llm.model import GPT, GPTConfig

def main():
    config = GPTConfig(vocab_size=100, block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = GPT(config)
    x = torch.randint(0, config.vocab_size, (4, 16))  # (B=4, T=16)
    logits, loss = model(x, targets=x)
    print("logits shape:", logits.shape)
    print("loss:", float(loss))

    y = model.generate(x[:, :4], max_new_tokens=10, temperature=1.0, top_k=20)
    print("generated shape:", y.shape)

if __name__ == "__main__":
    main()
