# llm/train.py
from __future__ import annotations
import os
import time
import math
import argparse
import torch

from llm.tokenizer import CharTokenizer, normalize_text
from llm.model import GPT, GPTConfig
from llm.data import TokenDataset


def estimate_loss(model, train_ds, val_ds, batch_size, eval_iters=50):
    model.eval()
    out = {}
    for split_name, ds in [("train", train_ds), ("val", val_ds)]:
        losses = []
        for _ in range(eval_iters):
            x, y = ds.get_batch(batch_size)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_path", type=str, default="data/train.txt")
    p.add_argument("--tokenizer_path", type=str, default="artifacts/char_tokenizer.json")
    p.add_argument("--out_dir", type=str, default="artifacts")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # model / data
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)

    # sampling
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--sample_tokens", type=int, default=300)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load tokenizer
    tok = CharTokenizer.load(args.tokenizer_path)

    # Load text -> ids
    text = open(args.train_path, "r", encoding="utf-8", errors="replace").read()
    text = normalize_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # Train/val split
    n = int(0.9 * len(ids))
    train_ids = ids[:n]
    val_ids = ids[n:]

    train_ds = TokenDataset(train_ids, args.block_size, device=args.device)
    val_ds = TokenDataset(val_ids, args.block_size, device=args.device)

    # Build model
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(cfg).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP only on CUDA
    use_amp = (args.device.startswith("cuda") and torch.cuda.is_available())
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"Device: {args.device} | AMP: {use_amp}")
    print(f"Tokens: {len(ids):,} | Vocab: {tok.vocab_size} | Params: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    for step in range(1, args.max_iters + 1):
        x, y = train_ds.get_batch(args.batch_size)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        if args.grad_clip is not None and args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # eval
        if step % args.eval_interval == 0 or step == 1:
            losses = estimate_loss(model, train_ds, val_ds, args.batch_size, args.eval_iters)
            dt = time.time() - t0
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} | last {loss.item():.4f} | {dt:.1f}s")
            # save checkpoint
            ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
            torch.save({
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "tokenizer_path": args.tokenizer_path,
                "step": step,
            }, ckpt_path)

        # sample text
        if step % args.sample_every == 0:
            model.eval()
            prompt = "Medical QA:\nQ: What is a fever?\nA:"
            idx = torch.tensor([tok.encode(prompt)], dtype=torch.long).to(args.device)
            out = model.generate(idx, max_new_tokens=args.sample_tokens, temperature=1.0, top_k=50)
            gen = tok.decode(out[0].tolist())
            print("\n--- SAMPLE ---")
            print(gen[:2000])
            print("--------------\n")
            model.train()

    # final save
    torch.save({"model": model.state_dict(), "config": cfg.__dict__},
               os.path.join(args.out_dir, "final_model.pt"))
    print("Training complete.")


if __name__ == "__main__":
    main()
