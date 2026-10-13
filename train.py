"""Train a character-level RNN on a plain text file.

Example:
    python train.py --data data/input.txt --epochs 20
"""
import argparse
import time

import torch
import torch.nn as nn

from model import CharRNN
from textutils import clean_text


def get_batch(data, batch_size, seq_len, device):
    """Pick random windows; target is the input shifted by one character."""
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, batch_size, seq_len, device, steps=20):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total = 0.0
    for _ in range(steps):
        x, y = get_batch(data, batch_size, seq_len, device)
        logits, _ = model(x)
        total += loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    model.train()
    return total / steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/input.txt")
    p.add_argument("--out", default="model.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--embed_size", type=int, default=64)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--raw", action="store_true",
                   help="do not clean the text (by default only letters, digits, "
                        "spaces and newlines are kept, so everything can be drawn)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    if not args.raw:
        text = clean_text(text)
    if len(text) < args.seq_len * 4:
        raise SystemExit("Dataset is too small. Add more text or lower --seq_len.")

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    print(f"Characters in dataset: {len(text):,} | unique characters: {len(chars)}")

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    if len(val_data) <= args.seq_len + 1:
        val_data = train_data  # tiny dataset: reuse train data for validation

    model = CharRNN(len(chars), args.embed_size, args.hidden_size,
                    args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    steps_per_epoch = max(1, len(train_data) // (args.batch_size * args.seq_len))
    best_val = float("inf")
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        running = 0.0
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, args.batch_size, args.seq_len, device)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # avoids exploding gradients
            optimizer.step()
            running += loss.item()

        train_loss = running / steps_per_epoch
        val_loss = evaluate(model, val_data, args.batch_size, args.seq_len, device)
        print(f"Epoch {epoch:3d}/{args.epochs} | train {train_loss:.3f} | "
              f"val {val_loss:.3f} | {time.time() - start:.0f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "chars": chars,
                "config": {
                    "embed_size": args.embed_size,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                },
            }, args.out)
            print(f"  saved best model to {args.out}")


if __name__ == "__main__":
    main()
