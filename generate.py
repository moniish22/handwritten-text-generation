"""Generate new text with a trained character-level RNN, and optionally draw it
as handwriting.

Examples:
    python generate.py --prime "The " --length 400
    python generate.py --prime "The " --length 400 --bank glyphs.npz --image_out out.png
    python generate.py --prime "The " --length 400 \
        --dataset data/english-handwritten-characters-dataset --image_out out.png
"""
import argparse
import random

import torch
import torch.nn.functional as F

from model import CharRNN


def load_model(path, device="cpu"):
    """Return (model, chars) from a checkpoint saved by train.py."""
    ckpt = torch.load(path, map_location=device)
    chars = ckpt["chars"]
    model = CharRNN(len(chars), **ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, chars


def generate_text(model, chars, prime="The ", length=400, temperature=0.8,
                  device="cpu"):
    """Let the RNN continue `prime` for `length` more characters."""
    stoi = {c: i for i, c in enumerate(chars)}
    prime = "".join(c for c in prime if c in stoi) or random.choice(chars)
    generated = prime
    with torch.no_grad():
        # Feed the prime text through the network to warm up its memory
        x = torch.tensor([[stoi[c] for c in prime]], device=device)
        logits, hidden = model(x, None)
        last = logits[:, -1, :]
        for _ in range(length):
            probs = F.softmax(last / max(temperature, 1e-4), dim=-1)
            idx = torch.multinomial(probs, 1)
            generated += chars[idx.item()]
            logits, hidden = model(idx, hidden)
            last = logits[:, -1, :]
    return generated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="model.pt")
    p.add_argument("--prime", default="The ", help="starting text")
    p.add_argument("--length", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="lower = safer/repetitive, higher = more random")
    p.add_argument("--seed", type=int, default=None, help="for repeatable output")
    p.add_argument("--dataset", default=None,
                   help="folder containing english.csv and Img/ (turns on drawing)")
    p.add_argument("--bank", default=None,
                   help="glyphs.npz made by build_glyph_bank.py (turns on drawing)")
    p.add_argument("--image_out", default="handwriting.png")
    p.add_argument("--line_width", type=int, default=1000, help="image width in pixels")
    p.add_argument("--ruled", action="store_true", help="draw notebook lines")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, chars = load_model(args.model, device)
    generated = generate_text(model, chars, args.prime, args.length,
                              args.temperature, device)
    print(generated)

    if args.dataset or args.bank:
        from handwriting import HandwritingRenderer
        renderer = HandwritingRenderer(dataset_dir=args.dataset, bank=args.bank,
                                       seed=args.seed)
        image = renderer.render(generated, line_width=args.line_width, ruled=args.ruled)
        image.save(args.image_out)
        print(f"\nSaved handwriting image to {args.image_out}")


if __name__ == "__main__":
    main()
