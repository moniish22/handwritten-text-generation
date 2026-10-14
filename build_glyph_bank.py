"""Pack the Kaggle handwriting images into ONE small file (glyphs.npz).

Run this once on your computer after downloading the dataset:

    python build_glyph_bank.py

The web app (app.py) then only needs glyphs.npz (about 1-3 MB), so you do not
have to upload the 3410 original images to GitHub.
"""
import argparse
import os

from handwriting import HandwritingRenderer

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="data/english-handwritten-characters-dataset")
p.add_argument("--out", default="glyphs.npz")
args = p.parse_args()

renderer = HandwritingRenderer(args.dataset)
renderer.save_bank(args.out)
print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")
