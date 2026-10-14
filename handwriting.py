"""Draw text as handwriting using the English Handwritten Characters dataset.

The dataset has 55 photos for each of 62 characters (a-z, A-Z, 0-9) and a
CSV (english.csv) with two columns: image path and label. For every character
in the text we pick one random photo of that character, tidy it up, add a
little random jitter, and paste the pieces side by side.
"""
import csv
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw


class HandwritingRenderer:
    def __init__(self, dataset_dir=None, csv_name="english.csv",
                 char_height=64, seed=None, bank=None):
        """Load from the raw Kaggle folder (dataset_dir) OR from a compact
        glyph bank file made by build_glyph_bank.py (bank)."""
        self.char_height = char_height
        self.rng = random.Random(seed)
        self.samples = {}  # label -> list of grayscale PIL images
        if bank:
            self._load_bank(bank)
        elif dataset_dir:
            self._load(dataset_dir, csv_name)
        else:
            raise ValueError("Give either dataset_dir or bank.")

    # ------------------------------------------------------------ glyph bank
    def save_bank(self, path):
        """Save the cleaned-up character images into one small .npz file."""
        arrays = {}
        for label, imgs in self.samples.items():
            for i, img in enumerate(imgs):
                arrays[f"{label}_{i}"] = np.asarray(img, dtype=np.uint8)
        np.savez_compressed(path, **arrays)

    def _load_bank(self, path):
        with np.load(path) as bank:
            for key in bank.files:
                label = key.rsplit("_", 1)[0]
                self.samples.setdefault(label, []).append(
                    Image.fromarray(bank[key]))
        self.char_height = next(iter(next(iter(self.samples.values())))).height

    # ------------------------------------------------------------------ load
    def _load(self, dataset_dir, csv_name):
        csv_path = os.path.join(dataset_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Could not find {csv_path}. Point --dataset at the folder that "
                f"contains {csv_name} and the Img/ folder."
            )
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            img = self._prepare(os.path.join(dataset_dir, row["image"]))
            if img is not None:
                self.samples.setdefault(str(row["label"]), []).append(img)

        if not self.samples:
            raise RuntimeError("No images could be loaded from the dataset.")
        print(f"Loaded {sum(len(v) for v in self.samples.values())} images "
              f"for {len(self.samples)} characters.")

    def _prepare(self, path):
        """Grayscale, make the paper white, trim empty sides, fix the height."""
        try:
            img = Image.open(path).convert("L")
        except (OSError, ValueError):
            return None
        img.thumbnail((300, 300))  # speeds everything up
        g = np.asarray(img, dtype=np.float32)
        if g.mean() < 127:         # dark background -> flip so paper is light
            g = 255.0 - g
        bg = float(np.percentile(g, 90))
        ink = g < bg - 60
        cols = np.where(ink.any(axis=0))[0]
        if len(cols) == 0:
            return None
        pad = 4
        x0 = max(int(cols[0]) - pad, 0)
        x1 = min(int(cols[-1]) + pad + 1, g.shape[1])
        frame = np.clip(g[:, x0:x1] * (255.0 / max(bg, 1.0)), 0, 255)
        # Only the sides are trimmed; the full height is kept so small letters
        # and tall letters keep their natural relative size and position.
        frame_img = Image.fromarray(frame.astype(np.uint8))
        new_w = max(1, round(frame_img.width * self.char_height / frame_img.height))
        return frame_img.resize((new_w, self.char_height), Image.Resampling.LANCZOS)

    def supported_chars(self):
        return "".join(sorted(self.samples))

    # ---------------------------------------------------------------- render
    def _glyph(self, ch, jitter):
        options = self.samples.get(ch)
        if not options:
            return None  # character not in the dataset (punctuation etc.)
        img = self.rng.choice(options)
        dy = 0
        if jitter:
            scale = self.rng.uniform(0.9, 1.1)
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))),
                             Image.Resampling.BICUBIC)
            img = img.rotate(self.rng.uniform(-4, 4), resample=Image.Resampling.BICUBIC,
                             expand=True, fillcolor=255)
            dy = self.rng.randint(-4, 4)
        return img, dy

    @staticmethod
    def _paste_dark(canvas, glyph, x, y):
        """Paste so the darker pixel wins (white paper never erases ink)."""
        x0, y0 = max(x, 0), max(y, 0)
        x1 = min(x + glyph.width, canvas.width)
        y1 = min(y + glyph.height, canvas.height)
        if x1 <= x0 or y1 <= y0:
            return
        part = glyph.crop((x0 - x, y0 - y, x1 - x, y1 - y))
        region = canvas.crop((x0, y0, x1, y1))
        canvas.paste(ImageChops.darker(region, part), (x0, y0))

    def render(self, text, line_width=1000, jitter=True, ruled=False):
        margin = 30
        line_h = int(self.char_height * 1.5)
        space_w = int(self.char_height * 0.5)
        max_w = line_width - 2 * margin

        # 1) turn every word into a list of glyphs
        paragraphs = []
        for para in text.split("\n"):
            words = []
            for word in para.split(" "):
                glyphs = [g for g in (self._glyph(c, jitter) for c in word) if g]
                if glyphs:
                    words.append(glyphs)
            paragraphs.append(words)

        # 2) place words on lines, wrapping when a line is full
        lines = []
        for words in paragraphs:
            x, line = 0, []
            for glyphs in words:
                est = sum(g.width + 2 for g, _ in glyphs)
                if line and x + est > max_w:
                    lines.append(line)
                    x, line = 0, []
                for g, dy in glyphs:
                    line.append((x, g, dy))
                    x += g.width + self.rng.randint(-2, 4)
                x += space_w
            lines.append(line)  # an empty list is simply a blank line

        # 3) draw
        height = 2 * margin + line_h * max(len(lines), 1)
        canvas = Image.new("L", (line_width, height), 255)
        if ruled:
            draw = ImageDraw.Draw(canvas)
            for i in range(len(lines)):
                y = margin + i * line_h + line_h - 8
                draw.line([(margin // 2, y), (line_width - margin // 2, y)], fill=225)
        for i, line in enumerate(lines):
            for x, g, dy in line:
                y = margin + i * line_h + (line_h - g.height) // 2 + dy
                self._paste_dark(canvas, g, margin + x, y)
        return canvas
