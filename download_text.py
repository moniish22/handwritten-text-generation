"""Download a small public text file (Tiny Shakespeare, ~1 MB) to train on.

    python download_text.py

You can use any other plain-text file instead: just save it as data/input.txt.
"""
import os
import urllib.request

URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/master/"
       "data/tinyshakespeare/input.txt")
DEST = os.path.join("data", "input.txt")

os.makedirs("data", exist_ok=True)
try:
    urllib.request.urlretrieve(URL, DEST)
    print(f"Saved {DEST} ({os.path.getsize(DEST):,} bytes)")
except Exception as e:  # network problems, blocked site, etc.
    print(f"Download failed: {e}")
    print(f"Open {URL} in your browser, save the page as {DEST}, and continue.")
