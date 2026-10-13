"""Small helpers for preparing text."""
import re


def clean_text(text):
    """Keep only letters, digits, spaces and newlines.

    The handwriting dataset only contains a-z, A-Z and 0-9, so we train the
    RNN on text it can actually draw. Apostrophes are removed ("don't" ->
    "dont"); other punctuation becomes a space.
    """
    text = text.replace("\r", "")
    text = re.sub(r"['’]", "", text)
    text = "".join(
        c if (c.isascii() and (c.isalnum() or c in " \n")) else " " for c in text
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"
