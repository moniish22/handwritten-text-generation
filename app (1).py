"""Web app: type a starting phrase, the RNN continues it, and it is drawn as
handwriting.

Run locally:   streamlit run app.py
Deploy free:   Streamlit Community Cloud (see README, Step 7)
"""
import io
import os

import streamlit as st

from handwriting import HandwritingRenderer
from textutils import clean_text

MODEL_PATH = "model.pt"
BANK_PATH = "glyphs.npz"

st.set_page_config(page_title="Handwriting Generator", page_icon="✍️", layout="wide")


@st.cache_resource(show_spinner="Loading handwriting samples...")
def get_bank_path():
    return BANK_PATH if os.path.exists(BANK_PATH) else None


@st.cache_resource(show_spinner="Loading the RNN...")
def get_rnn():
    if not os.path.exists(MODEL_PATH):
        return None
    from generate import load_model  # imported here so torch loads only if needed
    return load_model(MODEL_PATH, "cpu")


def draw(text, seed, width, ruled):
    renderer = HandwritingRenderer(bank=BANK_PATH, seed=seed)
    return renderer.render(text, line_width=width, ruled=ruled)


def show_result(text, seed, width, ruled, key):
    image = draw(text, seed, width, ruled)
    st.image(image, use_container_width=True)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    # `key` must be different for each tab, or Streamlit raises DuplicateElementId
    st.download_button("Download as PNG", buf.getvalue(), "handwriting.png",
                       "image/png", key=f"download_{key}")


st.title("✍️ Handwriting Generator")
st.caption("A character-level RNN writes new text, then real handwritten "
           "letters are used to draw it.")

if get_bank_path() is None:
    st.error("glyphs.npz is missing. Run `python build_glyph_bank.py` once and "
             "add the file to the project (see README).")
    st.stop()

with st.sidebar:
    st.header("Settings")
    width = st.slider("Page width (pixels)", 600, 1600, 1000, 50)
    ruled = st.checkbox("Notebook lines", value=True)
    seed = st.number_input("Random seed (same seed = same look)", 0, 99999, 42)

tab_ai, tab_own = st.tabs(["🤖 AI writes for me", "⌨️ I type my own text"])

with tab_ai:
    rnn = get_rnn()
    if rnn is None:
        st.warning("model.pt not found. Train the model first (README, Step 4) "
                   "and keep model.pt in the project folder.")
    else:
        c1, c2, c3 = st.columns([2, 1, 1])
        prime = c1.text_input("Start the text with", "The ")
        length = c2.slider("Length (characters)", 50, 800, 300, 50)
        temp = c3.slider("Creativity", 0.3, 1.3, 0.8, 0.1,
                         help="Low = safe and repetitive, high = creative but messy")
        if st.button("Generate", type="primary"):
            from generate import generate_text
            import torch
            torch.manual_seed(int(seed))
            model, chars = rnn
            with st.spinner("Writing..."):
                st.session_state["ai_text"] = generate_text(
                    model, chars, prime, length, temp, "cpu")
        if "ai_text" in st.session_state:
            with st.expander("Generated text", expanded=False):
                st.text(st.session_state["ai_text"])
            show_result(st.session_state["ai_text"], int(seed), width, ruled, "ai")

with tab_own:
    own = st.text_area("Type something (letters and digits are drawn; other "
                       "symbols become spaces)", "Hello World 2026", height=120)
    if st.button("Write it by hand", type="primary"):
        st.session_state["own_text"] = clean_text(own)
    if "own_text" in st.session_state:
        show_result(st.session_state["own_text"], int(seed), width, ruled, "own")
