# Handwritten-Style Text Generation with a Character-Level RNN

This project does two things:

1. **Learns to write.** A character-level RNN (an LSTM, built with PyTorch) reads a
   text file one character at a time and learns which character tends to come next.
   Afterwards it can invent brand-new text in the same style.
2. **Draws it as handwriting.** Every generated character is replaced by a real
   handwritten photo of that character from the *English Handwritten Characters*
   dataset (55 photos for each of a-z, A-Z, 0-9), with small random changes in size,
   tilt and position so no two letters look identical.

```
text file  ->  [ char-RNN trains ]  ->  generates new text  ->  [ handwriting.py ]  ->  handwriting.png
(data/input.txt)       train.py              generate.py          draws each letter from your images
```

## Why the project is built this way

The dataset is a set of **images of single characters**, so it cannot teach a model
how words and sentences flow. A model that maps one character to one image (the
approach in the original Kaggle notebook) ends up outputting a blurry average
picture of each character. Splitting the job fixes this: the RNN learns language
from text, and the real photos supply the handwriting. The output looks sharp
because it uses real pen strokes.

## Project layout

```
char-rnn-handwriting/
├── data/
│   ├── input.txt                                  <- text to train the RNN (Step 3)
│   └── english-handwritten-characters-dataset/    <- your Kaggle dataset (Step 3)
│       ├── english.csv
│       └── Img/ (3410 png files)
├── model.py            the neural network
├── train.py            trains the RNN on data/input.txt
├── generate.py         generates text (and draws it with --dataset)
├── handwriting.py      turns text into a handwriting image
├── textutils.py        cleans text so it only has letters/digits/spaces
├── download_text.py    downloads a sample text file for training
├── requirements.txt    packages to install
└── .gitignore
```

## Step 1: Install Python

Download Python 3.9 or newer from https://www.python.org/downloads/ . On Windows,
tick **"Add Python to PATH"** during installation. Check it works:

```bash
python --version
```

## Step 2: Create a virtual environment and install packages

A virtual environment keeps this project's packages separate from everything else.
Open a terminal **inside the project folder** and run:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows (Command Prompt / PowerShell)
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

You should now see `(venv)` at the start of your terminal line. Install the packages:

```bash
pip install -r requirements.txt
```

## Step 3: Get the data (two things)

**A. The handwriting images (you already use this on Kaggle).**
Search for "English Handwritten Characters Dataset" on kaggle.com, click
**Download**, and unzip it. Move the contents into
`data/english-handwritten-characters-dataset/` so that this path exists:

```
data/english-handwritten-characters-dataset/english.csv
data/english-handwritten-characters-dataset/Img/img001-001.png
```

**B. A text file for the RNN to learn from.** The images contain no sentences, so
the RNN needs ordinary text. Either run:

```bash
python download_text.py
```

(saves Tiny Shakespeare, about 1 MB, as `data/input.txt`), or save any plain-text
file of your own as `data/input.txt` (books from gutenberg.org work well; more text
means better results, 500,000+ characters is a good target). Your text is cleaned
automatically: only letters, digits, spaces and new lines are kept, because those are
the characters your dataset can draw.

## Step 4: Train the RNN

```bash
python train.py --data data/input.txt --epochs 20
```

What you will see: a line per epoch such as `Epoch 3/20 | train 1.62 | val 1.65`.
These numbers are the loss (the model's average mistake). **They should go down.**
Whenever the validation loss improves, the model is saved to `model.pt`.

Useful options:

| Option | Meaning | Default |
|---|---|---|
| `--epochs` | passes over the data | 20 |
| `--hidden_size` | network size (bigger = smarter but slower) | 256 |
| `--num_layers` | stacked LSTM layers | 2 |
| `--seq_len` | characters the model looks at per training window | 100 |
| `--batch_size` | windows per training step | 64 |

On a slow CPU, start small: `python train.py --epochs 5 --hidden_size 128`.
For speed, use a free GPU on Kaggle or Google Colab (see the box below).

## Step 5: Generate handwritten text

```bash
python generate.py --prime "The " --length 400 \
    --dataset data/english-handwritten-characters-dataset --image_out my_text.png
```

(On Windows Command Prompt, put it all on one line and drop the `\`.)

This prints the generated text and saves `my_text.png`. Useful options:

- `--prime "Once upon "` : the text the model continues from.
- `--temperature 0.5` safe/repetitive, `1.2` creative/messy (default 0.8).
- `--ruled` : draws notebook lines on the page.
- `--seed 42` : makes the result repeatable.
- `--line_width 1400` : wider page.

You can also draw any text you like without the RNN:

```python
from handwriting import HandwritingRenderer
r = HandwritingRenderer("data/english-handwritten-characters-dataset")
r.render("Hello World 2026").save("hello.png")
```

> **Running on Kaggle / Colab instead of your computer**
> Upload the project files, turn on a GPU (Kaggle: Settings > Accelerator; Colab:
> Runtime > Change runtime type), and use the dataset path
> `/kaggle/input/english-handwritten-characters-dataset` for `--dataset`. Kaggle
> notebooks need "Internet" switched on to run `download_text.py`.

## Step 6: Upload to GitHub

1. **Create a GitHub account** at https://github.com/signup .
2. **Install Git** from https://git-scm.com/downloads , then tell it who you are
   (once per computer):
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   ```
3. **Create the repository on GitHub:** click the **+** (top right) > **New repository**.
   Name it `char-rnn-handwriting`, choose Public, and leave "Add a README" and
   ".gitignore" **unchecked** (you already have them). Click **Create repository**.
4. **Push your code.** In your project folder (venv does not need to be active):
   ```bash
   git init
   git add .
   git commit -m "Character-level RNN for handwritten text generation"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/char-rnn-handwriting.git
   git push -u origin main
   ```
5. **Logging in:** GitHub no longer accepts your account password in the terminal.
   Create a token at GitHub > Settings > Developer settings > Personal access tokens >
   Tokens (classic) > Generate new token, tick **repo**, copy it, and paste it when the
   terminal asks for a password. (The GitHub Desktop app is an easier alternative:
   File > Add local repository > Publish.)
6. **Check:** refresh your repository page; you should see all the files.

What gets uploaded: `.gitignore` deliberately skips `venv/`, trained `*.pt` weights, and
the Kaggle image folder (large, and the dataset belongs to its authors). Anyone
cloning your repo can download the dataset themselves using Step 3.

**Update later** after changing code:
```bash
git add .
git commit -m "Describe what you changed"
git push
```

**Show off your result:** save a nice `my_text.png` in the repo folder, and add
`![sample](my_text.png)` to this README, then commit and push.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip` or `python` not found | Reinstall Python and tick "Add to PATH" (Windows), or try `python3` / `pip3` |
| `ModuleNotFoundError: torch` | Activate the venv (Step 2) and run `pip install -r requirements.txt` |
| `Could not find .../english.csv` | `--dataset` must point at the folder that directly contains `english.csv` |
| `Dataset is too small` | Use a bigger text file or lower `--seq_len` |
| Output is gibberish | Train longer (more `--epochs`) or use more text |
| Output repeats the same words | Raise `--temperature` to about 1.0 |
| Training is very slow | Lower `--hidden_size`, or use a Kaggle/Colab GPU |

## How it works (short version)

1. Each unique character in the text gets a number.
2. The text is cut into windows of `seq_len` characters. For each window the answer
   the model must predict is the same window shifted one character to the left.
3. An embedding layer turns each character number into a vector; LSTM layers carry
   a memory along the sequence; a final layer scores every possible next character.
4. To generate, the model scores the next character, one is picked at random according
   to those scores (temperature controls how adventurous), and it is fed back in.
5. `handwriting.py` trims each photo to the pen strokes, picks a random photo per
   letter, jitters it slightly, and places the letters in lines.

## Ideas to extend

- Train on your own writing so the *wording* matches you too.
- Add punctuation: photograph or draw your own `. , ! ?` samples and add them to
  `english.csv`.
- Go further with real pen-stroke generation (Alex Graves, "Generating Sequences With
  Recurrent Neural Networks") using the IAM On-Line handwriting dataset.
