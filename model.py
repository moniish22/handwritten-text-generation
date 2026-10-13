"""Character-level RNN (LSTM) model."""
import torch.nn as nn


class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=256,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len) of character ids
        out, hidden = self.rnn(self.embed(x), hidden)
        logits = self.fc(self.drop(out))  # (batch, seq_len, vocab_size)
        return logits, hidden
