import torch
import torch.nn as nn


class WardrobeNLPModel(nn.Module):
    """
    Shared LSTM encoder with three multi-task output heads:
      - occasion_head  : 6 classes (casual, formal, business, sport, party, outdoor)
      - weather_head   : 3 classes (hot, mild, cold)
      - style_head     : 6 classes (classic, streetwear, bohemian, minimalist, preppy, athletic)

    Samples where a dimension is absent from the text are trained with
    ignore_index in the loss, so the heads do not interfere with each other.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 num_occasions: int, num_weather: int, num_styles: int):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm         = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.occasion_head = nn.Linear(hidden_dim, num_occasions)
        self.weather_head  = nn.Linear(hidden_dim, num_weather)
        self.style_head    = nn.Linear(hidden_dim, num_styles)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        embedded = self.embedding(input_ids)          # (batch, seq_len, embed_dim)
        outputs, _ = self.lstm(embedded)              # outputs: (batch, seq_len, hidden_dim)
        # Mean pool so keywords anywhere in the sentence contribute equally
        h = outputs.mean(dim=1)                       # (batch, hidden_dim)
        return self.occasion_head(h), self.weather_head(h), self.style_head(h)
