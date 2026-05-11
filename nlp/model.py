import torch
import torch.nn as nn

class WardrobeNLPModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_occasions: int, num_weather: int):
        """
        Model Architecture:
        - Embedding Layer: Converts token indices to dense vectors.
        - LSTM Layer: Processes the sequence of embeddings.
        - Classifier Heads: Two separate linear layers for multi-task learning 
                            (predicting both Occasion and Weather simultaneously).
        """
        super(WardrobeNLPModel, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        
        # We use a simple LSTM, batch_first=True so input is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Classifier heads
        self.occasion_head = nn.Linear(hidden_dim, num_occasions)
        self.weather_head = nn.Linear(hidden_dim, num_weather)
        
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embed_dim)
        
        # lstm_out: (batch_size, seq_len, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # We use the final hidden state of the LSTM to make predictions
        # hidden[-1] gives the hidden state of the last layer (batch_size, hidden_dim)
        final_hidden = hidden[-1] 
        
        occasion_logits = self.occasion_head(final_hidden) # (batch_size, num_occasions)
        weather_logits = self.weather_head(final_hidden)   # (batch_size, num_weather)
        
        return occasion_logits, weather_logits
