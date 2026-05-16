"""
model.py — DistilBERT-based multi-task outfit intent classifier.

Architecture
------------
  DistilBertModel (pretrained, 6 transformer layers)
      ↓
  [CLS] hidden state  (768-dim)
      ↓
  Dropout(0.3)
      ↓
  ┌──────────────┬──────────────┬──────────────┐
  │ occasion_head│ weather_head │  style_head  │
  │  Linear→6   │  Linear→3   │  Linear→6   │
  └──────────────┴──────────────┴──────────────┘

Future-ready design
-------------------
  New heads (color, season, mood, NER, etc.) can be registered at any time
  via `model.add_head(name, num_classes)` without touching the encoder.
  Registered heads are saved/loaded automatically with the model state_dict.

Training strategy
-----------------
  Phase 1 — freeze bottom N layers, train heads + top layers (lr ≈ 2e-5)
  Phase 2 — unfreeze all layers, full fine-tune with lower lr (≈ 5e-6)

Backward compatibility
----------------------
  The legacy WardrobeNLPModel (BiLSTM) is kept at the bottom of this file
  so that existing code that imports it still works.
"""

import torch
import torch.nn as nn

# ── DistilBERT (primary model) ─────────────────────────────────────────────────

try:
    from transformers import DistilBertModel
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class DistilBertMultiTaskClassifier(nn.Module):
    """
    Multi-task intent classifier built on DistilBERT.

    Parameters
    ----------
    num_occasions : int   — number of occasion classes
    num_weather   : int   — number of weather classes
    num_styles    : int   — number of style classes
    dropout       : float — dropout probability before each head
    pretrained    : str   — HuggingFace model identifier
    """

    HIDDEN_DIM = 768          # DistilBERT hidden size (fixed)
    DEFAULT_PRETRAINED = "distilbert-base-uncased"

    def __init__(
        self,
        num_intents: int,
        num_occasions: int,
        num_weather: int,
        num_styles: int,
        dropout: float = 0.3,
        pretrained: str = DEFAULT_PRETRAINED,
    ):
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is required. Run: pip install transformers>=4.35.0"
            )
        super().__init__()

        self.encoder     = DistilBertModel.from_pretrained(pretrained)
        self.dropout     = nn.Dropout(dropout)

        # Standard classification heads
        self.intent_head   = nn.Linear(self.HIDDEN_DIM, num_intents)
        self.occasion_head = nn.Linear(self.HIDDEN_DIM, num_occasions)
        self.weather_head  = nn.Linear(self.HIDDEN_DIM, num_weather)
        self.style_head    = nn.Linear(self.HIDDEN_DIM, num_styles)

        # Extension registry — add new heads without touching the encoder
        # e.g.: model.add_head("color", 8) or model.add_head("mood", 5)
        self._extra_heads = nn.ModuleDict()

    # ── Extensibility ──────────────────────────────────────────────────────────

    def add_head(self, name: str, num_classes: int) -> None:
        """
        Register a new classification head at runtime.

        The head is immediately available via `forward_head()` and is included
        in the model's `state_dict` for saving/loading.

        Example
        -------
        model.add_head("color", 8)   # adds an 8-class color head
        model.add_head("mood", 5)    # adds a 5-class mood head
        """
        self._extra_heads[name] = nn.Linear(self.HIDDEN_DIM, num_classes)

    def forward_head(self, name: str, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
        """Run a single registered extra head and return its logits."""
        cls = self._encode(input_ids, attention_mask)
        return self._extra_heads[name](cls)

    # ── Layer freeze / unfreeze ────────────────────────────────────────────────

    def freeze_bottom_layers(self, n: int = 4) -> None:
        """
        Freeze the bottom N transformer layers of the encoder.
        Used in Phase 1 of training to protect general language representations
        and train only the task-specific layers + heads.
        """
        for i, layer in enumerate(self.encoder.transformer.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_all_encoder(self) -> None:
        """
        Unfreeze every encoder parameter.
        Used in Phase 2 of training for full fine-tuning with a lower LR.
        """
        for param in self.encoder.parameters():
            param.requires_grad = True

    def trainable_params(self) -> int:
        """Return the count of parameters that will be updated by the optimizer."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Core forward ───────────────────────────────────────────────────────────

    def _encode(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode input and return the [CLS] token representation."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state[:, 0, :])   # (B, 768)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        input_ids      : LongTensor (batch, seq_len)
        attention_mask : LongTensor (batch, seq_len)

        Returns
        -------
        (intent_logits, occasion_logits, weather_logits, style_logits)
        each shaped (batch, num_classes).
        """
        cls = self._encode(input_ids, attention_mask)
        return (
            self.intent_head(cls),
            self.occasion_head(cls),
            self.weather_head(cls),
            self.style_head(cls),
        )


# ── Legacy BiLSTM (kept for backward compatibility) ───────────────────────────

class WardrobeNLPModel(nn.Module):
    """
    Original BiLSTM multi-task classifier.
    Kept for backward compatibility — the DistilBERT model is preferred.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 num_occasions, num_weather, num_styles,
                 num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.lstm_drop = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.occasion_head = nn.Linear(out_dim, num_occasions)
        self.weather_head  = nn.Linear(out_dim, num_weather)
        self.style_head    = nn.Linear(out_dim, num_styles)

    def forward(self, input_ids):
        x = self.embed_drop(self.embedding(input_ids))
        out, _ = self.lstm(x)
        h = self.lstm_drop(out.mean(dim=1))
        return self.occasion_head(h), self.weather_head(h), self.style_head(h)
