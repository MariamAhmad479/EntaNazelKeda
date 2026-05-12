import json
import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import (
    get_dataloaders,
    OCCASIONS, WEATHER_CLASSES, STYLES,
    OCCASION_UNKNOWN_IDX, WEATHER_UNKNOWN_IDX, STYLE_UNKNOWN_IDX,
)
from nlp.model import WardrobeNLPModel


def masked_correct(logits, labels, unknown_idx):
    """Return (num_correct, num_valid) ignoring samples labelled as unknown."""
    mask = labels != unknown_idx
    if not mask.any():
        return 0, 0
    pred = logits[mask].argmax(dim=1)
    return (pred == labels[mask]).sum().item(), mask.sum().item()


def train_model(epochs=20, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Generating data and building vocabulary...")
    train_loader, val_loader, vocab = get_dataloaders(num_samples=20000, batch_size=batch_size)

    model = WardrobeNLPModel(
        vocab_size=vocab.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_occasions=len(OCCASIONS),
        num_weather=len(WEATHER_CLASSES),
        num_styles=len(STYLES),
    ).to(device)

    # ignore_index skips samples where a dimension was absent from the text
    criterion_occasion = nn.CrossEntropyLoss(ignore_index=OCCASION_UNKNOWN_IDX)
    criterion_weather  = nn.CrossEntropyLoss(ignore_index=WEATHER_UNKNOWN_IDX)
    criterion_style    = nn.CrossEntropyLoss(ignore_index=STYLE_UNKNOWN_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        occ_correct = occ_total = 0
        wea_correct = wea_total = 0
        sty_correct = sty_total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            occ_label = batch["occasion_label"].to(device)
            wea_label = batch["weather_label"].to(device)
            sty_label = batch["style_label"].to(device)

            optimizer.zero_grad()
            occ_logits, wea_logits, sty_logits = model(input_ids)

            loss = (criterion_occasion(occ_logits, occ_label)
                  + criterion_weather(wea_logits, wea_label)
                  + criterion_style(sty_logits, sty_label))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            c, n = masked_correct(occ_logits, occ_label, OCCASION_UNKNOWN_IDX)
            occ_correct += c; occ_total += n
            c, n = masked_correct(wea_logits, wea_label, WEATHER_UNKNOWN_IDX)
            wea_correct += c; wea_total += n
            c, n = masked_correct(sty_logits, sty_label, STYLE_UNKNOWN_IDX)
            sty_correct += c; sty_total += n

        train_occ = occ_correct / occ_total if occ_total else 0.0
        train_wea = wea_correct / wea_total if wea_total else 0.0
        train_sty = sty_correct / sty_total if sty_total else 0.0

        # Validation
        model.eval()
        val_loss = 0
        vocc_correct = vocc_total = 0
        vwea_correct = vwea_total = 0
        vsty_correct = vsty_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                occ_label = batch["occasion_label"].to(device)
                wea_label = batch["weather_label"].to(device)
                sty_label = batch["style_label"].to(device)

                occ_logits, wea_logits, sty_logits = model(input_ids)
                val_loss += (criterion_occasion(occ_logits, occ_label)
                           + criterion_weather(wea_logits, wea_label)
                           + criterion_style(sty_logits, sty_label)).item()

                c, n = masked_correct(occ_logits, occ_label, OCCASION_UNKNOWN_IDX)
                vocc_correct += c; vocc_total += n
                c, n = masked_correct(wea_logits, wea_label, WEATHER_UNKNOWN_IDX)
                vwea_correct += c; vwea_total += n
                c, n = masked_correct(sty_logits, sty_label, STYLE_UNKNOWN_IDX)
                vsty_correct += c; vsty_total += n

        val_occ = vocc_correct / vocc_total if vocc_total else 0.0
        val_wea = vwea_correct / vwea_total if vwea_total else 0.0
        val_sty = vsty_correct / vsty_total if vsty_total else 0.0

        print(
            f"Epoch [{epoch+1:2d}/{epochs}] "
            f"Loss: {total_loss/len(train_loader):.4f} "
            f"Train Acc (Occ/Wea/Sty): {train_occ:.2f}/{train_wea:.2f}/{train_sty:.2f} | "
            f"Val Loss: {val_loss/len(val_loader):.4f} "
            f"Val Acc (Occ/Wea/Sty): {val_occ:.2f}/{val_wea:.2f}/{val_sty:.2f}"
        )

    # Save weights, vocabulary, and config (so inference never hardcodes dims)
    os.makedirs('nlp/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'nlp/saved_models/nlp_model.pth')
    with open('nlp/saved_models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    model_config = {
        "vocab_size":    vocab.vocab_size,
        "embed_dim":     64,
        "hidden_dim":    128,
        "num_occasions": len(OCCASIONS),
        "num_weather":   len(WEATHER_CLASSES),
        "num_styles":    len(STYLES),
    }
    with open('nlp/saved_models/model_config.json', 'w') as f:
        json.dump(model_config, f, indent=2)

    print("Saved: nlp_model.pth, vocab.pkl, model_config.json")


if __name__ == "__main__":
    train_model(epochs=20)
