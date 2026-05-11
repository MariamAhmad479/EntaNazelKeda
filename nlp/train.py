import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import get_dataloaders, OCCASIONS, WEATHER_CLASSES
from nlp.model import WardrobeNLPModel

def train_model(epochs=10, batch_size=32, lr=0.001):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    print("Generating data and building vocabulary...")
    train_loader, val_loader, vocab = get_dataloaders(num_samples=2000, batch_size=batch_size)

    # Initialize model
    model = WardrobeNLPModel(
        vocab_size=vocab.vocab_size,
        embed_dim=64,
        hidden_dim=128,
        num_occasions=len(OCCASIONS),
        num_weather=len(WEATHER_CLASSES)
    ).to(device)

    # Loss and Optimizer
    criterion_occasion = nn.CrossEntropyLoss()
    criterion_weather = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_occ = 0
        correct_wea = 0
        total_samples = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            occ_label = batch["occasion_label"].to(device)
            wea_label = batch["weather_label"].to(device)

            optimizer.zero_grad()

            occ_logits, wea_logits = model(input_ids)

            loss_occ = criterion_occasion(occ_logits, occ_label)
            loss_wea = criterion_weather(wea_logits, wea_label)
            loss = loss_occ + loss_wea

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Accuracy calc
            _, predicted_occ = torch.max(occ_logits, 1)
            _, predicted_wea = torch.max(wea_logits, 1)
            correct_occ += (predicted_occ == occ_label).sum().item()
            correct_wea += (predicted_wea == wea_label).sum().item()
            total_samples += input_ids.size(0)

        train_occ_acc = correct_occ / total_samples
        train_wea_acc = correct_wea / total_samples
        
        # Validation Loop
        model.eval()
        val_loss = 0
        val_correct_occ = 0
        val_correct_wea = 0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                occ_label = batch["occasion_label"].to(device)
                wea_label = batch["weather_label"].to(device)

                occ_logits, wea_logits = model(input_ids)

                loss_occ = criterion_occasion(occ_logits, occ_label)
                loss_wea = criterion_weather(wea_logits, wea_label)
                loss = loss_occ + loss_wea

                val_loss += loss.item()
                
                _, predicted_occ = torch.max(occ_logits, 1)
                _, predicted_wea = torch.max(wea_logits, 1)
                val_correct_occ += (predicted_occ == occ_label).sum().item()
                val_correct_wea += (predicted_wea == wea_label).sum().item()
                val_samples += input_ids.size(0)

        val_occ_acc = val_correct_occ / val_samples
        val_wea_acc = val_correct_wea / val_samples

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {total_loss/len(train_loader):.4f} "
              f"Train Acc (Occ/Wea): {train_occ_acc:.2f}/{train_wea_acc:.2f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
              f"Val Acc (Occ/Wea): {val_occ_acc:.2f}/{val_wea_acc:.2f}")

    # Save the model and vocabulary
    os.makedirs('nlp/saved_models', exist_ok=True)
    torch.save(model.state_dict(), 'nlp/saved_models/nlp_model.pth')
    with open('nlp/saved_models/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Model and vocabulary saved in 'nlp/saved_models/'")

if __name__ == "__main__":
    train_model(epochs=15)
