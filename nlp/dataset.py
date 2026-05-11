import json
import random
import re
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader

OCCASIONS = ["casual", "formal", "business", "sport", "party", "outdoor"]
WEATHER_CLASSES = ["hot", "mild", "cold"]

# Synonyms for occasions and weather to expand vocabulary
OCCASION_SYNONYMS = {
    "casual": ["casual", "hangout", "mall", "everyday", "relaxing", "chill"],
    "formal": ["formal", "wedding", "black tie", "fancy", "elegant", "ceremony"],
    "business": ["business", "office", "work", "interview", "meeting", "job", "corporate"],
    "sport": ["sport", "gym", "workout", "running", "athletic", "exercise", "pool", "swim"],
    "party": ["party", "club", "dinner", "date", "night out", "celebration", "gathering"],
    "outdoor": ["outdoor", "hiking", "camping", "park", "picnic", "outside"]
}

WEATHER_SYNONYMS = {
    "hot": ["hot", "warm", "sunny", "boiling", "summer", "40 degrees", "sweating", "heat"],
    "mild": ["mild", "nice", "pleasant", "spring", "cool", "breezy", "perfect", "okay"],
    "cold": ["cold", "freezing", "chilly", "winter", "snow", "frosty", "10 degrees", "shivering"]
}

# Expanded Templates
TEMPLATES = [
    "I need an outfit for a {occasion_word} event, it's {weather_word} outside.",
    "What should I wear for {occasion_word}? The weather is {weather_word}.",
    "Give me some clothes for a {occasion_word} occasion, it is quite {weather_word}.",
    "I am going to a {occasion_word} and it feels {weather_word}.",
    "Suggest an outfit. It is {weather_word} and I'm going to a {occasion_word} place.",
    "Looking for {occasion_word} clothing suitable for {weather_word} weather.",
    "Need {occasion_word} wear for {weather_word} conditions.",
    "I want an outfit for an indoor {occasion_word}, the temperature is {weather_word}.",
    "No it's {weather_word}.",
    "What do you think I should wear to the {occasion_word}?",
    "I want to wear something {occasion_word} for a {occasion_word}.",
    "What do I wear to an {occasion_word}?",
    "What do I wear to a {occasion_word} to the {occasion_word}?"
]

def generate_synthetic_data(num_samples: int = 4000) -> List[Dict[str, str]]:
    """Generates synthetic training data using synonyms to increase vocabulary."""
    data = []
    for _ in range(num_samples):
        # Pick actual classes
        occasion_class = random.choice(OCCASIONS)
        weather_class = random.choice(WEATHER_CLASSES)
        
        # Pick synonyms representing those classes
        occasion_word = random.choice(OCCASION_SYNONYMS[occasion_class])
        weather_word = random.choice(WEATHER_SYNONYMS[weather_class])
        
        template = random.choice(TEMPLATES)
        # Randomly decide to drop weather or occasion to make model robust to partial info
        text = template.format(occasion_word=occasion_word, weather_word=weather_word)
        
        data.append({
            "text": text,
            "occasion": occasion_class,
            "weather": weather_class
        })
    return data

def preprocess_text(text: str) -> List[str]:
    """Simple tokenization: lowercasing and splitting by non-word characters."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, texts: List[str]):
        for text in texts:
            tokens = preprocess_text(text)
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        tokens = preprocess_text(text)
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

class WardrobeQueryDataset(Dataset):
    def __init__(self, data: List[Dict[str, str]], vocab: Vocabulary, max_len: int = 20):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

        self.occasion2idx = {occ: idx for idx, occ in enumerate(OCCASIONS)}
        self.weather2idx = {wea: idx for idx, wea in enumerate(WEATHER_CLASSES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_encoded = self.vocab.encode(item["text"])
        
        # Padding or truncating
        if len(text_encoded) < self.max_len:
            text_encoded += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(text_encoded))
        else:
            text_encoded = text_encoded[:self.max_len]

        occ_label = self.occasion2idx[item["occasion"]]
        wea_label = self.weather2idx[item["weather"]]

        return {
            "input_ids": torch.tensor(text_encoded, dtype=torch.long),
            "occasion_label": torch.tensor(occ_label, dtype=torch.long),
            "weather_label": torch.tensor(wea_label, dtype=torch.long)
        }

def get_dataloaders(num_samples=1000, batch_size=32, max_len=20):
    raw_data = generate_synthetic_data(num_samples)
    
    # Split 80/20 train/val
    train_size = int(0.8 * num_samples)
    train_data = raw_data[:train_size]
    val_data = raw_data[train_size:]
    
    vocab = Vocabulary()
    vocab.build_vocab([item["text"] for item in train_data])
    
    train_dataset = WardrobeQueryDataset(train_data, vocab, max_len)
    val_dataset = WardrobeQueryDataset(val_data, vocab, max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, vocab

if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders(num_samples=10)
    for batch in train_loader:
        print(batch)
        break
