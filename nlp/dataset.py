import random
import re
from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset, DataLoader

OCCASIONS = ["casual", "formal", "business", "sport", "party", "outdoor"]
WEATHER_CLASSES = ["hot", "mild", "cold"]
STYLES = ["classic", "streetwear", "bohemian", "minimalist", "preppy", "athletic"]

# Sentinel indices for "not mentioned in this text" — used with ignore_index in loss
OCCASION_UNKNOWN_IDX = len(OCCASIONS)       # 6
WEATHER_UNKNOWN_IDX  = len(WEATHER_CLASSES) # 3
STYLE_UNKNOWN_IDX    = len(STYLES)          # 6

OCCASION_SYNONYMS = {
    "casual":   ["casual", "hangout", "mall", "everyday", "relaxing", "chill",
                 "laid-back", "errands", "daytime", "informal", "cafe", "coffeeshop",
                 "coffee", "restaurant", "movies", "cinema", "brunch", "lunch",
                 "friends", "shopping", "stroll", "walk", "leisure", "sightseeing",
                 "outing", "picnic trip", "bookstore", "market", "farmers market",
                 "casual dinner", "get together", "rooftop", "terrace"],
    "formal":   ["formal", "wedding", "black tie", "fancy", "elegant", "ceremony",
                 "gala", "banquet", "black-tie", "prom", "graduation", "reception",
                 "charity dinner", "awards", "awards ceremony", "inauguration",
                 "charity gala", "formal event", "red carpet", "opera", "ballet"],
    "business": ["business", "office", "work", "interview", "meeting", "job",
                 "corporate", "conference", "presentation", "professional",
                 "client meeting", "networking", "seminar", "boardroom",
                 "sales pitch", "business lunch", "job fair", "workshop",
                 "team meeting", "annual review", "pitch", "startup event"],
    "sport":    ["sport", "gym", "workout", "running", "athletic", "exercise",
                 "pool", "swim", "training", "jogging", "yoga", "cycling",
                 "tennis", "basketball", "pilates", "spinning", "crossfit",
                 "football", "soccer", "volleyball", "golf", "boxing",
                 "climbing", "rowing", "marathon", "5k run", "bootcamp",
                 "fitness class", "aerobics", "weightlifting", "track"],
    "party":    ["party", "club", "dinner", "date", "night out", "celebration",
                 "gathering", "birthday", "nightclub", "evening", "cocktail party",
                 "house party", "mixer", "nightlife", "rave", "after party",
                 "birthday party", "festivity", "social event", "girls night",
                 "guys night", "rooftop party", "pool party"],
    "outdoor":  ["outdoor", "hiking", "camping", "park", "picnic", "outside",
                 "trail", "beach", "nature", "garden", "festival", "bbq",
                 "barbecue", "outdoor concert", "lakeside", "mountain",
                 "adventure", "road trip", "forest", "national park",
                 "countryside", "farm", "botanical garden", "waterfall",
                 "surfing", "kayaking", "rock climbing", "outdoor market"],
}

WEATHER_SYNONYMS = {
    "hot":  ["hot", "warm", "sunny", "boiling", "summer", "sweating", "heat",
             "scorching", "sweltering", "blazing", "humid", "tropical",
             "35 degrees", "38 degrees", "40 degrees", "over 30",
             "muggy", "steamy", "baking", "roasting", "heatwave",
             "melting", "very hot", "extremely hot", "really hot",
             "sweat", "90 degrees", "95 degrees", "stifling"],
    "mild": ["mild", "nice", "pleasant", "spring", "cool", "breezy", "windy",
             "perfect", "comfortable", "temperate", "moderate", "crisp", "refreshing",
             "20 degrees", "18 degrees", "15 degrees", "overcast", "cloudy",
             "partly cloudy", "foggy", "drizzle", "gusty", "changeable",
             "25 degrees", "22 degrees", "light wind", "cloudy day", "calm"],
    "cold": ["cold", "freezing", "chilly", "winter", "snow", "frosty", "shivering",
             "icy", "bitter", "frigid", "below zero", "sub-zero", "wintry",
             "5 degrees", "0 degrees", "minus", "arctic", "raw",
             "snowy", "blizzard", "sleet", "harsh", "bitter cold",
             "minus 10", "minus 5", "below freezing", "ice cold", "freezing cold"],
}

STYLE_SYNONYMS = {
    "classic":    ["classic", "traditional", "timeless", "elegant", "tailored",
                   "sophisticated", "refined", "polished", "conservative", "neat",
                   "dapper", "put-together", "sharp", "heritage", "gentlemanly",
                   "old school", "prim", "structured", "formal elegant"],
    "streetwear": ["streetwear", "urban", "street", "hype", "edgy", "trendy",
                   "graphic", "bold", "sneaker", "hypebeast",
                   "casual cool", "street style", "hype fashion", "skater",
                   "grunge", "off-duty cool", "hip hop", "urban style"],
    "bohemian":   ["bohemian", "boho", "hippie", "earthy", "flowy", "eclectic",
                   "artsy", "free-spirited", "vintage", "relaxed",
                   "festival style", "gypsy", "wanderlust", "hippie chic",
                   "earthy tones", "boho chic", "indie", "quirky"],
    "minimalist": ["minimalist", "minimal", "simple", "clean", "understated",
                   "basic", "sleek", "monochrome", "neutral", "modern",
                   "capsule", "zen", "stripped back", "less is more",
                   "muted", "low-key", "pared down", "monochromatic"],
    "preppy":     ["preppy", "prep", "collegiate", "ivy league", "smart casual",
                   "country club", "nautical", "polo", "chino", "clean-cut",
                   "blazer", "loafer", "button-down", "boat shoes",
                   "country style", "old money", "east coast", "classic preppy"],
    "athletic":   ["athletic", "sporty", "active", "fitness", "performance",
                   "workout", "running", "activewear", "gym wear", "sport style",
                   "athleisure", "tracksuit", "leggings", "tech wear",
                   "compression", "sportswear", "sports chic", "yoga wear"],
}

# Templates requiring BOTH occasion and weather
TEMPLATES_BOTH = [
    "I need an outfit for a {occasion_word} event, it's {weather_word} outside.",
    "What should I wear for {occasion_word}? The weather is {weather_word}.",
    "Give me some clothes for a {occasion_word} occasion, it is quite {weather_word}.",
    "I am going to a {occasion_word} and it feels {weather_word}.",
    "Suggest an outfit. It is {weather_word} and I'm going to a {occasion_word} place.",
    "Looking for {occasion_word} clothing suitable for {weather_word} weather.",
    "Need {occasion_word} wear for {weather_word} conditions.",
    "I want an outfit for an indoor {occasion_word}, the temperature is {weather_word}.",
    "Heading to a {occasion_word}, and the forecast says {weather_word}.",
    "Something for a {occasion_word} that works in {weather_word} weather, please.",
    "I have a {occasion_word} coming up and it's going to be {weather_word}.",
    "Planning for a {occasion_word} and the weather looks {weather_word}.",
    "Outfit for {occasion_word} in {weather_word} conditions?",
    "{occasion_word} session today, it's {weather_word}.",
    "Attending a {occasion_word} this evening, weather is {weather_word}.",
    "My {occasion_word} is in {weather_word} weather.",
    "Dressing for {occasion_word} today, it's {weather_word} outside.",
    "Help me dress for {occasion_word}, the weather is {weather_word}.",
    "I'll be at a {occasion_word}, it's {weather_word} out there.",
    "Dress me for {occasion_word} in {weather_word} weather.",
    "I'm heading to a {occasion_word}, it is {weather_word} today.",
    "What works for {occasion_word} when it's {weather_word}?",
    "I need clothes for {occasion_word} since it's {weather_word}.",
    "Going to {occasion_word}, weather is {weather_word}.",
    "For a {occasion_word} in {weather_word} conditions, what do you suggest?",
    "It's {weather_word} and I have a {occasion_word} planned.",
    "My {occasion_word} is tonight and the weather is {weather_word}.",
    "Help me pick something for {occasion_word}, it's {weather_word} out.",
    "I've got a {occasion_word} and it looks {weather_word} outside.",
    "What do I wear to {occasion_word}? It's {weather_word}.",
    "Preparing for {occasion_word}, the weather is {weather_word}.",
    "I need to look good for {occasion_word}, and it's {weather_word}.",
    "What's best for {occasion_word} given it's {weather_word}?",
    "Clothes for {occasion_word}, considering it's {weather_word}.",
    "Off to {occasion_word} today, it is {weather_word} out.",
    "I need a {occasion_word} outfit, it's {weather_word}.",
    "I want a {occasion_word} look, the weather is {weather_word}.",
    "Going to the {occasion_word}, it's {weather_word} outside.",
    "I need something {occasion_word}-appropriate, it's {weather_word}.",
]

# Templates requiring only occasion (weather label will be None)
TEMPLATES_OCCASION_ONLY = [
    "What do you think I should wear to the {occasion_word}?",
    "What do I wear to an {occasion_word}?",
    "Help me pick an outfit for {occasion_word}.",
    "What's appropriate to wear to a {occasion_word}?",
    "Got a {occasion_word} to attend, what should I wear?",
    "I need something to wear to a {occasion_word}.",
    "Outfit ideas for {occasion_word}?",
    "I'm doing {occasion_word} today, what do I wear?",
    "Dress code for {occasion_word}?",
    "{occasion_word} outfit suggestions?",
    "What should I wear for {occasion_word} today?",
    "I'm going to {occasion_word}, help me out.",
    "Need something for {occasion_word}.",
    "{occasion_word} today, what do I wear?",
    "Style tips for {occasion_word}?",
    "I have {occasion_word} later, any ideas?",
    "Heading to {occasion_word}, what outfit?",
    "Best outfit for {occasion_word}?",
    "I'll be at {occasion_word}, what should I put on?",
    "Going to {occasion_word} soon, suggest an outfit.",
    "I'm attending {occasion_word}, what to wear?",
    "Wardrobe ideas for {occasion_word}?",
    "What's the right outfit for {occasion_word}?",
    "Clothes for {occasion_word}?",
    "What to wear to {occasion_word}?",
    "I need a {occasion_word} outfit.",
    "I want a {occasion_word} look.",
    "Give me a {occasion_word} outfit.",
    "Going to the {occasion_word}.",
    "Off to {occasion_word}.",
]

# Templates requiring only weather (occasion label will be None)
TEMPLATES_WEATHER_ONLY = [
    "No it's {weather_word}.",
    "It's {weather_word} outside, what should I wear?",
    "The weather is {weather_word} today.",
    "It feels {weather_word}, any outfit suggestions?",
    "Just to clarify, it's {weather_word}.",
    "It's {weather_word} weather today, what should I put on?",
    "What do I wear in {weather_word} weather?",
    "The weather is {weather_word}, help me dress.",
    "{weather_word} outside today.",
    "It's so {weather_word} right now.",
    "Weather update: it's {weather_word}.",
    "It is {weather_word}, what should I wear?",
    "Dress me for {weather_word} weather.",
    "It's pretty {weather_word} out there.",
    "Today is {weather_word}, outfit ideas?",
]

# Templates requiring style + occasion + weather
TEMPLATES_STYLE_BOTH = [
    "I want a {style_word} outfit for {occasion_word}, it's {weather_word}.",
    "Looking for something {style_word} to wear to a {occasion_word} in {weather_word} weather.",
    "Can you suggest a {style_word} look for a {occasion_word} event? It's {weather_word}.",
    "I'm into {style_word} fashion. Need something for {occasion_word}, weather is {weather_word}.",
    "I prefer {style_word} looks. Need something for {occasion_word}, it's {weather_word}.",
    "Help me find a {style_word} outfit for {occasion_word} in {weather_word} weather.",
    "Going {style_word} to {occasion_word}, it's {weather_word}.",
    "I love {style_word} style. Heading to {occasion_word}, it is {weather_word}.",
    "Suggest a {style_word} look for {occasion_word}, the weather is {weather_word}.",
    "I want to dress {style_word} for {occasion_word}, it's {weather_word} outside.",
    "Something {style_word} for {occasion_word} in {weather_word} conditions?",
    "My style is {style_word}. I'm going to {occasion_word} and it's {weather_word}.",
]

# Templates requiring style + occasion (weather will be None)
TEMPLATES_STYLE_OCCASION = [
    "I'm going for a {style_word} vibe to the {occasion_word}.",
    "Something {style_word} for a {occasion_word}, please.",
    "I want a {style_word} style outfit for {occasion_word}.",
    "Suggest a {style_word} look for a {occasion_word}.",
    "I'm into {style_word} style, going to {occasion_word}.",
    "Need a {style_word} outfit for {occasion_word}.",
    "Give me a {style_word} look for {occasion_word}.",
    "I love {style_word} fashion, what works for {occasion_word}?",
    "Wearing something {style_word} to {occasion_word}.",
    "I want to look {style_word} at the {occasion_word}.",
]

# Templates requiring style + weather (occasion will be None)
TEMPLATES_STYLE_WEATHER = [
    "I want a {style_word} outfit, it's {weather_word}.",
    "Looking for {style_word} clothes for {weather_word} weather.",
    "What {style_word} outfits work for {weather_word} days?",
    "I want {style_word} clothes for {weather_word} weather.",
    "What {style_word} pieces work for {weather_word} conditions?",
    "Going {style_word} today, it's {weather_word}.",
    "Suggest {style_word} options for {weather_word} weather.",
    "I prefer {style_word} fashion. It's {weather_word} outside.",
]


def generate_synthetic_data(num_samples: int = 4000) -> List[Dict]:
    data = []
    for _ in range(num_samples):
        occasion_class = random.choice(OCCASIONS)
        weather_class  = random.choice(WEATHER_CLASSES)
        style_class    = random.choice(STYLES)

        occasion_word = random.choice(OCCASION_SYNONYMS[occasion_class])
        weather_word  = random.choice(WEATHER_SYNONYMS[weather_class])
        style_word    = random.choice(STYLE_SYNONYMS[style_class])

        r = random.random()
        if r < 0.38:
            # occasion + weather, no style
            template = random.choice(TEMPLATES_BOTH)
            text = template.format(occasion_word=occasion_word, weather_word=weather_word)
            record = {"text": text, "occasion": occasion_class,
                      "weather": weather_class, "style": None}
        elif r < 0.55:
            # style + occasion + weather
            template = random.choice(TEMPLATES_STYLE_BOTH)
            text = template.format(occasion_word=occasion_word,
                                   weather_word=weather_word, style_word=style_word)
            record = {"text": text, "occasion": occasion_class,
                      "weather": weather_class, "style": style_class}
        elif r < 0.65:
            # occasion only
            template = random.choice(TEMPLATES_OCCASION_ONLY)
            text = template.format(occasion_word=occasion_word)
            record = {"text": text, "occasion": occasion_class,
                      "weather": None, "style": None}
        elif r < 0.75:
            # weather only
            template = random.choice(TEMPLATES_WEATHER_ONLY)
            text = template.format(weather_word=weather_word)
            record = {"text": text, "occasion": None,
                      "weather": weather_class, "style": None}
        elif r < 0.85:
            # style + occasion, no weather
            template = random.choice(TEMPLATES_STYLE_OCCASION)
            text = template.format(occasion_word=occasion_word, style_word=style_word)
            record = {"text": text, "occasion": occasion_class,
                      "weather": None, "style": style_class}
        else:
            # style + weather, no occasion
            template = random.choice(TEMPLATES_STYLE_WEATHER)
            text = template.format(weather_word=weather_word, style_word=style_word)
            record = {"text": text, "occasion": None,
                      "weather": weather_class, "style": style_class}

        data.append(record)
    return data


def preprocess_text(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return [t for t in text.split() if t]  # filter empty strings from digit-stripping


class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.vocab_size = 2

    def build_vocab(self, texts: List[str]):
        for text in texts:
            for token in preprocess_text(text):
                if token not in self.word2idx:
                    self.word2idx[token] = self.vocab_size
                    self.idx2word[self.vocab_size] = token
                    self.vocab_size += 1

    def encode(self, text: str) -> List[int]:
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in preprocess_text(text)]


class WardrobeQueryDataset(Dataset):
    def __init__(self, data: List[Dict], vocab: Vocabulary, max_len: int = 25):
        self.data    = data
        self.vocab   = vocab
        self.max_len = max_len

        self.occasion2idx = {occ: idx for idx, occ in enumerate(OCCASIONS)}
        self.weather2idx  = {wea: idx for idx, wea in enumerate(WEATHER_CLASSES)}
        self.style2idx    = {sty: idx for idx, sty in enumerate(STYLES)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoded = self.vocab.encode(item["text"])

        if len(encoded) < self.max_len:
            encoded += [self.vocab.word2idx["<PAD>"]] * (self.max_len - len(encoded))
        else:
            encoded = encoded[:self.max_len]

        occ_label = (self.occasion2idx[item["occasion"]]
                     if item["occasion"] is not None else OCCASION_UNKNOWN_IDX)
        wea_label = (self.weather2idx[item["weather"]]
                     if item["weather"] is not None else WEATHER_UNKNOWN_IDX)
        sty_label = (self.style2idx[item["style"]]
                     if item["style"] is not None else STYLE_UNKNOWN_IDX)

        return {
            "input_ids":      torch.tensor(encoded, dtype=torch.long),
            "occasion_label": torch.tensor(occ_label, dtype=torch.long),
            "weather_label":  torch.tensor(wea_label, dtype=torch.long),
            "style_label":    torch.tensor(sty_label, dtype=torch.long),
        }


def get_dataloaders(num_samples=4000, batch_size=32, max_len=25):
    raw_data = generate_synthetic_data(num_samples)

    train_size = int(0.8 * num_samples)
    train_data = raw_data[:train_size]
    val_data   = raw_data[train_size:]

    vocab = Vocabulary()
    vocab.build_vocab([item["text"] for item in train_data])

    train_dataset = WardrobeQueryDataset(train_data, vocab, max_len)
    val_dataset   = WardrobeQueryDataset(val_data,   vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders(num_samples=10)
    for batch in train_loader:
        print(batch)
        break
