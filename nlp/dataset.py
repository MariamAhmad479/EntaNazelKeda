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
                 "outing", "picnic-trip", "bookstore", "market", "farmers-market",
                 "casual-dinner", "get-together", "rooftop", "terrace"],
    "formal":   ["formal", "wedding", "black-tie", "fancy", "elegant", "ceremony",
                 "gala", "banquet", "black-tie", "prom", "graduation", "reception",
                 "charity-dinner", "awards", "awards-ceremony", "inauguration",
                 "charity-gala", "formal-event", "red-carpet", "opera", "ballet"],
    "business": ["business", "office", "work", "interview", "meeting", "job",
                 "corporate", "conference", "presentation", "professional",
                 "client-meeting", "networking", "seminar", "boardroom",
                 "sales-pitch", "business-lunch", "job-fair", "workshop",
                 "team-meeting", "annual-review", "pitch", "startup-event"],
    "sport":    ["sport", "gym", "workout", "running", "athletic", "exercise",
                 "pool", "swim", "training", "jogging", "yoga", "cycling",
                 "tennis", "basketball", "pilates", "spinning", "crossfit",
                 "football", "soccer", "volleyball", "golf", "boxing",
                 "climbing", "rowing", "marathon", "5k-run", "bootcamp",
                 "fitness-class", "aerobics", "weightlifting", "track"],
    "party":    ["party", "club", "dinner", "date", "night-out", "celebration",
                 "gathering", "birthday", "nightclub", "evening", "cocktail-party",
                 "house-party", "mixer", "nightlife", "rave", "after-party",
                 "birthday-party", "festivity", "social-event", "girls-night",
                 "guys-night", "rooftop-party", "pool-party"],
    "outdoor":  ["outdoor", "hiking", "camping", "park", "picnic", "outside",
                 "trail", "beach", "nature", "garden", "festival", "bbq",
                 "barbecue", "outdoor-concert", "lakeside", "mountain",
                 "adventure", "road-trip", "forest", "national-park",
                 "countryside", "farm", "botanical-garden", "waterfall",
                 "surfing", "kayaking", "rock-climbing", "outdoor-market"],
}

WEATHER_SYNONYMS = {
    "hot":  ["hot", "warm", "sunny", "boiling", "summer", "sweating", "heat",
             "scorching", "sweltering", "blazing", "humid", "tropical",
             "35-degrees", "38-degrees", "40-degrees", "over-30",
             "muggy", "steamy", "baking", "roasting", "heatwave",
             "melting", "very-hot", "extremely-hot", "really-hot",
             "sweat", "90-degrees", "95-degrees", "stifling",
             # expanded
             "sizzling", "broiling", "torrid", "boiling-hot", "scorcher",
             "burning", "blazing-hot", "sticky", "breathless", "oppressive",
             "sauna-like", "dry-heat", "intense-heat", "furnace", "suffocating",
             "unbearable-heat", "too-hot", "really-warm", "super-warm", "desert-heat"],
    "mild": ["mild", "nice", "pleasant", "spring", "cool", "breezy", "windy",
             "perfect", "comfortable", "temperate", "moderate", "crisp", "refreshing",
             "20-degrees", "18-degrees", "15-degrees", "overcast", "cloudy",
             "partly-cloudy", "foggy", "drizzle", "gusty", "changeable",
             "25-degrees", "22-degrees", "light-wind", "cloudy-day", "calm",
             # expanded
             "fair", "nice-day", "good-weather", "pleasant-day", "not-too-hot",
             "not-too-cold", "average", "room-temperature", "bearable", "tolerable",
             "neither-hot-nor-cold", "in-between", "autumn-weather", "fall-weather",
             "spring-weather", "seasonal", "typical", "normal-weather", "standard",
             "lukewarm", "warmish", "coolish", "light-jacket-weather", "layering-weather"],
    "cold": ["cold", "freezing", "chilly", "winter", "snow", "frosty", "shivering",
             "icy", "bitter", "frigid", "below-zero", "sub-zero", "wintry",
             "5-degrees", "0-degrees", "minus", "arctic", "raw",
             "snowy", "blizzard", "sleet", "harsh", "bitter-cold",
             "minus-10", "minus-5", "below-freezing", "ice-cold", "freezing-cold",
             # expanded
             "nippy", "bone-chilling", "perishing", "cold-snap", "deep-winter",
             "freeze", "bitter-wind", "biting-cold", "numbing", "polar",
             "glacial", "wintry-blast", "frost", "black-ice", "heavy-snow",
             "hypothermia-weather", "coat-weather", "scarf-weather", "glove-weather",
             "really-cold", "super-cold", "extremely-cold", "so-cold", "very-chilly"],
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


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

FILLER_WORDS = [
    "um", "like", "basically", "honestly", "actually",
    "you know", "so", "well", "I think", "I mean",
    "kind of", "sort of", "really", "just", "maybe",
    "roughly", "approximately", "perhaps", "anyway", "right",
]


def augment_text(text: str, filler_prob: float = 0.4, swap_prob: float = 0.3) -> str:
    """
    Light augmentation applied offline during synthetic data generation.

    - Filler insertion : inserts a natural filler word at a random interior
      token boundary, simulating how real users write loosely.
    - Adjacent-word swap : swaps two adjacent middle-of-sentence words,
      making the model robust to slight word-order variation.

    Keywords (occasion / weather / style words) are preserved because
    augmentation touches only the surrounding sentence structure. Labels
    are unaffected.
    """
    words = text.split()
    if len(words) < 3:
        return text

    # filler insertion
    if random.random() < filler_prob:
        pos = random.randint(1, len(words) - 1)
        words.insert(pos, random.choice(FILLER_WORDS))

    # adjacent-word swap (avoid first/last position)
    if random.random() < swap_prob and len(words) > 3:
        pos = random.randint(1, len(words) - 2)
        words[pos], words[pos + 1] = words[pos + 1], words[pos]

    return " ".join(words)


def generate_synthetic_data(num_samples: int = 4000, augment: bool = True) -> List[Dict]:
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

        if augment:
            record["text"] = augment_text(record["text"])
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


def get_dataloaders(num_samples=4000, batch_size=32, max_len=25, augment=True):
    raw_data = generate_synthetic_data(num_samples, augment=augment)

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



# =============================================================================
# DistilBERT additions — appended to dataset.py
# =============================================================================

import random, re, json, os
from typing import List, Dict, Optional

INTENT_CLASSES = ["OUTFIT_REQUEST", "SMALL_TALK", "REJECTION", "CLARIFICATION", "FEEDBACK", "OTHER"]
INTENT_UNKNOWN_IDX = -100

# ── Realistic human-written queries ──────────────────────────────────────────

REALISTIC_QUERIES: List[Dict] = [
    # casual + cold
    {"text": "going out with friends and its freezing outside help 🥶", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "coffee run later its so cold idk what to put on", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "just chilling with my friend, weather is cold af", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "casual brunch tmrw but its supposed to be really chilly", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "going to the mall but its cold outside, outfit help??", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "movie night out with the girls, cold weather vibes", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "meeting some ppl, weather is freezing send help 😭", "occasion": "casual", "weather": "cold", "style": None},
    {"text": "want smth cozy but still cute for hanging out, its cold", "occasion": "casual", "weather": "cold", "style": None},
    # casual + hot
    {"text": "its boiling outside and i wanna look cute going out", "occasion": "casual", "weather": "hot", "style": None},
    {"text": "beach day with friends, what do i wear its SO hot 🔥", "occasion": "casual", "weather": "hot", "style": None},
    {"text": "just hanging out but its like 35 degrees 😭", "occasion": "casual", "weather": "hot", "style": None},
    {"text": "shopping trip but its so hot outside idk what to wear", "occasion": "casual", "weather": "hot", "style": None},
    {"text": "going for a walk later and its really warm, suggestions?", "occasion": "casual", "weather": "hot", "style": None},
    {"text": "picnic with friends, weather is insane rn so hot", "occasion": "casual", "weather": "hot", "style": None},
    # casual + mild
    {"text": "coffee date with my bestie, weather is pleasant today", "occasion": "casual", "weather": "mild", "style": None},
    {"text": "hanging out at the park, weather seems decent today", "occasion": "casual", "weather": "mild", "style": None},
    {"text": "casual plans today, weather is like meh not too hot not too cold", "occasion": "casual", "weather": "mild", "style": None},
    {"text": "brunch with some ppl, weather is actually really nice", "occasion": "casual", "weather": "mild", "style": None},
    # casual only
    {"text": "just need something casual and easy for today", "occasion": "casual", "weather": None, "style": None},
    {"text": "going out but like chill vibes only pls", "occasion": "casual", "weather": None, "style": None},
    {"text": "something simple for hanging out?? nothing too extra", "occasion": "casual", "weather": None, "style": None},
    {"text": "i'm meeting friends later, don't wanna overdress", "occasion": "casual", "weather": None, "style": None},
    {"text": "casual vibes for today pls 🙏", "occasion": "casual", "weather": None, "style": None},
    {"text": "coffee shop fit?? something cute but chill", "occasion": "casual", "weather": None, "style": None},
    {"text": "going to the cinema later what should i wear", "occasion": "casual", "weather": None, "style": None},
    {"text": "mall trip later, need outfit inspo", "occasion": "casual", "weather": None, "style": None},
    # formal + cold
    {"text": "need an outfit for a formal event but it's really cold outside", "occasion": "formal", "weather": "cold", "style": None},
    {"text": "going to a wedding tomorrow and its supposed to snow 😭", "occasion": "formal", "weather": "cold", "style": None},
    {"text": "black tie gala tonight but the weather is absolutely freezing", "occasion": "formal", "weather": "cold", "style": None},
    {"text": "fancy dinner but its cold af outside, what do i wear", "occasion": "formal", "weather": "cold", "style": None},
    {"text": "graduation ceremony and it's literally freezing out", "occasion": "formal", "weather": "cold", "style": None},
    # formal + hot
    {"text": "have a formal event but it's so hot i'm dying 💀", "occasion": "formal", "weather": "hot", "style": None},
    {"text": "wedding next week and its gonna be scorching hot", "occasion": "formal", "weather": "hot", "style": None},
    {"text": "fancy dinner tonight but its like 38 degrees outside", "occasion": "formal", "weather": "hot", "style": None},
    {"text": "need to look elegant but the heat is insane rn", "occasion": "formal", "weather": "hot", "style": None},
    # formal only
    {"text": "help me dress for a formal event pls", "occasion": "formal", "weather": None, "style": None},
    {"text": "what do i wear to a black tie event??", "occasion": "formal", "weather": None, "style": None},
    {"text": "going to a wedding this weekend, need outfit ideas", "occasion": "formal", "weather": None, "style": None},
    {"text": "fancy event tonight, what should i wear?", "occasion": "formal", "weather": None, "style": None},
    {"text": "need a formal outfit asap 😭", "occasion": "formal", "weather": None, "style": None},
    {"text": "something classy for a gala tonight?", "occasion": "formal", "weather": None, "style": None},
    # business + cold
    {"text": "important job interview tomorrow and its freezing outside", "occasion": "business", "weather": "cold", "style": None},
    {"text": "client meeting in this cold weather, what should i wear", "occasion": "business", "weather": "cold", "style": None},
    {"text": "work presentation today, its super cold outside rn", "occasion": "business", "weather": "cold", "style": None},
    {"text": "office today but the commute is gonna be freezing 🥶", "occasion": "business", "weather": "cold", "style": None},
    # business + hot
    {"text": "got an interview but its way too hot outside omg", "occasion": "business", "weather": "hot", "style": None},
    {"text": "work meeting today and its boiling 😭 what do i wear", "occasion": "business", "weather": "hot", "style": None},
    {"text": "office vibes but its like 35 degrees out, need help", "occasion": "business", "weather": "hot", "style": None},
    # business only
    {"text": "something simple but cute for uni tomorrow", "occasion": "business", "weather": None, "style": None},
    {"text": "job interview outfits?? need smth professional", "occasion": "business", "weather": None, "style": None},
    {"text": "office outfit ideas pls", "occasion": "business", "weather": None, "style": None},
    {"text": "have a big presentation at work tomorrow", "occasion": "business", "weather": None, "style": None},
    {"text": "what should i wear to my first day at the office", "occasion": "business", "weather": None, "style": None},
    {"text": "uni lectures tomorrow need something comfy but put together", "occasion": "business", "weather": None, "style": None},
    {"text": "professional outfit for an interview??", "occasion": "business", "weather": None, "style": None},
    # sport + hot
    {"text": "gym session today and its boiling outside 🔥", "occasion": "sport", "weather": "hot", "style": None},
    {"text": "going for a run but it's so hot today help", "occasion": "sport", "weather": "hot", "style": None},
    {"text": "workout later, outside is like a furnace 💀", "occasion": "sport", "weather": "hot", "style": None},
    {"text": "morning run planned but the heat is already crazy", "occasion": "sport", "weather": "hot", "style": None},
    {"text": "yoga in the park today, super warm out there", "occasion": "sport", "weather": "hot", "style": None},
    # sport + cold
    {"text": "outdoor run tmrw and its gonna be freezing 🥶", "occasion": "sport", "weather": "cold", "style": None},
    {"text": "gym in the morning but its so cold outside", "occasion": "sport", "weather": "cold", "style": None},
    {"text": "jogging in winter weather what should i wear", "occasion": "sport", "weather": "cold", "style": None},
    # sport only
    {"text": "hitting the gym what should i wear", "occasion": "sport", "weather": None, "style": None},
    {"text": "workout fit?? nothing too restrictive", "occasion": "sport", "weather": None, "style": None},
    {"text": "going for a run later, what do i wear", "occasion": "sport", "weather": None, "style": None},
    {"text": "yoga class today, need a comfortable outfit", "occasion": "sport", "weather": None, "style": None},
    # party + cold
    {"text": "date night but it's freezing outside, help 😭", "occasion": "party", "weather": "cold", "style": None},
    {"text": "birthday dinner and the weather is so cold tonight", "occasion": "party", "weather": "cold", "style": None},
    {"text": "going out to dinner, weather is really chilly", "occasion": "party", "weather": "cold", "style": None},
    {"text": "night out planned but its literally arctic outside", "occasion": "party", "weather": "cold", "style": None},
    # party + hot
    {"text": "pool party tonight in this crazy heat 🔥", "occasion": "party", "weather": "hot", "style": None},
    {"text": "going out tonight and its boiling, what to wear", "occasion": "party", "weather": "hot", "style": None},
    {"text": "dinner date tonight but its soooo hot outside", "occasion": "party", "weather": "hot", "style": None},
    # party only
    {"text": "date night outfit pls 🙏", "occasion": "party", "weather": None, "style": None},
    {"text": "going to a birthday party what should i wear", "occasion": "party", "weather": None, "style": None},
    {"text": "girls night out tonight, need outfit inspo", "occasion": "party", "weather": None, "style": None},
    {"text": "dinner date tonight, what should i put on", "occasion": "party", "weather": None, "style": None},
    {"text": "club night what do i wear?? something cute", "occasion": "party", "weather": None, "style": None},
    {"text": "i have dinner later idk what to wear help", "occasion": "party", "weather": None, "style": None},
    # outdoor + cold
    {"text": "hiking tomorrow and its gonna be freezing 🥶", "occasion": "outdoor", "weather": "cold", "style": None},
    {"text": "camping this weekend, weather looks really cold", "occasion": "outdoor", "weather": "cold", "style": None},
    {"text": "park walk but the weather is so chilly today", "occasion": "outdoor", "weather": "cold", "style": None},
    # outdoor + hot
    {"text": "beach day tomorrow it's gonna be so hot 🔥", "occasion": "outdoor", "weather": "hot", "style": None},
    {"text": "hiking in this heat, what do i even wear 😭", "occasion": "outdoor", "weather": "hot", "style": None},
    {"text": "bbq at the park, sun is intense today", "occasion": "outdoor", "weather": "hot", "style": None},
    # outdoor + mild
    {"text": "picnic in the park, weather is actually lovely today", "occasion": "outdoor", "weather": "mild", "style": None},
    {"text": "outdoor festival this weekend, weather seems okay", "occasion": "outdoor", "weather": "mild", "style": None},
    # outdoor only
    {"text": "hiking trip what should i wear", "occasion": "outdoor", "weather": None, "style": None},
    {"text": "camping this weekend, need practical outfit ideas", "occasion": "outdoor", "weather": None, "style": None},
    # weather only
    {"text": "idk what to wear it's kinda cold 😭", "occasion": None, "weather": "cold", "style": None},
    {"text": "its literally freezing outside rn, what do i wear", "occasion": None, "weather": "cold", "style": None},
    {"text": "brr its so cold today outfit ideas??", "occasion": None, "weather": "cold", "style": None},
    {"text": "its sooo hot omg 🥵 what should i wear", "occasion": None, "weather": "hot", "style": None},
    {"text": "boiling outside rn, need a cool outfit", "occasion": None, "weather": "hot", "style": None},
    {"text": "weather is nice today, what should i wear?", "occasion": None, "weather": "mild", "style": None},
    {"text": "kinda breezy outside today, outfit suggestions?", "occasion": None, "weather": "mild", "style": None},
    # style + occasion
    {"text": "going for a minimalist look for work tomorrow", "occasion": "business", "weather": None, "style": "minimalist"},
    {"text": "want smth streetwear-ish for hanging out with friends", "occasion": "casual", "weather": None, "style": "streetwear"},
    {"text": "classic elegant vibe for the dinner tonight", "occasion": "party", "weather": None, "style": "classic"},
    {"text": "athletic look for my gym session today", "occasion": "sport", "weather": None, "style": "athletic"},
    {"text": "boho vibes for the outdoor festival this weekend", "occasion": "outdoor", "weather": None, "style": "bohemian"},
    {"text": "preppy fit for uni tomorrow pls", "occasion": "business", "weather": None, "style": "preppy"},
    # all three
    {"text": "going out tonight, its cold, want something streetwear", "occasion": "party", "weather": "cold", "style": "streetwear"},
    {"text": "wedding this weekend its hot, need something classic", "occasion": "formal", "weather": "hot", "style": "classic"},
    {"text": "gym today and its boiling, full athletic fit", "occasion": "sport", "weather": "hot", "style": "athletic"},
    {"text": "office meeting today, its chilly, want a minimalist look", "occasion": "business", "weather": "cold", "style": "minimalist"},
    {"text": "hanging out later, nice weather, want something bohemian", "occasion": "casual", "weather": "mild", "style": "bohemian"},
    {"text": "hiking tmrw its cold, athletic and practical pls", "occasion": "outdoor", "weather": "cold", "style": "athletic"},
]


# ── Noise injection for realistic text ────────────────────────────────────────

_CONTRACTIONS = [
    ("it is", "its"), ("I am", "im"), ("do not", "dont"),
    ("I have", "ive"), ("I will", "ill"), ("going to", "gonna"),
    ("want to", "wanna"), ("I do not know", "idk"), ("something", "smth"),
    ("tomorrow", "tmrw"), ("tonight", "tonite"), ("because", "cuz"),
]
_EMOJIS_COLD  = ["🥶", "❄️", "😭", "🧥"]
_EMOJIS_HOT   = ["🔥", "🥵", "😅", "☀️"]
_EMOJIS_MISC  = ["😭", "💀", "✨", "🙏", "😩"]
_FILLERS_REAL = ["like", "lowkey", "ngl", "tbh", "fr", "rn", "honestly", "literally"]


def apply_realistic_noise(text: str, record: Dict, p: float = 0.5) -> str:
    """
    Apply social-media-style noise to a generated sentence.

    Transformations (each applied with probability p):
    - Contraction/slang substitution  (it is → its, going to → gonna …)
    - Lowercase everything
    - Remove some punctuation randomly
    - Inject a contextual emoji
    - Insert a real filler word (lowkey, ngl, tbh …)
    - Introduce a character-level typo in one non-keyword word
    """
    if random.random() > p:
        return text

    # Lowercase
    text = text.lower()

    # Contraction substitution
    for formal, informal in _CONTRACTIONS:
        if formal.lower() in text and random.random() < 0.6:
            text = text.replace(formal.lower(), informal)

    # Remove trailing punctuation randomly
    if text.endswith(".") and random.random() < 0.7:
        text = text[:-1]

    # Filler insertion
    if random.random() < 0.4:
        words = text.split()
        if len(words) > 3:
            pos = random.randint(1, len(words) - 1)
            words.insert(pos, random.choice(_FILLERS_REAL))
            text = " ".join(words)

    # Emoji injection
    if random.random() < 0.35:
        weather = record.get("weather")
        if weather == "cold":
            text += " " + random.choice(_EMOJIS_COLD)
        elif weather == "hot":
            text += " " + random.choice(_EMOJIS_HOT)
        else:
            text += " " + random.choice(_EMOJIS_MISC)

    # Simple typo: double a random letter in a non-short word
    if random.random() < 0.2:
        words = text.split()
        long_words = [i for i, w in enumerate(words) if len(w) > 4]
        if long_words:
            i = random.choice(long_words)
            w = words[i]
            pos = random.randint(1, len(w) - 1)
            words[i] = w[:pos] + w[pos] + w[pos:]   # double a char
            text = " ".join(words)

    return text


def generate_bert_dataset(
    num_synthetic: int = 10000,
    augment: bool = True,
    noise_prob: float = 0.4,
) -> List[Dict]:
    """
    Build the full training corpus for DistilBERT fine-tuning:

      1. `num_synthetic` template-generated samples (existing pipeline)
         with light structural augmentation (filler words, word swap).
      2. All REALISTIC_QUERIES (human-written, ~110 sentences).
      3. Noise applied to synthetic samples with probability `noise_prob`
         (contractions, emojis, typos, fillers).

    The realistic queries are added raw (noise already baked in naturally).
    """
    # Import here to avoid circular issues when this block is appended
    from nlp.dataset import generate_synthetic_data  # type: ignore

    synthetic = generate_synthetic_data(num_synthetic, augment=augment)

    # Apply realistic noise to synthetic samples
    if noise_prob > 0:
        for rec in synthetic:
            rec["text"] = apply_realistic_noise(rec["text"], rec, p=noise_prob)
            rec["intent"] = "OUTFIT_REQUEST"

    for rec in REALISTIC_QUERIES:
        rec["intent"] = "OUTFIT_REQUEST"

    def get_external_intent_data():
        cache_file = "nlp/intent_cache.json"
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                return json.load(f)

        from datasets import load_dataset
        results = []
        clinc = load_dataset("clinc_oos", "plus", split="train")
        smalltalk = ["greeting", "goodbye", "thank_you", "how_are_you", "nice_to_meet_you", "what_is_your_name"]
        rejection = ["no", "cancel", "stop"]
        
        # Sample limited OTHER from clinc
        other_samples = 0
        for item in clinc:
            intent_name = clinc.features['intent'].names[item['intent']]
            if intent_name in rejection:
                results.append({"text": item["text"], "intent": "REJECTION", "occasion": None, "weather": None, "style": None})
            elif intent_name in smalltalk:
                results.append({"text": item["text"], "intent": "SMALL_TALK", "occasion": None, "weather": None, "style": None})
            elif other_samples < 500:
                results.append({"text": item["text"], "intent": "OTHER", "occasion": None, "weather": None, "style": None})
                other_samples += 1

        banking = load_dataset("banking77", split="train")
        banking_samples = random.sample(list(banking), 500)
        for item in banking_samples:
            results.append({"text": item["text"], "intent": "OTHER", "occasion": None, "weather": None, "style": None})

        snips = load_dataset("snips_built_in_intents", split="train")
        num_snips = min(500, len(snips))
        snips_samples = random.sample(list(snips), num_snips)
        for item in snips_samples:
            results.append({"text": item["text"], "intent": "OTHER", "occasion": None, "weather": None, "style": None})

        # Add manual CLARIFICATION and FEEDBACK as HF datasets don't map perfectly
        clarification = ["what do you mean?", "i don't understand", "could you explain?", "can you repeat that?", "what?"]
        feedback = ["this is bad", "i hate this", "something else", "try again", "not my style", "different please"]
        for text in clarification:
            results.append({"text": text, "intent": "CLARIFICATION", "occasion": None, "weather": None, "style": None})
        for text in feedback:
            results.append({"text": text, "intent": "FEEDBACK", "occasion": None, "weather": None, "style": None})

        with open(cache_file, "w") as f:
            json.dump(results, f)
        return results

    external_intents = get_external_intent_data()
    combined = synthetic + REALISTIC_QUERIES + external_intents
    random.shuffle(combined)
    return combined


# ── BERT-compatible PyTorch Dataset ──────────────────────────────────────────

class WardrobeQueryDatasetBERT(Dataset):
    """
    PyTorch Dataset that tokenises text using a HuggingFace tokenizer
    (DistilBertTokenizerFast) and returns tensors ready for the
    DistilBertMultiTaskClassifier.

    Future-ready: pass `extra_label_keys` to include additional label columns
    (e.g. color, season) without subclassing.
    """

    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_len: int = 64,
        extra_label_keys: Optional[List[str]] = None,
    ):
        self.data      = data
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.extras    = extra_label_keys or []

        self.occasion2idx = {occ: idx for idx, occ in enumerate(OCCASIONS)}
        self.weather2idx  = {wea: idx for idx, wea in enumerate(WEATHER_CLASSES)}
        self.style2idx    = {sty: idx for idx, sty in enumerate(STYLES)}
        self.intent2idx   = {cls: idx for idx, cls in enumerate(INTENT_CLASSES)}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]

        enc = self.tokenizer(
            item["text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        occ_label = (self.occasion2idx[item["occasion"]]
                     if item.get("occasion") else OCCASION_UNKNOWN_IDX)
        wea_label = (self.weather2idx[item["weather"]]
                     if item.get("weather") else WEATHER_UNKNOWN_IDX)
        sty_label = (self.style2idx[item["style"]]
                     if item.get("style") else STYLE_UNKNOWN_IDX)
        int_label = (self.intent2idx[item["intent"]]
                     if item.get("intent") else INTENT_UNKNOWN_IDX)

        sample = {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "intent_label":   torch.tensor(int_label, dtype=torch.long),
            "occasion_label": torch.tensor(occ_label, dtype=torch.long),
            "weather_label":  torch.tensor(wea_label, dtype=torch.long),
            "style_label":    torch.tensor(sty_label, dtype=torch.long),
        }

        # Extra heads (future extensibility)
        for key in self.extras:
            sample[f"{key}_label"] = torch.tensor(
                item.get(key, -1), dtype=torch.long
            )

        return sample


def get_bert_dataloaders(
    num_synthetic: int = 10000,
    batch_size: int = 16,
    max_len: int = 64,
    val_split: float = 0.1,
    tokenizer=None,
):
    """
    Build train/val DataLoaders for DistilBERT fine-tuning.

    Parameters
    ----------
    num_synthetic : synthetic samples generated (realistic queries are added on top)
    batch_size    : samples per batch (16 recommended for DistilBERT on CPU)
    max_len       : tokenizer max sequence length
    val_split     : fraction of data to hold out for validation
    tokenizer     : HuggingFace tokenizer; loaded automatically if None

    Returns
    -------
    (train_loader, val_loader, tokenizer)
    """
    if tokenizer is None:
        from transformers import DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )

    all_data = generate_bert_dataset(num_synthetic=num_synthetic)
    split_at = int(len(all_data) * (1 - val_split))
    train_data, val_data = all_data[:split_at], all_data[split_at:]

    train_ds = WardrobeQueryDatasetBERT(train_data, tokenizer, max_len)
    val_ds   = WardrobeQueryDatasetBERT(val_data,   tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, tokenizer
