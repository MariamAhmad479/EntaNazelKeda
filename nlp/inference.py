"""
inference.py  --  Unified inference for both DistilBERT and BiLSTM backends.

Auto-detection
--------------
  If  nlp/saved_models_bert/model_config.json  exists  ->  DistilBERT  (preferred)
  Otherwise                                             ->  BiLSTM  (legacy)

Public API  (unchanged from v1)
--------------------------------
  nlp = NLPInference()
  result = nlp.predict("going out tonight its cold help")
  # {
  #   "occasion":      "party",
  #   "weather_class": "cold",
  #   "weather":       {"temperature": 4},
  #   "style":         None,
  #   "confidence":    {"occasion": 0.97, "weather": 0.91, "style": None}
  # }

Confidence scores
-----------------
  Each head's max-softmax probability is returned in "confidence".
  A value of None means the head did not fire (below its threshold).
"""

import json, os, re, sys
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import OCCASIONS, WEATHER_CLASSES, STYLES, INTENT_CLASSES
from recommendation_engine.location_weather import load_locations, get_location_details, map_category_to_occasion, fetch_realtime_weather

OCCASION_CONF_THRESH = 0.30
WEATHER_CONF_THRESH  = 0.30
STYLE_CONF_THRESH    = 0.30

BERT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models_bert")
LSTM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")


# ── Temperature regex (backend-agnostic) ──────────────────────────────────────

def _extract_explicit_temperature(text: str) -> Optional[float]:
    t = text.lower()
    m = re.search(r'(?:minus|negative)\s+(\d+(?:\.\d+)?)', t)
    if m:
        return -float(m.group(1))
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?\s*c|°\s*c)', t)
    if m:
        return float(m.group(1))
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:deg(?:rees?)?\s*f|°\s*f)', t)
    if m:
        return (float(m.group(1)) - 32) * 5 / 9
    m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°|degrees?)', t)
    if m:
        v = float(m.group(1))
        return v if v < 50 else (v - 32) * 5 / 9
    return None


def _temp_to_class(temp: float) -> str:
    if temp <= 15:
        return "cold"
    if temp <= 24:
        return "mild"
    return "hot"


# ── DistilBERT backend ────────────────────────────────────────────────────────

class _BertBackend:
    def __init__(self, model_dir: str):
        from transformers import DistilBertTokenizerFast
        from nlp.model import DistilBertMultiTaskClassifier

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
        self.model = DistilBertMultiTaskClassifier(
            num_intents=cfg["num_intents"],
            num_occasions=cfg["num_occasions"],
            num_weather=cfg["num_weather"],
            num_styles=cfg["num_styles"],
            dropout=cfg.get("dropout", 0.3),
        )
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "nlp_model.pth"), map_location=self.device)
        )
        self.model.to(self.device).eval()
        self.max_len = cfg.get("max_len", 64)

    def logits(self, text: str):
        enc = self.tokenizer(
            text, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ids  = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            i_lg, o_lg, w_lg, s_lg = self.model(ids, mask)
        return i_lg, o_lg, w_lg, s_lg


# ── BiLSTM backend (legacy) ───────────────────────────────────────────────────

class _LstmBackend:
    def __init__(self, model_dir: str):
        import pickle
        from nlp.dataset import preprocess_text
        from nlp.model import WardrobeNLPModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._preprocess = preprocess_text

        with open(os.path.join(model_dir, "vocab.pkl"), "rb") as f:
            self.vocab = pickle.load(f)
        with open(os.path.join(model_dir, "model_config.json")) as f:
            cfg = json.load(f)

        self.model = WardrobeNLPModel(**cfg)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "nlp_model.pth"), map_location=self.device)
        )
        self.model.to(self.device).eval()

    def logits(self, text: str):
        from nlp.dataset import OCCASION_UNKNOWN_IDX
        unk_idx = self.vocab.word2idx["<UNK>"]
        tokens  = self._preprocess(text)
        if not tokens:
            return None, None, None, None
        encoded = [self.vocab.word2idx.get(t, unk_idx) for t in tokens]
        if sum(1 for i in encoded if i == unk_idx) / len(encoded) > 0.6:
            return None, None, None, None
        ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
        with torch.no_grad():
            o_lg, w_lg, s_lg = self.model(ids)
        return None, o_lg, w_lg, s_lg


# ── Public NLPInference class ─────────────────────────────────────────────────

class NLPInference:
    """
    Unified inference class. Loads DistilBERT if available, else BiLSTM.

    Usage
    -----
    nlp = NLPInference()
    result = nlp.predict("need something for a date its freezing")
    """

    WEATHER_TO_TEMP = {"hot": 32, "mild": 18, "cold": 4}
    WEATHER_TO_CONDITION = {"hot": "sunny", "mild": "cloudy", "cold": "cloudy"}

    def __init__(self, model_dir: Optional[str] = None):
        # Auto-detect backend
        if model_dir is None:
            bert_cfg = os.path.join(BERT_MODEL_DIR, "model_config.json")
            if os.path.exists(bert_cfg):
                with open(bert_cfg) as f:
                    cfg = json.load(f)
                if cfg.get("backend") == "distilbert":
                    model_dir = BERT_MODEL_DIR
                else:
                    model_dir = LSTM_MODEL_DIR
            else:
                model_dir = LSTM_MODEL_DIR

        bert_cfg = os.path.join(model_dir, "model_config.json")
        is_bert  = False
        if os.path.exists(bert_cfg):
            with open(bert_cfg) as f:
                is_bert = json.load(f).get("backend") == "distilbert"

        if is_bert:
            self._backend = _BertBackend(model_dir)
            self.backend_name = "distilbert"
        else:
            self._backend = _LstmBackend(model_dir)
            self.backend_name = "bilstm"

        print(f"[NLPInference] backend={self.backend_name}, dir={model_dir}")

        self.idx2occasion = {i: o for i, o in enumerate(OCCASIONS)}
        self.idx2weather  = {i: w for i, w in enumerate(WEATHER_CLASSES)}
        self.idx2style    = {i: s for i, s in enumerate(STYLES)}
        self.idx2intent   = {i: c for i, c in enumerate(INTENT_CLASSES)}

        # Load locations for substring matching
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "egypt_places_dummy.csv")
        self.df_locations = load_locations(csv_path)

    # ── Core prediction ───────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Parse a free-text query into structured outfit parameters.

        Returns
        -------
        {
            "occasion":      str | None,
            "weather_class": str | None,
            "weather":       {"temperature": float} | None,
            "style":         str | None,
            "confidence":    {"occasion": float|None, "weather": float|None, "style": float|None},
        }
        """
        explicit_temp = _extract_explicit_temperature(text)

        i_lg, o_lg, w_lg, s_lg = self._backend.logits(text)

        # Backend returned None (OOV guard on BiLSTM or empty input)
        if o_lg is None:
            return self._null_result()

        if i_lg is not None:
            i_probs = F.softmax(i_lg, dim=1)
            i_conf, i_idx = i_probs.max(dim=1)
            intent = self.idx2intent[i_idx.item()]
            i_conf_v = i_conf.item()
        else:
            intent = "OUTFIT_REQUEST"
            i_conf_v = 1.0

        o_probs = F.softmax(o_lg, dim=1)
        w_probs = F.softmax(w_lg, dim=1)
        s_probs = F.softmax(s_lg, dim=1)

        o_conf, o_idx = o_probs.max(dim=1)
        w_conf, w_idx = w_probs.max(dim=1)
        s_conf, s_idx = s_probs.max(dim=1)

        o_conf_v = o_conf.item()
        w_conf_v = w_conf.item()
        s_conf_v = s_conf.item()

        occasion = (self.idx2occasion[o_idx.item()]
                    if o_idx.item() < len(OCCASIONS) and o_conf_v >= OCCASION_CONF_THRESH else None)
        style    = (self.idx2style[s_idx.item()]
                    if s_idx.item() < len(STYLES) and s_conf_v >= STYLE_CONF_THRESH else None)

        if explicit_temp is not None:
            weather_class = _temp_to_class(explicit_temp)
            temperature   = explicit_temp
            condition     = self.WEATHER_TO_CONDITION[weather_class]
        elif w_idx.item() < len(WEATHER_CLASSES) and w_conf_v >= WEATHER_CONF_THRESH:
            weather_class = self.idx2weather[w_idx.item()]
            temperature   = self.WEATHER_TO_TEMP[weather_class]
            condition     = self.WEATHER_TO_CONDITION[weather_class]
        else:
            weather_class = None
            temperature   = None
            condition     = None

        # ── Location Matching Override ─────────────────────────────────────────
        def normalize_str(s: str) -> str:
            import re
            s = s.lower()
            s = re.sub(r"[^\w\s]", " ", s)  # replace punctuation with space
            return " ".join(s.split())

        matched_location = None
        if not self.df_locations.empty:
            text_norm = normalize_str(text)
            # Strategy 1: Substring match on normalized strings
            for place in self.df_locations['place_name'].values:
                place_norm = normalize_str(str(place))
                if place_norm and place_norm in text_norm:
                    matched_location = place
                    break
            
            # Strategy 2: If no match, try token-set containment (excluding common stop words)
            if not matched_location:
                stop_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "of", "for", "with", "el"}
                text_words = set(text_norm.split())
                for place in self.df_locations['place_name'].values:
                    place_norm = normalize_str(str(place))
                    place_words = [w for w in place_norm.split() if w not in stop_words]
                    # If all significant words of the place name are in the query
                    if place_words and all(pw in text_words for pw in place_words):
                        matched_location = place
                        break
            
            # Strategy 3: If still no match, try no-space substring matching
            if not matched_location:
                text_no_space = text_norm.replace(" ", "")
                for place in self.df_locations['place_name'].values:
                    place_no_space = normalize_str(str(place)).replace(" ", "")
                    if len(place_no_space) >= 5 and place_no_space in text_no_space:
                        matched_location = place
                        break
        
        if matched_location:
            print(f"\n[NLPInference] Detected location: {matched_location}")
            loc_details = get_location_details(self.df_locations, matched_location)
            cat = loc_details.get("category", "")
            lat = loc_details.get("lat", 30.0444)
            lng = loc_details.get("lng", 31.2357)
            
            # Override occasion and weather using the realtime API
            occasion = map_category_to_occasion(cat)
            api_weather = fetch_realtime_weather(lat, lng)
            temperature = api_weather["temperature"]
            condition = api_weather["condition"]
            weather_class = _temp_to_class(temperature)
            
            # Force the intent to OUTFIT_REQUEST so the dialogue manager doesn't ignore it
            intent = "OUTFIT_REQUEST"
            i_conf_v = 1.0
            
            print(f"               Mapped to Occasion: {occasion}, Temp: {temperature}C")

        # ── Keyword Matching Fallback Override ─────────────────────────────────
        if not matched_location:
            import re
            text_lower = text.lower()
            
            def has_word(w_list, text_val):
                words = re.findall(r"[a-z]+", text_val)
                return any(w in words for w in w_list)

            # Occasion keywords matching
            occ_keywords = {
                "casual": ["casual", "hangout", "mall", "everyday", "relaxing", "chill", "informal", "cafe", "coffeeshop", "coffee", "restaurant", "movies", "cinema", "brunch", "lunch", "shopping", "stroll", "walk"],
                "formal": ["formal", "wedding", "black tie", "fancy", "elegant", "ceremony", "gala", "banquet", "prom", "graduation", "reception"],
                "business": ["business", "office", "work", "interview", "meeting", "job", "corporate", "conference", "presentation", "professional", "boardroom"],
                "sport": ["sport", "gym", "workout", "running", "athletic", "exercise", "pool", "swim", "training", "jogging", "yoga"],
                "party": ["party", "club", "dinner", "date", "night out", "celebration", "gathering", "birthday", "nightclub", "evening"],
                "outdoor": ["outdoor", "hiking", "camping", "park", "picnic", "outside", "trail", "beach", "nature"]
            }

            # Weather keywords matching
            wea_keywords = {
                "hot": ["hot", "warm", "sunny", "boiling", "summer", "heat", "scorching", "sweltering", "humid", "tropical"],
                "mild": ["mild", "nice", "pleasant", "spring", "cool", "breezy", "windy", "cloudy", "comfortable", "perfect"],
                "cold": ["cold", "freezing", "chilly", "winter", "snow", "frosty", "shivering", "icy", "arctic"]
            }

            # Check and override occasion
            found_occ = None
            for key, kws in occ_keywords.items():
                if has_word(kws, text_lower):
                    found_occ = key
                    break
            
            if found_occ:
                occasion = found_occ
                if o_conf_v < 0.6:
                    o_conf_v = 0.95

            # Check and override weather
            found_wea = None
            for key, kws in wea_keywords.items():
                if has_word(kws, text_lower):
                    found_wea = key
                    break

            if found_wea and explicit_temp is None:
                weather_class = found_wea
                temperature = self.WEATHER_TO_TEMP[weather_class]
                condition = self.WEATHER_TO_CONDITION[weather_class]
                if w_conf_v < 0.6:
                    w_conf_v = 0.95

            # Force intent to OUTFIT_REQUEST if we successfully matched occasion or weather keywords
            if found_occ or found_wea:
                if i_conf_v < 0.6 or intent in ["OTHER", "SMALL_TALK"]:
                    intent = "OUTFIT_REQUEST"
                    i_conf_v = 0.95

        # ── Color & Piece Extraction ──────────────────────────────────────────
        text_lower = text.lower()
        import re
        
        # Color match
        supported_colors = [
            "blue", "red", "black", "white", "green", "pink", "purple", "orange", 
            "brown", "grey", "gray", "beige", "navy", "turquoise", "cream", "khaki", 
            "gold", "silver"
        ]
        color = None
        words = re.findall(r"[a-z]+", text_lower)
        for col in supported_colors:
            if col in words:
                color = col
                if col == "gray":
                    color = "grey"
                break
                
        # Piece match
        piece_keywords = {
            "skirt": ["skirt", "skirts"],
            "pants": ["pants", "pants", "trouser", "trousers", "jean", "jeans"],
            "shorts": ["shorts"],
            "dress": ["dress", "dresses", "gown", "gowns"],
            "shirt": ["shirt", "shirts", "blouse", "blouses", "top", "tops", "tshirt", "tshirts", "t-shirt", "t-shirts"],
            "jacket": ["jacket", "jackets", "coat", "coats", "blazer", "blazers", "cardigan", "cardigans"],
            "accessory": ["accessory", "accessories", "bag", "bags", "hat", "hats", "sunglasses", "belt", "belts"],
            "shoes": ["shoes", "shoe", "boot", "boots", "sneaker", "sneakers", "sandal", "sandals", "heels", "heel", "flip flop", "flip flops", "espadrille", "espadrilles"]
        }
        piece = None
        for p_cat, p_kws in piece_keywords.items():
            if any(kw in words for kw in p_kws):
                piece = p_cat
                break
                
        # Force intent if color or piece is specified
        if color or piece:
            if i_conf_v < 0.6 or intent in ["OTHER", "SMALL_TALK"]:
                intent = "OUTFIT_REQUEST"
                i_conf_v = 0.95

        weather_dict = (
            {"temperature": temperature, "condition": condition}
            if temperature is not None else None
        )

        return {
            "intent":        intent,
            "occasion":      occasion,
            "weather_class": weather_class,
            "weather":       weather_dict,
            "style":         style,
            "color":         color,
            "piece":         piece,
            "confidence": {
                "intent":   i_conf_v,
                "occasion": round(o_conf_v, 3) if occasion else None,
                "weather":  round(w_conf_v, 3) if weather_class else None,
                "style":    round(s_conf_v, 3) if style else None,
            },
        }

    @staticmethod
    def _null_result():
        return {
            "intent":        "OTHER",
            "occasion":      None,
            "weather_class": None,
            "weather":       None,
            "style":         None,
            "color":         None,
            "piece":         None,
            "confidence":    {}
        }


# ── Coverage evaluation utility ───────────────────────────────────────────────

def evaluate_coverage(texts, model_dir: Optional[str] = None) -> dict:
    """
    Run a list of queries through predict() and report per-head coverage stats.
    """
    infer = NLPInference(model_dir=model_dir)
    n = len(texts)
    occ_hits = wea_hits = sty_hits = 0
    occ_cls: dict = {}
    wea_cls: dict = {}
    sty_cls: dict = {}

    for text in texts:
        r = infer.predict(text)
        if r["occasion"]:
            occ_hits += 1
            occ_cls[r["occasion"]] = occ_cls.get(r["occasion"], 0) + 1
        if r["weather_class"]:
            wea_hits += 1
            wea_cls[r["weather_class"]] = wea_cls.get(r["weather_class"], 0) + 1
        if r["style"]:
            sty_hits += 1
            sty_cls[r["style"]] = sty_cls.get(r["style"], 0) + 1

    return {
        "n_queries": n,
        "occasion": {"coverage": occ_hits / n if n else 0.0, "per_class": occ_cls},
        "weather":  {"coverage": wea_hits / n if n else 0.0, "per_class": wea_cls},
        "style":    {"coverage": sty_hits / n if n else 0.0, "per_class": sty_cls},
    }


# ── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    infer = NLPInference()
    messy_tests = [
        "idk what to wear its kinda cold",
        "something cute for dinner but not too much",
        "going out but like chill vibes",
        "gym today and its literally boiling outside",
        "need smth for a formal event its freezing",
        "date night outfit pls its cold",
        "I have a wedding tonight, it is absolutely freezing.",
        "Hitting the gym, it is 35 degrees outside.",
        "",
        "asdfjkl qwerty zzz",
    ]
    for t in messy_tests:
        r = infer.predict(t)
        print(f"Input : {t!r}")
        print(f"Output: occ={r['occasion']} wea={r['weather_class']} sty={r['style']} conf={r['confidence']}\n")
