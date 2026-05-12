import json
import os
import pickle
import re
import sys
from typing import Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import preprocess_text, OCCASIONS, WEATHER_CLASSES, STYLES
from nlp.model import WardrobeNLPModel

# Confidence thresholds: ~2× the random-guess baseline for each head
# 6 occasion classes → baseline 0.167, threshold 0.40
# 3 weather classes  → baseline 0.333, threshold 0.45
# 6 style classes    → baseline 0.167, threshold 0.35
OCCASION_CONF_THRESH = 0.25
WEATHER_CONF_THRESH  = 0.35
STYLE_CONF_THRESH    = 0.30


class NLPInference:
    def __init__(self, model_dir: str = 'nlp/saved_models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(os.path.join(model_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)

        with open(os.path.join(model_dir, 'model_config.json')) as f:
            cfg = json.load(f)

        self.model = WardrobeNLPModel(**cfg)
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, 'nlp_model.pth'), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.idx2occasion = {idx: occ for idx, occ in enumerate(OCCASIONS)}
        self.idx2weather  = {idx: wea for idx, wea in enumerate(WEATHER_CLASSES)}
        self.idx2style    = {idx: sty for idx, sty in enumerate(STYLES)}

        # Mid-band values that land firmly inside each context_filter temperature range:
        #   hot  → ≥25°C (summer band)
        #   mild → 16-24°C (spring/autumn band)
        #   cold → ≤5°C (winter band)   ← old value 10 was in the wrong autumn/spring band
        self.weather_to_temp = {"hot": 32, "mild": 18, "cold": 4}

    # ------------------------------------------------------------------
    # Rule-based temperature extraction (runs before the model)
    # ------------------------------------------------------------------

    def _extract_explicit_temperature(self, text: str) -> Optional[float]:
        """Return a Celsius float if an explicit temperature is found, else None."""
        t = text.lower()

        m = re.search(r'(?:minus|negative)\s+(\d+(?:\.\d+)?)', t)
        if m:
            return -float(m.group(1))

        m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?\s+c(?:elsius)?)', t)
        if m:
            return float(m.group(1))

        m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°\s*f|degrees?\s+f(?:ahrenheit)?)', t)
        if m:
            return (float(m.group(1)) - 32) * 5 / 9

        # Bare "X degrees" or "X°" — assume Celsius if value < 50, Fahrenheit otherwise
        m = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:°|degrees?)', t)
        if m:
            v = float(m.group(1))
            return v if v < 50 else (v - 32) * 5 / 9

        return None

    def _temp_to_class(self, temp: float) -> str:
        if temp <= 15:
            return "cold"
        if temp <= 24:
            return "mild"
        return "hot"

    # ------------------------------------------------------------------
    # Main prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> dict:
        """
        Parse a free-text query into structured outfit parameters.

        Returns:
            {
                "occasion":      str | None,
                "weather_class": str | None,
                "weather":       {"temperature": float} | None,
                "style":         str | None,
            }
        """
        explicit_temp = self._extract_explicit_temperature(text)

        tokens = preprocess_text(text)
        if not tokens:
            return {"occasion": None, "weather_class": None, "weather": None, "style": None}

        unk_idx = self.vocab.word2idx["<UNK>"]
        encoded = [self.vocab.word2idx.get(t, unk_idx) for t in tokens]

        # Abstain when the query is mostly unrecognised words
        oov_ratio = sum(1 for idx in encoded if idx == unk_idx) / len(encoded)
        if oov_ratio > 0.6:
            return {"occasion": None, "weather_class": None, "weather": None, "style": None}
        input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            occ_logits, wea_logits, sty_logits = self.model(input_ids)

            occ_probs = F.softmax(occ_logits, dim=1)
            wea_probs = F.softmax(wea_logits, dim=1)
            sty_probs = F.softmax(sty_logits, dim=1)

            occ_conf, occ_idx = occ_probs.max(dim=1)
            wea_conf, wea_idx = wea_probs.max(dim=1)
            sty_conf, sty_idx = sty_probs.max(dim=1)

        occasion = (self.idx2occasion[occ_idx.item()]
                    if occ_conf.item() >= OCCASION_CONF_THRESH else None)

        if explicit_temp is not None:
            temperature   = explicit_temp
            weather_class = self._temp_to_class(explicit_temp)
        elif wea_conf.item() >= WEATHER_CONF_THRESH:
            weather_class = self.idx2weather[wea_idx.item()]
            temperature   = self.weather_to_temp[weather_class]
        else:
            weather_class = None
            temperature   = None

        style   = (self.idx2style[sty_idx.item()]
                   if sty_conf.item() >= STYLE_CONF_THRESH else None)
        weather = {"temperature": temperature} if temperature is not None else None

        return {
            "occasion":      occasion,
            "weather_class": weather_class,
            "weather":       weather,
            "style":         style,
        }


if __name__ == "__main__":
    infer = NLPInference()
    tests = [
        "I have a wedding tonight, it is absolutely freezing.",
        "Hitting the gym, it is 35 degrees outside.",
        "Something minimal for a casual hangout, nice weather.",
        "I need a formal outfit for tonight, it's very cold.",
        "What should I wear to the gym? It's pretty hot outside.",
        "",
        "asdfjkl qwerty zzz",
    ]
    for t in tests:
        print(f"Input:  {t!r}")
        print(f"Output: {infer.predict(t)}\n")
