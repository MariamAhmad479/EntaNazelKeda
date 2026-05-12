# NLP Component — Handoff Document

**Owner:** NLP team  
**Status:** Complete — 62/62 tests passing  
**Last trained:** 2026-05-13 (20,000 samples, 20 epochs)

---

## What This Component Does

Parses a free-text user query into structured outfit parameters consumed by the recommendation engine:

```python
{
    "occasion":      str | None,   # "casual" | "formal" | "business" | "sport" | "party" | "outdoor"
    "weather_class": str | None,   # "hot" | "mild" | "cold"
    "weather":       {"temperature": float} | None,
    "style":         str | None,   # "classic" | "streetwear" | "bohemian" | "minimalist" | "preppy" | "athletic"
}
```

`None` means the user did not mention that dimension — the recommendation engine should treat it as "no preference."

---

## File Structure

```
nlp/
├── __init__.py          # Exports NLPInference
├── dataset.py           # Synthetic data generation, vocabulary, PyTorch dataset
├── model.py             # WardrobeNLPModel (LSTM + 3 output heads)
├── train.py             # Training script — run this to produce saved_models/
├── inference.py         # NLPInference class — used at runtime
├── saved_models/
│   ├── nlp_model.pth    # Trained weights
│   ├── vocab.pkl        # Vocabulary (word → index)
│   └── model_config.json  # Model dimensions (loaded at inference time)
└── HANDOFF.md           # This file

nlp_test.py              # Interactive manual test script (project root)
tests/
└── test_nlp.py          # 62 automated tests
```

---

## How to Use (Integration)

```python
from nlp import NLPInference

nlp = NLPInference("nlp/saved_models")
result = nlp.predict("I need a formal outfit for a wedding, it's freezing outside")
# → {"occasion": "formal", "weather_class": "cold", "weather": {"temperature": 4}, "style": None}
```

The `weather["temperature"]` value is in **Celsius** and maps directly into the recommendation engine's `context_filter` temperature bands:
- `hot` → 32°C (≥ 25°C band)
- `mild` → 18°C (16–24°C band)
- `cold` → 4°C (≤ 15°C band)

If the user types an explicit temperature (e.g. "35 degrees", "90°F"), the regex extractor overrides the model and returns the exact converted value.

---

## Model Architecture

**Multi-task LSTM classifier** with a shared encoder and three independent output heads.

```
Input text
    ↓
preprocess_text()        # lowercase, strip punctuation/digits, tokenise
    ↓
Embedding layer          # vocab_size × 64
    ↓
LSTM (hidden=128)        # unidirectional, batch_first
    ↓
Mean pool over sequence  # gives equal weight to every token position
    ↓
┌──────────────┬──────────────┬──────────────┐
│ occasion_head│ weather_head │  style_head  │
│  Linear→6   │  Linear→3   │  Linear→6   │
└──────────────┴──────────────┴──────────────┘
```

**Key design choices:**
- **Mean pooling** (not last-hidden-state): ensures keywords anywhere in the sentence (beginning, middle, end) contribute equally. Critical for queries like "I need a *formal* outfit for a *wedding*".
- **Three separate loss functions** with `ignore_index`: samples where a dimension is absent from the text do not penalise that head. This lets each head specialise independently.
- **Confidence thresholds** per head: softmax probability must exceed the threshold or the head returns `None`. Prevents forced predictions on ambiguous or OOV inputs.
- **OOV guard**: if >60% of input tokens are unknown words, all heads return `None` immediately (no model call).
- **Regex pre-pass**: explicit temperatures ("35°C", "90°F", "minus 10 degrees") are extracted before the model runs and override the weather head.

**Final validation accuracy (20 epochs, 20k samples):**
| Head | Accuracy |
|---|---|
| Occasion | 99% |
| Weather | 91% |
| Style | 100% |

---

## Training Data

Entirely **synthetic** — generated from template pools × synonym lists. No external dataset is needed or used.

**Template pools** (each pool assigns only the labels its templates contain):
| Pool | Dimensions labelled | Share |
|---|---|---|
| `TEMPLATES_BOTH` | occasion + weather | 38% |
| `TEMPLATES_STYLE_BOTH` | style + occasion + weather | 17% |
| `TEMPLATES_OCCASION_ONLY` | occasion | 10% |
| `TEMPLATES_WEATHER_ONLY` | weather | 10% |
| `TEMPLATES_STYLE_OCCASION` | style + occasion | 10% |
| `TEMPLATES_STYLE_WEATHER` | style + weather | 15% |

**Synonym coverage (selected examples):**
- Casual: `cafe`, `coffeeshop`, `restaurant`, `movies`, `brunch`, `shopping`, `stroll`, `outing`, …
- Sport: `gym`, `yoga`, `cycling`, `tennis`, `crossfit`, `pilates`, `basketball`, …
- Outdoor: `hiking`, `camping`, `beach`, `festival`, `bbq`, `national park`, …
- Hot: `humid`, `scorching`, `muggy`, `heatwave`, `35 degrees`, `90 degrees`, …
- Mild: `breezy`, `windy`, `cloudy`, `overcast`, `drizzle`, …
- Cold: `freezing`, `arctic`, `blizzard`, `minus 10`, `below freezing`, …

---

## Confidence Thresholds

```python
OCCASION_CONF_THRESH = 0.25   # 6 classes, random baseline = 0.167
WEATHER_CONF_THRESH  = 0.35   # 3 classes, random baseline = 0.333
STYLE_CONF_THRESH    = 0.30   # 6 classes, random baseline = 0.167
```

Lower than typical because the model must also handle partial queries (occasion-only, weather-only) where the absent head legitimately has low confidence.

---

## How to Retrain

```bash
# From the project root, using the project venv:
D:\nlp_env\Scripts\python.exe nlp/train.py
```

This regenerates `nlp/saved_models/nlp_model.pth`, `vocab.pkl`, and `model_config.json`. Training takes ~2–3 minutes on CPU. Adjust `num_samples` in `train.py` (currently 20,000) or `epochs` (currently 20) as needed.

---

## How to Run Tests

```bash
# All 62 tests (unit + integration):
D:\nlp_env\Scripts\python.exe -m pytest tests/test_nlp.py -v -p no:cacheprovider

# Unit tests only (no trained model needed):
D:\nlp_env\Scripts\python.exe -m pytest tests/test_nlp.py -v -p no:cacheprovider -k "not Integration"
```

The `-p no:cacheprovider` flag prevents pytest from writing a cache to disk (required if C: drive is near full).

---

## Manual Testing

```bash
D:\nlp_env\Scripts\python.exe nlp_test.py
```

Type any free-text query and press Enter. Type `quit` to exit.

---

## What Was Fixed / Built (vs. Original Code)

| Problem | Fix |
|---|---|
| Templates assigned labels for dimensions absent from the text | Split into 6 typed template pools; `None` labels for absent dimensions |
| Style never extracted | Added `STYLES`, `STYLE_SYNONYMS`, `style_head`, style output in `predict()` |
| `cold` mapped to 10°C (wrong recommendation engine band) | Changed to 4°C (firmly in ≤15°C band) |
| Explicit temperature numbers ignored | Added `_extract_explicit_temperature()` regex pre-pass |
| Model forced a prediction on every input | Added per-head softmax confidence thresholds |
| OOV-heavy inputs still predicted above threshold | Added >60% OOV guard — returns all `None` |
| Empty tokens from digit-stripping | `preprocess_text` now filters with `if t` |
| Last-hidden-state LSTM missed keywords early in sentence | Switched to mean pooling over all LSTM outputs |
| `nlp/` not importable as a package | Created `nlp/__init__.py` |
| No tests | Created `tests/test_nlp.py` with 62 tests across 7 groups |
| Dataset too small (4k samples) | Expanded to 20k training samples |
| Vocab too narrow (few synonyms) | Added 30+ synonyms per category + weather/style |
| Too few templates (narrow phrasing) | Added 50+ new templates across all pools |

---

## Known Limitations

- **Vocabulary is closed**: words not seen during training become `<UNK>`. Very unusual phrasing or brand names (e.g. "Supreme drop", "Patagonia fleece") will not parse correctly. Retrain with expanded synonyms to add new vocabulary.
- **Weather accuracy is 91%** (lower than the other heads) because weather words overlap with style/occasion contexts (e.g. "summer" = hot weather, but also a style vibe). This is the main remaining error source.
- **No conversational memory**: each `predict()` call is stateless. Follow-up queries ("actually it's colder") work only if the weather word appears in the follow-up text itself.
- **Single language**: English only.

---

## Integration Notes for `main.py` / Recommendation Engine

The NLP output now includes `style`. Whoever maintains `main.py` can pass it through:

```python
result = nlp.predict(user_query)
outfits = get_outfits(
    occasion=result["occasion"],
    weather=result["weather"],
    style=result["style"],          # ← new field, was missing before
)
```

`style` will be `None` when the user does not mention a style preference — the recommendation engine should treat this as "any style acceptable."
