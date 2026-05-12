# Smart Wardrobe — Recommendation Engine

**Person 2 Component** for the ML Project: an outfit recommendation engine that matches wardrobe items into outfits using machine learning.

## Features

- **Wardrobe Management** — Load/save/query a digital wardrobe (JSON)
- **Feature Encoding** — Convert clothing attributes → numerical vectors (one-hot, multi-hot, normalised scalars)
- **K-Means Clustering** — Group similar clothes with auto-k selection (Elbow Method)
- **Compatibility Scoring** — Multi-factor outfit scoring:
  - Color harmony (HSV complementary/analogous analysis)
  - Style coherence
  - Formality matching
  - Cosine similarity of feature vectors
- **Outfit Generation** — Slot-based outfit assembly (top + bottom + shoes, or dress + shoes)
- **Context Filtering** — Filter by occasion, weather/temperature, and style
- **User Feedback** — Accept/reject loop with XGBoost weight retraining

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python -m recommendation_engine.api

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
recommendation_engine/
├── data_models.py       # Enums & ClothingItem dataclass
├── wardrobe_manager.py  # Load/save/query wardrobe
├── feature_encoder.py   # Item → feature vector encoding
├── clustering.py        # K-Means clustering
├── compatibility.py     # Outfit compatibility scoring
├── outfit_generator.py  # Candidate outfit generation
├── context_filter.py    # Occasion/weather/style filters
├── feedback.py          # Accept/reject + weight learning
└── api.py               # Public API (for Person 3)
```

## API Usage (for Person 3)

```python
from recommendation_engine import RecommendationAPI

api = RecommendationAPI("data/sample_wardrobe.json")

# Get outfit recommendations
outfits = api.get_outfits(occasion="formal", weather={"temperature": 20}, top_n=5)

# Submit feedback
api.submit_feedback(outfits[0]["outfit_id"], "accept")

# Retrain weights from feedback
api.retrain()
```

## Tech Stack

- Python 3.9+
- scikit-learn (K-Means, cosine similarity)
- XGBoost (feedback-based weight learning)
- NumPy / Pandas
