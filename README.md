# AI Stylist Application

**Faster, Smarter, Personal.**

An AI-powered virtual stylist that digitizes your wardrobe, understands natural-language style requests, and generates personalized, context-aware outfit recommendations — accounting for weather, location, occasion, and your evolving fashion taste.

## Team

- Judy Elsheikh
- Rokaya Alsakka
- Mariam Mohamed

## The Problem

Choosing outfits daily is still a frustrating manual process. Users struggle with:

- Decision fatigue every morning
- Forgetting what exists inside their wardrobe
- Matching outfits correctly
- Dressing appropriately for weather or occasions
- Receiving generic recommendations that ignore personal taste

Existing fashion apps largely lack real personalization, context awareness, and an intelligent understanding of style.

## Our Solution

An AI-powered stylist that understands your wardrobe, your fashion preferences, your location, live weather conditions, and your requested style or occasion. The system digitizes clothes automatically, analyzes user requests conversationally, and generates personalized outfit recommendations in real time.

## Core Features

### 1. Smart Clothing Digitization
Users upload photos of their clothes directly into the app. The system automatically:
- Removes the background
- Detects clothing type
- Identifies colors
- Predicts season and usage
- Stores the item inside a digital wardrobe

This eliminates manual wardrobe organization entirely.

### 2. Weather & Location Awareness
The system considers real-time weather and location before generating outfits:
1. User enters a destination
2. The app retrieves live weather data
3. The recommendation engine filters out unsuitable clothing automatically

*Example: heavy winter jackets are excluded during hot weather.*

### 3. Conversational AI Stylist
Users communicate with the system using natural language, e.g. *"I need a cozy preppy outfit for a cold business meeting."* The NLP model extracts:
- Occasion
- Weather
- Style preference

...then generates suitable outfit combinations instantly.

### 4. Intelligent Outfit Generation
The recommendation engine combines clothing items into complete outfits by considering:
- Color harmony
- Style compatibility
- Season compatibility
- Formality consistency

Invalid combinations are automatically prevented.

### 5. Personalized Learning System
Users can accept or reject suggested outfits, and the system learns continuously from these interactions — recommendations become increasingly personalized to the user's fashion preferences over time.

## System Architecture

### Computer Vision Pipeline
Clothing digitization runs through three stages:

1. **Background Segmentation** — Uses the U-2-Net segmentation model to isolate the clothing item from the image, removing shadows, hangers, and room clutter to produce a clean, centered image.
2. **Multi-Attribute CNN Classification** — A MobileNetV2-based CNN predicts multiple clothing attributes simultaneously (category, article type, color, season, usage) in a single forward pass.
3. **YOLOv11 Validation & Error Correction** — A YOLOv11 object detection layer validates CNN predictions and corrects major classification errors by detecting actual clothing boundaries.
4. **Confidence-Based Hybrid Prediction** — Combines both models: when YOLO confidence is high, it can directly override incorrect CNN predictions; at lower confidence, YOLO acts as a semantic family filter restricting the classifier to logically related categories rather than fully replacing the prediction.

### Clothing Recommendation Engine
- **Feature Encoding** — Each garment is converted into a 24-dimensional numerical feature vector encoding category, occasion, season, warmth level, formality level, and color.
- **K-Means Clustering** — Groups wardrobe items into style-based clusters for efficient recommendation; the optimal number of clusters is determined via the Elbow Method.
- **Similarity Search (KNN + Cosine Similarity)** — Identifies stylistically compatible items by measuring closeness between clothing vectors in feature space.
- **Color Harmony Engine** — Applies color theory (monochromatic, complementary, analogous, triadic) to generate visually balanced outfits.
- **Weather & Location Pipeline** — Integrates GeoPy/Nominatim and the Open-Meteo API to convert user locations into coordinates and retrieve live weather data for context-aware filtering.
- **Global Store Fallback Recommendation** — If no suitable outfit exists in the user's wardrobe, the system generates fallback outfit suggestions from H&M based on occasion and weather.

### NLP Intent Understanding
- **Transformer-Based Language Model (DistilBERT)** — A lightweight transformer that understands user input in context, using sub-word tokenization to handle typos and informal language while maintaining real-time performance.
- **Intent Detection** — Identifies the user's goal (outfit request, question, feedback) to route messages to the correct response logic.
- **Occasion Detection** — Classifies requests into occasions such as casual, formal, business, party, or sportswear.
- **Weather & Context Understanding** — Detects environmental cues from text and maps them to hot, cold, or mild conditions.
- **Style Preference Extraction** — Extracts style descriptors such as streetwear, minimalist, preppy, or athletic.
- **Training & Data Strategy** — Trained on a custom synthetic query dataset (built to address data scarcity) combined with public intent datasets, using augmentation techniques like slang, emojis, and typos to handle noisy real-world input.

### Personalized Recommendation Learning
- **User Feedback Collection** — Records accepted and rejected outfit recommendations.
- **Feature-Based Re-Ranking (XGBoost)** — Evaluates outfit quality using color harmony, style coherence, formality matching, and item similarity scores.
- **Dynamic Personalization** — The system automatically retrains as feedback accumulates, adapting recommendation behavior to each user's evolving fashion preferences.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Computer Vision | U-2-Net, MobileNetV2, YOLOv11 |
| NLP | DistilBERT |
| Machine Learning | K-Means, K-Nearest Neighbors, XGBoost |
| APIs | GeoPy / Nominatim, Open-Meteo |

## How It Works — End to End

1. User uploads photos of their clothes → CV pipeline digitizes and classifies each item into the wardrobe.
2. User sends a natural-language request (e.g. style, occasion, destination) → NLP pipeline extracts intent, occasion, style, and weather context.
3. Recommendation engine encodes wardrobe items, clusters by style, and searches for compatible combinations using color harmony, formality, and season rules — filtered by live weather/location data.
4. If no suitable in-wardrobe outfit exists, the system falls back to H&M suggestions.
5. User accepts or rejects the outfit → feedback feeds the XGBoost re-ranking model, which continuously personalizes future recommendations.

## Future Work

- Expand fallback retail integrations beyond H&M
- Improve YOLOv11/CNN hybrid accuracy on low-quality or poorly lit images
- Broaden the NLP intent dataset for greater linguistic and cultural coverage
- Add multi-user/household wardrobe support
