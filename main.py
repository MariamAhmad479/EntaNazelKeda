import os
import sys

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI
from nlp.inference import NLPInference

# ---------------------------------------------------------------------------
# Dialogue States
# ---------------------------------------------------------------------------
AWAITING_QUERY      = "AWAITING_QUERY"
CLARIFICATION_NEEDED = "CLARIFICATION_NEEDED"
SHOWING_RESULTS     = "SHOWING_RESULTS"


# ---------------------------------------------------------------------------
# WardrobeChatbot — NLP-driven dialogue manager
# ---------------------------------------------------------------------------
class WardrobeChatbot:
    def __init__(self, api: RecommendationAPI, nlp_model: NLPInference):
        self.api  = api
        self.nlp  = nlp_model
        self.state = AWAITING_QUERY
        self.context = {
            "occasion":     None,
            "weather":      None,   # dict {"temperature": int, "condition": str} or None
            "weather_class": None,  # "HOT" / "MILD" / "COLD"
            "style":        None,
        }
        self.last_recommendations = []

    # ------------------------------------------------------------------
    def _reset_context(self):
        self.context = {
            "occasion":      None,
            "weather":       None,
            "weather_class": None,
            "style":         None,
        }
        self.last_recommendations = []
        self.state = AWAITING_QUERY

    # ── Rejection / feedback keyword pre-filter ──────────────────────
    # The NLP model hallucinates weather+occasion from rejection phrases
    # (e.g. "I don't like these" → mild+casual with 0.80 confidence).
    # We intercept those BEFORE the NLP to protect context integrity.
    _REJECTION_KEYWORDS = [
        "don't like", "dont like", "do not like",
        "not these", "not this", "hate this", "hate these",
        "these are bad", "this is bad", "not good",
        "show me something", "something else", "different outfit",
        "other options", "other outfits", "try again",
        "none of these", "none of those",
        "no i don't", "no i dont", "nah", "nope",
    ]
    _FEEDBACK_KEYWORDS = [
        "more casual", "more formal", "more sporty",
        "less formal", "less casual",
        "make it", "something more", "something less",
        "prefer", "rather have",
    ]

    @staticmethod
    def _detect_rejection(text: str) -> str | None:
        """Return 'REJECTION', 'FEEDBACK', or None based on keyword matching."""
        t = text.lower()
        for kw in WardrobeChatbot._FEEDBACK_KEYWORDS:
            if kw in t:
                return "FEEDBACK"
        for kw in WardrobeChatbot._REJECTION_KEYWORDS:
            if kw in t:
                return "REJECTION"
        # Short inputs while showing results are almost always rejections
        # e.g. "no", "nope", "nah", "no thanks"
        words = t.split()
        if len(words) <= 3 and any(w in t for w in ["no", "nope", "nah", "bad", "ugh", "eh"]):
            return "REJECTION"
        return None

    # ------------------------------------------------------------------
    def handle_input(self, user_input: str) -> str:
        import random
        raw_text = user_input.strip()

        # ── 0. Rule-based pre-filter (runs BEFORE NLP) ───────────────
        # Catches rejection/feedback phrases the model misclassifies.
        rule_intent = self._detect_rejection(raw_text)
        if rule_intent == "REJECTION":
            if self.state == SHOWING_RESULTS and self.context["occasion"] and self.context["weather_class"]:
                return self._generate_recommendations(is_retry=True)
            self._reset_context()
            return "Okay, let's start fresh. What kind of outfit are you looking for?"
        if rule_intent == "FEEDBACK":
            if self.state == SHOWING_RESULTS and self.context["occasion"] and self.context["weather_class"]:
                return self._generate_recommendations(is_retry=True)
            return "Could you be more specific? Like 'make it more casual' or 'something for the cold'?"

        # ── 1. NLP call ──────────────────────────────────────────────
        print("\n* Thinking...")
        extracted = self.nlp.predict(raw_text)

        intent       = extracted.get("intent", "OTHER")
        conf         = extracted.get("confidence", {})
        intent_conf  = conf.get("intent") or 0.0

        if intent_conf < 0.6:
            intent = "OTHER"

        occ_conf = conf.get("occasion") or 0.0
        wea_conf = conf.get("weather")  or 0.0
        sty_conf = conf.get("style")    or 0.0

        # ── 2. Intent routing: terminal intents (no context change) ──
        if intent == "SMALL_TALK":
            greetings = [
                "Hello! I'm your Smart Wardrobe Assistant. Tell me where you're going and what the weather is like!",
                "Hi there! Ready to pick an outfit?",
                "Hey! Where are you heading today?",
            ]
            if self.state == SHOWING_RESULTS:
                self._reset_context()
                return "Glad I could help! Let me know when you need another outfit."
            return random.choice(greetings)

        if intent == "OTHER":
            if self.state == SHOWING_RESULTS:
                self._reset_context()
                return "Okay, I've cleared your previous search. What kind of outfit do you need now?"
            return ("I didn't quite catch any outfit details. "
                    "Do you want outfit suggestions or need help deciding what to wear?")

        if intent == "CLARIFICATION":
            return (
                "I can help you pick an outfit from your wardrobe!\n"
                "Just tell me the occasion (e.g., formal, casual, sport, party) "
                "and the weather (hot, cold, mild).\n"
                "Example: 'I need something for a formal event and it's freezing outside.'"
            )

        # ── 3. REJECTION / FEEDBACK: handle BEFORE touching context ──
        if intent == "REJECTION":
            if self.state == SHOWING_RESULTS and self.context["occasion"] and self.context["weather_class"]:
                return self._generate_recommendations(is_retry=True)
            self._reset_context()
            return "Okay, let's start fresh. What kind of outfit are you looking for?"

        if intent == "FEEDBACK":
            if self.state == SHOWING_RESULTS and self.context["occasion"] and self.context["weather_class"]:
                return self._generate_recommendations(is_retry=True)
            return "Could you be more specific? Like 'make it more casual' or 'something for the cold'?"

        # ── 4. Context update for OUTFIT_REQUEST ─────────────────────
        #
        # We trust the NLP model's extractions because it is now trained
        # to explicitly output an UNKNOWN class when a slot is absent.
        updated_something = False

        if extracted.get("occasion"):
            self.context["occasion"] = extracted["occasion"]
            updated_something = True

        if extracted.get("weather_class"):
            self.context["weather_class"] = extracted["weather_class"]
            self.context["weather"]       = extracted.get("weather")
            updated_something = True

        if extracted.get("style"):
            self.context["style"] = extracted["style"]
            updated_something = True

        # ── 6. Partial information → ask follow-up ───────────────────
        if not self.context["occasion"] and not self.context["weather_class"]:
            self.state = AWAITING_QUERY
            return random.choice([
                "I need to know the occasion and the weather. Could you tell me more?",
                "What's the occasion and weather like?",
                "Could you specify the event type and temperature?",
            ])

        if self.context["occasion"] and not self.context["weather_class"]:
            self.state = CLARIFICATION_NEEDED
            occ = self.context["occasion"].upper()
            return random.choice([
                f"Got it, {occ}! What's the weather going to be like?",
                f"Okay, {occ}. Is it hot, mild, or cold outside?",
            ])

        if not self.context["occasion"] and self.context["weather_class"]:
            self.state = CLARIFICATION_NEEDED
            wea = self.context["weather"]
            temp = f" ({wea['temperature']}°C)" if wea and "temperature" in wea else ""
            wcls = self.context["weather_class"].upper()
            return random.choice([
                f"Understood, it's {wcls}{temp}. What's the occasion?",
                f"Got the weather ({wcls}). Where are you heading?",
            ])

        # ── 7. Full context → generate ───────────────────────────────
        self.state = SHOWING_RESULTS
        return self._generate_recommendations()

    # ------------------------------------------------------------------
    def _generate_recommendations(self, is_retry: bool = False) -> str:
        occ     = self.context["occasion"]
        wea     = self.context["weather"]       # dict or None
        wea_cls = self.context["weather_class"]

        # Fetch extra outfits so we have alternates for retries
        outfits = self.api.get_outfits(occasion=occ, weather=wea, top_n=6)

        # ── Progressive fallback if weather filter left no complete outfit ──
        if not outfits and wea is not None:
            # Relax season constraint but keep warmth preference via scoring
            outfits = self.api.get_outfits(occasion=occ, weather=None, top_n=6)
            if outfits:
                # Re-rank: prefer items with higher warmth for cold, lower for hot
                target_warmth = 4 if wea["temperature"] <= 15 else 1
                def _warmth_penalty(o):
                    avg_w = sum(
                        item.get("warmth_level", 2) for item in o["items"]
                    ) / max(len(o["items"]), 1)
                    return abs(avg_w - target_warmth)
                outfits.sort(key=_warmth_penalty)

        if not outfits:
            return (
                f"I couldn't find any suitable {occ.upper()} outfits "
                f"for {(wea_cls or 'any').upper()} weather in your wardrobe.\n"
                f"Try uploading more items with: upload <image_path>\n"
            )

        if is_retry and len(outfits) > 1:
            selected = outfits[1:4] if len(outfits) >= 4 else outfits[1:]
            response = "\nNo problem! Here are some alternative outfit options:\n"
        else:
            selected = outfits[:3]
            temp_str = f" ({wea['temperature']}°C)" if wea and "temperature" in wea else ""
            response  = f"\n[Extracted] Occasion: {occ.upper()}, Weather: {(wea_cls or 'N/A').upper()}{temp_str}\n"
            response += "\n* Here are your outfit recommendations:\n"

        self.last_recommendations = selected

        for i, outfit in enumerate(selected, 1):
            response += f"\nOutfit #{i} (Score: {outfit['score']:.2f})\n"
            response += f"  Summary: {outfit['summary']}\n"
            for item in outfit["items"]:
                img = item.get("image_path")
                src = f"  [img: {os.path.basename(img)}]" if img else ""
                wlvl = item.get("warmth_level", "?")
                response += f"    - {item['name']} ({item['color_name']} {item['category']}) warmth={wlvl}{src}\n"

        return response


# ── Category-aware defaults for uploaded items ──────────────────────────────
_UPLOAD_DEFAULTS = {
    "shorts":    {"warmth_level": 1, "seasons": ["summer"],
                  "formality_level": 1, "occasions": ["casual", "sport"]},
    "dress":     {"warmth_level": 2, "seasons": ["summer", "spring"],
                  "formality_level": 3, "occasions": ["casual", "party", "formal"]},
    "shirt":     {"warmth_level": 2, "seasons": ["summer", "spring", "autumn"],
                  "formality_level": 2, "occasions": ["casual", "formal", "business"]},
    "skirt":     {"warmth_level": 2, "seasons": ["summer", "spring"],
                  "formality_level": 2, "occasions": ["casual", "party"]},
    "shoes":     {"warmth_level": 2, "seasons": ["summer", "spring", "autumn"],
                  "formality_level": 2, "occasions": ["casual", "formal", "sport", "party"]},
    "pants":     {"warmth_level": 3, "seasons": ["spring", "autumn", "winter"],
                  "formality_level": 3, "occasions": ["casual", "formal", "business"]},
    "jacket":    {"warmth_level": 4, "seasons": ["autumn", "winter"],
                  "formality_level": 3, "occasions": ["casual", "formal", "business", "outdoor"]},
    "accessory": {"warmth_level": 2, "seasons": ["summer", "spring", "autumn", "winter"],
                  "formality_level": 2, "occasions": ["casual", "formal", "sport", "party"]},
}

_UPLOAD_FALLBACK = {
    "warmth_level": 2,
    "seasons": ["summer", "spring", "autumn", "winter"],
    "formality_level": 2,
    "occasions": ["casual", "formal", "sport", "party"],
}


def handle_upload(api: RecommendationAPI, img_path: str) -> str:
    """Scan an image with the vision model and add it to the wardrobe."""
    if not os.path.exists(img_path):
        return f"Error: Image not found at '{img_path}'"

    print(f"\n  Scanning {os.path.basename(img_path)} with Vision Model...")
    try:
        # First, ask the vision model what category the item is
        vision_result = api.analyze_image(img_path)
        detected_cat  = vision_result.get("category", "")

        # Pick category-appropriate defaults so warmth/seasons are sensible
        cat_defaults = _UPLOAD_DEFAULTS.get(detected_cat, _UPLOAD_FALLBACK)

        item_dict = {
            "name":            f"Item from {os.path.basename(img_path)}",
            "color_rgb":       [128, 128, 128],
            "color_name":      "grey",
            "pattern":         "solid",
            "style":           "classic",
            "occasions":       cat_defaults["occasions"],
            "seasons":         cat_defaults["seasons"],
            "warmth_level":    cat_defaults["warmth_level"],
            "formality_level": cat_defaults["formality_level"],
        }
        item_id = api.add_item_from_image(img_path, item_dict)
        item    = api._wardrobe.get_item(item_id)
        summary = api.get_wardrobe_summary()
        return (
            f"  Added: {item.name} ({item.category.value}) — "
            f"warmth={item.warmth_level}, seasons={[s.value for s in item.seasons]}\n"
            f"  Wardrobe now has {summary['total_items']} items."
        )
    except Exception as e:
        return f"  Error processing image: {e}"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------
def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir     = os.path.join(project_root, "data")

    # ── 1. Load real wardrobe ────────────────────────────────────────
    # Prefer my_wardrobe.json (real vision-scanned data).
    # Fall back to sample_wardrobe.json if it doesn't exist yet.
    my_wardrobe_path     = os.path.join(data_dir, "my_wardrobe.json")
    sample_wardrobe_path = os.path.join(data_dir, "sample_wardrobe.json")

    if os.path.exists(my_wardrobe_path):
        wardrobe_file = my_wardrobe_path
    elif os.path.exists(sample_wardrobe_path):
        wardrobe_file = sample_wardrobe_path
        print("[Wardrobe] my_wardrobe.json not found — falling back to sample_wardrobe.json")
    else:
        print(f"Error: No wardrobe file found in {data_dir}")
        sys.exit(1)

    print("Initializing Smart Wardrobe Recommendation System...")
    api = RecommendationAPI(wardrobe_file)
    summary = api.get_wardrobe_summary()
    print(f"[Wardrobe] Loaded {summary['total_items']} items from {os.path.basename(wardrobe_file)}")
    print(f"[Wardrobe] Categories: {summary['by_category']}")

    # ── 2. Load NLP model ────────────────────────────────────────────
    try:
        nlp_model = NLPInference()
    except Exception as e:
        print(f"\nError loading NLP model: {e}")
        print("Please run 'python nlp/train.py' first.")
        sys.exit(1)
    print("\nLoaded NLP Inference Model.")

    bot = WardrobeChatbot(api, nlp_model)

    # ── 3. Interactive loop ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SMART WARDROBE AI ASSISTANT")
    print("=" * 60)
    print("  Chat naturally to get outfit recommendations.")
    print("  Example: 'I need a formal outfit, it's freezing outside.'")
    print("  Commands:")
    print("    upload <image_path>  — add a clothing item from an image")
    print("    quit / exit / q      — exit the assistant")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # ── Upload command ───────────────────────────────────────────
        if user_input.lower().startswith("upload "):
            img_path = user_input[7:].strip().strip('"').strip("'")
            print(handle_upload(api, img_path))
            continue

        # ── NLP chatbot ──────────────────────────────────────────────
        response = bot.handle_input(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
