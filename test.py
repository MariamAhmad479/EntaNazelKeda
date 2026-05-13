"""
Test the recommendation engine with weather-based outfit generation.
Uses the real wardrobe built from Person 1's vision data.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI

# Load the vision-based wardrobe
wardrobe_path = os.path.join("data", "vision_wardrobe.json")
api = RecommendationAPI(wardrobe_path)

summary = api.get_wardrobe_summary()
print(f"Wardrobe: {summary['total_items']} items")
print(f"Categories: {summary['by_category']}\n")

# ---------------------------------------------------------------
# Test 1: Hot summer day - casual outing
# ---------------------------------------------------------------
print("=" * 60)
print("SCENARIO 1: Hot summer day, casual outing (32C, sunny)")
print("=" * 60)
outfits = api.get_outfits(
    occasion="casual",
    weather={"temperature": 32, "condition": "sunny"},
    top_n=3
)
for i, o in enumerate(outfits, 1):
    print(f"\n  Outfit #{i} (Score: {o['score']:.3f})")
    for item in o["items"]:
        print(f"    - {item['name']} ({item['color_name']} {item['category']})")

# ---------------------------------------------------------------
# Test 2: Cold winter evening - formal event
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SCENARIO 2: Cold winter evening, formal event (3C, cloudy)")
print("=" * 60)
outfits = api.get_outfits(
    occasion="formal",
    weather={"temperature": 3, "condition": "cloudy"},
    top_n=3
)
for i, o in enumerate(outfits, 1):
    print(f"\n  Outfit #{i} (Score: {o['score']:.3f})")
    for item in o["items"]:
        print(f"    - {item['name']} ({item['color_name']} {item['category']})")

# ---------------------------------------------------------------
# Test 3: Mild spring morning - sports
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SCENARIO 3: Mild spring morning, sports (18C, clear)")
print("=" * 60)
outfits = api.get_outfits(
    occasion="sport",
    weather={"temperature": 18, "condition": "clear"},
    top_n=3
)
for i, o in enumerate(outfits, 1):
    print(f"\n  Outfit #{i} (Score: {o['score']:.3f})")
    for item in o["items"]:
        print(f"    - {item['name']} ({item['color_name']} {item['category']})")

# ---------------------------------------------------------------
# Test 4: Rainy autumn day - party
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SCENARIO 4: Rainy autumn day, party (12C, rainy)")
print("=" * 60)
outfits = api.get_outfits(
    occasion="party",
    weather={"temperature": 12, "condition": "rainy"},
    top_n=3
)
for i, o in enumerate(outfits, 1):
    print(f"\n  Outfit #{i} (Score: {o['score']:.3f})")
    for item in o["items"]:
        print(f"    - {item['name']} ({item['color_name']} {item['category']})")

# ---------------------------------------------------------------
# Test 5: No weather filter - just occasion
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("SCENARIO 5: Business meeting (no weather filter)")
print("=" * 60)
outfits = api.get_outfits(occasion="business", top_n=3)
for i, o in enumerate(outfits, 1):
    print(f"\n  Outfit #{i} (Score: {o['score']:.3f})")
    for item in o["items"]:
        print(f"    - {item['name']} ({item['color_name']} {item['category']})")

print("\n" + "=" * 60)
print("All scenarios completed!")
print("=" * 60)