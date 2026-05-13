"""Quick integration test for the vision wardrobe."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine.api import RecommendationAPI

# Test 1: Load from vision wardrobe JSON
print("=== Test: Load vision_wardrobe.json ===")
api = RecommendationAPI("data/vision_wardrobe.json")
summary = api.get_wardrobe_summary()
print(f"Total items: {summary['total_items']}")
print(f"By category: {summary['by_category']}")

# Test 2: Generate formal outfits
print("\n=== Test: Formal outfits ===")
outfits = api.get_outfits(occasion="formal", top_n=3)
print(f"Found {len(outfits)} formal outfits")
for i, o in enumerate(outfits, 1):
    print(f"  #{i} Score: {o['score']:.3f}")
    print(f"     {o['summary'][:120]}")

# Test 3: Generate casual outfits with warm weather
print("\n=== Test: Casual + hot weather ===")
outfits = api.get_outfits(occasion="casual", weather={"temperature": 30, "condition": "sunny"}, top_n=3)
print(f"Found {len(outfits)} casual summer outfits")
for i, o in enumerate(outfits, 1):
    print(f"  #{i} Score: {o['score']:.3f}")
    print(f"     {o['summary'][:120]}")

# Test 4: Generate sport outfits
print("\n=== Test: Sport outfits ===")
outfits = api.get_outfits(occasion="sport", top_n=3)
print(f"Found {len(outfits)} sport outfits")
for i, o in enumerate(outfits, 1):
    print(f"  #{i} Score: {o['score']:.3f}")
    print(f"     {o['summary'][:120]}")

# Test 5: Clusters
print("\n=== Test: Clusters ===")
clusters = api.get_clusters()
print(f"Number of clusters: {len(clusters)}")
for cid, members in clusters.items():
    print(f"  Cluster {cid}: {len(members)} items")

print("\nAll integration tests passed!")
