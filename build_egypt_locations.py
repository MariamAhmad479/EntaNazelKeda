# -*- coding: utf-8 -*-
"""
build_egypt_locations.py
========================
One-time data-collection script.
Uses Geopy + Nominatim (OpenStreetMap) — NO API key required.

Run once:
    python build_egypt_locations.py

Output: data/egypt_locations.json
"""

import json
import os
import time

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ── Nominatim requires a unique user_agent string (your app name) ────────────
geolocator = Nominatim(user_agent="SmartWardrobeAI_EgyptLocations/1.0")
geocode     = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)  # Nominatim ToS: ≤1 req/sec

# ---------------------------------------------------------------------------
# Seed list — Egyptian Governorates, cities, neighbourhoods, and landmarks
# The script will geocode each one live, so coordinates are always accurate.
# ---------------------------------------------------------------------------
SEED_PLACES = [
    # ── Cairo Governorate ──────────────────────────────────────────────────
    ("Cairo",                 "Cairo",     "city"),
    ("Zamalek",               "Cairo",     "neighbourhood"),
    ("Maadi",                 "Cairo",     "neighbourhood"),
    ("Heliopolis",            "Cairo",     "neighbourhood"),
    ("Nasr City",             "Cairo",     "neighbourhood"),
    ("Downtown Cairo",        "Cairo",     "neighbourhood"),
    ("Mohandessin",           "Cairo",     "neighbourhood"),
    ("Garden City Cairo",     "Cairo",     "neighbourhood"),
    ("New Cairo",             "Cairo",     "neighbourhood"),
    ("Khan el-Khalili",       "Cairo",     "tourist_attraction"),
    ("The Egyptian Museum",   "Cairo",     "museum"),
    ("Cairo Tower",           "Cairo",     "tourist_attraction"),
    ("City Stars Mall",       "Cairo",     "shopping_mall"),
    ("Mall of Egypt",         "Cairo",     "shopping_mall"),
    ("Al-Azhar Mosque",       "Cairo",     "mosque"),
    ("Muhammad Ali Mosque",   "Cairo",     "mosque"),
    ("Al-Azhar Park",         "Cairo",     "park"),
    ("Cairo Jazz Club",       "Cairo",     "night_club"),
    ("Four Seasons Cairo",    "Cairo",     "hotel"),
    ("Kempinski Nile Hotel",  "Cairo",     "hotel"),
    ("Cairo International Stadium", "Cairo", "stadium"),
    ("American University Cairo",   "Cairo", "university"),
    ("Cairo University",             "Cairo", "university"),

    # ── Giza Governorate ──────────────────────────────────────────────────
    ("Giza",                  "Giza",      "city"),
    ("Dokki",                 "Giza",      "neighbourhood"),
    ("6th of October City",   "Giza",      "city"),
    ("Sheikh Zayed City",     "Giza",      "city"),
    ("Giza Pyramids Complex", "Giza",      "tourist_attraction"),
    ("Sphinx",                "Giza",      "tourist_attraction"),
    ("Solar Boat Museum",     "Giza",      "museum"),
    ("Dream Park",            "Giza",      "park"),
    ("Marriott Mena House",   "Giza",      "hotel"),

    # ── Alexandria Governorate ────────────────────────────────────────────
    ("Alexandria",            "Alexandria", "city"),
    ("Bibliotheca Alexandrina","Alexandria","museum"),
    ("Qaitbay Citadel",       "Alexandria","tourist_attraction"),
    ("Montaza Palace",        "Alexandria","tourist_attraction"),
    ("Montaza Beach",         "Alexandria","beach"),
    ("Stanley Beach",         "Alexandria","beach"),
    ("San Stefano Mall",      "Alexandria","shopping_mall"),
    ("Mosque of Abu al-Abbas","Alexandria","mosque"),
    ("Alexandria National Museum","Alexandria","museum"),
    ("Roastery Cafe Alexandria","Alexandria","cafe"),
    ("Sidi Gaber",            "Alexandria","neighbourhood"),
    ("Smouha",                "Alexandria","neighbourhood"),
    ("Miami Beach Alexandria","Alexandria","beach"),
    ("Agami Beach",           "Alexandria","beach"),

    # ── Luxor Governorate ─────────────────────────────────────────────────
    ("Luxor",                 "Luxor",     "city"),
    ("Karnak Temple",         "Luxor",     "tourist_attraction"),
    ("Luxor Temple",          "Luxor",     "tourist_attraction"),
    ("Valley of the Kings",   "Luxor",     "tourist_attraction"),
    ("Hatshepsut Temple",     "Luxor",     "tourist_attraction"),
    ("Luxor Museum",          "Luxor",     "museum"),
    ("Colossi of Memnon",     "Luxor",     "tourist_attraction"),
    ("Winter Palace Hotel",   "Luxor",     "hotel"),
    ("Sofitel Winter Palace", "Luxor",     "hotel"),

    # ── Aswan Governorate ─────────────────────────────────────────────────
    ("Aswan",                 "Aswan",     "city"),
    ("Philae Temple",         "Aswan",     "tourist_attraction"),
    ("Abu Simbel",            "Aswan",     "tourist_attraction"),
    ("Nubian Museum",         "Aswan",     "museum"),
    ("Aswan High Dam",        "Aswan",     "tourist_attraction"),
    ("Movenpick Aswan",       "Aswan",     "hotel"),
    ("Sofitel Legend Cataract Aswan","Aswan","hotel"),
    ("Ferial Garden Aswan",   "Aswan",     "park"),
    ("Elephantine Island",    "Aswan",     "tourist_attraction"),

    # ── Red Sea Governorate ───────────────────────────────────────────────
    ("Hurghada",              "Hurghada",  "city"),
    ("El Gouna",              "Hurghada",  "city"),
    ("Hurghada Marina",       "Hurghada",  "tourist_attraction"),
    ("Hurghada Grand Aquarium","Hurghada", "tourist_attraction"),
    ("Sindbad Beach Hurghada","Hurghada",  "beach"),
    ("Senzo Mall Hurghada",   "Hurghada",  "shopping_mall"),
    ("Titanic Palace Hurghada","Hurghada", "hotel"),
    ("Steigenberger Aldau",   "Hurghada",  "hotel"),
    ("Makadi Bay",            "Hurghada",  "beach"),
    ("Soma Bay",              "Hurghada",  "beach"),

    # ── South Sinai Governorate ───────────────────────────────────────────
    ("Sharm el-Sheikh",       "Sharm el-Sheikh","city"),
    ("Dahab",                 "Sharm el-Sheikh","city"),
    ("Naama Bay",             "Sharm el-Sheikh","tourist_attraction"),
    ("Ras Mohammed Park",     "Sharm el-Sheikh","park"),
    ("Sharks Bay Sharm",      "Sharm el-Sheikh","beach"),
    ("Four Seasons Sharm",    "Sharm el-Sheikh","hotel"),
    ("Savoy Sharm el-Sheikh", "Sharm el-Sheikh","hotel"),
    ("Farsha Cafe Sharm",     "Sharm el-Sheikh","cafe"),

    # ── North Sinai Governorate ───────────────────────────────────────────
    ("Arish",                 "Arish",     "city"),
    ("Arish Beach",           "Arish",     "beach"),

    # ── Ismailia Governorate ──────────────────────────────────────────────
    ("Ismailia",              "Ismailia",  "city"),
    ("Ismailia Lake",         "Ismailia",  "park"),

    # ── Suez Governorate ──────────────────────────────────────────────────
    ("Suez",                  "Suez",      "city"),
    ("Suez Canal",            "Suez",      "tourist_attraction"),

    # ── Port Said Governorate ─────────────────────────────────────────────
    ("Port Said",             "Port Said", "city"),
    ("Port Said Corniche",    "Port Said", "park"),

    # ── Dakahlia Governorate ──────────────────────────────────────────────
    ("Mansoura",              "Mansoura",  "city"),
    ("Mansoura University",   "Mansoura",  "university"),

    # ── Gharbia Governorate ───────────────────────────────────────────────
    ("Tanta",                 "Tanta",     "city"),
    ("Tanta Mosque",          "Tanta",     "mosque"),

    # ── Sharqia Governorate ───────────────────────────────────────────────
    ("Zagazig",               "Zagazig",   "city"),
    ("Zagazig University",    "Zagazig",   "university"),

    # ── Qalyubia Governorate ──────────────────────────────────────────────
    ("Banha",                 "Banha",     "city"),
    ("Shibin El Kom",         "Menofia",   "city"),

    # ── Kafr El Sheikh ────────────────────────────────────────────────────
    ("Kafr el-Sheikh",        "Kafr el-Sheikh","city"),

    # ── Damietta Governorate ──────────────────────────────────────────────
    ("Damietta",              "Damietta",  "city"),
    ("New Damietta",          "Damietta",  "city"),
    ("Damietta Beach",        "Damietta",  "beach"),

    # ── Beheira Governorate ───────────────────────────────────────────────
    ("Damanhur",              "Beheira",   "city"),

    # ── Monufia Governorate ───────────────────────────────────────────────
    ("Shebin el-Kom",         "Monufia",   "city"),

    # ── Beni Suef Governorate ─────────────────────────────────────────────
    ("Beni Suef",             "Beni Suef", "city"),

    # ── Fayoum Governorate ───────────────────────────────────────────────
    ("Fayoum",                "Fayoum",    "city"),
    ("Wadi El Rayan",         "Fayoum",    "park"),
    ("Lake Qarun",            "Fayoum",    "tourist_attraction"),

    # ── Minya Governorate ─────────────────────────────────────────────────
    ("Minya",                 "Minya",     "city"),
    ("Beni Hassan Tombs",     "Minya",     "tourist_attraction"),

    # ── Asyut Governorate ─────────────────────────────────────────────────
    ("Asyut",                 "Asyut",     "city"),
    ("Asyut University",      "Asyut",     "university"),

    # ── Sohag Governorate ─────────────────────────────────────────────────
    ("Sohag",                 "Sohag",     "city"),
    ("Abydos Temple",         "Sohag",     "tourist_attraction"),

    # ── Qena Governorate ──────────────────────────────────────────────────
    ("Qena",                  "Qena",      "city"),
    ("Dendera Temple",        "Qena",      "tourist_attraction"),

    # ── Matruh Governorate ────────────────────────────────────────────────
    ("Mersa Matruh",          "Matruh",    "city"),
    ("Agiba Beach",           "Matruh",    "beach"),
    ("Cleopatra Beach Matruh","Matruh",    "beach"),

    # ── New Valley Governorate ────────────────────────────────────────────
    ("Kharga",                "New Valley","city"),
    ("Siwa Oasis",            "New Valley","tourist_attraction"),

    # ── South Sinai extra ─────────────────────────────────────────────────
    ("Saint Catherine Monastery","Sinai",  "tourist_attraction"),
    ("Mount Sinai",           "Sinai",     "tourist_attraction"),

    # ── New Administrative Capital ────────────────────────────────────────
    ("New Administrative Capital", "Cairo","city"),
]

# ---------------------------------------------------------------------------
# Category → occasion mapping (same logic as the original module)
# ---------------------------------------------------------------------------
CAT_TO_OCCASION = {
    "tourist_attraction": "casual",
    "park":               "casual",
    "beach":              "casual",
    "shopping_mall":      "casual",
    "cafe":               "casual",
    "city":               "casual",
    "neighbourhood":      "casual",
    "gym":                "sport",
    "spa":                "sport",
    "stadium":            "sport",
    "night_club":         "party",
    "mosque":             "formal",
    "museum":             "formal",
    "university":         "formal",
    "hotel":              "business",
    "restaurant":         "business",
}

def category_to_occasion(cat: str) -> str:
    return CAT_TO_OCCASION.get(cat.lower(), "casual")

# ---------------------------------------------------------------------------
# Main geocoding loop
# ---------------------------------------------------------------------------
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", "egypt_locations.json")

def build():
    results = []
    total   = len(SEED_PLACES)
    failed  = []

    print(f"\nBuilding Egypt locations database ({total} places)...")
    print("   Using Nominatim/OpenStreetMap - NO API key required\n")

    for idx, (place_name, governorate, category) in enumerate(SEED_PLACES, 1):
        query = f"{place_name}, Egypt"
        try:
            loc = geocode(query, language="en")
            if loc:
                results.append({
                    "place_name":  place_name,
                    "governorate": governorate,
                    "category":    category,
                    "occasion":    category_to_occasion(category),
                    "address":     loc.address,
                    "lat":         round(loc.latitude,  6),
                    "lng":         round(loc.longitude, 6),
                })
                print(f"  [{idx:03d}/{total}] OK  {place_name:40s} -> {loc.latitude:.4f}, {loc.longitude:.4f}")
            else:
                failed.append(place_name)
                print(f"  [{idx:03d}/{total}] SKIP {place_name} - no result, skipped")
        except Exception as e:
            failed.append(place_name)
            print(f"  [{idx:03d}/{total}] ERR {place_name} - error: {e}")

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} places -> {OUTPUT_PATH}")
    if failed:
        print(f"  {len(failed)} places failed: {failed}")

if __name__ == "__main__":
    build()
