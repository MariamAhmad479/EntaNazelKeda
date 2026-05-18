"""
location_weather.py
===================
Real Egyptian location look-up powered by Geopy + Nominatim (OpenStreetMap).

No API key is required.  Nominatim is a free, open geocoding service backed
by OpenStreetMap data.  We use a user_agent string as required by their ToS.

Flow
----
1. On first import the module loads ``data/egypt_locations.json`` (generated
   by ``build_egypt_locations.py``) into an in-memory lookup table.
2. ``get_location_details(place_name)`` does a case-insensitive substring
   search against the pre-built list first (instant, offline).
3. If the place is not in the local list it falls back to a live Nominatim
   geocode call, which returns lat/lng + a best-guess occasion category.
4. ``fetch_realtime_weather(lat, lng)`` remains unchanged — it calls the
   free Open-Meteo API exactly as before.

Recommendation-engine contract
-------------------------------
Every function that returns a location dict returns the **same 7 fields**:

    {
        "place_name":  str,          # human-readable name
        "governorate": str,          # Egyptian governorate / city
        "category":    str,          # OSM place type (tourist_attraction, …)
        "occasion":    str,          # mapped engine occasion (casual, formal, …)
        "address":     str,          # full address string
        "lat":         float,
        "lng":         float,
    }

Plus, ``get_location_with_weather()`` adds two more fields from Open-Meteo:

    {
        "temperature": float,        # °C (live, from Open-Meteo)
        "condition":   str,          # e.g. "sunny", "rainy", "cloudy"
    }
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level cache – loaded once on first import
# ---------------------------------------------------------------------------
_LOCATIONS: list[dict] = []
_LOCATION_INDEX: dict[str, dict] = {}  # lower-case place_name → record

_DATA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "egypt_locations.json",
)


def _load_locations() -> None:
    """Load egypt_locations.json into the module-level cache (idempotent)."""
    global _LOCATIONS, _LOCATION_INDEX
    if _LOCATIONS:
        return  # already loaded

    if os.path.exists(_DATA_FILE):
        try:
            with open(_DATA_FILE, encoding="utf-8") as f:
                _LOCATIONS = json.load(f)
            _LOCATION_INDEX = {
                rec["place_name"].lower(): rec for rec in _LOCATIONS
            }
        except Exception as e:
            print(f"[LocationWeather] Warning – could not load {_DATA_FILE}: {e}")
    else:
        print(
            "[LocationWeather] egypt_locations.json not found. "
            "Run build_egypt_locations.py once to generate it. "
            "Live Nominatim geocoding will be used as a fallback."
        )


# ---------------------------------------------------------------------------
# Category → occasion mapping (mirrors build_egypt_locations.py)
# ---------------------------------------------------------------------------
_CAT_TO_OCCASION: dict[str, str] = {
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


def map_category_to_occasion(category: str) -> str:
    """Maps an OSM place category to a recommendation-engine occasion string."""
    return _CAT_TO_OCCASION.get(str(category).lower(), "casual")


# ---------------------------------------------------------------------------
# OSM/Nominatim type → our category
# ---------------------------------------------------------------------------
def _osm_type_to_category(raw: dict) -> str:
    """Convert Nominatim's raw ``type`` / ``class`` fields to our category."""
    osm_type  = str(raw.get("type",  "")).lower()
    osm_class = str(raw.get("class", "")).lower()

    mapping = {
        "attraction":  "tourist_attraction",
        "tourism":     "tourist_attraction",
        "museum":      "museum",
        "place_of_worship": "mosque",
        "mosque":      "mosque",
        "church":      "formal",
        "mall":        "shopping_mall",
        "retail":      "shopping_mall",
        "beach":       "beach",
        "park":        "park",
        "restaurant":  "restaurant",
        "cafe":        "cafe",
        "bar":         "night_club",
        "nightclub":   "night_club",
        "hotel":       "hotel",
        "gym":         "gym",
        "university":  "university",
        "city":        "city",
        "town":        "city",
        "village":     "city",
        "suburb":      "neighbourhood",
        "neighbourhood": "neighbourhood",
    }

    for key in (osm_type, osm_class):
        if key in mapping:
            return mapping[key]
    return "city"  # safe default


# ---------------------------------------------------------------------------
# Local lookup
# ---------------------------------------------------------------------------

def search_locations(query: str, max_results: int = 10) -> list[dict]:
    """Return up to *max_results* records whose place_name contains *query*."""
    _load_locations()
    q = query.strip().lower()
    matches = [
        rec for rec in _LOCATIONS
        if q in rec["place_name"].lower()
        or q in rec.get("governorate", "").lower()
        or q in rec.get("address", "").lower()
    ]
    return matches[:max_results]


def get_location_details(place_name: str) -> Optional[dict]:
    """
    Return the 7-field location dict for *place_name*.

    1. Tries exact (case-insensitive) match in the local JSON.
    2. Tries substring search in the local JSON.
    3. Falls back to a live Nominatim geocode call (requires internet).
    4. Returns None if everything fails.
    """
    _load_locations()

    # 1. Exact match
    key = place_name.strip().lower()
    if key in _LOCATION_INDEX:
        return dict(_LOCATION_INDEX[key])

    # 2. Substring match (first hit)
    for rec in _LOCATIONS:
        if key in rec["place_name"].lower() or key in rec.get("address", "").lower():
            return dict(rec)

    # 3. Live Nominatim fallback
    return _nominatim_lookup(place_name)


def _nominatim_lookup(place_name: str) -> Optional[dict]:
    """Geocode *place_name* via Nominatim and return a normalised dict."""
    try:
        from geopy.geocoders import Nominatim
        geo = Nominatim(user_agent="SmartWardrobeAI/1.0")
        query = place_name if "egypt" in place_name.lower() else f"{place_name}, Egypt"
        loc = geo.geocode(query, language="en", timeout=8)
        if loc is None:
            return None

        raw      = loc.raw
        category = _osm_type_to_category(raw)
        return {
            "place_name":  place_name,
            "governorate": raw.get("display_name", "").split(",")[-2].strip()
                           if "," in raw.get("display_name", "") else "Egypt",
            "category":    category,
            "occasion":    map_category_to_occasion(category),
            "address":     loc.address,
            "lat":         round(loc.latitude,  6),
            "lng":         round(loc.longitude, 6),
        }
    except ImportError:
        print("[LocationWeather] geopy not installed. Run: pip install geopy")
        return None
    except Exception as e:
        print(f"[LocationWeather] Nominatim lookup failed for '{place_name}': {e}")
        return None


# ---------------------------------------------------------------------------
# All locations list (for UI dropdowns, etc.)
# ---------------------------------------------------------------------------

def get_all_locations() -> list[dict]:
    """Return the full list of pre-built Egyptian locations."""
    _load_locations()
    return list(_LOCATIONS)


def get_all_place_names() -> list[str]:
    """Return sorted list of all place names (useful for Streamlit selectboxes)."""
    _load_locations()
    return sorted(rec["place_name"] for rec in _LOCATIONS)


def get_governorates() -> list[str]:
    """Return a sorted, deduplicated list of Egyptian governorates/cities."""
    _load_locations()
    govs = sorted({rec.get("governorate", "") for rec in _LOCATIONS if rec.get("governorate")})
    return govs


# ---------------------------------------------------------------------------
# Weather (unchanged — uses Open-Meteo, no API key required)
# ---------------------------------------------------------------------------

def map_weathercode_to_condition(weathercode: int) -> str:
    """Maps WMO weather codes to human-readable condition strings."""
    if weathercode <= 3:
        return "sunny" if weathercode == 0 else "cloudy"
    elif weathercode in (45, 48):
        return "cloudy"  # fog
    elif 50 <= weathercode <= 69:
        return "rainy"
    elif 70 <= weathercode <= 79:
        return "snowy"
    elif 80 <= weathercode <= 99:
        return "rainy"
    return "clear"


def fetch_realtime_weather(lat: float, lng: float) -> dict:
    """
    Fetch live weather from the Open-Meteo API (free, no key required).

    Returns
    -------
    dict with keys ``temperature`` (float, °C) and ``condition`` (str).
    Falls back to safe defaults if the request fails.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lng}&current_weather=true"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SmartWardrobe/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data    = json.loads(response.read().decode("utf-8"))
            current = data.get("current_weather", {})
            temp    = current.get("temperature", 22.0)
            code    = current.get("weathercode", 0)
            return {"temperature": temp, "condition": map_weathercode_to_condition(code)}
    except Exception as e:
        print(f"[LocationWeather] Weather fetch failed: {e}")
        return {"temperature": 22.0, "condition": "clear"}


# ---------------------------------------------------------------------------
# Convenience: look up a place AND fetch live weather in one call
# ---------------------------------------------------------------------------

def get_location_with_weather(place_name: str) -> Optional[dict]:
    """
    Look up *place_name* and append real-time weather.

    Returns a merged dict with all 7 location fields **plus**:
        ``temperature`` – current temp in °C
        ``condition``   – weather condition string

    Returns None if the location cannot be resolved.
    """
    details = get_location_details(place_name)
    if details is None:
        return None

    weather = fetch_realtime_weather(details["lat"], details["lng"])
    return {**details, **weather}
