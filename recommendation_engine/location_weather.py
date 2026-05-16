import os
import json
import urllib.request
import pandas as pd

def load_locations(csv_path: str) -> pd.DataFrame:
    """Loads the dummy location dataset."""
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path)

def get_location_details(df: pd.DataFrame, place_name: str) -> dict:
    """Gets details for a specific place name."""
    row = df[df['place_name'] == place_name]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def map_category_to_occasion(category: str) -> str:
    """Maps the dataset location category to an engine occasion."""
    category = str(category).lower()
    
    casual_cats = ['tourist_attraction', 'park', 'beach', 'shopping_mall', 'cafe']
    sport_cats = ['gym', 'spa']
    party_cats = ['night_club']
    formal_cats = ['mosque', 'museum']
    business_cats = ['hotel', 'restaurant']
    
    if category in casual_cats:
        return 'casual'
    elif category in sport_cats:
        return 'sport'
    elif category in party_cats:
        return 'party'
    elif category in formal_cats:
        return 'formal'
    elif category in business_cats:
        return 'business'
    
    return 'casual' # default fallback

def map_weathercode_to_condition(weathercode: int) -> str:
    """Maps WMO weather codes to string conditions."""
    if weathercode <= 3:
        return "sunny" if weathercode == 0 else "cloudy"
    elif weathercode in [45, 48]:
        return "cloudy" # fog
    elif 50 <= weathercode <= 69:
        return "rainy"
    elif 70 <= weathercode <= 79:
        return "snowy"
    elif 80 <= weathercode <= 99:
        return "rainy"
    return "clear"

def fetch_realtime_weather(lat: float, lng: float) -> dict:
    """Fetches real-time weather from Open-Meteo API."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lng}&current_weather=true"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'SmartWardrobe/1.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            current = data.get("current_weather", {})
            temp = current.get("temperature", 22.0)
            code = current.get("weathercode", 0)
            condition = map_weathercode_to_condition(code)
            return {"temperature": temp, "condition": condition}
    except Exception as e:
        print(f"Error fetching weather: {e}")
        # Return a safe fallback if the API fails
        return {"temperature": 22.0, "condition": "clear"}
