import os
import json
import pandas as pd
import uuid

# Mapping from H&M to our engine's ClothingCategory
def map_category(product_group, product_type):
    group = str(product_group).lower()
    ptype = str(product_type).lower()
    
    if group == 'garment upper body':
        if 'jacket' in ptype or 'coat' in ptype or 'hoodie' in ptype or 'blazer' in ptype:
            return 'jacket'
        return 'shirt'
    elif group == 'garment lower body':
        if 'skirt' in ptype:
            return 'skirt'
        elif 'shorts' in ptype:
            return 'shorts'
        return 'pants'
    elif group == 'garment full body':
        return 'dress'
    elif group == 'shoes':
        return 'shoes'
    else:
        return 'accessory'

def map_pattern(appearance):
    app = str(appearance).lower()
    if 'solid' in app: return 'solid'
    if 'stripe' in app: return 'striped'
    if 'check' in app: return 'plaid'
    if 'floral' in app or 'pattern' in app: return 'floral'
    return 'solid'

def run_conversion():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'articles.csv')
    json_path = os.path.join(base_dir, 'data', 'hm_catalog.json')
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out underwear and nightwear so we get mostly street clothes
    exclude = ['Underwear', 'Nightwear', 'Swimwear', 'Socks & Tights', 'Items', 'Stationery']
    df = df[~df['product_group_name'].isin(exclude)]
    
    print(f"Found {len(df)} suitable garments. Sampling 10,000 for the global catalog...")
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
    
    items = []
    for _, row in df.iterrows():
        cat = map_category(row['product_group_name'], row['product_type_name'])
        pat = map_pattern(row['graphical_appearance_name'])
        
        # Color mapping (dummy rgb)
        cname = str(row['perceived_colour_master_name']).lower()
        if cname == 'black': rgb = [0,0,0]
        elif cname == 'white': rgb = [255,255,255]
        elif cname == 'red': rgb = [255,0,0]
        elif cname == 'blue': rgb = [0,0,255]
        else: rgb = [128,128,128] # grey fallback
        
        # Determine occasion based on department name (rough heuristic)
        dept = str(row['department_name']).lower()
        if 'sport' in dept or 'active' in dept:
            occ = 'sport'
        elif 'blazer' in dept or 'tailored' in dept or 'suit' in dept:
            occ = 'business'
        elif 'dress' in dept:
            occ = 'formal'
        else:
            occ = 'casual'
            
        item = {
            "id": f"hm_{row['article_id']}",
            "name": str(row['prod_name']),
            "category": cat,
            "color_rgb": rgb,
            "color_name": str(row['colour_group_name']).lower(),
            "pattern": pat,
            "style": "classic",
            "occasions": [occ],
            "seasons": ["spring", "summer", "autumn", "winter"],
            "warmth_level": 2 if cat in ['shirt', 'skirt', 'shorts'] else 3,
            "formality_level": 4 if occ in ['business', 'formal'] else 2,
            "image_features": None,
            "image_path": None
        }
        items.append(item)
        
    print(f"Writing {json_path}...")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(items, f, indent=2)
    print("Done!")

if __name__ == "__main__":
    run_conversion()
