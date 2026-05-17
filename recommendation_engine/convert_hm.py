import os
import json
import pandas as pd
import uuid
import re

# Mapped colors palette from vision module for high-quality RGB values
COLOUR_RGB_MAP = {
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "off white": (255, 250, 240),
    "grey": (160, 160, 160),
    "grey melange": (180, 180, 180),
    "charcoal": (54, 69, 79),
    "navy blue": (0, 0, 128),
    "blue": (70, 130, 180),
    "dark blue": (25, 25, 112),
    "light blue": (173, 216, 230),
    "turquoise blue": (0, 206, 209),
    "teal": (0, 128, 128),
    "red": (180, 30, 30),
    "maroon": (128, 0, 0),
    "burgundy": (128, 0, 32),
    "pink": (255, 182, 193),
    "magenta": (255, 0, 144),
    "lavender": (180, 160, 220),
    "purple": (128, 0, 128),
    "green": (34, 139, 34),
    "olive": (107, 142, 35),
    "khaki": (195, 176, 145),
    "yellow": (255, 215, 0),
    "orange": (255, 140, 0),
    "brown": (139, 90, 43),
    "coffee brown": (75, 54, 33),
    "tan": (210, 180, 140),
    "beige": (210, 180, 140),
    "cream": (255, 253, 208),
    "gold": (212, 175, 55),
    "silver": (192, 192, 192),
    "copper": (184, 115, 51),
    "bronze": (205, 127, 50),
    "steel": (113, 121, 126),
    "nude": (227, 187, 164),
    "peach": (255, 218, 185),
    "coral": (255, 127, 80),
    "rust": (183, 65, 14),
    "lime green": (50, 205, 50),
    "sea green": (46, 139, 87),
    "fluorescent green": (0, 255, 0),
    "multi": (128, 128, 128),
    "mushroom brown": (150, 120, 90),
    "taupe": (72, 60, 50),
    "mauve": (224, 176, 255),
    "skin": (227, 187, 164),
}

def extract_category(product_group, product_type, name, desc):
    pg = str(product_group).lower()
    pt = str(product_type).lower()
    nm = str(name).lower()
    ds = str(desc).lower()
    
    # Check shoes first
    if 'shoes' in pg or any(x in pt or x in nm or x in ds for x in ['shoe', 'sneaker', 'boot', 'sandal', 'heel', 'flip flop', 'flats', 'slippers']):
        return 'shoes'
        
    # Check dress/jumpsuit/full body
    if pg == 'garment full body' or any(x in pt or x in nm or x in ds for x in ['dress', 'jumpsuit', 'romper', 'saree', 'gown', 'robe']):
        return 'dress'
        
    # Check jacket/outerwear
    if any(x in pt or x in nm or x in ds for x in ['jacket', 'coat', 'hoodie', 'blazer', 'cardigan', 'sweater', 'sweatshirt', 'trench', 'parka', 'outerwear', 'poncho', 'anorak', 'waistcoat']):
        return 'jacket'
        
    # Check shorts
    if 'shorts' in pt or 'shorts' in nm or ('shorts' in ds and 'pants' not in ds) or 'hotpants' in nm:
        return 'shorts'
        
    # Check skirt
    if 'skirt' in pt or 'skirt' in nm or 'skirt' in ds:
        return 'skirt'
        
    # Check pants
    if pg == 'garment lower body' or any(x in pt or x in nm or x in ds for x in ['pants', 'trousers', 'jeans', 'leggings', 'chinos', 'joggers', 'sweatpants']):
        return 'pants'
        
    # Check shirt/top
    if pg == 'garment upper body' or any(x in pt or x in nm or x in ds for x in ['shirt', 'tshirt', 't-shirt', 'top', 'blouse', 'camisole', 'tank', 'vest', 'pullover', 'jersey', 'cardigan']):
        return 'shirt'
        
    if pg == 'accessories' or pg == 'bags' or pg == 'items':
        return 'accessory'
        
    return 'accessory'

def extract_gender(index_group, index, section, dept):
    ig = str(index_group).lower()
    ind = str(index).lower()
    sec = str(section).lower()
    dp = str(dept).lower()

    # Menswear check
    if 'menswear' in ig or 'menswear' in ind:
        return 'male'
    if 'men' in sec or 'boy' in sec or 'boys' in sec:
        return 'male'

    # Ladieswear check
    if 'ladieswear' in ig or 'ladieswear' in ind or 'divided' in ig or 'divided' in ind:
        return 'female'
    if 'lingeries' in ind or 'ladies' in sec or 'girl' in sec or 'girls' in sec or 'mama' in sec:
        return 'female'

    # Default to unisex
    return 'unisex'

def extract_color(color_group, perceived_color):
    cg = str(color_group).lower().strip()
    pc = str(perceived_color).lower().strip()
    
    # First try to match the detailed colour group name
    if cg in COLOUR_RGB_MAP:
        return COLOUR_RGB_MAP[cg], cg
    # Then try perceived master name
    if pc in COLOUR_RGB_MAP:
        return COLOUR_RGB_MAP[pc], pc
        
    # Fallbacks / substring matches
    for name, rgb in COLOUR_RGB_MAP.items():
        if name in cg or cg in name:
            return rgb, name
    for name, rgb in COLOUR_RGB_MAP.items():
        if name in pc or pc in name:
            return rgb, name
            
    return (128, 128, 128), cg if cg else "grey"

def extract_pattern(graphical_appearance, name, desc):
    ga = str(graphical_appearance).lower()
    nm = str(name).lower()
    ds = str(desc).lower()
    
    if 'floral' in ga or 'floral' in nm or 'floral' in ds or 'flower' in ds:
        return 'floral'
    if any(x in ga or x in nm or x in ds for x in ['stripe', 'striped', 'stripes']):
        return 'striped'
    if any(x in ga or x in nm or x in ds for x in ['check', 'checked', 'plaid', 'tartan', 'gingham']):
        return 'plaid'
    if any(x in ga or x in nm or x in ds for x in ['print', 'printed', 'graphic', 'logo', 'illustration']):
        return 'graphic'
    if any(x in ga or x in nm or x in ds for x in ['abstract', 'camo', 'camouflage', 'tie-dye', 'animal', 'leopard', 'zebra']):
        return 'abstract'
    if 'solid' in ga or 'plain' in ds or 'solid colour' in ds or 'solid color' in ds:
        return 'solid'
        
    return 'solid'

def extract_seasons_and_warmth(category, name, product_type, dept, desc):
    nm = str(name).lower()
    pt = str(product_type).lower()
    dp = str(dept).lower()
    ds = str(desc).lower()
    
    # Default seasons
    seasons = ["spring", "summer", "autumn", "winter"]
    warmth = 3
    
    # Score each season based on keywords in description, name, type, department
    summer_keywords = [
        'summer', 'sleeveless', 'strap', 'linen', 'beach', 'chiffon', 'sun', 
        'lightweight', 'thin', 'halterneck', 'bikini', 'crop', 'cropped', 
        'swimwear', 'sandals', 'shorts', 'camisole', 'tank top', 'backless'
    ]
    winter_keywords = [
        'winter', 'wool', 'knit', 'heavy', 'fleece', 'coat', 'down', 'padded', 
        'parka', 'fur', 'thermal', 'warm', 'brushed inside', 'cashmere', 
        'turtleneck', 'mock neck', 'thick', 'flannel', 'velvet', 'heavyweight',
        'long sleeves', 'long-sleeved', 'outerwear', 'sweatshirt', 'sweater', 'hoodie'
    ]
    
    summer_score = sum(1 for kw in summer_keywords if kw in ds or kw in nm or kw in pt or kw in dp)
    winter_score = sum(1 for kw in winter_keywords if kw in ds or kw in nm or kw in pt or kw in dp)
    
    # Special rule adjustments for specific categories
    if category == 'shorts' or category == 'skirt':
        summer_score += 2
        winter_score -= 2
    elif category == 'jacket' and any(x in pt or x in nm for x in ['coat', 'padded', 'down', 'parka']):
        winter_score += 3
        summer_score -= 3
        
    # Category-based default overrides if strong signals exist
    if winter_score > summer_score + 1:
        # winter-focused
        seasons = ["winter", "autumn"]
        if winter_score > 3:
            warmth = 5 if category in ['jacket', 'pants'] else 4
        else:
            warmth = 4 if category in ['jacket', 'pants'] else 3
    elif summer_score > winter_score + 1:
        # summer-focused
        seasons = ["summer", "spring"]
        warmth = 1 if 'sleeveless' in ds or 'short' in ds or 'tank' in nm else 2
    else:
        # Multi-season basic
        seasons = ["spring", "summer", "autumn", "winter"]
        # Adjust warmth based on category
        if category in ['shorts', 'skirt']:
            warmth = 2
        elif category == 'jacket':
            warmth = 3
        elif category == 'shirt':
            warmth = 2
        else:
            warmth = 3
            
    # Specific override logic for winter/summer dresses
    if category == 'dress':
        if 'sleeveless' in ds or 'strap' in ds or 'linen' in ds or 'summer' in ds:
            seasons = ["summer"]
            warmth = 1
        elif 'knit' in ds or 'wool' in ds or 'warm' in ds or 'long sleeve' in ds:
            seasons = ["winter", "autumn"]
            warmth = 3
            
    return seasons, warmth

def extract_style_and_occasions(category, name, dept, desc):
    nm = str(name).lower()
    dp = str(dept).lower()
    ds = str(desc).lower()
    
    # Defaults
    style = 'classic'
    occasions = ['casual']
    formality = 2
    
    # Analyze occasions
    sport_kws = ['sport', 'active', 'run', 'stretch', 'training', 'gym', 'yoga', 'workout', 'performance', 'outdoor']
    business_kws = ['blazer', 'tailored', 'suit', 'office', 'workwear', 'business', 'smart', 'creased', 'trousers', 'collared']
    formal_kws = ['evening', 'cocktail', 'wedding', 'gala', 'formal', 'silk', 'lace', 'tuxedo', 'elegant', 'dressy']
    party_kws = ['party', 'sequin', 'glitter', 'sparkle', 'festive', 'metallic', 'club', 'celebration']
    outdoor_kws = ['outdoor', 'hiking', 'waterproof', 'windproof', 'utility', 'cargo', 'trekking', 'safari']
    
    occ_set = set()
    
    if any(kw in ds or kw in nm or kw in dp for kw in sport_kws):
        occ_set.add('sport')
    if any(kw in ds or kw in nm or kw in dp for kw in business_kws):
        occ_set.add('business')
    if any(kw in ds or kw in nm or kw in dp for kw in formal_kws):
        occ_set.add('formal')
    if any(kw in ds or kw in nm or kw in dp for kw in party_kws):
        occ_set.add('party')
    if any(kw in ds or kw in nm or kw in dp for kw in outdoor_kws):
        occ_set.add('outdoor')
        
    if not occ_set:
        occ_set.add('casual')
    else:
        # Casual is a default for most things except purely formal/business
        if len(occ_set) < 3:
            occ_set.add('casual')
            
    occasions = list(occ_set)
    
    # Determine Style tag (classic, streetwear, bohemian, minimalist, preppy, athletic)
    if 'sport' in occ_set or 'active' in dp:
        style = 'athletic'
    elif 'business' in occ_set or 'formal' in occ_set:
        style = 'classic'
    elif 'lace' in ds or 'floral' in ds or 'boho' in ds or 'bohemian' in ds or 'embroidery' in ds or 'ruffle' in ds:
        style = 'bohemian'
    elif any(x in ds or x in nm for x in ['oversized', 'relaxed', 'hoodie', 'sweatshirt', 'cargo', 'denim', 'jeans', 'streetwear']):
        style = 'streetwear'
    elif any(x in ds or x in nm for x in ['basic', 'minimal', 'clean', 'simple', 'classic']):
        style = 'minimalist'
    elif any(x in ds or x in nm or x in dp for x in ['preppy', 'polo', 'stripe', 'knitvest', 'oxford', 'blazer']):
        style = 'preppy'
    else:
        style = 'classic'
        
    # Formality level (1 to 5)
    if 'formal' in occ_set:
        formality = 5 if category in ['dress', 'jacket'] else 4
    elif 'business' in occ_set:
        formality = 4
    elif 'party' in occ_set:
        formality = 3
    elif 'sport' in occ_set:
        formality = 1
    else:
        formality = 2
        
    return style, occasions, formality

def run_conversion():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, 'data', 'articles.csv')
    json_path = os.path.join(base_dir, 'data', 'hm_catalog.json')
    
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter out underwear and nightwear so we get mostly street clothes
    exclude = ['Underwear', 'Nightwear', 'Swimwear', 'Socks & Tights', 'Items', 'Stationery']
    df = df[~df['product_group_name'].isin(exclude)]
    
    # Robust word-boundary-aware innerwear filter
    innerwear_keywords = {
        "bra", "bras", "bralette", "bralettes", "underwear", "brief", "briefs",
        "panty", "panties", "trunk", "trunks", "boxer", "boxers", "socks", "sock",
        "tight", "tights", "stocking", "stockings", "nightwear", "sleepwear",
        "loungewear", "pajamas", "pyjamas", "undergarment", "undergarments"
    }
    
    def is_innerwear(row) -> bool:
        text = f"{row.get('product_group_name', '')} {row.get('product_type_name', '')} {row.get('prod_name', '')} {row.get('department_name', '')}".lower()
        tokens = set(re.findall(r'[a-z]+', text))
        return not tokens.isdisjoint(innerwear_keywords)
        
    df = df[~df.apply(is_innerwear, axis=1)]
    
    print(f"Found {len(df)} suitable garments. Sampling 10,000 for the global catalog...")
    if len(df) > 10000:
        df = df.sample(10000, random_state=42)
    
    items = []
    for _, row in df.iterrows():
        desc = str(row.get('detail_desc', ''))
        name = str(row.get('prod_name', ''))
        pg = str(row.get('product_group_name', ''))
        pt = str(row.get('product_type_name', ''))
        dept = str(row.get('department_name', ''))
        
        # 1. Extract true Category using the description and product groups
        cat = extract_category(pg, pt, name, desc)
        
        # 2. Extract Color RGB & Name using our high-quality palette
        rgb, cname = extract_color(row.get('colour_group_name', ''), row.get('perceived_colour_master_name', ''))
        
        # 3. Extract Pattern
        pat = extract_pattern(row.get('graphical_appearance_name', ''), name, desc)
        
        # 4. Extract Seasons & Warmth rating by scanning description/name/dept
        seasons, warmth = extract_seasons_and_warmth(cat, name, pt, dept, desc)
        
        # 5. Extract Style, Occasions & Formality
        style, occasions, formality = extract_style_and_occasions(cat, name, dept, desc)
        
        # Extract gender using our high-accuracy metadata mapping
        gender = extract_gender(row.get('index_group_name', ''), row.get('index_name', ''), row.get('section_name', ''), dept)
        
        item = {
            "id": f"hm_{row['article_id']}",
            "name": name,
            "category": cat,
            "color_rgb": list(rgb),
            "color_name": cname,
            "pattern": pat,
            "style": style,
            "occasions": occasions,
            "seasons": seasons,
            "warmth_level": warmth,
            "formality_level": formality,
            "gender": gender,
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
