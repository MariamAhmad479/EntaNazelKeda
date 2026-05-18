import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "vision_multi_output_model.keras"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


try:
    from vision.yolo_detector import predict_with_yolo
    YOLO_AVAILABLE = True
except Exception as e:
    print("YOLO detector not available:", e)
    YOLO_AVAILABLE = False


YOLO_ALLOWED_ARTICLES = {
    "shirt_blouse": {
        "Shirts", "Tshirts", "Tops", "Blouses", "Kurtas", "Tunics"
    },
    "top_tshirt_sweatshirt": {
        "Tshirts", "Tops", "Sweatshirts", "Shirts"
    },
    "sweater": {
        "Sweaters", "Sweatshirts"
    },
    "cardigan": {
        "Cardigan", "Sweaters", "Shrug"
    },
    "jacket": {
        "Jackets", "Blazers", "Coats"
    },
    "vest": {
        "Waistcoat", "Jackets"
    },
    "pants": {
        "Jeans", "Trousers", "Track Pants", "Leggings", "Jeggings", "Capris"
    },
    "shorts": {
        "Shorts"
    },
    "skirt": {
        "Skirts"
    },
    "coat": {
        "Coats", "Jackets", "Blazers"
    },
    "dress": {
        "Dresses", "Jumpsuit", "Rompers", "Sarees"
    },
    "jumpsuit": {
        "Jumpsuit", "Dresses", "Rompers"
    },
}


YOLO_SUBCATEGORY_MAP = {
    "shirt_blouse": "Topwear",
    "top_tshirt_sweatshirt": "Topwear",
    "sweater": "Topwear",
    "cardigan": "Topwear",
    "jacket": "Topwear",
    "vest": "Topwear",
    "pants": "Bottomwear",
    "shorts": "Bottomwear",
    "skirt": "Bottomwear",
    "coat": "Topwear",
    "dress": "Apparel",
    "jumpsuit": "Apparel",
}


ACCESSORY_ARTICLE_TYPES = {
    "Casual Shoes",
    "Sports Shoes",
    "Formal Shoes",
    "Sandals",
    "Flip Flops",
    "Heels",
    "Flats",
    "Booties",
    "Handbags",
    "Backpacks",
    "Duffel Bag",
    "Laptop Bag",
    "Clutches",
    "Wallets",
    "Caps",
    "Hat",
    "Sunglasses",
    "Watches",
    "Belts",
    "Scarves",
}

YOLO_DIRECT_MAP = {
    "jacket": "Jackets",
    "coat": "Coats",
    "dress": "Dresses",
    "pants": "Trousers",
    "shorts": "Shorts",
    "skirt": "Skirts",
    "jumpsuit": "Jumpsuit",
    "sweater": "Sweaters",
    "cardigan": "Cardigan",
    "vest": "Waistcoat",
    "shirt_blouse": "Shirts",
    "top_tshirt_sweatshirt": "Tshirts",
    }


def remove_background_make_white(image):
    if not REMBG_AVAILABLE:
        return image.convert("RGB")

    try:
        image_rgba = image.convert("RGBA")
        no_bg = remove(image_rgba)

        white_bg = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
        white_bg.paste(no_bg, mask=no_bg.split()[3])

        return white_bg.convert("RGB")

    except Exception as e:
        print("Background removal failed:", e)
        return image.convert("RGB")


def get_article_probabilities(article_prediction_array):
    article_classes = encoders["articleType"].classes_

    return {
        article_classes[i]: float(article_prediction_array[i])
        for i in range(len(article_classes))
    }


def choose_best_allowed_article(article_probs, allowed_articles, original_article):
    available_allowed = {
        article: prob
        for article, prob in article_probs.items()
        if article in allowed_articles
    }

    if original_article in allowed_articles:
        return original_article, article_probs.get(original_article, 0.0)

    if not available_allowed:
        return original_article, article_probs.get(original_article, 0.0)

    best_article = max(available_allowed, key=available_allowed.get)
    return best_article, available_allowed[best_article]


def apply_yolo_family_filter(result, article_probs, image_source):
    print("\n--- DEBUG: Entering YOLO Family Filter ---")
    if not YOLO_AVAILABLE:
        print("DEBUG: YOLO not available.")
        return result

    try:
        yolo_result = predict_with_yolo(image_source)
        print(f"DEBUG: YOLO Result: {yolo_result}")
    except Exception as e:
        print("DEBUG: YOLO prediction failed:", e)
        return result

    if not yolo_result.get("yoloDetected"):
        print("DEBUG: YOLO did not detect anything.")
        return result

    yolo_class = yolo_result.get("yoloClass")
    print(f"DEBUG: YOLO Class Detected: {yolo_class}")

    if yolo_class not in YOLO_ALLOWED_ARTICLES:
        print(f"DEBUG: YOLO Class '{yolo_class}' not in allowed articles map.")
        return result

    classifier_thinks_accessory = result["articleType"] in ACCESSORY_ARTICLE_TYPES

    if classifier_thinks_accessory and yolo_result.get("yoloConfidence", 0) < 0.60:
        return result

    allowed_articles = YOLO_ALLOWED_ARTICLES[yolo_class]

    if yolo_result.get("yoloConfidence", 0) >= 0.80:
        result["articleType"] = YOLO_DIRECT_MAP[yolo_class]
        result["articleTypeConfidence"] = yolo_result.get("yoloConfidence", 0)

    if yolo_class in YOLO_SUBCATEGORY_MAP:
        result["subCategory"] = YOLO_SUBCATEGORY_MAP[yolo_class]

    best_article, best_confidence = choose_best_allowed_article(
        article_probs=article_probs,
        allowed_articles=allowed_articles,
        original_article=result["articleType"]
    )

    result["articleType"] = best_article
    result["articleTypeConfidence"] = best_confidence

    if yolo_class in YOLO_SUBCATEGORY_MAP:
        result["subCategory"] = YOLO_SUBCATEGORY_MAP[yolo_class]

    print("YOLO CLASS:", yolo_class)
    print("CLASSIFIER ARTICLE:", result["articleType"])
    print("ALLOWED ARTICLES:", allowed_articles)
    print("FINAL ARTICLE:", best_article)

    return result


def fix_prediction(prediction):
    article = prediction["articleType"]
    sub = prediction["subCategory"]

    if sub == "Topwear" and article in {"Bra", "Briefs", "Boxers", "Panties"}:
        prediction["articleType"] = "Unknown Topwear"
        prediction["rawArticleType"] = article

    return prediction


def classify_category(article_type, sub_category):
    art = str(article_type).lower()
    sub = str(sub_category).lower()

    if any(w in art for w in ["dress", "saree", "jumpsuit", "romper"]):
        return "dress"

    if any(w in art for w in ["shirt", "tshirt", "top", "blouse", "kurta", "sweater", "sweatshirt"]):
        return "shirt"

    if any(w in art for w in ["pant", "jeans", "trouser", "legging", "jogger"]):
        return "pants"

    if "short" in art:
        return "shorts"

    if "skirt" in art:
        return "skirt"

    if any(w in art for w in ["shoe", "sneaker", "boot", "sandal", "heel", "flip flop"]):
        return "shoes"

    if any(w in art for w in ["jacket", "coat", "hoodie", "blazer", "cardigan", "waistcoat"]):
        return "jacket"

    if any(w in art for w in ["bag", "handbag", "backpack", "duffel", "wallet", "clutch"]):
        return "bag"

    if "top" in sub:
        return "shirt"

    if "bottom" in sub:
        return "pants"

    if "shoe" in sub or "sandal" in sub or "flip" in sub:
        return "shoes"

    return "accessory"


def predict_and_clean_item(image, use_background_removal=False, yolo_class=None):
    image_source = image

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    cleaned_image = image

    if use_background_removal:
        cleaned_image = remove_background_make_white(image)

    resized_image = cleaned_image.resize(IMG_SIZE)

    image_array = np.array(resized_image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    sub_idx = np.argmax(predictions[0])
    article_idx = np.argmax(predictions[1])
    colour_idx = np.argmax(predictions[2])
    season_idx = np.argmax(predictions[3])
    usage_idx = np.argmax(predictions[4])

    subcategory = encoders["subCategory"].inverse_transform([sub_idx])[0]
    article_type = encoders["articleType"].inverse_transform([article_idx])[0]
    base_colour = encoders["baseColour"].inverse_transform([colour_idx])[0]
    season = encoders["season"].inverse_transform([season_idx])[0]
    usage = encoders["usage"].inverse_transform([usage_idx])[0]

    article_probs = get_article_probabilities(predictions[1][0])

    result = {
        "subCategory": subcategory,
        "subCategoryConfidence": float(np.max(predictions[0])),

        "articleType": article_type,
        "rawArticleType": article_type,
        "articleTypeConfidence": float(np.max(predictions[1])),

        "baseColour": base_colour,
        "baseColourConfidence": float(np.max(predictions[2])),

        "season": season,
        "seasonConfidence": float(np.max(predictions[3])),

        "usage": usage,
        "usageConfidence": float(np.max(predictions[4])),
    }

    result = apply_yolo_family_filter(result, article_probs, image_source)
    result = fix_prediction(result)


    print("\n--- Prediction Results ---")

    for key, value in result.items():
        print(f"{key}: {value}")

    print("--------------------------\n")

    return result, cleaned_image


def predict_item(image, use_background_removal=False, yolo_class=None):
    result, _ = predict_and_clean_item(
        image,
        use_background_removal=use_background_removal,
        yolo_class=yolo_class
    )

    return result


class VisionModel:
    def analyze(self, image_source, yolo_class=None):
        res, _ = predict_and_clean_item(
            image_source,
            use_background_removal=False,
            yolo_class=yolo_class
        )

        analysis = {
            "category": classify_category(
                res["articleType"],
                res["subCategory"]
            ),

            "subCategory": res["subCategory"],
            "subCategoryConfidence": res["subCategoryConfidence"],

            "articleType": res["articleType"],
            "rawArticleType": res["rawArticleType"],
            "articleTypeConfidence": res["articleTypeConfidence"],

            "baseColour": res["baseColour"],
            "baseColourConfidence": res["baseColourConfidence"],

            "season": res["season"],
            "seasonConfidence": res["seasonConfidence"],

            "usage": res["usage"],
            "usageConfidence": res["usageConfidence"],

            "image_features": None
        }

        return analysis