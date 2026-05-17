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


def remove_background_make_white(image):
    if not REMBG_AVAILABLE:
        return image.convert("RGB")

    try:
        image_rgba = image.convert("RGBA")
        no_bg = remove(image_rgba)

        white_bg = Image.new(
            "RGBA",
            no_bg.size,
            (255, 255, 255, 255)
        )

        white_bg.paste(
            no_bg,
            mask=no_bg.split()[3]
        )

        return white_bg.convert("RGB")

    except Exception as e:
        print("Background removal failed:", e)
        return image.convert("RGB")


def predict_and_clean_item(image, use_background_removal=False):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    cleaned_image = image

    if use_background_removal:
        cleaned_image = remove_background_make_white(image)

    resized_image = cleaned_image.resize(IMG_SIZE)

    # IMPORTANT:
    # Use /255.0 because your final trained model used this preprocessing.
    image_array = np.array(resized_image).astype("float32")

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

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

    sub_conf = float(np.max(predictions[0]))
    article_conf = float(np.max(predictions[1]))
    colour_conf = float(np.max(predictions[2]))
    season_conf = float(np.max(predictions[3]))
    usage_conf = float(np.max(predictions[4]))

    display_article_type = article_type

    result = {
        "subCategory": subcategory,
        "subCategoryConfidence": sub_conf,

        "articleType": display_article_type,
        "rawArticleType": article_type,
        "articleTypeConfidence": article_conf,

        "baseColour": base_colour,
        "baseColourConfidence": colour_conf,

        "season": season,
        "seasonConfidence": season_conf,

        "usage": usage,
        "usageConfidence": usage_conf,
    }

    return result, cleaned_image


def predict_item(image, use_background_removal=False):
    result, _ = predict_and_clean_item(
        image,
        use_background_removal=use_background_removal
    )

    return result


class VisionModel:
    def analyze(self, image_source):
        res, _ = predict_and_clean_item(
            image_source,
            use_background_removal=False
        )

        art = res["articleType"].lower()
        raw_art = res["rawArticleType"].lower()
        cat_str = res["subCategory"].lower()

        category = "accessory"

        if any(w in raw_art or w in art for w in ["dress", "saree", "jumpsuit", "romper"]):
            category = "dress"

        elif any(w in raw_art or w in art for w in ["shirt", "tshirt", "top", "blouse", "kurta"]):
            category = "shirt"

        elif any(w in raw_art or w in art for w in ["pant", "jeans", "trouser", "legging", "jogger"]):
            category = "pants"

        elif "short" in raw_art or "short" in art:
            category = "shorts"

        elif "skirt" in raw_art or "skirt" in art:
            category = "skirt"

        elif any(w in raw_art or w in art for w in ["shoe", "sneaker", "boot", "sandal", "heel", "flip flop"]):
            category = "shoes"

        elif any(w in raw_art or w in art for w in ["jacket", "coat", "sweater", "sweatshirt", "hoodie", "blazer"]):
            category = "jacket"

        elif "top" in cat_str:
            category = "shirt"

        elif "bottom" in cat_str:
            category = "pants"

        elif "shoe" in cat_str or "sandal" in cat_str or "flip" in cat_str:
            category = "shoes"

        analysis = {
            "category": category,

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