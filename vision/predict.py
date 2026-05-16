import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Background removal
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "vision_multi_output_model.keras"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

IMG_SIZE = (224, 224)


# ==========================================
# LOAD MODEL
# ==========================================

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# LOAD ENCODERS
# ==========================================

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


# ==========================================
# BACKGROUND REMOVAL
# ==========================================

def remove_background_make_white(image):
    """
    Removes background and places the item on a white background.
    If rembg fails, returns the original image.
    """

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


# ==========================================
# PREDICTION FUNCTIONS
# ==========================================

def predict_and_clean_item(image, use_background_removal=True):
    """
    Runs prediction and returns (result_dict, cleaned_highres_image).
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    # Remove background and put clothing on white background
    cleaned_image = image
    if use_background_removal:
        cleaned_image = remove_background_make_white(image)

    # Resize for model input
    resized_image = cleaned_image.resize(IMG_SIZE)

    # Preprocess
    image_array = np.array(resized_image).astype("float32")
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    # Decode labels
    subcategory = encoders["subCategory"].inverse_transform([np.argmax(predictions[0])])[0]
    article_type = encoders["articleType"].inverse_transform([np.argmax(predictions[1])])[0]
    base_colour = encoders["baseColour"].inverse_transform([np.argmax(predictions[2])])[0]
    season = encoders["season"].inverse_transform([np.argmax(predictions[3])])[0]
    usage = encoders["usage"].inverse_transform([np.argmax(predictions[4])])[0]

    result = {
        "subCategory": subcategory,
        "articleType": article_type,
        "baseColour": base_colour,
        "season": season,
        "usage": usage,
    }

    return result, cleaned_image


def predict_item(image, use_background_removal=True):
    """Backwards compatibility: returns only the prediction result."""
    res, _ = predict_and_clean_item(image, use_background_removal)
    return res


# ==========================================
# VISION MODEL CLASS
# ==========================================

class VisionModel:
    """
    Encapsulated Vision Model interface for the recommendation engine.
    """

    def analyze(self, image_source):
        """
        Analyzes an image and returns a dict with category mapping and features.
        """
        # image_source can be path or Image object
        res, _ = predict_and_clean_item(image_source)

        # Map to Recommendation Engine categories
        # (This avoids circular imports and provides the keys RecommendationAPI expects)
        art = res["articleType"].lower()
        cat_str = res["subCategory"].lower()
        
        category = "accessory"
        if any(w in art for w in ["dress", "saree", "jumpsuit", "romper"]):
            category = "dress"
        elif any(w in art for w in ["shirt", "tshirt", "top", "blouse", "kurta"]):
            category = "shirt"
        elif any(w in art for w in ["pant", "jeans", "trouser", "legging", "jogger"]):
            category = "pants"
        elif "short" in art:
            category = "shorts"
        elif "skirt" in art:
            category = "skirt"
        elif any(w in art for w in ["shoe", "sneaker", "boot", "sandal", "heel", "flip flop"]):
            category = "shoes"
        elif any(w in art for w in ["jacket", "coat", "sweater", "sweatshirt", "hoodie", "blazer"]):
            category = "jacket"
        elif "top" in cat_str:
            category = "shirt"
        elif "bottom" in cat_str:
            category = "pants"

        # Construct final output
        analysis = {
            "category": category,
            "subCategory": res["subCategory"],
            "articleType": res["articleType"],
            "baseColour": res["baseColour"],
            "season": res["season"],
            "usage": res["usage"],
            "image_features": None # Can be extended to extract 2048-D features if needed
        }
        
        return analysis