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
# PREDICTION FUNCTION
# ==========================================

def predict_item(image, use_background_removal=True):

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    # Remove background and put clothing on white background
    if use_background_removal:
        image = remove_background_make_white(image)

    # Resize
    image = image.resize(IMG_SIZE)

    # IMPORTANT:
    # Use same preprocessing used during training.
    image_array = np.array(image).astype("float32")

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

    subcategory = encoders["subCategory"].inverse_transform(
        [np.argmax(predictions[0])]
    )[0]

    article_type = encoders["articleType"].inverse_transform(
        [np.argmax(predictions[1])]
    )[0]

    base_colour = encoders["baseColour"].inverse_transform(
        [np.argmax(predictions[2])]
    )[0]

    season = encoders["season"].inverse_transform(
        [np.argmax(predictions[3])]
    )[0]

    usage = encoders["usage"].inverse_transform(
        [np.argmax(predictions[4])]
    )[0]

    result = {
        "subCategory": subcategory,
        "articleType": article_type,
        "baseColour": base_colour,
        "season": season,
        "usage": usage,
    }

    return result