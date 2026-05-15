import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

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
# PREDICTION FUNCTION
# ==========================================

def predict_item(image):

    print("predict.py loaded successfully")
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    result = {

        "subCategory":
        encoders["subCategory"].inverse_transform(
            [np.argmax(predictions[0])]
        )[0],

        "articleType":
        encoders["articleType"].inverse_transform(
            [np.argmax(predictions[1])]
        )[0],

        "season":
        encoders["season"].inverse_transform(
            [np.argmax(predictions[2])]
        )[0],

        "usage":
        encoders["usage"].inverse_transform(
            [np.argmax(predictions[3])]
        )[0],
         
        "baseColour":
        encoders["baseColour"].inverse_transform(
            [np.argmax(predictions[4])]
        )[0],
    }

    return result