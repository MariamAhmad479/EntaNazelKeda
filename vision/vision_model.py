import os
import numpy as np
import joblib

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


class VisionModel:
    def __init__(self):
        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(224, 224, 3)
        )

        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=base_model.output
        )

        self.classifier = joblib.load("vision/saved/clothing_classifier.pkl")
        self.label_encoder = joblib.load("vision/saved/label_encoder.pkl")

    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = self.feature_extractor.predict(img_array, verbose=0)
        return features.flatten()

    def map_category(self, kaggle_label):
        mapping = {
            "Tshirts": "shirt",
            "Shirts": "shirt",
            "Tops": "shirt",

            "Jeans": "pants",
            "Pants": "pants",

            "Dresses": "dress",
            "Skirts": "skirt",

            "Casual Shoes": "shoes",
            "Sports Shoes": "shoes",
            "Heels": "shoes",
        }

        return mapping.get(kaggle_label, "accessory")

    def analyze(self, img_path):
        features = self.extract_features(img_path)

        prediction = self.classifier.predict(features.reshape(1, -1))[0]
        kaggle_label = self.label_encoder.inverse_transform([prediction])[0]

        confidence = None
        if hasattr(self.classifier, "predict_proba"):
            probabilities = self.classifier.predict_proba(features.reshape(1, -1))[0]
            confidence = float(np.max(probabilities))

        category = self.map_category(kaggle_label)

        return {
            "category": category,
            "kaggle_label": kaggle_label,
            "confidence": confidence,
            "image_features": features.tolist()
        }