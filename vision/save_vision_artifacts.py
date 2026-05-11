"""
save_vision_artifacts.py
========================
Run this cell at the END of EntaNazelKeda_visionmodel.ipynb (after the
classifier and label_encoder have been trained) to persist the model
artifacts needed by the recommendation engine.

Copy-paste the cell below into Colab or run this script locally.
"""

# ── CELL TO ADD AT THE END OF YOUR COLAB NOTEBOOK ──────────────────────────
# (Everything below this comment is a single notebook cell)

import os, sys

# Make sure the project root is importable
PROJECT_ROOT = "/content/EntaNazelKeda"   # adjust if your Colab path differs
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Install joblib if not already present (it usually is with sklearn)
# !pip install joblib  # uncomment if needed

from vision.vision_model import VisionModel

# Save the trained artifacts to vision/saved/
# `classifier` and `label_encoder` must already be defined in the notebook.
VisionModel.save_artifacts(classifier, label_encoder, save_dir=os.path.join(PROJECT_ROOT, "vision", "saved"))

print("Done! Files written:")
print("  vision/saved/classifier.pkl")
print("  vision/saved/label_encoder.pkl")

# ── OPTIONAL: Quick sanity check ────────────────────────────────────────────
# vm = VisionModel(saved_dir=os.path.join(PROJECT_ROOT, "vision", "saved"))
# result = vm.analyze(data_sample.loc[0, "image_path"])
# print("Category:", result["category"])
# print("Kaggle label:", result["kaggle_label"])
# print("Confidence:", result["confidence"])
# print("Feature vector (first 5):", result["image_features"][:5])
