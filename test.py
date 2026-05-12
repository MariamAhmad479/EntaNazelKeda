from vision import VisionModel

model = VisionModel()
result = model.analyze("test_img.png")

print(result["category"])
print(result["kaggle_label"])
print(result["confidence"])
print(len(result["image_features"]))