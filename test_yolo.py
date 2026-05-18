from vision.yolo_detector import detect_and_crop
from vision.predict import predict_item

items = detect_and_crop("new outfits/download (2).jpg", conf=0.35)

print("Detected items:", len(items))

for i, item in enumerate(items):
    crop = item["crop"]
    yolo_class = item["class_name"]

    crop_path = f"temp_crop_{i}_{yolo_class}.jpg"
    crop.save(crop_path)

    prediction = predict_item(crop_path, yolo_class=yolo_class)

    print("\nYOLO class:", yolo_class)
    print("YOLO confidence:", item["confidence"])
    print("Classifier prediction:")
    print(prediction)