import os
import sys
from PIL import Image

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from vision.predict import predict_item
    print("Vision engine imported successfully.\n")
except ImportError as e:
    print(f"Error: Could not import vision engine: {e}")
    sys.exit(1)

def test_vision_on_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    print(f"Testing vision model on: {image_path}")
    try:
        image = Image.open(image_path)
        prediction = predict_item(image)
        
        print("\n--- Prediction Results ---")
        for key, value in prediction.items():
            print(f"{key}: {value}")
        print("--------------------------\n")
        
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    # Test with a sample image if provided as argument, otherwise use default test.jpg
    target_image = "test.jpg"
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    
    test_vision_on_image(target_image)
