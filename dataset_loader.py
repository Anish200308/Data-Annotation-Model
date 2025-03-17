import requests
import zipfile
import os
from io import BytesIO
from config import COCO_SAMPLE_URL

# Correct path based on the actual dataset structure
DATA_DIR = os.path.join("data", "coco2017", "coco", "images")

def download_and_extract_dataset():
    # Check if dataset already exists
    val_dir = os.path.join(DATA_DIR, 'val2017')
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        print("COCO 2017 validation dataset already exists. Skipping download.")
        return
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Downloading COCO 2017 validation dataset...")

    response = requests.get(COCO_SAMPLE_URL, stream=True)
    if response.status_code == 200:
        zip_data = BytesIO(response.content)
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            zip_ref.extractall(os.path.join("data", "coco2017"))  # Fix extraction path
        print("Dataset downloaded and extracted.")
    else:
        raise Exception(f"Failed to download dataset. Status code: {response.status_code}")

def load_sample_image():
    # Updated path to match actual file location
    image_path = os.path.join(DATA_DIR, 'val2017', '000000002532.jpg')
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Sample image not found at {image_path}")

    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image at {image_path}")
    
    return image
