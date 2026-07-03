from ultralytics import YOLO
import cv2
import os

# ==========================================
# PATHS
# ==========================================

MODEL_PATH = "/content/drive/MyDrive/VisDrone_Project/runs/YOLOv8_VisDrone/weights/best.pt"

IMAGE_PATH = "/content/drive/MyDrive/VisDrone_Project/data/VisDrone2019-DET-val/images"

OUTPUT_DIR = "/content/drive/MyDrive/VisDrone_Project/outputs/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)

# ==========================================
# IMAGE LIST
# ==========================================

image_files = sorted(os.listdir(IMAGE_PATH))

# Predict first 10 images
for image_name in image_files[:10]:

    image_path = os.path.join(
        IMAGE_PATH,
        image_name
    )

    results = model.predict(
        source=image_path,
        conf=0.25,
        save=False
    )

    annotated = results[0].plot()

    save_path = os.path.join(
        OUTPUT_DIR,
        image_name
    )

    cv2.imwrite(
        save_path,
        annotated
    )

print("Prediction Completed!")