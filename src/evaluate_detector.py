from ultralytics import YOLO
import os

# ==========================================
# PATHS
# ==========================================

MODEL_PATH = "/content/drive/MyDrive/VisDrone_Project/runs/YOLOv8_VisDrone/weights/best.pt"

DATA_YAML = "/content/VisDrone_Data/visdrone.yaml"

OUTPUT_DIR = "/content/drive/MyDrive/VisDrone_Project/outputs/evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# LOAD MODEL
# ==========================================

print("Loading trained model...")

model = YOLO(MODEL_PATH)

print("Model Loaded Successfully!")

# ==========================================
# EVALUATION
# ==========================================

print("\nEvaluating on Validation Dataset...\n")

metrics = model.val(

    data=DATA_YAML,

    imgsz=640,

    batch=8,

    conf=0.25,

    iou=0.5,

    project=OUTPUT_DIR,

    name="detector_metrics",

    exist_ok=True
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==============================")
print("DETECTOR EVALUATION RESULTS")
print("==============================")

print(f"mAP@50      : {metrics.box.map50:.4f}")

print(f"mAP@50-95   : {metrics.box.map:.4f}")

print(f"Precision   : {metrics.box.mp:.4f}")

print(f"Recall      : {metrics.box.mr:.4f}")

print("==============================")