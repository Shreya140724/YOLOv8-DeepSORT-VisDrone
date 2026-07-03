from ultralytics import YOLO

MODEL_PATH = "/content/drive/MyDrive/VisDrone_Project/runs/YOLOv8_VisDrone/weights/best.pt"

VIDEO_PATH = "/content/drive/MyDrive/VisDrone_Project/Test.mp4"

model = YOLO(MODEL_PATH)

model.predict(
    source=VIDEO_PATH,
    save=True,
    conf=0.25,
    imgsz=640
)

print("Video Prediction Completed!")