import cv2
import os
import time

from collections import defaultdict

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ==========================================================
# PATHS
# ==========================================================

MODEL_PATH = "/content/drive/MyDrive/VisDrone_Project/runs/YOLOv8_VisDrone/weights/best.pt"

VIDEO_PATH = "/content/drive/MyDrive/VisDrone_Project/Test.mp4"

OUTPUT_VIDEO = "/content/drive/MyDrive/VisDrone_Project/outputs/tracking/tracked_video.mp4"

os.makedirs(
    os.path.dirname(OUTPUT_VIDEO),
    exist_ok=True
)

# ==========================================================
# PARAMETERS
# ==========================================================

CONFIDENCE = 0.45

tracker = DeepSort(
    max_age=40,
    n_init=2,
    max_iou_distance=0.6,
    max_cosine_distance=0.4
)

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading YOLO...")

model = YOLO(MODEL_PATH)

CLASS_NAMES = model.names

print("YOLO Loaded Successfully")

# ==========================================================
# CLASS COLORS
# ==========================================================

CLASS_COLORS = {

    "pedestrian": (0,255,0),

    "people": (0,180,0),

    "bicycle": (255,255,0),

    "car": (255,0,0),

    "van": (255,128,0),

    "truck": (0,0,255),

    "tricycle": (255,0,255),

    "awning-tricycle": (180,0,255),

    "bus": (0,255,255),

    "motor": (128,255,0)

}

# ==========================================================
# VIDEO
# ==========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width,height)
)

frame_count = 0

start_time = time.time()

unique_tracks = defaultdict(set)

print("Tracking Started...")
# ==========================================================
# MAIN LOOP
# ==========================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_count += 1

    # ------------------------------------------------------
    # YOLO Detection
    # ------------------------------------------------------

    results = model.predict(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )[0]

    detections = []

    if results.boxes is not None:

        for box in results.boxes:

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            confidence = float(box.conf[0])

            class_id = int(box.cls[0])

            if confidence < CONFIDENCE:
                continue

            detections.append(
                (
                    [x1, y1, x2 - x1, y2 - y1],
                    confidence,
                    class_id
                )
            )

    # ------------------------------------------------------
    # DeepSORT Update
    # ------------------------------------------------------

    tracks = tracker.update_tracks(
        detections,
        frame=frame
    )

    active_tracks = 0

    current_counts = defaultdict(int)
    # ------------------------------------------------------
    # DRAW TRACKS
    # ------------------------------------------------------

    for track in tracks:

        if not track.is_confirmed():
            continue

        active_tracks += 1

        track_id = str(track.track_id)

        class_id = track.get_det_class()

        if class_id is None:
            continue

        class_name = CLASS_NAMES[int(class_id)]

        unique_tracks[class_name].add(track_id)

        current_counts[class_name] += 1

        # Bounding Box
        left, top, right, bottom = track.to_ltrb()

        left = int(left)
        top = int(top)
        right = int(right)
        bottom = int(bottom)

        # Class Color
        color = CLASS_COLORS.get(
            class_name,
            (255,255,255)
        )

        # Draw Bounding Box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            color,
            2
        )

        # Label
        label = f"{class_name} #{track_id}"

        (w, h), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1
        )

        cv2.rectangle(
            frame,
            (left, top - 22),
            (left + w + 5, top),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (left + 2, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

    # ------------------------------------------------------
    # FPS
    # ------------------------------------------------------

    elapsed = time.time() - start_time

    fps_now = frame_count / elapsed if elapsed > 0 else 0

    cv2.putText(
        frame,
        f"FPS : {fps_now:.2f}",
        (20,35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,255),
        2
    )

    # ------------------------------------------------------
    # FRAME NUMBER
    # ------------------------------------------------------

    cv2.putText(
        frame,
        f"Frame : {frame_count}",
        (20,65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,0),
        2
    )

    # ------------------------------------------------------
    # ACTIVE TRACKS
    # ------------------------------------------------------

    cv2.putText(
        frame,
        f"Active Tracks : {active_tracks}",
        (20,95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0,255,0),
        2
    )

    # ------------------------------------------------------
    # LIVE CLASS COUNTS
    # ------------------------------------------------------

    y = 130

    for cls_name in sorted(current_counts.keys()):

        cv2.putText(
            frame,
            f"{cls_name}: {current_counts[cls_name]}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        y += 25

    # ------------------------------------------------------
    # SAVE FRAME
    # ------------------------------------------------------

    writer.write(frame)

    if frame_count % 100 == 0:

        print(f"Processed {frame_count} frames...")
        # ==========================================================
# RELEASE RESOURCES
# ==========================================================

cap.release()
writer.release()

print("\n========================================")
print("Tracking Completed Successfully!")
print("========================================")

# ==========================================================
# PERFORMANCE SUMMARY
# ==========================================================

total_time = time.time() - start_time

avg_fps = frame_count / total_time if total_time > 0 else 0

print(f"\nProcessed Frames : {frame_count}")
print(f"Execution Time   : {total_time:.2f} seconds")
print(f"Average FPS      : {avg_fps:.2f}")

print(f"\nTracked video saved at:\n{OUTPUT_VIDEO}")

# ==========================================================
# SAVE UNIQUE OBJECT COUNTS
# ==========================================================

statistics_dir = "/content/drive/MyDrive/VisDrone_Project/outputs/statistics"

os.makedirs(
    statistics_dir,
    exist_ok=True
)

stats_path = os.path.join(
    statistics_dir,
    "object_statistics.csv"
)

with open(stats_path, "w") as f:

    f.write("Class,Unique Objects\n")

    total_objects = 0

    for cls in sorted(unique_tracks.keys()):

        count = len(unique_tracks[cls])

        total_objects += count

        f.write(f"{cls},{count}\n")

# ==========================================================
# PRINT SUMMARY
# ==========================================================

print("\n=========== OBJECT SUMMARY ===========")

for cls in sorted(unique_tracks.keys()):

    count = len(unique_tracks[cls])

    print(f"{cls:<20} : {count}")

print("--------------------------------------")
print(f"Total Unique Objects : {total_objects}")
print("======================================")

print(f"\nStatistics CSV saved to:\n{stats_path}")