import os
import cv2
import pandas as pd

# ==========================================
# PATHS
# ==========================================

VIDEO_PATH = "/content/drive/MyDrive/VisDrone_Project/outputs/tracking/tracked_video.mp4"

CSV_PATH = "/content/drive/MyDrive/VisDrone_Project/outputs/statistics/object_statistics.csv"

# ==========================================
# VIDEO INFORMATION
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = total_frames / fps if fps > 0 else 0

cap.release()

# ==========================================
# OBJECT STATISTICS
# ==========================================

df = pd.read_csv(CSV_PATH)

total_objects = df["Unique Objects"].sum()

# ==========================================
# PRINT RESULTS
# ==========================================

print("=" * 60)
print("TRACKER EVALUATION")
print("=" * 60)

print(f"Resolution      : {width} x {height}")
print(f"Total Frames    : {total_frames}")
print(f"FPS             : {fps:.2f}")
print(f"Video Duration  : {duration:.2f} sec")
print(f"Unique Objects  : {total_objects}")

print("\nClass-wise Counts")
print("-" * 60)

for _, row in df.iterrows():

    print(f"{row['Class']:<20} : {row['Unique Objects']}")

print("=" * 60)

report_path = "/content/drive/MyDrive/VisDrone_Project/outputs/statistics/tracker_report.txt"

with open(report_path, "w") as f:
    f.write("TRACKER EVALUATION\n")
    f.write("=" * 40 + "\n")
    f.write(f"Resolution      : {width} x {height}\n")
    f.write(f"Total Frames    : {total_frames}\n")
    f.write(f"FPS             : {fps:.2f}\n")
    f.write(f"Video Duration  : {duration:.2f} sec\n")
    f.write(f"Unique Objects  : {total_objects}\n\n")

    f.write("Class-wise Counts\n")
    f.write("-" * 40 + "\n")

    for _, row in df.iterrows():
        f.write(f"{row['Class']:<20} : {row['Unique Objects']}\n")

print(f"\nTracker report saved to:\n{report_path}")