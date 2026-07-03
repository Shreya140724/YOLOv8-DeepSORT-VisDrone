import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# PATHS
# ==========================================

CSV_PATH = "/content/drive/MyDrive/VisDrone_Project/outputs/statistics/object_statistics.csv"

OUTPUT_DIR = "/content/drive/MyDrive/VisDrone_Project/outputs/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Object Statistics")
print(df)

# ==========================================
# BAR CHART
# ==========================================

plt.figure(figsize=(10,6))

plt.bar(
    df["Class"],
    df["Unique Objects"]
)

plt.title("Unique Object Count by Class")

plt.xlabel("Object Class")

plt.ylabel("Count")

plt.xticks(rotation=45)

plt.tight_layout()

bar_path = os.path.join(
    OUTPUT_DIR,
    "object_count_bar.png"
)

plt.savefig(bar_path)

plt.close()

# ==========================================
# PIE CHART
# ==========================================

plt.figure(figsize=(8,8))

plt.pie(
    df["Unique Objects"],
    labels=df["Class"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Object Class Distribution")

pie_path = os.path.join(
    OUTPUT_DIR,
    "object_distribution_pie.png"
)

plt.savefig(pie_path)

plt.close()

# ==========================================
# TOP 5 OBJECTS
# ==========================================

top5 = df.sort_values(
    by="Unique Objects",
    ascending=False
).head(5)

plt.figure(figsize=(8,5))

plt.barh(
    top5["Class"],
    top5["Unique Objects"]
)

plt.title("Top 5 Detected Object Classes")

plt.xlabel("Unique Objects")

plt.tight_layout()

top5_path = os.path.join(
    OUTPUT_DIR,
    "top5_objects.png"
)

plt.savefig(top5_path)

plt.close()

print(f"Top-5 Chart Saved : {top5_path}")

# ==========================================
# PRINT SUMMARY
# ==========================================

print("\nVisualization Completed Successfully!")

print(f"\nBar Chart Saved : {bar_path}")

print(f"Pie Chart Saved : {pie_path}")