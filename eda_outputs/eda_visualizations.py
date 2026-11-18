from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR

CSV_FILE = BASE_DIR / "trip_statistics.csv"

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(CSV_FILE)

print("\n========== EDA VISUALIZATION ==========")
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

# ==========================================
# 1. TRIP DURATION DISTRIBUTION
# ==========================================
plt.figure(figsize=(10, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.hist(
        subset["duration_sec"],
        bins=15,
        alpha=0.6,
        label=split
    )

plt.title("Distribution of Trip Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Trips")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR / "trip_duration_distribution.png", dpi=300)
plt.show()

# ==========================================
# 2. ROAD NODES DISTRIBUTION
# ==========================================
plt.figure(figsize=(10, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.hist(
        subset["nodes"],
        bins=15,
        alpha=0.6,
        label=split
    )

plt.title("Distribution of Road Network Nodes per Trip")
plt.xlabel("Number of Nodes")
plt.ylabel("Number of Trips")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR / "road_nodes_distribution.png", dpi=300)
plt.show()

# ==========================================
# 3. ROAD ARCS DISTRIBUTION
# ==========================================
plt.figure(figsize=(10, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.hist(
        subset["arcs"],
        bins=15,
        alpha=0.6,
        label=split
    )

plt.title("Distribution of Road Network Arcs per Trip")
plt.xlabel("Number of Road Arcs")
plt.ylabel("Number of Trips")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR / "road_arcs_distribution.png", dpi=300)
plt.show()

# ==========================================
# 4. GPS POINTS VS TRIP DURATION
# ==========================================
plt.figure(figsize=(10, 6))

for split in df["split"].unique():
    subset = df[df["split"] == split]

    plt.scatter(
        subset["gps_points"],
        subset["duration_sec"],
        alpha=0.7,
        label=split
    )

plt.title("GPS Points vs Trip Duration")
plt.xlabel("Number of GPS Points")
plt.ylabel("Trip Duration (seconds)")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR / "gps_vs_duration.png", dpi=300)
plt.show()

# ==========================================
# SUMMARY
# ==========================================
print("\n========== GRAPH FILES GENERATED ==========")

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(file.name)

print("\nEDA visualization completed successfully!")