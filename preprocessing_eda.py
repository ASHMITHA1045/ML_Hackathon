
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")

OUTPUT_DIR = Path("eda_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

SPLITS = ["train", "student_test"]

# =========================================================
# DATA LOADING
# =========================================================
def read_file(file_path):
    if not file_path.exists():
        return None

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )


def load_trip(trip_folder):
    trip_id = trip_folder.name

    track = read_file(trip_folder / f"{trip_id}.track")
    nodes = read_file(trip_folder / f"{trip_id}.nodes")
    arcs = read_file(trip_folder / f"{trip_id}.arcs")
    route = read_file(trip_folder / f"{trip_id}.route")

    if track is not None and track.shape[1] >= 3:
        track = track.iloc[:, :3].copy()
        track.columns = ["longitude", "latitude", "timestamp"]

    if nodes is not None and nodes.shape[1] >= 2:
        nodes = nodes.iloc[:, :2].copy()
        nodes.columns = ["longitude", "latitude"]

    if arcs is not None and arcs.shape[1] >= 2:
        arcs = arcs.iloc[:, :2].copy()
        arcs.columns = ["from_node", "to_node"]

    if route is not None:
        route = route.iloc[:, :1].copy()
        route.columns = ["arc_id"]

    return track, nodes, arcs, route


# =========================================================
# PREPROCESSING / DATA QUALITY CHECKS
# =========================================================
trip_stats = []
quality_rows = []
all_tracks = []

for split in SPLITS:

    split_folder = DATASET / split

    if not split_folder.exists():
        print(f"Missing folder: {split_folder}")
        continue

    trip_folders = sorted(
        p for p in split_folder.iterdir() if p.is_dir()
    )

    print(f"\nProcessing {split}: {len(trip_folders)} trips")

    for folder in trip_folders:

        trip_id = folder.name

        track, nodes, arcs, route = load_trip(folder)

        if track is None:
            print(f"Skipping {trip_id}: track file missing")
            continue

        if track.empty or not {"longitude", "latitude", "timestamp"}.issubset(track.columns):
            print(f"Skipping {trip_id}: invalid track file")
            continue

        # -------------------------------------------------
        # Missing values
        # -------------------------------------------------
        missing_gps = int(track.isna().sum().sum())
        missing_nodes = int(nodes.isna().sum().sum()) if nodes is not None else None
        missing_arcs = int(arcs.isna().sum().sum()) if arcs is not None else None
        missing_route = int(route.isna().sum().sum()) if route is not None else None

        # -------------------------------------------------
        # Duplicate rows
        # -------------------------------------------------
        duplicate_gps = int(track.duplicated().sum())
        duplicate_nodes = int(nodes.duplicated().sum()) if nodes is not None else None
        duplicate_arcs = int(arcs.duplicated().sum()) if arcs is not None else None
        duplicate_route = int(route.duplicated().sum()) if route is not None else None

        # -------------------------------------------------
        # Coordinate validation
        # -------------------------------------------------
        valid_coordinates = (
            track["longitude"].between(-180, 180)
            & track["latitude"].between(-90, 90)
        )

        invalid_coordinates = int((~valid_coordinates).sum())

        # -------------------------------------------------
        # Timestamp preprocessing
        # -------------------------------------------------
        timestamps = pd.to_numeric(
            track["timestamp"], errors="coerce"
        )

        invalid_timestamps = int(timestamps.isna().sum())

        time_diff = timestamps.diff().dropna()

        negative_intervals = int((time_diff < 0).sum())
        zero_intervals = int((time_diff == 0).sum())

        positive_intervals = time_diff[time_diff > 0]

        mean_interval = (
            positive_intervals.mean()
            if len(positive_intervals) > 0
            else 0
        )

        max_interval = (
            positive_intervals.max()
            if len(positive_intervals) > 0
            else 0
        )

        # -------------------------------------------------
        # Trip duration
        # -------------------------------------------------
        duration = (
            timestamps.max() - timestamps.min()
            if timestamps.notna().any()
            else 0
        )

        # -------------------------------------------------
        # GPS bounds
        # -------------------------------------------------
        trip_stats.append({
            "split": split,
            "trip_id": trip_id,
            "gps_points": len(track),
            "nodes": len(nodes) if nodes is not None else 0,
            "arcs": len(arcs) if arcs is not None else 0,
            "route_segments": len(route) if route is not None else 0,
            "duration_sec": duration,
            "longitude_min": track["longitude"].min(),
            "longitude_max": track["longitude"].max(),
            "latitude_min": track["latitude"].min(),
            "latitude_max": track["latitude"].max(),
            "mean_gps_interval_sec": mean_interval,
            "max_gps_interval_sec": max_interval
        })

        # -------------------------------------------------
        # Quality report
        # -------------------------------------------------
        quality_rows.append({
            "split": split,
            "trip_id": trip_id,
            "gps_rows": len(track),
            "missing_gps_values": missing_gps,
            "duplicate_gps_rows": duplicate_gps,
            "invalid_coordinates": invalid_coordinates,
            "invalid_timestamps": invalid_timestamps,
            "negative_time_intervals": negative_intervals,
            "zero_time_intervals": zero_intervals,
            "missing_node_values": missing_nodes,
            "duplicate_node_rows": duplicate_nodes,
            "missing_arc_values": missing_arcs,
            "duplicate_arc_rows": duplicate_arcs,
            "missing_route_values": missing_route,
            "duplicate_route_rows": duplicate_route
        })

        # Store a sample of GPS observations for plots
        sample = track.copy()
        sample["split"] = split
        sample["trip_id"] = trip_id
        all_tracks.append(sample)

        print(f"Processed trip: {trip_id}")


# =========================================================
# SAVE REPORTS
# =========================================================
trip_df = pd.DataFrame(trip_stats)
quality_df = pd.DataFrame(quality_rows)

trip_df.to_csv(OUTPUT_DIR / "trip_statistics.csv", index=False)
quality_df.to_csv(OUTPUT_DIR / "data_quality_report.csv", index=False)

print("\n========================================")
print("DATA PREPROCESSING COMPLETED")
print("========================================")

print("\nTrip Statistics:")
print(trip_df.groupby("split")["gps_points"].agg(["count", "mean", "min", "max"]))

print("\nData Quality Summary:")
if not quality_df.empty:
    print(
        quality_df.select_dtypes(include="number")
        .sum()
        .to_string()
    )

# =========================================================
# EDA 1: GPS POINTS PER TRIP
# =========================================================
if not trip_df.empty:

    plt.figure(figsize=(10, 6))

    for split in SPLITS:
        subset = trip_df[trip_df["split"] == split]

        if not subset.empty:
            plt.hist(
                subset["gps_points"],
                bins=20,
                alpha=0.6,
                label=split
            )

    plt.title("Distribution of GPS Points per Trip")
    plt.xlabel("Number of GPS Points")
    plt.ylabel("Number of Trips")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gps_points_distribution.png", dpi=300)
    plt.show()

# =========================================================
# EDA 2: TRIP DURATION DISTRIBUTION
# =========================================================
if not trip_df.empty:

    plt.figure(figsize=(10, 6))

    for split in SPLITS:
        subset = trip_df[trip_df["split"] == split]

        if not subset.empty:
            plt.hist(
                subset["duration_sec"],
                bins=20,
                alpha=0.6,
                label=split
            )

    plt.title("Trip Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Trips")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "trip_duration_distribution.png", dpi=300)
    plt.show()

# =========================================================
# EDA 3: ROAD NETWORK SIZE
# =========================================================
if not trip_df.empty:

    plt.figure(figsize=(10, 6))

    plt.scatter(
        trip_df["nodes"],
        trip_df["arcs"],
        alpha=0.7
    )

    plt.title("Road Network: Nodes vs Arcs")
    plt.xlabel("Number of Nodes")
    plt.ylabel("Number of Arcs")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "nodes_vs_arcs.png", dpi=300)
    plt.show()

# =========================================================
# EDA 4: GPS COORDINATE DISTRIBUTION
# =========================================================
if all_tracks:

    gps_df = pd.concat(all_tracks, ignore_index=True)

    # Sampling keeps the plot responsive for large datasets
    if len(gps_df) > 100000:
        gps_df = gps_df.sample(100000, random_state=42)

    plt.figure(figsize=(10, 7))

    for split in SPLITS:
        subset = gps_df[gps_df["split"] == split]

        if not subset.empty:
            plt.scatter(
                subset["longitude"],
                subset["latitude"],
                alpha=0.25,
                s=4,
                label=split
            )

    plt.title("GPS Coordinate Distribution")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gps_spatial_distribution.png", dpi=300)
    plt.show()

# =========================================================
# EDA 5: GPS TIME INTERVAL DISTRIBUTION
# =========================================================
if all_tracks:

    intervals = []

    for track in all_tracks:
        timestamps = pd.to_numeric(
            track["timestamp"], errors="coerce"
        )

        diffs = timestamps.diff().dropna()
        diffs = diffs[diffs > 0]

        if not diffs.empty:
            intervals.extend(diffs.tolist())

    if intervals:

        # Sample if the dataset is large
        interval_series = pd.Series(intervals)

        if len(interval_series) > 100000:
            interval_series = interval_series.sample(
                100000, random_state=42
            )

        plt.figure(figsize=(10, 6))

        plt.hist(
            interval_series,
            bins=50
        )

        plt.title("GPS Sampling Interval Distribution")
        plt.xlabel("Time Difference (seconds)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "gps_time_intervals.png", dpi=300)
        plt.show()

# =========================================================
# FINAL OUTPUT
# =========================================================
print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY")
print("========================================")
print("Output folder:", OUTPUT_DIR.resolve())

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
