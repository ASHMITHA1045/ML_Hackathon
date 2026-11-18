from pathlib import Path
import pandas as pd

DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")

rows = []

for split in ["train", "student_test"]:

    root = DATASET / split

    for trip_folder in sorted(p for p in root.iterdir() if p.is_dir()):

        trip_id = trip_folder.name

        track_file = trip_folder / f"{trip_id}.track"
        nodes_file = trip_folder / f"{trip_id}.nodes"
        arcs_file = trip_folder / f"{trip_id}.arcs"
        route_file = trip_folder / f"{trip_id}.route"

        track = pd.read_csv(track_file, sep=r"\s+", header=None)
        nodes = pd.read_csv(nodes_file, sep=r"\s+", header=None)
        arcs = pd.read_csv(arcs_file, sep=r"\s+", header=None)
        route = pd.read_csv(route_file, sep=r"\s+", header=None)

        rows.append({
            "split": split,
            "trip_id": trip_id,
            "gps_points": len(track),
            "nodes": len(nodes),
            "arcs": len(arcs),
            "route_segments": len(route),
            "duration_sec": track.iloc[-1, 2] - track.iloc[0, 2],
            "longitude_min": track.iloc[:, 0].min(),
            "longitude_max": track.iloc[:, 0].max(),
            "latitude_min": track.iloc[:, 1].min(),
            "latitude_max": track.iloc[:, 1].max()
        })

df = pd.DataFrame(rows)

print("\n========== DATASET SUMMARY ==========")
print(df.groupby("split")[["gps_points", "nodes", "arcs", "route_segments", "duration_sec"]].agg(["mean", "min", "max"]))

print("\n========== FIRST 10 TRIPS ==========")
print(df.head(10).to_string(index=False))

df.to_csv("dataset_statistics.csv", index=False)

print("\nSaved: dataset_statistics.csv")