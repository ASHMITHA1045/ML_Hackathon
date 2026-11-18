from pathlib import Path

DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")

for folder in ["student_test", "train"]:
    root = DATASET / folder

    print(f"\n========== {folder.upper()} ==========")

    if not root.exists():
        print("Folder not found:", root)
        continue

    trips = sorted(p for p in root.iterdir() if p.is_dir())

    print("Number of trips:", len(trips))

    for trip in trips[:3]:
        print(f"\nTrip: {trip.name}")

        for file in sorted(trip.iterdir()):
            if file.is_file():
                print(file.name, "|", file.stat().st_size, "bytes")