from pathlib import Path

DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")

# Inspect one training trip
trip_id = "00000001"
trip_folder = DATASET / "train" / trip_id

for ext in [".track", ".nodes", ".arcs", ".route"]:
    file_path = trip_folder / f"{trip_id}{ext}"

    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 10:
                break
            print(line.rstrip())