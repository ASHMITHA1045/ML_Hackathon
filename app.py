
import streamlit as st
import pandas as pd
from pathlib import Path
import streamlit.components.v1 as components
import sys

# =========================================================
# CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide"
)

DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")
PROJECT_DIR = Path(__file__).resolve().parent
MAP_DIR = PROJECT_DIR / "maps"
MAP_DIR.mkdir(exist_ok=True)

# Import your existing map-generation function
from visualize import generate_map

# =========================================================
# HEADER
# =========================================================
st.title("🗺️ PathSnap")
st.caption("GPS Trajectory Reconstruction & Map Matching")

# =========================================================
# DATASET COUNTS
# =========================================================
train_dir = DATASET / "train"
test_dir = DATASET / "student_test"

train_trips = sorted(
    p.name for p in train_dir.iterdir() if p.is_dir()
) if train_dir.exists() else []

test_trips = sorted(
    p.name for p in test_dir.iterdir() if p.is_dir()
) if test_dir.exists() else []

c1, c2, c3 = st.columns(3)

c1.metric("Training Trips", len(train_trips))
c2.metric("Test Trips", len(test_trips))
c3.metric("Dataset", "GPS + Road Network")

st.divider()

# =========================================================
# TRIP EXPLORER
# =========================================================
st.subheader("🚗 Trip Explorer")

if not test_trips:
    st.error("No test trips found in the student_test folder.")
    st.stop()

selected_trip = st.selectbox(
    "Select a test trip",
    test_trips
)

col1, col2 = st.columns([1, 2])

# ---------------------------------------------------------
# Selected trip file information
# ---------------------------------------------------------
with col1:
    st.info(f"Selected Trip: {selected_trip}")

    st.write("### Dataset Files")

    trip_folder = test_dir / selected_trip

    for ext in [".track", ".nodes", ".arcs", ".route"]:
        file_path = trip_folder / f"{selected_trip}{ext}"

        if file_path.exists():
            st.success(f"✓ {file_path.name}")
        else:
            st.error(f"✗ {file_path.name} missing")

# ---------------------------------------------------------
# Trip statistics
# ---------------------------------------------------------
with col2:
    st.write("### Trip Statistics")

    track_file = trip_folder / f"{selected_trip}.track"

    if track_file.exists():
        track = pd.read_csv(
            track_file,
            sep=r"\s+",
            header=None,
            names=["longitude", "latitude", "timestamp"]
        )

        st.metric("GPS Observations", len(track))

        duration = track["timestamp"].max() - track["timestamp"].min()
        st.metric("Trip Duration", f"{duration:.2f} seconds")

        st.write(
            f"**Longitude Range:** "
            f"{track['longitude'].min():.6f} to "
            f"{track['longitude'].max():.6f}"
        )

        st.write(
            f"**Latitude Range:** "
            f"{track['latitude'].min():.6f} to "
            f"{track['latitude'].max():.6f}"
        )

st.divider()

# =========================================================
# DYNAMIC MAP GENERATION
# =========================================================
st.subheader("🗺️ GPS & Road Network Visualization")

map_file = MAP_DIR / f"{selected_trip}_map.html"

# Generate map only when the selected trip changes
if (
    "current_map_trip" not in st.session_state
    or st.session_state.current_map_trip != selected_trip
    or not map_file.exists()
):

    with st.spinner(f"Generating map for {selected_trip}..."):

        try:
            generate_map(
                trip_id=selected_trip,
                split="student_test",
                output_file=str(map_file)
            )

            st.session_state.current_map_trip = selected_trip

        except Exception as e:
            st.error(f"Map generation failed: {e}")
            st.stop()

# ---------------------------------------------------------
# Refresh map button
# ---------------------------------------------------------
if st.button("🔄 Refresh Selected Trip Map"):

    with st.spinner("Refreshing map..."):

        try:
            generate_map(
                trip_id=selected_trip,
                split="student_test",
                output_file=str(map_file)
            )

            st.session_state.current_map_trip = selected_trip
            st.success("Map refreshed successfully!")

        except Exception as e:
            st.error(f"Map refresh failed: {e}")

# ---------------------------------------------------------
# Display map
# ---------------------------------------------------------
if map_file.exists():

    map_html = map_file.read_text(encoding="utf-8")

    components.html(
        map_html,
        height=650,
        scrolling=True
    )

else:
    st.warning("Map file has not been generated yet.")

st.divider()

# =========================================================
# DATASET SUMMARY
# =========================================================
st.subheader("📊 Dataset Summary")

stats_file = PROJECT_DIR / "dataset_statistics.csv"

if stats_file.exists():

    stats = pd.read_csv(stats_file)

    st.dataframe(
        stats,
        use_container_width=True,
        hide_index=True
    )

else:
    st.warning("dataset_statistics.csv not found.")

st.divider()

# =========================================================
# FOOTER
# =========================================================
st.caption("PathSnap | GPS Trajectory Reconstruction & Map Matching")





# =========================================================
# DATA QUALITY CHECKS
# =========================================================
st.divider()
st.subheader("🔍 GPS Data Quality & EDA")

all_tracks = []

for split_name, split_dir in [
    ("train", train_dir),
    ("student_test", test_dir)
]:
    if split_dir.exists():
        for trip_folder in split_dir.iterdir():
            if not trip_folder.is_dir():
                continue

            track_file = trip_folder / f"{trip_folder.name}.track"

            if track_file.exists():
                df = pd.read_csv(
                    track_file,
                    sep=r"\s+",
                    header=None,
                    names=["longitude", "latitude", "timestamp"]
                )

                df["trip_id"] = trip_folder.name
                df["split"] = split_name
                all_tracks.append(df)

if all_tracks:

    gps_df = pd.concat(all_tracks, ignore_index=True)

    # Missing values
    missing_values = gps_df[
        ["longitude", "latitude", "timestamp"]
    ].isnull().sum().sum()

    # Duplicate GPS observations
    duplicate_rows = gps_df.duplicated(
        subset=["longitude", "latitude", "timestamp"]
    ).sum()

    # Invalid coordinates
    invalid_coordinates = (
        (gps_df["longitude"] < -180) |
        (gps_df["longitude"] > 180) |
        (gps_df["latitude"] < -90) |
        (gps_df["latitude"] > 90)
    ).sum()

    # Timestamp analysis
    gps_df["timestamp"] = pd.to_numeric(
        gps_df["timestamp"], errors="coerce"
    )

    invalid_timestamps = gps_df["timestamp"].isna().sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total GPS Points", len(gps_df))
    c2.metric("Missing Values", int(missing_values))
    c3.metric("Duplicate Points", int(duplicate_rows))
    c4.metric("Invalid Coordinates", int(invalid_coordinates))

    st.write("### Timestamp Check")
    st.metric("Invalid Timestamps", int(invalid_timestamps))

    st.write("### GPS Dataset Preview")
    st.dataframe(
        gps_df.head(10),
        use_container_width=True
    )

else:
    st.warning("No GPS track data found.")

    # =========================================================
# GPS DATA VISUALIZATION
# =========================================================
st.divider()
st.subheader("📈 GPS Data Visualization")

# ---------------------------------------------------------
# 1. GPS Point Distribution
# ---------------------------------------------------------
st.write("### 1. GPS Point Distribution")

plot_df = gps_df.sample(
    n=min(30000, len(gps_df)),
    random_state=42
)

st.scatter_chart(
    plot_df,
    x="longitude",
    y="latitude",
    height=450
)

st.caption("Showing a sample of up to 30,000 GPS observations.")

# ---------------------------------------------------------
# 2. GPS Observations per Trip
# ---------------------------------------------------------
st.write("### 2. GPS Observations per Trip")

trip_summary = gps_df.groupby(
    ["split", "trip_id"]
).agg(
    gps_points=("timestamp", "count"),
    start_time=("timestamp", "min"),
    end_time=("timestamp", "max")
).reset_index()

trip_summary["duration_seconds"] = (
    trip_summary["end_time"] - trip_summary["start_time"]
)

trip_summary["trip_label"] = (
    trip_summary["split"] + "_" + trip_summary["trip_id"]
)

top_trips = trip_summary.nlargest(
    20, "gps_points"
).set_index("trip_label")

st.bar_chart(
    top_trips["gps_points"],
    height=400
)

# ---------------------------------------------------------
# 3. Trip Duration Distribution
# ---------------------------------------------------------
st.write("### 3. Trip Duration Distribution")

duration_df = trip_summary[
    trip_summary["duration_seconds"] >= 0
].copy()

duration_df["duration_minutes"] = (
    duration_df["duration_seconds"] / 60
)

st.bar_chart(
    duration_df.set_index("trip_label")["duration_minutes"],
    height=400
)

# ---------------------------------------------------------
# 4. Trip Summary Table
# ---------------------------------------------------------
st.write("### 4. Trip Summary")

st.dataframe(
    trip_summary[
        ["split", "trip_id", "gps_points", "duration_seconds"]
    ].sort_values("gps_points", ascending=False),
    use_container_width=True,
    hide_index=True
)

# ---------------------------------------------------------
# 5. GPS Track Visualization
# ---------------------------------------------------------
st.write("### 5. GPS Track Visualization")

if not gps_df.empty:
    selected_trip = st.selectbox(
        "Select a Trip ID",
        gps_df["trip_id"].unique()
    )

    trip_data = gps_df[gps_df["trip_id"] == selected_trip]

    st.map(
        trip_data,
        latitude="latitude",
        longitude="longitude"
    )

