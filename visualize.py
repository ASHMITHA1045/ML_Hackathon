
from pathlib import Path
import pandas as pd
import folium

# =========================================================
# CONFIGURATION
# =========================================================
DATASET = Path(r"D:\Sem5_Labs\ML-Class\Student folder\Student folder")


# =========================================================
# LOAD TRIP DATA
# =========================================================
def load_trip(trip_id, split="train"):
    """Load all files associated with one trip."""

    trip_folder = DATASET / split / trip_id

    track = pd.read_csv(
        trip_folder / f"{trip_id}.track",
        sep=r"\s+",
        header=None
    )

    nodes = pd.read_csv(
        trip_folder / f"{trip_id}.nodes",
        sep=r"\s+",
        header=None
    )

    arcs = pd.read_csv(
        trip_folder / f"{trip_id}.arcs",
        sep=r"\s+",
        header=None
    )

    route = pd.read_csv(
        trip_folder / f"{trip_id}.route",
        sep=r"\s+",
        header=None
    )

    # Assign column names
    track.columns = ["longitude", "latitude", "timestamp"]
    nodes.columns = ["longitude", "latitude"]
    arcs.columns = ["from_node", "to_node"]
    route.columns = ["arc_id"]

    return track, nodes, arcs, route


# =========================================================
# GENERATE MAP
# =========================================================
def generate_map(trip_id, split="train", output_file=None):
    """Generate an interactive map for the selected trip."""

    track, nodes, arcs, route = load_trip(trip_id, split)

    # -----------------------------------------------------
    # GPS BOUNDS
    # -----------------------------------------------------
    min_lon = track["longitude"].min()
    max_lon = track["longitude"].max()

    min_lat = track["latitude"].min()
    max_lat = track["latitude"].max()

    center_lat = track["latitude"].mean()
    center_lon = track["longitude"].mean()

    margin = 0.005

    # -----------------------------------------------------
    # CREATE BASE MAP
    # -----------------------------------------------------
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
        control_scale=True
    )

    # Esri street map tiles (alternative to blocked OSM tiles)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Esri World street map",
        overlay=False,
        control=True
    ).add_to(m)

    # -----------------------------------------------------
    # RAW GPS TRAJECTORY
    # -----------------------------------------------------
    gps_points = list(
        zip(track["latitude"], track["longitude"])
    )

    gps_layer = folium.FeatureGroup(
        name="Raw GPS Trajectory",
        show=True
    )

    # GPS trajectory line
    folium.PolyLine(
        gps_points,
        color="blue",
        weight=4,
        opacity=0.85,
        tooltip="Raw GPS Trajectory"
    ).add_to(gps_layer)

    # Start marker
    folium.Marker(
        gps_points[0],
        tooltip="Trip Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(gps_layer)

    # End marker
    folium.Marker(
        gps_points[-1],
        tooltip="Trip End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(gps_layer)

    gps_layer.add_to(m)

    # -----------------------------------------------------
    # ROAD NETWORK
    # -----------------------------------------------------
    road_layer = folium.FeatureGroup(
        name="Road Network",
        show=True
    )

    valid_road_edges = 0

    for _, arc in arcs.iterrows():

        from_id = int(arc["from_node"])
        to_id = int(arc["to_node"])

        # Validate node indices
        if not (
            0 <= from_id < len(nodes)
            and 0 <= to_id < len(nodes)
        ):
            continue

        p1 = nodes.iloc[from_id]
        p2 = nodes.iloc[to_id]

        # Skip roads outside the trip area
        if (
            max(p1["longitude"], p2["longitude"]) < min_lon - margin
            or min(p1["longitude"], p2["longitude"]) > max_lon + margin
            or max(p1["latitude"], p2["latitude"]) < min_lat - margin
            or min(p1["latitude"], p2["latitude"]) > max_lat + margin
        ):
            continue

        # Draw road edge
        folium.PolyLine(
            [
                [p1["latitude"], p1["longitude"]],
                [p2["latitude"], p2["longitude"]]
            ],
            color="gray",
            weight=1,
            opacity=0.45,
            tooltip="Road Network"
        ).add_to(road_layer)

        valid_road_edges += 1

    road_layer.add_to(m)

    # -----------------------------------------------------
    # REFERENCE / GROUND-TRUTH ROUTE
    # -----------------------------------------------------
    route_layer = folium.FeatureGroup(
        name="Reference Route",
        show=True
    )

    valid_route_edges = 0

    for arc_id in route["arc_id"]:

        arc_id = int(arc_id)

        # Validate arc index
        if not (0 <= arc_id < len(arcs)):
            continue

        arc = arcs.iloc[arc_id]

        from_id = int(arc["from_node"])
        to_id = int(arc["to_node"])

        # Validate node indices
        if not (
            0 <= from_id < len(nodes)
            and 0 <= to_id < len(nodes)
        ):
            continue

        p1 = nodes.iloc[from_id]
        p2 = nodes.iloc[to_id]

        # Draw reference route
        folium.PolyLine(
            [
                [p1["latitude"], p1["longitude"]],
                [p2["latitude"], p2["longitude"]]
            ],
            color="orange",
            weight=5,
            opacity=0.9,
            tooltip=f"Reference Route - Arc {arc_id}"
        ).add_to(route_layer)

        valid_route_edges += 1

    route_layer.add_to(m)

    # -----------------------------------------------------
    # MAP LAYER CONTROL
    # -----------------------------------------------------
    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    # -----------------------------------------------------
    # AUTO FIT MAP TO GPS TRAJECTORY
    # -----------------------------------------------------
    m.fit_bounds([
        [min_lat - margin, min_lon - margin],
        [max_lat + margin, max_lon + margin]
    ])

    # -----------------------------------------------------
    # SAVE MAP
    # -----------------------------------------------------
    if output_file is None:
        output_file = f"trip_{split}_{trip_id}_map.html"

    m.save(output_file)

    # -----------------------------------------------------
    # TRIP STATISTICS
    # -----------------------------------------------------
    duration = (
        track["timestamp"].max()
        - track["timestamp"].min()
    )

    print("\n========== TRIP STATISTICS ==========")
    print("Trip ID:", trip_id)
    print("Split:", split)
    print("GPS observations:", len(track))
    print("Road nodes:", len(nodes))
    print("Road arcs:", len(arcs))
    print("Road edges displayed:", valid_road_edges)
    print("Reference route segments:", len(route))
    print("Reference route edges drawn:", valid_route_edges)
    print("Trip duration (seconds):", duration)
    print("Longitude range:", min_lon, "to", max_lon)
    print("Latitude range:", min_lat, "to", max_lat)
    print("Map saved:", output_file)

    return m, track, nodes, arcs, route


# =========================================================
# RUN DIRECTLY
# =========================================================
if __name__ == "__main__":

    TRIP_ID = "00000001"
    SPLIT = "train"

    generate_map(
        trip_id=TRIP_ID,
        split=SPLIT
    )
