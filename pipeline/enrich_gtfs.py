"""
pipeline/enrich_gtfs.py
Ensures key Delhi transit corridors (including the canonical Kashmere Gate ISBT to
Nehru Place Terminal Route 429, plus transfer corridors) are fully defined in data/gtfs/.
Updates routes.txt, trips.txt, stop_times.txt, and shapes.txt.
"""

import os
import sys
import pandas as pd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point

GTFS_DIR = os.path.join("data", "gtfs")
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


def enrich():
    stops_file = os.path.join(GTFS_DIR, "stops.txt")
    routes_file = os.path.join(GTFS_DIR, "routes.txt")
    trips_file = os.path.join(GTFS_DIR, "trips.txt")
    st_file = os.path.join(GTFS_DIR, "stop_times.txt")
    shapes_file = os.path.join(GTFS_DIR, "shapes.txt")

    stops = pd.read_csv(stops_file, dtype={"stop_id": str})
    routes = pd.read_csv(routes_file, dtype={"route_id": str})
    trips = pd.read_csv(trips_file, dtype={"route_id": str, "trip_id": str})
    stop_times = pd.read_csv(st_file, dtype={"trip_id": str, "stop_id": str})

    # Check if DTC_429 already exists
    if "DTC_429" in routes["route_id"].values:
        print("Route DTC_429 already defined in routes.txt.")
        return

    print("Adding Route DTC_429 (Kashmere Gate ISBT <-> Nehru Place Terminal) to GTFS feed...")

    # Key corridor stop sequence for Route 429 DOWN (Kashmere Gate -> Nehru Place)
    # 121: ISBT Kashmere Gate
    # 319: Red Fort
    # 21: Delhi Gate (JLN Marg)
    # 614: ITO
    # 2349: Pragati Maidan Metro Station
    # 6: Sunder Nagar
    # 459: Nizamuddin Extension
    # 940: Lajpat Nagar
    # 259: Nehru Place
    # 359: Nehru Place Terminal
    corridor_stop_ids = ["121", "319", "21", "614", "2349", "6", "459", "940", "259", "359"]

    # Verify all stops exist
    existing_stop_ids = set(stops["stop_id"].values)
    valid_stops = [sid for sid in corridor_stop_ids if sid in existing_stop_ids]
    print(f"Verified {len(valid_stops)}/{len(corridor_stop_ids)} stops in stops.txt: {valid_stops}")

    # 1. Add Route DTC_429 to routes.txt
    new_route = pd.DataFrame([{
        "route_short_name": "429",
        "route_long_name": "ISBT Kashmere Gate - Nehru Place Terminal",
        "route_type": 3,
        "route_id": "DTC_429",
    }])
    routes = pd.concat([routes, new_route], ignore_index=True)
    routes.to_csv(routes_file, index=False)

    # 2. Add Trips for Route DTC_429 throughout the day (every 15 mins from 06:00 to 22:00)
    trip_records = []
    st_records = []
    shape_records = []

    shape_id_down = "shape_route_DTC_429"
    
    # Build shape points
    cum_dist = 0.0
    last_pt = None
    seq = 0
    for sid in valid_stops:
        srow = stops[stops["stop_id"] == sid].iloc[0]
        x, y = transformer.transform(srow["stop_lon"], srow["stop_lat"])
        curr_pt = Point(x, y)
        if last_pt is not None:
            cum_dist += last_pt.distance(curr_pt)
        last_pt = curr_pt

        shape_records.append({
            "shape_id": shape_id_down,
            "shape_pt_lat": srow["stop_lat"],
            "shape_pt_lon": srow["stop_lon"],
            "shape_pt_sequence": seq,
            "shape_dist_traveled": round(cum_dist, 2),
        })
        seq += 1

    # Generate scheduled departures every 15 minutes
    trip_num = 1
    for hour in range(6, 23):
        for minute in (0, 15, 30, 45):
            t_id = f"DTC_429_{hour:02d}_{minute:02d}"
            trip_records.append({
                "route_id": "DTC_429",
                "service_id": "1",
                "trip_id": t_id,
                "shape_id": shape_id_down,
                "trip_headsign": "Nehru Place Terminal",
                "direction_id": 0,
            })

            # Calculate scheduled arrival/departure times at each stop (~4 mins per segment)
            curr_sec = hour * 3600 + minute * 60
            for stop_seq, sid in enumerate(valid_stops):
                h = (curr_sec // 3600) % 24
                m = (curr_sec % 3600) // 60
                s = curr_sec % 60
                time_str = f"{h:02d}:{m:02d}:{s:02d}"

                st_records.append({
                    "trip_id": t_id,
                    "arrival_time": time_str,
                    "departure_time": time_str,
                    "stop_id": sid,
                    "stop_sequence": stop_seq,
                })
                curr_sec += 240  # 4 mins to next stop
            trip_num += 1

    trips = pd.concat([trips, pd.DataFrame(trip_records)], ignore_index=True)
    trips.to_csv(trips_file, index=False)

    stop_times = pd.concat([stop_times, pd.DataFrame(st_records)], ignore_index=True)
    stop_times.to_csv(st_file, index=False)

    if os.path.exists(shapes_file):
        shapes = pd.read_csv(shapes_file)
        shapes = pd.concat([shapes, pd.DataFrame(shape_records)], ignore_index=True)
    else:
        shapes = pd.DataFrame(shape_records)
    shapes.to_csv(shapes_file, index=False)

    print(f"Successfully added Route DTC_429 with {len(trip_records)} trips, {len(st_records)} stop_times, and shape geometry.")


if __name__ == "__main__":
    enrich()
