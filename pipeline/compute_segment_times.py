"""
pipeline/compute_segment_times.py
Phase 2: Map-matching and Segment Travel Times.
Extracts per-trip monotonic shape progress, interpolates stop arrival times,
separates stop dwell times from running times, discards anomalies, and logs discard reasons.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, LineString

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.load_gtfs import GTFSFeed

logger = logging.getLogger("segment_times")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

CLEAN_GPS_DIR = os.path.join("data", "parquet", "clean_gps")
SEGMENT_OBS_DIR = os.path.join("data", "parquet", "segment_observations")
LOGS_DIR = os.path.join("data", "parquet")

# Thresholds
DWELL_SPEED_THRESHOLD_KMH = 2.0  # Speed below this is considered stopped/dwelling
DWELL_RADIUS_M = 40.0             # Radius around stop to measure dwell
MAX_SEGMENT_SPEED_KMH = 80.0     # Discard running speeds above this
MIN_SEGMENT_SPEED_KMH = 1.0      # Discard running speeds below this
MIN_PINGS_PER_SEGMENT = 2         # Discard segments with fewer pings
JITTER_BACKWARD_MAX_M = 30.0     # Backward jitter clamp threshold
MIN_STOP_SPACING_M = 150.0       # Minimum spacing between consecutive stops
MIN_TRIP_SPAN_M = 400.0          # Minimum displacement for a moving bus trip


def enforce_monotonic_progress(p_dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforces monotonic shape progress along the trip.
    Clamps short backward jumps (<= JITTER_BACKWARD_MAX_M) to current max distance.
    Returns (cleaned_dists, valid_mask).
    """
    clean_d = p_dists.copy()
    valid_mask = np.ones(len(p_dists), dtype=bool)
    max_d = -1.0

    for i, d in enumerate(clean_d):
        if d >= max_d:
            max_d = d
        elif (max_d - d) <= JITTER_BACKWARD_MAX_M:
            clean_d[i] = max_d
        else:
            valid_mask[i] = False

    return clean_d, valid_mask


def process_clean_pings_for_segments(
    parquet_path: str, feed: GTFSFeed, output_dir: str = SEGMENT_OBS_DIR, max_trips: int = None
) -> Dict[str, Any]:
    """
    Processes a daily clean_pings.parquet file and calculates segment observations.
    """
    fname = os.path.basename(os.path.dirname(parquet_path))
    date_str = fname.replace("date=", "")
    logger.info(f"Processing segment times for date {date_str} from {parquet_path}...")
    t0 = time.time()

    df = pd.read_parquet(parquet_path)
    if df.empty:
        logger.warning(f"Empty dataframe in {parquet_path}")
        return {"date": date_str, "status": "empty"}

    day_type = df["day_type"].iloc[0]

    # Pre-build GTFS route_stops_map for routes with known GTFS stop sequences
    st_merged = feed.stop_times.merge(feed.trips[["trip_id", "route_id"]], on="trip_id")
    st_merged = st_merged.merge(feed.stops[["stop_id", "stop_name", "x_32643", "y_32643"]], on="stop_id")
    st_merged = st_merged.sort_values(["trip_id", "stop_sequence"])

    route_stops_map = {}
    for r_id, grp in st_merged.groupby("route_id"):
        shape = feed.get_shape_for_trip("", route_id=str(r_id))
        if shape is None or shape.is_empty:
            continue
        first_tid = grp["trip_id"].iloc[0]
        t_stops = grp[grp["trip_id"] == first_tid].sort_values("stop_sequence").copy()
        pts = shapely.points(t_stops["x_32643"].values, t_stops["y_32643"].values)
        t_stops["stop_dist_along"] = shapely.line_locate_point(shape, pts)
        t_stops = t_stops.sort_values("stop_dist_along").drop_duplicates(subset=["stop_id"])
        if len(t_stops) >= 2:
            route_stops_map[str(r_id)] = t_stops

    logger.info(f"Indexed {len(route_stops_map)} GTFS route stop sequences.")

    # Spatial stops index
    s_pts = shapely.points(feed.stops["x_32643"].values, feed.stops["y_32643"].values)

    # Filter moving trips
    logger.info("Identifying active moving bus trips...")
    trip_summary = df.groupby(["vehicle_id", "trip_id"]).agg(
        pings=("ts_utc", "count"),
        min_x=("x_32643", "min"), max_x=("x_32643", "max"),
        min_y=("y_32643", "min"), max_y=("y_32643", "max"),
        r_id=("route_id", "first")
    ).reset_index()

    trip_summary["span"] = np.sqrt(
        (trip_summary["max_x"] - trip_summary["min_x"]) ** 2 +
        (trip_summary["max_y"] - trip_summary["min_y"]) ** 2
    )
    moving_trips = trip_summary[(trip_summary["span"] >= MIN_TRIP_SPAN_M) & (trip_summary["pings"] >= 8)].copy()
    logger.info(f"Found {len(moving_trips):,} active moving trips out of {len(trip_summary):,} total.")

    if max_trips:
        moving_trips = moving_trips.head(max_trips)

    discard_too_few_pings = 0
    discard_invalid_time = 0
    discard_speed_high = 0
    discard_speed_low = 0
    discard_bad_length = 0

    segment_records = []

    # Map-match each moving trip
    moving_trip_keys = set(zip(moving_trips["vehicle_id"], moving_trips["trip_id"]))
    grouped = df.groupby(["vehicle_id", "trip_id"])

    for (v_id, t_id), trip_pings in grouped:
        if (v_id, t_id) not in moving_trip_keys:
            continue

        r_id = str(trip_pings["route_id"].iloc[0])
        trip_pings_sorted = trip_pings.sort_values("ts_utc").reset_index(drop=True)

        # Check if route has defined GTFS stops and shape
        if r_id in route_stops_map:
            sub_stops = route_stops_map[r_id]
            p_dists = trip_pings_sorted["dist_along_shape_m"].values
        else:
            # Spatial trajectory matching
            raw_pts = list(zip(trip_pings_sorted["x_32643"].values, trip_pings_sorted["y_32643"].values))
            dedup_pts = [raw_pts[0]]
            for p in raw_pts[1:]:
                if p != dedup_pts[-1]:
                    dedup_pts.append(p)

            if len(dedup_pts) < 2:
                continue

            traj_line = LineString(dedup_pts)
            if traj_line.length < 500.0:
                continue

            dists_to_stops = shapely.distance(traj_line, s_pts)
            matched_stops = feed.stops[dists_to_stops <= 60.0].copy()
            if len(matched_stops) < 2:
                continue

            m_pts = shapely.points(matched_stops["x_32643"].values, matched_stops["y_32643"].values)
            matched_stops["stop_dist_along"] = shapely.line_locate_point(traj_line, m_pts)
            matched_stops = matched_stops.sort_values("stop_dist_along").drop_duplicates(subset=["stop_name"])

            filtered_stops = []
            last_sd = -1e9
            for _, s in matched_stops.iterrows():
                if s["stop_dist_along"] >= last_sd + MIN_STOP_SPACING_M:
                    filtered_stops.append(s)
                    last_sd = s["stop_dist_along"]

            if len(filtered_stops) < 2:
                continue

            sub_stops = pd.DataFrame(filtered_stops).reset_index(drop=True)
            # Project pings onto traj_line
            trip_p_pts = shapely.points(trip_pings_sorted["x_32643"].values, trip_pings_sorted["y_32643"].values)
            p_dists = shapely.line_locate_point(traj_line, trip_p_pts)

        # Monotonic enforcement
        clean_dists, valid_mask = enforce_monotonic_progress(p_dists)
        if valid_mask.sum() < 2:
            continue

        p_dists_clean = clean_dists[valid_mask]
        p_times = trip_pings_sorted["ts_utc"].values[valid_mask]
        p_speeds = trip_pings_sorted["reported_speed"].values[valid_mask]
        p_xs = trip_pings_sorted["x_32643"].values[valid_mask]
        p_ys = trip_pings_sorted["y_32643"].values[valid_mask]

        min_obs_d = p_dists_clean[0]
        max_obs_d = p_dists_clean[-1]

        # Filter stops strictly within observed bounds
        stops_in_range = sub_stops[(sub_stops["stop_dist_along"] >= min_obs_d) & (sub_stops["stop_dist_along"] <= max_obs_d)].reset_index(drop=True)
        if len(stops_in_range) < 2:
            continue

        # Interpolate stop crossing times
        stop_cross_times = []
        for _, s_row in stops_in_range.iterrows():
            sd = s_row["stop_dist_along"]
            idx = np.searchsorted(p_dists_clean, sd)
            if idx == 0:
                t_cross = p_times[0]
            elif idx >= len(p_dists_clean):
                t_cross = p_times[-1]
            else:
                d0, d1 = p_dists_clean[idx - 1], p_dists_clean[idx]
                t0_p, t1_p = p_times[idx - 1], p_times[idx]
                if d1 == d0:
                    t_cross = t0_p
                else:
                    t_cross = t0_p + (t1_p - t0_p) * ((sd - d0) / (d1 - d0))
            stop_cross_times.append(t_cross)

        # Compute segment intervals
        for s_idx in range(len(stops_in_range) - 1):
            s_from = stops_in_range.iloc[s_idx]
            s_to = stops_in_range.iloc[s_idx + 1]

            t_dep = stop_cross_times[s_idx]
            t_arr = stop_cross_times[s_idx + 1]

            seg_len = s_to["stop_dist_along"] - s_from["stop_dist_along"]
            elapsed_time = t_arr - t_dep

            if seg_len <= 10.0 or seg_len > 15000.0:
                discard_bad_length += 1
                continue

            if elapsed_time <= 0 or elapsed_time > 7200:
                discard_invalid_time += 1
                continue

            seg_mask = (p_dists_clean >= s_from["stop_dist_along"]) & (p_dists_clean <= s_to["stop_dist_along"])
            pings_in_seg = int(seg_mask.sum())

            if pings_in_seg < MIN_PINGS_PER_SEGMENT:
                discard_too_few_pings += 1
                continue

            # Compute dwell time at stop_from
            sx, sy = s_from["x_32643"], s_from["y_32643"]
            from_dists = np.sqrt((p_xs - sx) ** 2 + (p_ys - sy) ** 2)
            dwell_mask = (from_dists <= DWELL_RADIUS_M) & (p_speeds < DWELL_SPEED_THRESHOLD_KMH)

            dwell_time_s = 0.0
            dwell_idx = np.where(dwell_mask)[0]
            if len(dwell_idx) >= 2:
                dwell_time_s = float(p_times[dwell_idx[-1]] - p_times[dwell_idx[0]])
                dwell_time_s = min(dwell_time_s, elapsed_time * 0.8)

            running_time_s = max(1.0, elapsed_time - dwell_time_s)
            running_speed_kmh = (seg_len / running_time_s) * 3.6

            if running_speed_kmh > MAX_SEGMENT_SPEED_KMH:
                discard_speed_high += 1
                continue
            if running_speed_kmh < MIN_SEGMENT_SPEED_KMH:
                discard_speed_low += 1
                continue

            dep_ts_ist = pd.to_datetime(t_dep + 19800, unit="s")
            hour_bucket = dep_ts_ist.hour

            seg_id = f"{r_id}_{s_from['stop_id']}_{s_to['stop_id']}"

            segment_records.append({
                "segment_id": seg_id,
                "route_id": str(r_id),
                "from_stop_id": str(s_from["stop_id"]),
                "to_stop_id": str(s_to["stop_id"]),
                "from_stop_name": str(s_from["stop_name"]),
                "to_stop_name": str(s_to["stop_name"]),
                "trip_id": str(t_id),
                "vehicle_id": str(v_id),
                "date": date_str,
                "day_type": day_type,
                "hour_bucket": int(hour_bucket),
                "departure_ts_ist": dep_ts_ist,
                "segment_length_m": round(float(seg_len), 2),
                "running_time_s": round(float(running_time_s), 2),
                "dwell_time_s": round(float(dwell_time_s), 2),
                "total_time_s": round(float(elapsed_time), 2),
                "running_speed_kmh": round(float(running_speed_kmh), 2),
                "pings_count": pings_in_seg,
            })

    # Save to Parquet
    out_date_dir = os.path.join(output_dir, f"date={date_str}")
    os.makedirs(out_date_dir, exist_ok=True)
    out_parquet = os.path.join(out_date_dir, "segment_obs.parquet")

    if segment_records:
        seg_df = pd.DataFrame(segment_records)
        seg_df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")
        file_size_mb = round(os.path.getsize(out_parquet) / (1024 * 1024), 2)
    else:
        seg_df = pd.DataFrame()
        file_size_mb = 0.0

    t1 = time.time()
    stats = {
        "date": date_str,
        "day_type": day_type,
        "valid_segments_saved": len(seg_df),
        "discard_too_few_pings": discard_too_few_pings,
        "discard_invalid_time": discard_invalid_time,
        "discard_speed_high": discard_speed_high,
        "discard_speed_low": discard_speed_low,
        "discard_bad_length": discard_bad_length,
        "duration_sec": round(t1 - t0, 2),
        "file_size_mb": file_size_mb,
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    summary_path = os.path.join(LOGS_DIR, f"segment_stats_{date_str}.json")
    with open(summary_path, "w") as sf:
        json.dump(stats, sf, indent=2)

    logger.info(f"Phase 2 completed for {date_str} in {stats['duration_sec']}s: {stats['valid_segments_saved']:,} segment observations saved ({file_size_mb} MB).")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract segment running and dwell times from clean GPS pings.")
    parser.add_argument("--date", type=str, default="2026-10-01", help="Date in YYYY-MM-DD")
    parser.add_argument("--all", action="store_true", help="Process all dates in data/parquet/clean_gps/")
    parser.add_argument("--max-trips", type=int, default=None, help="Optional max trips to process")
    args = parser.parse_args()

    feed = GTFSFeed()

    files = []
    if args.all:
        files = sorted(glob.glob(os.path.join(CLEAN_GPS_DIR, "date=*", "clean_pings.parquet")))
    else:
        target = os.path.join(CLEAN_GPS_DIR, f"date={args.date}", "clean_pings.parquet")
        if os.path.exists(target):
            files.append(target)
        else:
            logger.error(f"Clean pings not found for {args.date} at {target}")
            return

    all_stats = []
    for f in files:
        s = process_clean_pings_for_segments(f, feed, max_trips=args.max_trips)
        all_stats.append(s)

    print("\n" + "=" * 60)
    print("PHASE 2 SEGMENT TRAVEL TIMES SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"Date: {s.get('date')} ({s.get('day_type')})")
        print(f"  Valid segments saved:     {s.get('valid_segments_saved', 0):,}")
        print(f"  Discarded (<2 pings):     {s.get('discard_too_few_pings', 0):,}")
        print(f"  Discarded (invalid time): {s.get('discard_invalid_time', 0):,}")
        print(f"  Discarded (speed >80k):   {s.get('discard_speed_high', 0):,}")
        print(f"  Discarded (speed <1k):    {s.get('discard_speed_low', 0):,}")
        print(f"  Discarded (bad length):   {s.get('discard_bad_length', 0):,}")
        print(f"  Output Parquet size:      {s.get('file_size_mb', 0)} MB (in {s.get('duration_sec', 0)}s)")
        print("-" * 60)


if __name__ == "__main__":
    main()
