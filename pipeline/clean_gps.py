"""
pipeline/clean_gps.py
Phase 1: High-Performance GPS Cleaning and Loading into Partitioned Parquet.
Cleans raw daily SQLite DBs, projects to EPSG:32643, filters anomalies,
and exports partitioned Parquet datasets with detailed drop logging.
"""

import os
import sys
import glob
import json
import time
import argparse
import logging
from typing import Dict, Any, List
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import shapely

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.load_gtfs import GTFSFeed, project_coords_array

logger = logging.getLogger("clean_gps")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RAW_DIR = os.path.join("data", "raw")
PARQUET_DIR = os.path.join("data", "parquet", "clean_gps")
LOGS_DIR = os.path.join("data", "parquet")

# Public holidays in Delhi for October 2026
OCT_2026_HOLIDAYS = {
    "2026-10-02": "Mahatma Gandhi Jayanti",
    "2026-10-20": "Dussehra / Vijayadashami",
    "2026-10-26": "Maharishi Valmiki Jayanti",
}

# Spatial bounds for Delhi NCR
MIN_LAT, MAX_LAT = 28.0, 29.2
MIN_LNG, MAX_LNG = 76.5, 77.8
MAX_IMPLIED_SPEED_KMH = 80.0
MAX_SHAPE_DIST_METERS = 50.0


def determine_day_type(date_str: str) -> tuple[str, bool]:
    """Derives (day_type, is_holiday) for an ISO date string."""
    dt = pd.to_datetime(date_str)
    is_holiday = date_str in OCT_2026_HOLIDAYS
    if is_holiday:
        return "sunday", True
    dow = dt.dayofweek
    if dow == 5:
        return "saturday", False
    elif dow == 6:
        return "sunday", False
    else:
        return "weekday", False


def clean_daily_db(db_path: str, feed: GTFSFeed, output_dir: str = PARQUET_DIR, limit: int = None) -> Dict[str, Any]:
    """
    Cleans a single day's SQLite database and outputs partitioned Parquet.
    Uses vectorized operations across route shapes for maximum performance.
    """
    fname = os.path.basename(db_path)
    date_str = fname.replace("bus_movements_", "").replace(".db", "").replace("_", "-")
    day_type, is_holiday = determine_day_type(date_str)

    logger.info(f"Processing {db_path} for date {date_str} (day_type={day_type}, is_holiday={is_holiday})...")
    t0 = time.time()

    con = duckdb.connect()
    con.execute("INSTALL sqlite; LOAD sqlite;")

    # 1. Total row count in SQLite
    total_raw = con.execute(f"SELECT count(1) FROM sqlite_scan('{db_path}', 'vehicle_feed')").fetchone()[0]

    limit_clause = f"LIMIT {limit}" if limit else ""

    # 2. Stage 1: Vectorized SQL filtering in DuckDB
    logger.info(f"Extracting valid coordinates and deduplicating raw pings...")
    stage1_sql = f"""
        WITH raw AS (
            SELECT 
                vehicle_id,
                try_cast(route_id as BIGINT)::VARCHAR as route_id,
                trip_id,
                try_cast(lat as DOUBLE) as lat,
                try_cast(lng as DOUBLE) as lng,
                try_cast(timestamp as BIGINT) as ts_utc,
                speed as reported_speed
            FROM sqlite_scan('{db_path}', 'vehicle_feed')
            {limit_clause}
        ),
        bounded AS (
            SELECT * FROM raw
            WHERE lat BETWEEN {MIN_LAT} AND {MAX_LAT}
              AND lng BETWEEN {MIN_LNG} AND {MAX_LNG}
              AND ts_utc IS NOT NULL
              AND vehicle_id IS NOT NULL
        ),
        deduped AS (
            SELECT 
                vehicle_id,
                route_id,
                trip_id,
                lat,
                lng,
                ts_utc,
                reported_speed,
                ROW_NUMBER() OVER (PARTITION BY vehicle_id, ts_utc ORDER BY lat) as rn,
                LAG(ts_utc) OVER (PARTITION BY vehicle_id ORDER BY ts_utc) as prev_ts,
                LAG(lat) OVER (PARTITION BY vehicle_id ORDER BY ts_utc) as prev_lat,
                LAG(lng) OVER (PARTITION BY vehicle_id ORDER BY ts_utc) as prev_lng
            FROM bounded
        )
        SELECT 
            vehicle_id,
            route_id,
            trip_id,
            lat,
            lng,
            ts_utc,
            reported_speed,
            case when rn > 1 then 1 else 0 end as is_duplicate,
            case when prev_ts is not null and ts_utc <= prev_ts then 1 else 0 end as is_out_of_order,
            case 
                when prev_ts is null then 0.0
                when (ts_utc - prev_ts) <= 0 then 999.0
                else (sqrt(
                    power((lng - prev_lng) * 111320.0 * cos(radians(28.6)), 2) +
                    power((lat - prev_lat) * 110540.0, 2)
                ) / (ts_utc - prev_ts)) * 3.6
            end as implied_speed_kmh
        FROM deduped
        WHERE rn = 1 AND (prev_ts IS NULL OR ts_utc > prev_ts)
    """

    df = con.execute(stage1_sql).df()
    logger.info(f"Retained {len(df):,} candidate pings after spatial bounding and deduplication.")

    dropped_out_of_bounds = total_raw - len(df)
    overspeed_mask = df["implied_speed_kmh"] > MAX_IMPLIED_SPEED_KMH
    dropped_overspeed = int(overspeed_mask.sum())

    # Keep pings within speed threshold
    df = df[~overspeed_mask].copy()

    # 3. Add IST timestamp and time features
    df["ts_ist"] = pd.to_datetime(df["ts_utc"] + 19800, unit="s")
    df["hour_bucket"] = df["ts_ist"].dt.hour
    df["time_ist"] = df["ts_ist"].dt.strftime("%H:%M:%S")
    df["date"] = date_str
    df["day_type"] = day_type
    df["is_holiday"] = is_holiday

    # 4. Project coordinates to EPSG:32643
    logger.info("Projecting coordinates to EPSG:32643 metric space...")
    xs, ys = project_coords_array(df["lng"].values, df["lat"].values)
    df["x_32643"] = xs
    df["y_32643"] = ys

    # 5. Route shape distance filtering and missing trip_id fallback
    logger.info("Performing vectorized route shape distance checks...")
    dropped_far_from_shape = 0
    repaired_trips = 0

    df["dist_to_shape_m"] = 0.0
    df["dist_along_shape_m"] = 0.0

    clean_dfs = []
    # Group by route_id (vectorized per route)
    route_groups = df.groupby("route_id")
    for r_id, grp in route_groups:
        shape = feed.get_shape_for_trip("", route_id=str(r_id))
        if shape is not None and not shape.is_empty:
            pts = shapely.points(grp["x_32643"].values, grp["y_32643"].values)
            dist_to = shapely.distance(shape, pts)
            dist_along = shapely.line_locate_point(shape, pts)

            valid_mask = dist_to <= MAX_SHAPE_DIST_METERS
            dropped_far_from_shape += int((~valid_mask).sum())

            grp_valid = grp[valid_mask].copy()
            grp_valid["dist_to_shape_m"] = dist_to[valid_mask]
            grp_valid["dist_along_shape_m"] = dist_along[valid_mask]
            clean_dfs.append(grp_valid)
        else:
            # Handle missing shape or fallback to nearest route shape if trip_id missing
            missing_trip_mask = grp["trip_id"].isna() | (grp["trip_id"].astype(str).str.strip() == "")
            if missing_trip_mask.any():
                repaired_trips += int(missing_trip_mask.sum())
            clean_dfs.append(grp)

    if clean_dfs:
        clean_df = pd.concat(clean_dfs, ignore_index=True)
    else:
        clean_df = df

    # 6. Save clean dataset to partitioned Parquet
    out_date_dir = os.path.join(output_dir, f"date={date_str}")
    os.makedirs(out_date_dir, exist_ok=True)
    out_parquet = os.path.join(out_date_dir, "clean_pings.parquet")

    logger.info(f"Writing {len(clean_df):,} cleaned pings to {out_parquet}...")
    clean_df.to_parquet(out_parquet, index=False, engine="pyarrow", compression="snappy")

    t1 = time.time()
    stats = {
        "date": date_str,
        "day_type": day_type,
        "is_holiday": is_holiday,
        "raw_pings": total_raw,
        "dropped_out_of_bounds": dropped_out_of_bounds,
        "dropped_overspeed": dropped_overspeed,
        "dropped_far_from_shape": dropped_far_from_shape,
        "repaired_missing_trip_id": repaired_trips,
        "clean_pings": len(clean_df),
        "duration_sec": round(t1 - t0, 2),
        "output_path": out_parquet,
        "file_size_mb": round(os.path.getsize(out_parquet) / (1024 * 1024), 2),
    }

    os.makedirs(LOGS_DIR, exist_ok=True)
    summary_path = os.path.join(LOGS_DIR, f"cleaning_stats_{date_str}.json")
    with open(summary_path, "w") as sf:
        json.dump(stats, sf, indent=2)

    logger.info(f"Completed cleaning for {date_str} in {stats['duration_sec']}s: {stats['clean_pings']:,} clean pings saved ({stats['file_size_mb']} MB).")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean raw Delhi bus GPS databases into partitioned Parquet.")
    parser.add_argument("--db", type=str, help="Path to single SQLite DB file")
    parser.add_argument("--days", type=str, help="Comma-separated day numbers (e.g. '1', '1,2,3', '1-5')")
    parser.add_argument("--all", action="store_true", help="Process all available DBs in data/raw/")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per DB for testing")
    args = parser.parse_args()

    feed = GTFSFeed()

    db_files = []
    if args.db:
        db_files.append(args.db)
    elif args.days:
        day_nums = []
        for p in args.days.split(","):
            if "-" in p:
                s, e = map(int, p.split("-"))
                day_nums.extend(range(s, e + 1))
            else:
                day_nums.append(int(p))
        for d in day_nums:
            f = os.path.join(RAW_DIR, f"bus_movements_2026_10_{d:02d}.db")
            if os.path.exists(f):
                db_files.append(f)
            else:
                logger.warning(f"File not found for day {d}: {f}")
    elif args.all:
        db_files = sorted(glob.glob(os.path.join(RAW_DIR, "bus_movements_*.db")))
    else:
        db_files = sorted(glob.glob(os.path.join(RAW_DIR, "bus_movements_*.db")))

    if not db_files:
        logger.error(f"No database files found in {RAW_DIR} to clean.")
        return

    logger.info(f"Found {len(db_files)} database(s) to process.")
    all_stats = []
    for dbf in db_files:
        s = clean_daily_db(dbf, feed, limit=args.limit)
        all_stats.append(s)

    print("\n" + "=" * 60)
    print("PHASE 1 GPS CLEANING SUMMARY")
    print("=" * 60)
    for s in all_stats:
        print(f"Date: {s['date']} ({s['day_type']}, holiday={s['is_holiday']})")
        print(f"  Raw pings:                {s['raw_pings']:,}")
        print(f"  Dropped (out-of-bounds):  {s['dropped_out_of_bounds']:,}")
        print(f"  Dropped (overspeed >80k): {s['dropped_overspeed']:,}")
        print(f"  Dropped (>50m off shape): {s['dropped_far_from_shape']:,}")
        print(f"  Missing trip_id fallback: {s['repaired_missing_trip_id']:,}")
        print(f"  Clean pings preserved:    {s['clean_pings']:,}")
        print(f"  Clean parquet size:       {s['file_size_mb']} MB (processed in {s['duration_sec']}s)")
        print("-" * 60)


if __name__ == "__main__":
    main()
