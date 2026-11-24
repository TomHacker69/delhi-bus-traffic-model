"""
model/travel_time_model.py
Phase 3: Travel-Time Model with Dwell Model, 4-Level Fallback, and Live GTFS-RT Blending.
"""

import os
import sys
import glob
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import duckdb

logger = logging.getLogger("travel_time_model")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MODEL_DIR = os.path.join("data", "model")
SEGMENT_OBS_DIR = os.path.join("data", "parquet", "segment_observations")

DEFAULT_STOP_DWELL_SEC = 20.0
# Delhi baseline speeds by hour (km/h) for urban bus transit
HOURLY_DELHI_BASELINE_SPEEDS = {
    0: 28.0, 1: 30.0, 2: 32.0, 3: 32.0, 4: 30.0, 5: 26.0,
    6: 22.0, 7: 18.0, 8: 13.0, 9: 11.5, 10: 12.0, 11: 13.5,
    12: 15.0, 13: 16.0, 14: 16.5, 15: 15.0, 16: 14.0, 17: 12.0,
    18: 11.0, 19: 11.5, 20: 13.0, 21: 16.0, 22: 20.0, 23: 24.0,
}


class TravelTimeModel:
    """
    Time-dependent travel time and dwell model.
    Provides hierarchical fallback and live GTFS-RT blending.
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.segment_model: pd.DataFrame = pd.DataFrame()
        self.dwell_model: pd.DataFrame = pd.DataFrame()
        self.route_speed_model: pd.DataFrame = pd.DataFrame()
        self.allday_segment_model: pd.DataFrame = pd.DataFrame()

        # Fast memory lookups
        self.seg_lookup: Dict[Tuple[str, str, int], Dict[str, float]] = {}
        self.allday_lookup: Dict[Tuple[str, str], float] = {}
        self.route_lookup: Dict[Tuple[str, str, int], float] = {}
        self.dwell_lookup: Dict[Tuple[str, str, int], float] = {}
        self.dwell_stop_lookup: Dict[str, float] = {}

        # Live speed ratio cache: segment_id -> (speed_ratio, last_updated_ts)
        self.live_speed_ratios: Dict[str, Tuple[float, float]] = {}

        self.is_loaded = False
        self.load_or_train()

    def train(self, obs_glob: str = os.path.join(SEGMENT_OBS_DIR, "date=*", "segment_obs.parquet")):
        """
        Trains model tables from segment observations using DuckDB.
        """
        logger.info(f"Training travel-time model from {obs_glob}...")
        matching_files = glob.glob(obs_glob)
        if not matching_files:
            logger.warning(f"No segment observation files found at {obs_glob}. Initializing with baseline.")
            self._init_synthetic_baseline()
            return

        con = duckdb.connect()

        # 1. Segment-level hourly model: (segment_id, day_type, hour_bucket)
        seg_sql = f"""
            SELECT 
                segment_id,
                day_type,
                hour_bucket,
                count(1) as sample_count,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY running_time_s), 2) as median_running_time_s,
                round(percentile_cont(0.25) WITHIN GROUP (ORDER BY running_time_s), 2) as p25_running_time_s,
                round(percentile_cont(0.75) WITHIN GROUP (ORDER BY running_time_s), 2) as p75_running_time_s,
                round(avg(running_time_s), 2) as mean_running_time_s,
                round(avg(running_speed_kmh), 2) as avg_speed_kmh,
                round(avg(segment_length_m), 2) as segment_length_m,
                first(route_id) as route_id,
                first(from_stop_id) as from_stop_id,
                first(to_stop_id) as to_stop_id
            FROM read_parquet('{obs_glob}')
            GROUP BY segment_id, day_type, hour_bucket
        """
        self.segment_model = con.execute(seg_sql).df()

        # 2. All-day segment model: (segment_id, day_type)
        allday_sql = f"""
            SELECT 
                segment_id,
                day_type,
                count(1) as sample_count,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY running_time_s), 2) as allday_median_running_time_s,
                round(avg(segment_length_m), 2) as segment_length_m
            FROM read_parquet('{obs_glob}')
            GROUP BY segment_id, day_type
        """
        self.allday_segment_model = con.execute(allday_sql).df()

        # 3. Route-level speed model: (route_id, day_type, hour_bucket)
        route_sql = f"""
            SELECT 
                route_id,
                day_type,
                hour_bucket,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY running_speed_kmh), 2) as route_median_speed_kmh
            FROM read_parquet('{obs_glob}')
            GROUP BY route_id, day_type, hour_bucket
        """
        self.route_speed_model = con.execute(route_sql).df()

        # 4. Stop dwell model: (from_stop_id, day_type, hour_bucket)
        dwell_sql = f"""
            SELECT 
                from_stop_id as stop_id,
                day_type,
                hour_bucket,
                count(1) as dwell_count,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY dwell_time_s), 2) as median_dwell_s
            FROM read_parquet('{obs_glob}')
            GROUP BY from_stop_id, day_type, hour_bucket
        """
        self.dwell_model = con.execute(dwell_sql).df()

        # Save model tables to Parquet
        seg_out = os.path.join(self.model_dir, "segment_model.parquet")
        allday_out = os.path.join(self.model_dir, "allday_model.parquet")
        route_out = os.path.join(self.model_dir, "route_speed_model.parquet")
        dwell_out = os.path.join(self.model_dir, "dwell_model.parquet")

        self.segment_model.to_parquet(seg_out, index=False)
        self.allday_segment_model.to_parquet(allday_out, index=False)
        self.route_speed_model.to_parquet(route_out, index=False)
        self.dwell_model.to_parquet(dwell_out, index=False)

        logger.info(f"Model saved: {len(self.segment_model):,} segment-hour profiles.")
        self._build_fast_lookups()

    def _init_synthetic_baseline(self):
        """Initializes default lookups when no observations have been processed yet."""
        self.is_loaded = True

    def load_or_train(self):
        """Loads saved model files or trains if missing."""
        seg_file = os.path.join(self.model_dir, "segment_model.parquet")
        if os.path.exists(seg_file):
            logger.info(f"Loading trained models from {self.model_dir}...")
            self.segment_model = pd.read_parquet(seg_file)
            self.allday_segment_model = pd.read_parquet(os.path.join(self.model_dir, "allday_model.parquet"))
            self.route_speed_model = pd.read_parquet(os.path.join(self.model_dir, "route_speed_model.parquet"))
            self.dwell_model = pd.read_parquet(os.path.join(self.model_dir, "dwell_model.parquet"))
            self._build_fast_lookups()
        else:
            self.train()

    def _build_fast_lookups(self):
        """Converts model DataFrames into fast dictionary lookups."""
        logger.info("Building in-memory fast lookups for travel-time model...")
        # 1. Segment-hour lookup
        for _, r in self.segment_model.iterrows():
            key = (str(r["segment_id"]), str(r["day_type"]), int(r["hour_bucket"]))
            self.seg_lookup[key] = {
                "sample_count": int(r["sample_count"]),
                "median_running_time_s": float(r["median_running_time_s"]),
                "p25_running_time_s": float(r["p25_running_time_s"]),
                "p75_running_time_s": float(r["p75_running_time_s"]),
                "avg_speed_kmh": float(r["avg_speed_kmh"]),
                "segment_length_m": float(r["segment_length_m"]),
                "route_id": str(r["route_id"]),
            }

        # 2. All-day lookup
        for _, r in self.allday_segment_model.iterrows():
            key = (str(r["segment_id"]), str(r["day_type"]))
            self.allday_lookup[key] = float(r["allday_median_running_time_s"])

        # 3. Route speed lookup
        for _, r in self.route_speed_model.iterrows():
            key = (str(r["route_id"]), str(r["day_type"]), int(r["hour_bucket"]))
            self.route_lookup[key] = float(r["route_median_speed_kmh"])

        # 4. Dwell lookups
        for _, r in self.dwell_model.iterrows():
            key = (str(r["stop_id"]), str(r["day_type"]), int(r["hour_bucket"]))
            self.dwell_lookup[key] = float(r["median_dwell_s"])
            self.dwell_stop_lookup[str(r["stop_id"])] = float(r["median_dwell_s"])

        self.is_loaded = True
        logger.info(f"Model loaded with {len(self.seg_lookup):,} segment-hour profiles.")

    def update_live_speed(self, segment_id: str, observed_speed_kmh: float, current_ts: float):
        """
        Updates live speed observation from GTFS-RT.
        Computes ratio relative to historical speed.
        """
        self.live_speed_ratios[segment_id] = (observed_speed_kmh, current_ts)

    def get_dwell_time(self, stop_id: str, day_type: str, hour: int) -> float:
        """Returns modeled dwell time for a stop."""
        key = (str(stop_id), day_type, hour)
        if key in self.dwell_lookup:
            return max(5.0, self.dwell_lookup[key])
        if str(stop_id) in self.dwell_stop_lookup:
            return max(5.0, self.dwell_stop_lookup[str(stop_id)])
        return DEFAULT_STOP_DWELL_SEC

    def predict_segment_travel_time(
        self,
        segment_id: str,
        day_type: str,
        hour: int,
        segment_length_m: float = None,
        route_id: str = None,
        departure_time_minutes_ahead: float = 0.0,
        scheduled_time_s: float = None,
    ) -> Dict[str, Any]:
        """
        Predicts running time, dwell time, and total time with 4-level fallback hierarchy.
        Returns prediction details including confidence and fallback_level.
        """
        hour = int(hour) % 24
        day_type = str(day_type).lower()
        seg_id = str(segment_id)

        running_time_s = None
        confidence = "HIGH"
        fallback_level = "L0_EXACT_SEGMENT_HOUR"
        p25_s = None
        p75_s = None
        seg_len = segment_length_m

        # Level 0: Direct match on (segment_id, day_type, hour) with >= 3 samples
        key0 = (seg_id, day_type, hour)
        if key0 in self.seg_lookup:
            rec = self.seg_lookup[key0]
            if rec["sample_count"] >= 3:
                running_time_s = rec["median_running_time_s"]
                p25_s = rec["p25_running_time_s"]
                p75_s = rec["p75_running_time_s"]
                seg_len = rec["segment_length_m"]
                confidence = "HIGH"
                fallback_level = "L0_EXACT_SEGMENT_HOUR"

        # Level 1: Adjacent hours (hour-1, hour+1)
        if running_time_s is None:
            adj_hours = [(hour - 1) % 24, (hour + 1) % 24]
            adj_times = []
            for h in adj_hours:
                k = (seg_id, day_type, h)
                if k in self.seg_lookup:
                    adj_times.append(self.seg_lookup[k]["median_running_time_s"])
                    if seg_len is None:
                        seg_len = self.seg_lookup[k]["segment_length_m"]
            if adj_times:
                running_time_s = float(np.mean(adj_times))
                p25_s = running_time_s * 0.85
                p75_s = running_time_s * 1.20
                confidence = "MEDIUM_HIGH"
                fallback_level = "L1_ADJACENT_HOURS"

        # Level 2: All-day median for this day_type
        if running_time_s is None:
            k_allday = (seg_id, day_type)
            if k_allday in self.allday_lookup:
                running_time_s = self.allday_lookup[k_allday]
                p25_s = running_time_s * 0.80
                p75_s = running_time_s * 1.25
                confidence = "MEDIUM"
                fallback_level = "L2_ALL_DAY_MEDIAN"

        # Level 3: Route-level average speed multiplied by segment length
        if running_time_s is None and route_id:
            k_route = (str(route_id), day_type, hour)
            if k_route in self.route_lookup and seg_len:
                speed_kmh = self.route_lookup[k_route]
                speed_ms = max(1.0, speed_kmh / 3.6)
                running_time_s = seg_len / speed_ms
                p25_s = running_time_s * 0.80
                p75_s = running_time_s * 1.30
                confidence = "LOW"
                fallback_level = "L3_ROUTE_SPEED"

        # Level 4: Scheduled GTFS time or City Baseline Speed
        if running_time_s is None:
            if scheduled_time_s and scheduled_time_s > 0:
                running_time_s = float(scheduled_time_s)
            else:
                city_speed_kmh = HOURLY_DELHI_BASELINE_SPEEDS.get(hour, 15.0)
                speed_ms = city_speed_kmh / 3.6
                length_m = seg_len if seg_len and seg_len > 0 else 800.0  # default 800m segment
                running_time_s = length_m / speed_ms
            p25_s = running_time_s * 0.75
            p75_s = running_time_s * 1.35
            confidence = "VERY_LOW"
            fallback_level = "L4_SCHEDULED"

        # Live GTFS-RT Blending
        live_applied = False
        speed_ratio = 1.0
        if seg_id in self.live_speed_ratios:
            live_speed, live_ts = self.live_speed_ratios[seg_id]
            # Historical speed
            hist_speed = (seg_len / running_time_s) * 3.6 if seg_len and running_time_s > 0 else 15.0
            if hist_speed > 0:
                raw_ratio = live_speed / hist_speed
                # Linear decay over 60 mins: decay_weight goes from 1.0 down to 0.0 at 60 mins
                decay_weight = max(0.0, 1.0 - (departure_time_minutes_ahead / 60.0))
                speed_ratio = 1.0 + (raw_ratio - 1.0) * decay_weight
                # Clamp ratio to [0.4, 2.5]
                speed_ratio = float(np.clip(speed_ratio, 0.4, 2.5))
                running_time_s = running_time_s / speed_ratio
                live_applied = True

        # Stop dwell time
        from_stop = seg_id.split("_")[1] if "_" in seg_id else ""
        dwell_s = self.get_dwell_time(from_stop, day_type, hour)
        total_time_s = running_time_s + dwell_s

        return {
            "segment_id": seg_id,
            "day_type": day_type,
            "hour_bucket": hour,
            "running_time_s": round(running_time_s, 1),
            "dwell_time_s": round(dwell_s, 1),
            "total_time_s": round(total_time_s, 1),
            "p25_running_time_s": round(p25_s, 1) if p25_s else None,
            "p75_running_time_s": round(p75_s, 1) if p75_s else None,
            "confidence": confidence,
            "fallback_level": fallback_level,
            "live_applied": live_applied,
            "speed_ratio": round(speed_ratio, 2),
            "segment_length_m": round(seg_len, 1) if seg_len else None,
        }


if __name__ == "__main__":
    model = TravelTimeModel()
    print("Testing travel time model predictions across peak vs off-peak hours:")
    for h in [8, 14, 23]:
        pred = model.predict_segment_travel_time(
            segment_id="429_121_319",
            day_type="weekday",
            hour=h,
            segment_length_m=1200.0,
            route_id="429"
        )
        print(f"  Hour {h:02d}:00 -> Total: {pred['total_time_s']}s (running: {pred['running_time_s']}s, dwell: {pred['dwell_time_s']}s) [Confidence: {pred['confidence']}, Level: {pred['fallback_level']}]")
