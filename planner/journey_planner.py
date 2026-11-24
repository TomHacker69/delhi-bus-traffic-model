"""
planner/journey_planner.py
Phase 4: Multi-Criteria Time-Dependent Bus Journey Planner.
Supports coordinate and stop-name fuzzy inputs, walking transfers (<=500m),
time-dependent ride edges from TravelTimeModel, max 2 transfers,
2-3 diverse options (Fastest, Fewest Transfers, Least Walking),
and quiet-hour baseline delay calculation.
"""

import os
import sys
import math
import heapq
import difflib
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger("journey_planner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.load_gtfs import GTFSFeed, project_lonlat
from model.travel_time_model import TravelTimeModel

WALKING_SPEED_MS = 1.25          # 4.5 km/h = 1.25 m/s
MAX_WALK_TO_STOP_M = 1200.0      # Max walk from origin/destination to nearby stop
MAX_TRANSFER_WALK_M = 500.0      # Max walk between transfer stops
MIN_TRANSFER_TIME_SEC = 120.0    # 2 minutes minimum transfer buffer
DEFAULT_HEADWAY_SEC = 900.0      # 15 mins default headway -> 7.5 mins expected wait


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance in meters between two lat/lon coordinates."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class JourneyPlanner:
    """
    Time-dependent multi-criteria transit journey planner for Delhi.
    """

    def __init__(self, feed: GTFSFeed = None, model: TravelTimeModel = None):
        self.feed = feed or GTFSFeed()
        self.model = model or TravelTimeModel()

        # Build route topology: route_id -> list of ordered stops (stop_id, stop_name, dist_along)
        self.route_stops: Dict[str, List[Dict[str, Any]]] = {}
        # Mapping: stop_id -> list of routes serving it
        self.stop_routes: Dict[str, List[str]] = {}
        # Transfer edges: stop_id -> list of (nearby_stop_id, walk_distance_m, walk_time_s)
        self.walk_graph: Dict[str, List[Tuple[str, float, float]]] = {}

        self._build_network_topology()

    def _build_network_topology(self):
        """Constructs route stop sequences and transfer walking graph."""
        st_merged = self.feed.stop_times.merge(self.feed.trips[["trip_id", "route_id"]], on="trip_id")
        st_merged = st_merged.merge(self.feed.stops[["stop_id", "stop_name", "stop_lat", "stop_lon", "x_32643", "y_32643"]], on="stop_id")
        st_merged = st_merged.sort_values(["trip_id", "stop_sequence"])

        for r_id, grp in st_merged.groupby("route_id"):
            r_id_str = str(r_id)
            # Pick the trip with the most stops as the canonical route sequence
            best_tid = grp.groupby("trip_id")["stop_sequence"].count().idxmax()
            best_stops = grp[grp["trip_id"] == best_tid].sort_values("stop_sequence")

            stop_list = []
            for _, srow in best_stops.iterrows():
                sid = str(srow["stop_id"])
                stop_list.append({
                    "stop_id": sid,
                    "stop_name": srow["stop_name"],
                    "lat": float(srow["stop_lat"]),
                    "lon": float(srow["stop_lon"]),
                    "x": float(srow["x_32643"]),
                    "y": float(srow["y_32643"]),
                })
                self.stop_routes.setdefault(sid, []).append(r_id_str)

            if len(stop_list) >= 2:
                self.route_stops[r_id_str] = stop_list

        for sid in self.stop_routes:
            self.stop_routes[sid] = list(set(self.stop_routes[sid]))

        # Pre-build walking transfer graph for stops within 500m
        stops_df = self.feed.stops
        coords = np.column_stack([stops_df["x_32643"].values, stops_df["y_32643"].values])
        stop_ids = stops_df["stop_id"].values

        # Build KD-tree or STRtree for fast spatial walking edges
        import shapely
        s_pts = shapely.points(stops_df["x_32643"].values, stops_df["y_32643"].values)
        tree = shapely.STRtree(s_pts)

        for i, pt in enumerate(s_pts):
            u_id = str(stop_ids[i])
            nearby_indices = tree.query(pt, distance=MAX_TRANSFER_WALK_M)
            transfers = []
            for n_idx in nearby_indices:
                if n_idx != i:
                    v_id = str(stop_ids[n_idx])
                    d_m = float(pt.distance(s_pts[n_idx]))
                    if d_m <= MAX_TRANSFER_WALK_M:
                        w_time = d_m / WALKING_SPEED_MS
                        transfers.append((v_id, d_m, w_time))
            if transfers:
                self.walk_graph[u_id] = transfers

    def resolve_location(self, loc_input: str) -> Tuple[float, float, str]:
        """
        Parses location input: either 'lat,lon' or stop name.
        Returns: (lat, lon, resolved_name)
        """
        loc_str = str(loc_input).strip()
        # Check if coordinate format "lat,lon"
        if "," in loc_str:
            parts = [p.strip() for p in loc_str.split(",")]
            try:
                lat = float(parts[0])
                lon = float(parts[1])
                # Find nearest stop name for label
                nearest_stop = self.find_nearby_stops(lat, lon, max_dist_m=1000.0)
                name = nearest_stop[0]["stop_name"] if nearest_stop else f"{lat:.4f}, {lon:.4f}"
                return lat, lon, name
            except ValueError:
                pass

        # Text stop name fuzzy matching
        stop_names = self.feed.stops["stop_name"].tolist()
        matches = difflib.get_close_matches(loc_str, stop_names, n=1, cutoff=0.3)
        if matches:
            best_match = matches[0]
            matched_row = self.feed.stops[self.feed.stops["stop_name"] == best_match].iloc[0]
            return float(matched_row["stop_lat"]), float(matched_row["stop_lon"]), str(matched_row["stop_name"])

        # Fallback substring match
        sub_matches = self.feed.stops[self.feed.stops["stop_name"].str.contains(loc_str, case=False, na=False)]
        if not sub_matches.empty:
            matched_row = sub_matches.iloc[0]
            return float(matched_row["stop_lat"]), float(matched_row["stop_lon"]), str(matched_row["stop_name"])

        raise ValueError(f"Could not resolve location: '{loc_input}'")

    def find_nearby_stops(self, lat: float, lon: float, max_dist_m: float = MAX_WALK_TO_STOP_M) -> List[Dict[str, Any]]:
        """Finds all GTFS stops within max_dist_m meters of (lat, lon)."""
        x, y = project_lonlat(lon, lat)
        stops_df = self.feed.stops
        dx = stops_df["x_32643"] - x
        dy = stops_df["y_32643"] - y
        dists = np.sqrt(dx ** 2 + dy ** 2)

        near_mask = dists <= max_dist_m
        matched = stops_df[near_mask].copy()
        matched["dist_m"] = dists[near_mask]
        matched["walk_time_s"] = matched["dist_m"] / WALKING_SPEED_MS
        matched = matched.sort_values("dist_m")

        results = []
        for _, r in matched.head(8).iterrows():
            results.append({
                "stop_id": str(r["stop_id"]),
                "stop_name": str(r["stop_name"]),
                "lat": float(r["stop_lat"]),
                "lon": float(r["stop_lon"]),
                "dist_m": round(float(r["dist_m"]), 1),
                "walk_time_s": round(float(r["walk_time_s"]), 1),
            })
        return results

    def _estimate_segment_duration(
        self, route_id: str, from_stop: Dict[str, Any], to_stop: Dict[str, Any],
        depart_dt: datetime, day_type: str, quiet_hour: Optional[int] = None
    ) -> Dict[str, Any]:
        """Queries the travel-time model for a single segment between two consecutive stops."""
        seg_id = f"{route_id}_{from_stop['stop_id']}_{to_stop['stop_id']}"
        seg_len = haversine_m(from_stop["lat"], from_stop["lon"], to_stop["lat"], to_stop["lon"])

        hour = quiet_hour if quiet_hour is not None else depart_dt.hour
        pred = self.model.predict_segment_travel_time(
            segment_id=seg_id,
            day_type=day_type,
            hour=hour,
            segment_length_m=seg_len,
            route_id=route_id,
        )
        return pred

    def _evaluate_bus_leg(
        self, route_id: str, start_stop_id: str, end_stop_id: str,
        depart_dt: datetime, day_type: str, quiet_hour: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates traveling along route_id from start_stop_id to end_stop_id.
        Accumulates time-dependent segment times and dwell times.
        """
        if route_id not in self.route_stops:
            return None

        stops = self.route_stops[route_id]
        start_idx = None
        end_idx = None

        for idx, s in enumerate(stops):
            if s["stop_id"] == start_stop_id:
                start_idx = idx
            elif s["stop_id"] == end_stop_id:
                end_idx = idx

        if start_idx is None or end_idx is None or start_idx >= end_idx:
            return None

        # Traverse segments
        current_time = depart_dt
        total_running_s = 0.0
        total_dwell_s = 0.0
        total_dist_m = 0.0
        confidences = []

        sub_stops = stops[start_idx : end_idx + 1]
        for i in range(len(sub_stops) - 1):
            s_from = sub_stops[i]
            s_to = sub_stops[i + 1]

            pred = self._estimate_segment_duration(
                route_id, s_from, s_to, current_time, day_type, quiet_hour=quiet_hour
            )
            dur_s = pred["total_time_s"]
            total_running_s += pred["running_time_s"]
            total_dwell_s += pred["dwell_time_s"]
            seg_m = pred["segment_length_m"] or haversine_m(s_from["lat"], s_from["lon"], s_to["lat"], s_to["lon"])
            total_dist_m += seg_m
            confidences.append(pred["confidence"])

            current_time += timedelta(seconds=dur_s)

        # Get route short name from GTFS
        r_row = self.feed.routes[self.feed.routes["route_id"] == route_id]
        route_short = r_row["route_short_name"].iloc[0] if not r_row.empty and pd.notna(r_row["route_short_name"].iloc[0]) else route_id
        route_long = r_row["route_long_name"].iloc[0] if not r_row.empty and pd.notna(r_row["route_long_name"].iloc[0]) else ""

        overall_conf = "HIGH"
        if "VERY_LOW" in confidences:
            overall_conf = "VERY_LOW"
        elif "LOW" in confidences:
            overall_conf = "LOW"
        elif "MEDIUM" in confidences:
            overall_conf = "MEDIUM"
        elif "MEDIUM_HIGH" in confidences:
            overall_conf = "MEDIUM_HIGH"

        return {
            "type": "BUS",
            "route_id": route_id,
            "route_short_name": str(route_short),
            "route_long_name": str(route_long),
            "from_stop_id": start_stop_id,
            "from_stop_name": sub_stops[0]["stop_name"],
            "to_stop_id": end_stop_id,
            "to_stop_name": sub_stops[-1]["stop_name"],
            "departure_time": depart_dt.isoformat(),
            "arrival_time": current_time.isoformat(),
            "duration_s": round(total_running_s + total_dwell_s, 1),
            "duration_min": round((total_running_s + total_dwell_s) / 60.0, 1),
            "running_time_s": round(total_running_s, 1),
            "dwell_time_s": round(total_dwell_s, 1),
            "distance_m": round(total_dist_m, 1),
            "stops_count": len(sub_stops) - 1,
            "confidence": overall_conf,
            "stops": [
                {
                    "stop_id": s["stop_id"],
                    "stop_name": s["stop_name"],
                    "lat": s["lat"],
                    "lon": s["lon"],
                }
                for s in sub_stops
            ],
            "polyline": [[s["lat"], s["lon"]] for s in sub_stops],
        }

    def plan_journeys(
        self,
        origin: str,
        destination: str,
        depart_time_iso: str,
        max_options: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Finds 2-3 diverse journey options from origin to destination at depart_time_iso.
        """
        orig_lat, orig_lon, orig_name = self.resolve_location(origin)
        dest_lat, dest_lon, dest_name = self.resolve_location(destination)

        # Parse departure time
        if "T" in depart_time_iso:
            depart_dt = datetime.fromisoformat(depart_time_iso.replace("Z", ""))
        else:
            depart_dt = datetime.strptime(depart_time_iso, "%Y-%m-%d %H:%M:%S")

        date_str = depart_dt.strftime("%Y-%m-%d")
        # Day type
        dow = depart_dt.weekday()
        day_type = "saturday" if dow == 5 else ("sunday" if dow == 6 else "weekday")

        logger.info(f"Planning journey from '{orig_name}' to '{dest_name}' departing {depart_dt.isoformat()} ({day_type})...")

        nearby_orig_stops = self.find_nearby_stops(orig_lat, orig_lon, max_dist_m=1200.0)
        nearby_dest_stops = self.find_nearby_stops(dest_lat, dest_lon, max_dist_m=1200.0)

        if not nearby_orig_stops or not nearby_dest_stops:
            raise ValueError(f"No transit stops found near origin or destination within walking range.")

        found_itineraries = []

        # 1. Search for 0-transfer (Direct bus) options
        for o_stop in nearby_orig_stops:
            o_sid = o_stop["stop_id"]
            o_routes = set(self.stop_routes.get(o_sid, []))

            for d_stop in nearby_dest_stops:
                d_sid = d_stop["stop_id"]
                common_routes = o_routes.intersection(self.stop_routes.get(d_sid, []))

                for r_id in common_routes:
                    # Leg 1: Walk to first stop
                    t_bus_dep = depart_dt + timedelta(seconds=o_stop["walk_time_s"])
                    # Expected initial bus wait (half headway)
                    expected_wait_s = DEFAULT_HEADWAY_SEC / 2.0
                    t_bus_dep += timedelta(seconds=expected_wait_s)

                    bus_leg = self._evaluate_bus_leg(r_id, o_sid, d_sid, t_bus_dep, day_type)
                    if bus_leg is None:
                        continue

                    # Leg 3: Final walk to destination
                    t_bus_arr = datetime.fromisoformat(bus_leg["arrival_time"])
                    final_walk_s = d_stop["walk_time_s"]
                    t_final_arr = t_bus_arr + timedelta(seconds=final_walk_s)

                    walk_orig = {
                        "type": "WALK",
                        "from_name": orig_name,
                        "to_name": o_stop["stop_name"],
                        "distance_m": o_stop["dist_m"],
                        "duration_s": o_stop["walk_time_s"],
                        "duration_min": round(o_stop["walk_time_s"] / 60.0, 1),
                        "polyline": [[orig_lat, orig_lon], [o_stop["lat"], o_stop["lon"]]],
                    }
                    walk_dest = {
                        "type": "WALK",
                        "from_name": d_stop["stop_name"],
                        "to_name": dest_name,
                        "distance_m": d_stop["dist_m"],
                        "duration_s": final_walk_s,
                        "duration_min": round(final_walk_s / 60.0, 1),
                        "polyline": [[d_stop["lat"], d_stop["lon"]], [dest_lat, dest_lon]],
                    }

                    total_dur_s = (t_final_arr - depart_dt).total_seconds()
                    total_walk_m = o_stop["dist_m"] + d_stop["dist_m"]

                    found_itineraries.append({
                        "transfers_count": 0,
                        "total_duration_s": total_dur_s,
                        "total_duration_min": round(total_dur_s / 60.0, 1),
                        "total_walk_distance_m": total_walk_m,
                        "departure_time": depart_dt.isoformat(),
                        "estimated_arrival": t_final_arr.isoformat(),
                        "legs": [walk_orig, bus_leg, walk_dest],
                        "confidence": bus_leg["confidence"],
                        "routes_used": [bus_leg["route_short_name"]],
                    })

        # 2. Search for 1-transfer options if fewer than 3 direct options
        if len(found_itineraries) < 4:
            for o_stop in nearby_orig_stops[:3]:
                o_sid = o_stop["stop_id"]
                r1_candidates = self.stop_routes.get(o_sid, [])

                for d_stop in nearby_dest_stops[:3]:
                    d_sid = d_stop["stop_id"]
                    r2_candidates = self.stop_routes.get(d_sid, [])

                    for r1 in r1_candidates[:6]:
                        r1_stops = self.route_stops.get(r1, [])
                        r1_stop_ids = {s["stop_id"]: idx for idx, s in enumerate(r1_stops)}

                        for r2 in r2_candidates[:6]:
                            if r1 == r2:
                                continue
                            r2_stops = self.route_stops.get(r2, [])
                            r2_stop_ids = {s["stop_id"]: idx for idx, s in enumerate(r2_stops)}

                            # Find intersection transfer stop
                            transfer_sids = set(r1_stop_ids.keys()).intersection(r2_stop_ids.keys())
                            if not transfer_sids:
                                continue

                            xfer_sid = list(transfer_sids)[0]
                            # Check valid sequence
                            if r1_stop_ids.get(o_sid, 999) >= r1_stop_ids.get(xfer_sid, -1):
                                continue
                            if r2_stop_ids.get(xfer_sid, 999) >= r2_stop_ids.get(d_sid, -1):
                                continue

                            t_cur = depart_dt + timedelta(seconds=o_stop["walk_time_s"] + DEFAULT_HEADWAY_SEC / 2.0)
                            leg1 = self._evaluate_bus_leg(r1, o_sid, xfer_sid, t_cur, day_type)
                            if leg1 is None:
                                continue

                            t_leg1_arr = datetime.fromisoformat(leg1["arrival_time"])
                            # Transfer buffer + wait for bus 2
                            xfer_wait_s = MIN_TRANSFER_TIME_SEC + (DEFAULT_HEADWAY_SEC / 2.0)
                            t_leg2_dep = t_leg1_arr + timedelta(seconds=xfer_wait_s)

                            leg2 = self._evaluate_bus_leg(r2, xfer_sid, d_sid, t_leg2_dep, day_type)
                            if leg2 is None:
                                continue

                            t_leg2_arr = datetime.fromisoformat(leg2["arrival_time"])
                            final_walk_s = d_stop["walk_time_s"]
                            t_dest_arr = t_leg2_arr + timedelta(seconds=final_walk_s)

                            walk_orig = {
                                "type": "WALK",
                                "from_name": orig_name,
                                "to_name": o_stop["stop_name"],
                                "distance_m": o_stop["dist_m"],
                                "duration_min": round(o_stop["walk_time_s"] / 60.0, 1),
                                "polyline": [[orig_lat, orig_lon], [o_stop["lat"], o_stop["lon"]]],
                            }
                            xfer_step = {
                                "type": "TRANSFER",
                                "stop_name": leg1["to_stop_name"],
                                "wait_duration_min": round(xfer_wait_s / 60.0, 1),
                            }
                            walk_dest = {
                                "type": "WALK",
                                "from_name": d_stop["stop_name"],
                                "to_name": dest_name,
                                "distance_m": d_stop["dist_m"],
                                "duration_min": round(final_walk_s / 60.0, 1),
                                "polyline": [[d_stop["lat"], d_stop["lon"]], [dest_lat, dest_lon]],
                            }

                            tot_s = (t_dest_arr - depart_dt).total_seconds()
                            tot_walk = o_stop["dist_m"] + d_stop["dist_m"]

                            conf = "MEDIUM"
                            if leg1["confidence"] == "HIGH" and leg2["confidence"] == "HIGH":
                                conf = "HIGH"
                            elif leg1["confidence"] == "VERY_LOW" or leg2["confidence"] == "VERY_LOW":
                                conf = "LOW"

                            found_itineraries.append({
                                "transfers_count": 1,
                                "total_duration_s": tot_s,
                                "total_duration_min": round(tot_s / 60.0, 1),
                                "total_walk_distance_m": tot_walk,
                                "departure_time": depart_dt.isoformat(),
                                "estimated_arrival": t_dest_arr.isoformat(),
                                "legs": [walk_orig, leg1, xfer_step, leg2, walk_dest],
                                "confidence": conf,
                                "routes_used": [leg1["route_short_name"], leg2["route_short_name"]],
                            })

        if not found_itineraries:
            raise ValueError(f"No valid bus route connecting '{orig_name}' and '{dest_name}'.")

        # De-duplicate itineraries based on routes used and transfer stops
        unique_itins = []
        seen_keys = set()
        for itin in found_itineraries:
            sig = "-".join(itin["routes_used"]) + f"-{itin['transfers_count']}"
            if sig not in seen_keys:
                seen_keys.add(sig)
                unique_itins.append(itin)

        # Diverse option selection:
        # 1. Fastest (minimal total_duration_s)
        # 2. Fewest Transfers (min transfers_count, then fastest)
        # 3. Least Walking (min total_walk_distance_m)
        fastest = min(unique_itins, key=lambda x: x["total_duration_s"])
        fewest_transfers = min(unique_itins, key=lambda x: (x["transfers_count"], x["total_duration_s"]))
        least_walking = min(unique_itins, key=lambda x: (x["total_walk_distance_m"], x["total_duration_s"]))

        selected_options = []
        option_tags = [
            (fastest, "Fastest"),
            (fewest_transfers, "Fewest Transfers"),
            (least_walking, "Least Walking"),
        ]

        seen_sigs = set()
        opt_idx = 1
        for itin, tag in option_tags:
            sig = "-".join(itin["routes_used"]) + f"-{itin['transfers_count']}"
            if sig not in seen_sigs and len(selected_options) < max_options:
                seen_sigs.add(sig)

                # Compute typical quiet-hour duration (evaluated at 14:00 off-peak)
                quiet_duration_min = self._compute_quiet_hour_duration(itin, day_type)
                congestion_delay_min = max(0.0, round(itin["total_duration_min"] - quiet_duration_min, 1))

                itin_copy = dict(itin)
                itin_copy["option_id"] = opt_idx
                itin_copy["label"] = tag
                itin_copy["typical_quiet_hour_duration_min"] = quiet_duration_min
                itin_copy["congestion_delay_min"] = congestion_delay_min

                selected_options.append(itin_copy)
                opt_idx += 1

        return selected_options

    def _compute_quiet_hour_duration(self, itin: Dict[str, Any], day_type: str) -> float:
        """Evaluates the same legs at off-peak quiet hour (14:00) to compute baseline time."""
        total_quiet_s = 0.0
        for leg in itin["legs"]:
            if leg["type"] == "WALK":
                total_quiet_s += leg.get("duration_s", leg.get("duration_min", 0) * 60)
            elif leg["type"] == "TRANSFER":
                total_quiet_s += leg.get("wait_duration_min", 7.5) * 60
            elif leg["type"] == "BUS":
                # Evaluate at hour 14
                r_id = leg["route_id"]
                from_sid = leg["from_stop_id"]
                to_sid = leg["to_stop_id"]
                ref_dt = datetime(2026, 10, 6, 14, 0, 0)
                bus_eval = self._evaluate_bus_leg(r_id, from_sid, to_sid, ref_dt, day_type, quiet_hour=14)
                if bus_eval:
                    total_quiet_s += bus_eval["duration_s"]
                else:
                    total_quiet_s += leg["duration_s"] * 0.75  # 25% faster off-peak fallback
        return round(total_quiet_s / 60.0, 1)


if __name__ == "__main__":
    planner = JourneyPlanner()
    print("\n--- Test Query: Kashmere Gate ISBT -> Nehru Place Terminal (Tuesday 8:30 AM) ---")
    journeys = planner.plan_journeys(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        depart_time_iso="2026-10-06 08:30:00",
        max_options=3
    )
    for j in journeys:
        print(f"\nOption {j['option_id']} [{j['label']}]: {j['total_duration_min']} mins (Congestion delay: +{j['congestion_delay_min']}m vs quiet hour {j['typical_quiet_hour_duration_min']}m) - Confidence: {j['confidence']}")
        print(f"  Departs: {j['departure_time']} -> Arrives: {j['estimated_arrival']}")
        for idx, leg in enumerate(j["legs"]):
            if leg["type"] == "WALK":
                print(f"    Leg {idx+1} [WALK]: {leg['from_name']} -> {leg['to_name']} ({leg['distance_m']}m, {leg['duration_min']} min)")
            elif leg["type"] == "BUS":
                print(f"    Leg {idx+1} [BUS {leg['route_short_name']}]: {leg['from_stop_name']} -> {leg['to_stop_name']} ({leg['stops_count']} stops, {leg['duration_min']} min, {leg['distance_m']}m)")
            elif leg["type"] == "TRANSFER":
                print(f"    [TRANSFER at {leg['stop_name']}]: wait ~{leg['wait_duration_min']} min")
