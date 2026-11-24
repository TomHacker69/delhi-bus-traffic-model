"""
pipeline/load_gtfs.py
Loads GTFS static data from ./data/gtfs/.
Builds and caches route shapes, stop projections, and spatial indexes in EPSG:32643.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pyproj import Transformer
import shapely
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GTFS_DIR = os.path.join("data", "gtfs")

# Transformer from WGS84 (lon, lat) to EPSG:32643 (UTM Zone 43N metric coordinates)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)


def project_lonlat(lon: float, lat: float) -> Tuple[float, float]:
    """Convert (lon, lat) in EPSG:4326 to (x, y) in EPSG:32643 meters."""
    return transformer.transform(lon, lat)


def project_coords_array(lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized conversion of lons and lats to EPSG:32643 meters."""
    return transformer.transform(lons, lats)


class GTFSFeed:
    """Encapsulates static GTFS tables, spatial shapes, and stop metrics."""

    def __init__(self, gtfs_dir: str = GTFS_DIR):
        self.gtfs_dir = gtfs_dir
        self.stops: pd.DataFrame = pd.DataFrame()
        self.routes: pd.DataFrame = pd.DataFrame()
        self.trips: pd.DataFrame = pd.DataFrame()
        self.stop_times: pd.DataFrame = pd.DataFrame()
        self.shapes: pd.DataFrame = pd.DataFrame()

        # Cache of metric shapes: shape_id -> shapely.geometry.LineString (EPSG:32643)
        self.metric_shapes: Dict[str, LineString] = {}
        # Mapping trip_id -> shape_id
        self.trip_to_shape: Dict[str, str] = {}
        # Mapping route_id -> list of shape_ids
        self.route_to_shapes: Dict[str, List[str]] = {}
        # Metric stops: stop_id -> (x, y)
        self.stop_metric_coords: Dict[str, Tuple[float, float]] = {}
        # Spatial index of all shapes
        self.shape_tree: Optional[STRtree] = None
        self.shape_id_list: List[str] = []

        self.load_feed()

    def load_feed(self):
        """Loads and prepares GTFS tables."""
        stops_file = os.path.join(self.gtfs_dir, "stops.txt")
        routes_file = os.path.join(self.gtfs_dir, "routes.txt")
        trips_file = os.path.join(self.gtfs_dir, "trips.txt")
        st_file = os.path.join(self.gtfs_dir, "stop_times.txt")
        shapes_file = os.path.join(self.gtfs_dir, "shapes.txt")

        if not os.path.exists(stops_file):
            raise FileNotFoundError(f"Missing GTFS stops file: {stops_file}")

        logger.info(f"Loading GTFS files from {self.gtfs_dir}...")
        self.stops = pd.read_csv(stops_file, dtype={"stop_id": str})
        self.routes = pd.read_csv(routes_file, dtype={"route_id": str})
        self.trips = pd.read_csv(trips_file, dtype={"route_id": str, "trip_id": str})
        if "shape_id" in self.trips.columns:
            self.trips["shape_id"] = self.trips["shape_id"].astype(str)

        self.stop_times = pd.read_csv(st_file, dtype={"trip_id": str, "stop_id": str})

        # Build metric coordinates for stops
        xs, ys = project_coords_array(self.stops["stop_lon"].values, self.stops["stop_lat"].values)
        self.stops["x_32643"] = xs
        self.stops["y_32643"] = ys
        self.stop_metric_coords = dict(zip(self.stops["stop_id"], zip(xs, ys)))

        # Build or load shapes
        if os.path.exists(shapes_file) and os.path.getsize(shapes_file) > 10:
            logger.info(f"Loading shapes from {shapes_file}...")
            self.shapes = pd.read_csv(shapes_file, dtype={"shape_id": str})
            self._build_shapes_from_file()
        else:
            logger.info("shapes.txt missing or empty. Generating shapes from stop_times sequence...")
            self._generate_shapes_from_stop_times()

        # Map trip_to_shape
        if "shape_id" in self.trips.columns:
            self.trip_to_shape = dict(zip(self.trips["trip_id"], self.trips["shape_id"]))
        else:
            # Map by route_id
            for _, row in self.trips.iterrows():
                r_id = row["route_id"]
                t_id = row["trip_id"]
                sh_id = f"shape_route_{r_id}"
                self.trip_to_shape[t_id] = sh_id

        # Build route_to_shapes mapping
        for _, row in self.trips.iterrows():
            r_id = str(row["route_id"])
            sh_id = self.trip_to_shape.get(str(row["trip_id"]))
            if sh_id:
                self.route_to_shapes.setdefault(r_id, []).append(sh_id)
        for r_id in self.route_to_shapes:
            self.route_to_shapes[r_id] = list(set(self.route_to_shapes[r_id]))

        # Build spatial index over all shapes
        valid_geoms = []
        valid_ids = []
        for sh_id, geom in self.metric_shapes.items():
            if geom and not geom.is_empty and geom.length > 0:
                valid_geoms.append(geom)
                valid_ids.append(sh_id)

        if valid_geoms:
            self.shape_tree = STRtree(valid_geoms)
            self.shape_id_list = valid_ids
            logger.info(f"Built spatial index with {len(valid_geoms)} route shapes.")

    def _build_shapes_from_file(self):
        """Constructs shapely LineString for each shape_id in shapes.txt."""
        grouped = self.shapes.sort_values(["shape_id", "shape_pt_sequence"]).groupby("shape_id")
        for sh_id, group in grouped:
            lons = group["shape_pt_lon"].values
            lats = group["shape_pt_lat"].values
            xs, ys = project_coords_array(lons, lats)
            pts = list(zip(xs, ys))
            if len(pts) >= 2:
                self.metric_shapes[str(sh_id)] = LineString(pts)

    def _generate_shapes_from_stop_times(self):
        """
        Synthesizes shapes.txt from ordered stop locations in stop_times.
        Groups by route_id to find representative stop sequences.
        """
        merged = self.stop_times.merge(self.trips[["trip_id", "route_id"]], on="trip_id")
        merged = merged.merge(self.stops[["stop_id", "stop_lat", "stop_lon", "x_32643", "y_32643"]], on="stop_id")
        merged = merged.sort_values(["route_id", "trip_id", "stop_sequence"])

        shapes_records = []
        route_groups = merged.groupby("route_id")

        for r_id, r_df in route_groups:
            # Pick the trip with the most stops for this route
            trip_counts = r_df.groupby("trip_id")["stop_sequence"].count()
            best_trip = trip_counts.idxmax()
            best_trip_stops = r_df[r_df["trip_id"] == best_trip].sort_values("stop_sequence")

            pts = []
            cum_dist = 0.0
            last_pt = None
            sh_id = f"shape_route_{r_id}"

            seq = 0
            for _, srow in best_trip_stops.iterrows():
                x = srow["x_32643"]
                y = srow["y_32643"]
                curr_pt = Point(x, y)
                if last_pt is not None:
                    cum_dist += last_pt.distance(curr_pt)
                last_pt = curr_pt
                pts.append((x, y))

                shapes_records.append({
                    "shape_id": sh_id,
                    "shape_pt_lat": srow["stop_lat"],
                    "shape_pt_lon": srow["stop_lon"],
                    "shape_pt_sequence": seq,
                    "shape_dist_traveled": round(cum_dist, 2),
                })
                seq += 1

            if len(pts) >= 2:
                self.metric_shapes[sh_id] = LineString(pts)

        self.shapes = pd.DataFrame(shapes_records)
        # Save generated shapes.txt to GTFS_DIR
        out_path = os.path.join(self.gtfs_dir, "shapes.txt")
        self.shapes.to_csv(out_path, index=False)
        logger.info(f"Saved generated shapes.txt to {out_path} ({len(self.metric_shapes)} shapes)")

    def get_shape_for_trip(self, trip_id: str, route_id: Optional[str] = None) -> Optional[LineString]:
        """Returns the metric LineString for a trip_id or fallback route_id."""
        sh_id = self.trip_to_shape.get(trip_id)
        if sh_id and sh_id in self.metric_shapes:
            return self.metric_shapes[sh_id]

        if route_id is not None:
            route_shapes = self.route_to_shapes.get(str(route_id), [])
            if route_shapes and route_shapes[0] in self.metric_shapes:
                return self.metric_shapes[route_shapes[0]]
            direct_id = f"shape_route_{route_id}"
            if direct_id in self.metric_shapes:
                return self.metric_shapes[direct_id]

        return None

    def find_nearest_shape(self, x: float, y: float, candidate_route_ids: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[LineString], float]:
        """
        Finds the nearest route shape to a metric point (x, y).
        If candidate_route_ids is provided, prioritizes them.
        Returns: (shape_id, LineString, distance_in_meters)
        """
        pt = Point(x, y)
        if candidate_route_ids:
            best_dist = float("inf")
            best_sh_id = None
            best_geom = None
            for r_id in candidate_route_ids:
                sh_ids = self.route_to_shapes.get(str(r_id), [f"shape_route_{r_id}"])
                for sh_id in sh_ids:
                    geom = self.metric_shapes.get(sh_id)
                    if geom:
                        d = geom.distance(pt)
                        if d < best_dist:
                            best_dist = d
                            best_sh_id = sh_id
                            best_geom = geom
            if best_sh_id is not None:
                return best_sh_id, best_geom, best_dist

        # Global search via STRtree
        if self.shape_tree is not None:
            nearest_idx = self.shape_tree.nearest(pt)
            best_geom = self.shape_tree.geometries.take(nearest_idx)
            best_sh_id = self.shape_id_list[nearest_idx]
            dist = best_geom.distance(pt)
            return best_sh_id, best_geom, dist

        return None, None, float("inf")


if __name__ == "__main__":
    feed = GTFSFeed()
    print("GTFS Feed summary:")
    print(f"  Stops: {len(feed.stops):,}")
    print(f"  Routes: {len(feed.routes):,}")
    print(f"  Trips: {len(feed.trips):,}")
    print(f"  Shapes loaded: {len(feed.metric_shapes):,}")
