"""
tests/test_core_components.py
Unit tests for key pipeline and model components:
1. Shape projection in EPSG:32643
2. Monotonic progress and segment-time extraction
3. 4-level fallback hierarchy in TravelTimeModel
4. End-to-end journey planning query
"""

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.load_gtfs import GTFSFeed, project_coords_array, project_lonlat
from pipeline.compute_segment_times import enforce_monotonic_progress
from model.travel_time_model import TravelTimeModel
from planner.journey_planner import JourneyPlanner


def test_shape_projection():
    """Verifies projection of coordinates to EPSG:32643 metric space and distance calculations."""
    # Delhi coordinates: Kashmere Gate (77.229367, 28.666867)
    x, y = project_lonlat(77.229367, 28.666867)
    assert 700000.0 < x < 750000.0, f"Expected UTM Easting in Delhi range, got {x}"
    assert 3150000.0 < y < 3200000.0, f"Expected UTM Northing in Delhi range, got {y}"

    # Test vectorized projection
    lons = np.array([77.229367, 77.257157])
    lats = np.array([28.666867, 28.548512])
    xs, ys = project_coords_array(lons, lats)
    assert len(xs) == 2 and len(ys) == 2

    # Metric distance between Kashmere Gate and Nehru Place (~13.4 km straight-line)
    pt1 = Point(xs[0], ys[0])
    pt2 = Point(xs[1], ys[1])
    dist_km = pt1.distance(pt2) / 1000.0
    assert 12.0 < dist_km < 15.0, f"Expected ~13.4 km straight-line, got {dist_km:.2f} km"
    print(f"\n[PASS] test_shape_projection: Kashmere Gate to Nehru Place straight-line = {dist_km:.2f} km")


def test_segment_time_extraction():
    """Verifies monotonic progress filtering and backward jitter handling."""
    # Simulated noisy GPS sequence with jitter and backward jumps
    raw_dists = np.array([0.0, 150.0, 300.0, 290.0, 450.0, 600.0, 500.0, 750.0])
    clean_dists, valid_mask = enforce_monotonic_progress(raw_dists)

    # Small backward jitter (290m vs max 300m: 10m backward) should be clamped to 300m
    assert clean_dists[3] == 300.0, f"Expected clamped jitter at 300.0, got {clean_dists[3]}"
    assert valid_mask[3] == True

    # Large backward jump (500m vs max 600m: 100m backward) should be dropped
    assert valid_mask[6] == False, "Expected 100m backward jump to be marked invalid"

    # Monotonicity check
    valid_dists = clean_dists[valid_mask]
    assert np.all(np.diff(valid_dists) >= 0), "Cleaned distances must be monotonically non-decreasing"
    print(f"\n[PASS] test_segment_time_extraction: Monotonic progress enforced, dropped large backward outlier.")


def test_fallback_logic():
    """Verifies the 4-level fallback hierarchy and confidence scores."""
    model = TravelTimeModel()

    # 1. Level 0 / Level 2: Known observed segment
    pred_obs = model.predict_segment_travel_time(
        segment_id="789_3550_1651",
        day_type="weekday",
        hour=5,
    )
    assert pred_obs["running_time_s"] > 0
    assert pred_obs["confidence"] in ["HIGH", "MEDIUM_HIGH", "MEDIUM"]
    print(f"\n[PASS] test_fallback_logic (observed): {pred_obs['fallback_level']} -> {pred_obs['running_time_s']}s (Confidence: {pred_obs['confidence']})")

    # 2. Level 4: Completely unseen segment (must fall back to Level 4 scheduled / baseline)
    pred_unseen = model.predict_segment_travel_time(
        segment_id="UNSEEN_SEGMENT_99999",
        day_type="weekday",
        hour=8,
        segment_length_m=1000.0,
    )
    assert pred_unseen["confidence"] == "VERY_LOW"
    assert pred_unseen["fallback_level"] == "L4_SCHEDULED"
    assert pred_unseen["running_time_s"] > 0
    print(f"[PASS] test_fallback_logic (unseen): {pred_unseen['fallback_level']} -> {pred_unseen['running_time_s']}s (Confidence: {pred_unseen['confidence']})")


def test_end_to_end_journey():
    """Verifies full end-to-end journey planning query from Kashmere Gate to Nehru Place."""
    planner = JourneyPlanner()
    journeys = planner.plan_journeys(
        origin="ISBT Kashmere Gate",
        destination="Nehru Place Terminal",
        depart_time_iso="2026-10-06 08:30:00",
        max_options=3,
    )
    assert len(journeys) >= 1, "Expected at least 1 valid journey option"
    opt = journeys[0]
    assert opt["total_duration_min"] > 0
    assert opt["typical_quiet_hour_duration_min"] > 0
    assert opt["congestion_delay_min"] >= 0
    assert len(opt["legs"]) >= 2
    print(f"\n[PASS] test_end_to_end_journey: Found {len(journeys)} option(s).")
    print(f"  Fastest: {opt['total_duration_min']} mins (quiet hour: {opt['typical_quiet_hour_duration_min']} mins, delay: +{opt['congestion_delay_min']} mins)")


if __name__ == "__main__":
    test_shape_projection()
    test_segment_time_extraction()
    test_fallback_logic()
    test_end_to_end_journey()
    print("\nALL 4 CORE UNIT TESTS PASSED SUCCESSFULLY!")
