"""
tests/test_api_endpoints.py
Unit tests for FastAPI endpoints: /health, /stops, /model/segment, and /journeys.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "data_mode" in data
    assert data["gtfs_stops_count"] > 0
    assert data["gtfs_routes_count"] > 0
    print("\n[PASS] /health returned:", data)


def test_stops_autocomplete():
    response = client.get("/stops?q=Kashmere&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) > 0
    assert any("kashmere" in s["stop_name"].lower() for s in data)
    print("\n[PASS] /stops returned:", len(data), "stops. First:", data[0])


def test_model_segment():
    response = client.get("/model/segment/DTC_429_121_319?day_type=weekday&hour=8&length_m=1200")
    assert response.status_code == 200
    data = response.json()
    assert data["segment_id"] == "DTC_429_121_319"
    assert data["hour_bucket"] == 8
    assert data["running_time_s"] > 0
    assert "confidence" in data
    print("\n[PASS] /model/segment returned:", data)


def test_journeys_kashmere_to_nehru_place():
    response = client.get(
        "/journeys?from=ISBT Kashmere Gate&to=Nehru Place Terminal&depart=2026-10-06T08:30:00&max_options=3"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["origin"]["name"] != ""
    assert data["destination"]["name"] != ""
    assert data["data_mode"] in ["historical-only", "live+historical"]
    assert len(data["options"]) >= 1

    opt1 = data["options"][0]
    assert opt1["total_duration_min"] > 0
    assert opt1["typical_quiet_hour_duration_min"] > 0
    assert "legs" in opt1
    print("\n[PASS] /journeys returned", len(data["options"]), "options.")
    print("  Option 1:", opt1["label"], f"{opt1['total_duration_min']} mins (Congestion delay: +{opt1['congestion_delay_min']}m)")


if __name__ == "__main__":
    test_health()
    test_stops_autocomplete()
    test_model_segment()
    test_journeys_kashmere_to_nehru_place()
    print("\nALL API TESTS PASSED SUCCESSFULLY!")
