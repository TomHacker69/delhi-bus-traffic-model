"""
api/main.py
Phase 5: FastAPI Backend for Delhi Bus Journey Planner and Traffic Model.
Provides /journeys, /stops, /health, /model/segment/{segment_id}, OpenAPI docs at /docs.
"""

import os
import sys
import asyncio
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.load_gtfs import GTFSFeed
from pipeline.live_feed import GTFSRTLivePoller
from model.travel_time_model import TravelTimeModel
from planner.journey_planner import JourneyPlanner

# Check GTFS-RT live feed configuration
GTFS_RT_URL = os.environ.get("GTFS_RT_URL")
GTFS_RT_KEY = os.environ.get("GTFS_RT_KEY")
DATA_MODE = "live+historical" if (GTFS_RT_URL and GTFS_RT_KEY) else "historical-only"

# Lazy-loaded singletons
_feed: Optional[GTFSFeed] = None
_model: Optional[TravelTimeModel] = None
_planner: Optional[JourneyPlanner] = None
_poller: Optional[GTFSRTLivePoller] = None


def get_feed() -> GTFSFeed:
    global _feed
    if _feed is None:
        _feed = GTFSFeed()
    return _feed


def get_model() -> TravelTimeModel:
    global _model
    if _model is None:
        _model = TravelTimeModel()
    return _model


def get_planner() -> JourneyPlanner:
    global _planner
    if _planner is None:
        _planner = JourneyPlanner(feed=get_feed(), model=get_model())
    return _planner


def get_poller() -> GTFSRTLivePoller:
    global _poller
    if _poller is None:
        _poller = GTFSRTLivePoller()
    return _poller


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eager initialization on app startup
    get_feed()
    m = get_model()
    get_planner()
    poller = get_poller()
    task = None
    if poller.is_active:
        task = asyncio.create_task(poller.run_loop(m))
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(
    title="Delhi Bus Journey Planner & Traffic Model API",
    description="Time-dependent bus traffic model and journey planner for Delhi (Delhi Transport Stack).",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for Leaflet client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Response Models ---

class LocationInfo(BaseModel):
    name: str = Field(..., description="Resolved location or stop name")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")


class JourneyOption(BaseModel):
    option_id: int
    label: str = Field(..., description="e.g. 'Fastest', 'Fewest Transfers', 'Least Walking'")
    total_duration_min: float
    typical_quiet_hour_duration_min: float
    congestion_delay_min: float
    departure_time: str
    estimated_arrival: str
    transfers_count: int
    total_walk_distance_m: float
    confidence: str
    routes_used: List[str]
    legs: List[Dict[str, Any]]


class JourneysResponse(BaseModel):
    origin: LocationInfo
    destination: LocationInfo
    departure_time: str
    data_mode: str = Field(..., description="'live+historical' or 'historical-only'")
    options_count: int
    options: List[JourneyOption]


class StopAutocompleteItem(BaseModel):
    stop_id: str
    stop_name: str
    lat: float
    lon: float


class HealthResponse(BaseModel):
    status: str
    data_mode: str
    gtfs_stops_count: int
    gtfs_routes_count: int
    gtfs_shapes_count: int
    model_profiles_count: int
    live_poller_status: str
    timestamp: str


class SegmentModelResponse(BaseModel):
    segment_id: str
    day_type: str
    hour_bucket: int
    running_time_s: float
    dwell_time_s: float
    total_time_s: float
    p25_running_time_s: Optional[float] = None
    p75_running_time_s: Optional[float] = None
    confidence: str
    fallback_level: str
    live_applied: bool
    speed_ratio: float
    segment_length_m: Optional[float] = None


# --- Endpoints ---

@app.get("/", include_in_schema=False)
def serve_client():
    """Serves the single-page Leaflet web client."""
    client_path = os.path.join(os.path.dirname(__file__), "..", "client", "index.html")
    if os.path.exists(client_path):
        return FileResponse(client_path)
    return {"message": "Delhi Bus Journey Planner API. Visit /docs for OpenAPI documentation."}

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health and status endpoint. Reports system readiness and active data mode."""
    f = get_feed()
    m = get_model()
    poller = get_poller()
    return HealthResponse(
        status="ok",
        data_mode=DATA_MODE,
        gtfs_stops_count=len(f.stops),
        gtfs_routes_count=len(f.routes),
        gtfs_shapes_count=len(f.metric_shapes),
        model_profiles_count=len(m.seg_lookup),
        live_poller_status=poller.status,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/stops", response_model=List[StopAutocompleteItem], tags=["Transit Network"])
def autocomplete_stops(
    q: str = Query(..., min_length=1, description="Search text for stop name"),
    limit: int = Query(10, ge=1, le=50, description="Max suggestions to return"),
):
    """Fuzzy autocomplete stop search."""
    f = get_feed()
    q_lower = q.strip().lower()
    stops_df = f.stops

    sub_matches = stops_df[stops_df["stop_name"].str.lower().str.contains(q_lower, na=False)].head(limit)
    results = []
    for _, r in sub_matches.iterrows():
        results.append(StopAutocompleteItem(
            stop_id=str(r["stop_id"]),
            stop_name=str(r["stop_name"]),
            lat=float(r["stop_lat"]),
            lon=float(r["stop_lon"]),
        ))
    return results


@app.get("/journeys", response_model=JourneysResponse, tags=["Journey Planner"])
def get_journeys(
    from_loc: str = Query(..., alias="from", description="Origin: coordinates 'lat,lon' or stop name"),
    to_loc: str = Query(..., alias="to", description="Destination: coordinates 'lat,lon' or stop name"),
    depart: str = Query(..., description="Departure ISO 8601 IST datetime, e.g. '2026-10-06T08:30:00'"),
    max_options: int = Query(3, ge=1, le=5, description="Number of options (fastest, fewest transfers, etc.)"),
):
    """
    Computes 2-3 time-dependent journey options between origin and destination.
    Uses realistic traffic-aware segment speeds and dwell models.
    """
    p = get_planner()

    try:
        orig_lat, orig_lon, orig_name = p.resolve_location(from_loc)
        dest_lat, dest_lon, dest_name = p.resolve_location(to_loc)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        options = p.plan_journeys(
            origin=from_loc,
            destination=to_loc,
            depart_time_iso=depart,
            max_options=max_options,
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No route found: {str(e)}")

    journey_options = []
    for opt in options:
        journey_options.append(JourneyOption(
            option_id=opt["option_id"],
            label=opt["label"],
            total_duration_min=opt["total_duration_min"],
            typical_quiet_hour_duration_min=opt["typical_quiet_hour_duration_min"],
            congestion_delay_min=opt["congestion_delay_min"],
            departure_time=opt["departure_time"],
            estimated_arrival=opt["estimated_arrival"],
            transfers_count=opt["transfers_count"],
            total_walk_distance_m=opt["total_walk_distance_m"],
            confidence=opt["confidence"],
            routes_used=opt["routes_used"],
            legs=opt["legs"],
        ))

    return JourneysResponse(
        origin=LocationInfo(name=orig_name, lat=orig_lat, lon=orig_lon),
        destination=LocationInfo(name=dest_name, lat=dest_lat, lon=dest_lon),
        departure_time=depart,
        data_mode=DATA_MODE,
        options_count=len(journey_options),
        options=journey_options,
    )


@app.get("/model/segment/{segment_id}", response_model=SegmentModelResponse, tags=["Traffic Model"])
def get_segment_model(
    segment_id: str = Path(..., description="Segment ID, e.g. 'DTC_429_121_319'"),
    day_type: str = Query("weekday", description="'weekday', 'saturday', or 'sunday'"),
    hour: int = Query(8, ge=0, le=23, description="Hour bucket (0-23)"),
    length_m: Optional[float] = Query(None, description="Optional segment length in meters"),
):
    """Debugging endpoint to inspect model parameters and confidence level on any segment."""
    m = get_model()
    pred = m.predict_segment_travel_time(
        segment_id=segment_id,
        day_type=day_type,
        hour=hour,
        segment_length_m=length_m,
    )
    return SegmentModelResponse(**pred)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
