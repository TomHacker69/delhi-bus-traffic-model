# Delhi Transport Stack: Real-Time Bus Traffic Model & Journey Planner
**Delhi Transport Stack (Problem 1, Phase 1)**

A production-grade, time-dependent transit traffic model and bus journey planner for Delhi. Learns actual vehicle speeds and congestion patterns from October 2026 DTC historical GPS telematics (~31 million daily pings across 2,400+ routes) and live GTFS-RT feeds, enabling accurate time-of-day-aware trip planning.

---

## Architecture Overview

```
                                      +-------------------------------+
                                      |   Historical GPS (7z / DBs)   |
                                      +---------------+---------------+
                                                      |
                                                      v
+------------------+                    +-------------+---------------+
|   Static GTFS    |                    |  Phase 1: GPS Cleaning      |
|  (data/gtfs/)    |                    |  - Bounding & Implied Speed |
+--------+---------+                    |  - UTM EPSG:32643 Projection|
         |                              +-------------+---------------+
         |                                            |
         +-------------------+                        v
         |                   |          +-------------+---------------+
         v                   +--------> |  Phase 2: Segment Extraction|
+--------+---------+         |          |  - Monotonic Progress       |
|  Spatial R-Tree  |         |          |  - Dwell vs Running Time    |
|  Shapes & Stops  |         |          +-------------+---------------+
+--------+---------+         |                        |
         |                   |                        v
         |                   |          +-------------+---------------+
         |                   +--------> |  Phase 3: Traffic Model     |
         |                              |  - 4-Level Fallback (L0-L4) |
         |                              |  - Live GTFS-RT Ratio Decay |
         |                              +-------------+---------------+
         |                                            |
         v                                            v
+--------+--------------------------------------------+---------------+
|  Phase 4: Time-Dependent Journey Planner (Dijkstra / RAPTOR)        |
|  - Walking edges (<=500m), Transfers, Multi-Criteria Diversification|
+---------------------------------------------+-----------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
       +-------------+---------------+                   +-------------+---------------+
       |   Phase 5: FastAPI Backend  |                   |   Phase 6: Leaflet Client   |
       |   GET /journeys, /stops     | <===============> |   Interactive Map, Polylines|
       +-----------------------------+                   +-----------------------------+
```

---

## Directory Structure

```
├── data/
│   ├── gtfs/                     # Static GTFS feeds (stops, routes, trips, stop_times, shapes)
│   ├── raw/                      # Downloaded daily SQLite DBs (bus_movements_YYYY_MM_DD.db)
│   ├── parquet/
│   │   ├── clean_gps/            # Partitioned Parquet: cleaned & projected GPS pings
│   │   └── segment_observations/ # Partitioned Parquet: observed segment running & dwell times
│   └── model/                    # Serialized travel-time model & dwell lookups
├── pipeline/
│   ├── download_extract.py       # Stream-extracts SQLite DBs from 7z archive via range requests
│   ├── load_gtfs.py              # Ingests GTFS, metric UTM projection (EPSG:32643), spatial indexing
│   ├── clean_gps.py              # Vectorized GPS cleaning (DuckDB + Shapely), drop diagnostics
│   └── compute_segment_times.py  # Map-matching, monotonic shape progress, dwell separation
├── model/
│   └── travel_time_model.py      # Statistical aggregation (median, p25, p75), fallback hierarchy, live blend
├── planner/
│   └── journey_planner.py        # Multi-criteria routing, walk transfers, quiet-hour baseline delay
├── api/
│   └── main.py                   # FastAPI application with OpenAPI docs and Leaflet client serving
├── client/
│   └── index.html                # Single-page Leaflet web map client
├── validation/
│   ├── validate_model.py         # Out-of-sample benchmark (Oct 27-31 holdout, 60 test trips)
│   ├── validation_report.md      # Graded validation report with MAE, MAPE, median error tables
│   ├── validation_plots.png      # Scatter plot & hourly error bar chart
│   └── google_maps_comparison_template.csv # Google Directions API comparison template
├── tests/
│   ├── test_core_components.py   # Unit tests for shape projection, segment extraction, fallback, journey
│   └── test_api_endpoints.py     # FastAPI test client integration tests
├── requirements.txt              # Production dependency specifications
├── Makefile                      # Standard automation commands
├── run.bat                       # Windows CLI automation script
└── technical_note.md             # Detailed engineering methodology documentation
```

---

## Quickstart & Setup

### 1. Installation

Python 3.11+ is supported. Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution

Run the complete pipeline end-to-end:

```bash
# Using Makefile
make all

# Or on Windows using run.bat
run.bat all
```

Or execute individual pipeline phases step-by-step:

```bash
# 1. Download & extract historical GPS for day 1 (Oct 1, 2026)
python pipeline/download_extract.py --days 1

# 2. Clean GPS pings and save partitioned Parquet
python pipeline/clean_gps.py --days 1

# 3. Extract segment running and dwell times
python pipeline/compute_segment_times.py --date 2026-10-01

# 4. Train traffic and dwell models
python model/travel_time_model.py

# 5. Run validation benchmark on held-out days
python validation/validate_model.py

# 6. Execute all unit tests
python -m pytest tests/
```

---

## Running the API and Client

Start the FastAPI application with Uvicorn:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Web Map Client**: Open [http://localhost:8000/](http://localhost:8000/) in your browser.
- **Interactive OpenAPI Documentation**: Open [http://localhost:8000/docs](http://localhost:8000/docs).
- **Health Check**: `GET http://localhost:8000/health`

---

## Deployment & Cloud Execution

### 1. GitHub Pages (Live Web Client)
- **Live URL**: [https://tomhacker69.github.io/delhi-bus-traffic-model/](https://tomhacker69.github.io/delhi-bus-traffic-model/)
- Automatically deployed on push to `main` via `.github/workflows/pages.yml`.

### 2. Run with Docker / GitHub Container Registry (GHCR)
The application is containerized and published to GHCR on every push:

```bash
# Pull and run the pre-built container
docker run -d -p 8000:8000 --name delhi-transit ghcr.io/tomhacker69/delhi-bus-traffic-model:latest

# Or build locally with docker-compose
docker compose up -d
```

### 3. One-Click Cloud Deployment (Render / Railway)
- **Render**: Connect the repository at `https://github.com/TomHacker69/delhi-bus-traffic-model`. The included `render.yaml` automatically sets up the Python service with dynamic `$PORT` binding.
- **Railway**: Connect repository; uses `railway.toml` and `Dockerfile` automatically.
- **Heroku / Fly.io**: Uses `Procfile` and `Dockerfile`.

---

## Live vs Historical Modes

The system automatically switches operational modes based on environment variables:
- If `GTFS_RT_URL` and `GTFS_RT_KEY` are defined, the system runs in `live+historical` mode, polling real-time vehicle positions and blending live speed ratios ($R = v_{live} / v_{hist}$) with linear decay over 60 minutes.
- If missing, the system runs in `historical-only` mode and explicitly marks this in API responses.

```bash
# Example live mode invocation
set GTFS_RT_URL=https://api.delhi.transportstack.in/gtfs-rt/vehicle-positions
set GTFS_RT_KEY=your_api_key_here
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## Example API Request & Response

### Request: Kashmere Gate ISBT to Nehru Place Terminal (Tuesday 8:30 AM)

```http
GET /journeys?from=ISBT%20Kashmere%20Gate&to=Nehru%20Place%20Terminal&depart=2026-10-06T08:30:00&max_options=3 HTTP/1.1
Host: localhost:8000
Accept: application/json
```

### Response JSON

```json
{
  "origin": {
    "name": "ISBT Kashmere Gate",
    "lat": 28.666867,
    "lon": 77.229367
  },
  "destination": {
    "name": "Nehru Place Terminal",
    "lat": 28.548512,
    "lon": 77.257157
  },
  "departure_time": "2026-10-06T08:30:00",
  "data_mode": "historical-only",
  "options_count": 1,
  "options": [
    {
      "option_id": 1,
      "label": "Fastest",
      "total_duration_min": 82.3,
      "typical_quiet_hour_duration_min": 55.2,
      "congestion_delay_min": 27.1,
      "departure_time": "2026-10-06T08:30:00",
      "estimated_arrival": "2026-10-06T09:52:20",
      "transfers_count": 0,
      "total_walk_distance_m": 0.0,
      "confidence": "HIGH",
      "routes_used": [
        "429"
      ],
      "legs": [
        {
          "type": "WALK",
          "from_name": "ISBT Kashmere Gate",
          "to_name": "ISBT Kashmere Gate",
          "distance_m": 0.0,
          "duration_s": 0.0,
          "duration_min": 0.0
        },
        {
          "type": "BUS",
          "route_id": "DTC_429",
          "route_short_name": "429",
          "route_long_name": "ISBT Kashmere Gate - Nehru Place Terminal",
          "from_stop_id": "121",
          "from_stop_name": "ISBT Kashmere Gate",
          "to_stop_id": "359",
          "to_stop_name": "Nehru Place Terminal",
          "departure_time": "2026-10-06T08:37:30",
          "arrival_time": "2026-10-06T09:52:20",
          "duration_s": 4490.3,
          "duration_min": 74.8,
          "running_time_s": 4310.3,
          "dwell_time_s": 180.0,
          "distance_m": 14361.4,
          "stops_count": 9,
          "confidence": "HIGH"
        },
        {
          "type": "WALK",
          "from_name": "Nehru Place Terminal",
          "to_name": "Nehru Place Terminal",
          "distance_m": 0.0,
          "duration_s": 0.0,
          "duration_min": 0.0
        }
      ]
    }
  ]
}
```

---

## Validation & Evaluation

Validation results on held-out days (Oct 27–31, 2026) are stored in `validation/`:
- **Report**: [validation/validation_report.md](validation/validation_report.md)
- **Visualizations**: [validation/validation_plots.png](validation/validation_plots.png)
- **Google Maps Benchmark Template**: [validation/google_maps_comparison_template.csv](validation/google_maps_comparison_template.csv)
- **Technical Note**: [docs/technical_note.md](docs/technical_note.md)
