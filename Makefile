.PHONY: install download clean segments train api test validate all

install:
	pip install -r requirements.txt

download:
	python pipeline/download_extract.py --days 1

clean:
	python pipeline/clean_gps.py --days 1

segments:
	python pipeline/compute_segment_times.py --date 2026-10-01

train:
	python model/travel_time_model.py

test:
	python -m pytest tests/

validate:
	python validation/validate_model.py

api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

all: clean segments train validate test
