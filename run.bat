@echo off
REM Windows run script for Delhi Transport Stack Journey Planner

if "%1"=="" goto help
if "%1"=="install" goto install
if "%1"=="download" goto download
if "%1"=="clean" goto clean
if "%1"=="segments" goto segments
if "%1"=="train" goto train
if "%1"=="test" goto test
if "%1"=="validate" goto validate
if "%1"=="api" goto api
if "%1"=="all" goto all

:install
pip install -r requirements.txt
goto end

:download
python pipeline/download_extract.py --days 1
goto end

:clean
python pipeline/clean_gps.py --days 1
goto end

:segments
python pipeline/compute_segment_times.py --date 2026-10-01
goto end

:train
python model/travel_time_model.py
goto end

:test
python -m pytest tests/
goto end

:validate
python validation/validate_model.py
goto end

:api
uvicorn api.main:app --host 0.0.0.0 --port 8000
goto end

:all
call %0 clean
call %0 segments
call %0 train
call %0 validate
call %0 test
goto end

:help
echo Usage: run.bat [install|download|clean|segments|train|test|validate|api|all]

:end
