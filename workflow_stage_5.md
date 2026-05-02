# Stage 5 Work Breakdown & Architecture Wiring

To successfully wire the newly updated frontend and backend, and formally complete **Stage 5** of the EnviroSense AI project, the tasks are distributed as follows among the team. 
This ensures zero overlap and perfectly complements the existing systems.

## 1. Ashu - Edge Layer (IoT & Hardware)
**Focus:** Secure continuous ESP32 data stream and validate physical data ingestion.
- **Task:** Verify that the ESP32 is correctly publishing `pm1_0_pcs`, `pm2_5_pcs`, and `pm10_pcs` over MQTT to the broker.
- **Wiring check:** Make sure the JSON payload perfectly matches the keys expected by `subscriber.py` (`normalize_payload` function).

## 2. Pragnya - Persistence & API Layer (Backend)
**Focus:** Ensure the database schema matches the expanded FastAPI metrics pipeline.
- **Task:** We updated the FastAPI backend (`backend/api/main.py`) to query true `pm1_0` and `pm10_0` metrics from the `clean_data` database instead of mocking them. You need to verify that TimescaleDB/PostgreSQL is properly creating the `clean_data` views derived from `sensor_data`.
- **Wiring check:** Validate the query `SELECT time, device_id, pm1_0, pm2_5, pm10_0, temperature, humidity FROM clean_data` runs successfully without SQL errors.

## 3. Pratishtha - Inference Layer (Data Science / AI)
**Focus:** ML models and anomaly integration.
- **Task:** The frontend UI now expects real regime matrices, drift series, anomaly scores, and model RMSE/MAE metrics. Since the backend now relies on `model_features`, `regime_profiles`, and `pragnya_drift_metrics` tables, you must ensure the Python jobs running your Autoencoders and LSTM models are correctly populating these DB tables.
- **Wiring check:** Monitor the pipeline for 24 hours to ensure that the data written by your ML jobs maps directly to the UI forecast widgets.

## 4. Anushka - Presentation Layer (Frontend PWA)
**Focus:** UI data binding & polish.
- **Task:** The frontend React code in `frontend-pwa/src/lib/pipeline.ts` is currently simulating the API. You must toggle the app into live mode by setting `VITE_ENABLE_LIVE_API=true` and `VITE_API_BASE_URL` in your `.env.production` file.
- **Wiring check:** Coordinate with Rachna to point the frontend to the deployed VPS API. Verify that the UI renders the true `pm10` values instead of the `1.35x` mock ratio.

## 5. Rachna - Deployment & DevOps (VPS)
**Focus:** Final server configuration, NGINX routing, and Service isolation.
- **Task:** We have heavily cleaned up the python virtual environment to isolate the pipeline `requirements.txt`. You must set up `systemd` or `pm2` services to keep `subscriber.py`, `uvicorn` (FastAPI), and the built frontend PWA running 24/7 on the VPS. 
- **Wiring check:** Execute the newly created `start_services.sh` script to verify everything runs smoothly, then migrate those commands into robust standard background services. Set up NGINX to reverse proxy the UI (port 3000) and Backend (port 8000), linking it perfectly together.
