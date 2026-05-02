#!/bin/bash
# start_services.sh
# Starts the EnviroSense AI application suite.

# Frontend Env Setup
export VITE_ENABLE_LIVE_API=true
export VITE_API_BASE_URL="http://localhost:8000"

echo "Stopping any existing background jobs..."
pkill -f "uvicorn backend.api.main" || true
pkill -f "python subscriber.py" || true
pkill -f "npm run preview" || true

cd /home/pragnya/EnviroSenseAI/frontend-pwa
echo "Starting Frontend PWA on port 3000..."
nohup npm run preview -- --port 3000 --host 0.0.0.0 > frontend.log 2>&1 &

cd /home/pragnya/EnviroSenseAI
source .venv/bin/activate

echo "Starting FastAPI Backend on port 8000..."
nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

echo "Starting MQTT Subscriber for edge devices..."
nohup python subscriber.py > subscriber.log 2>&1 &

echo "✅ All services successfully launched!"
echo "- UI hosted on http://localhost:3000"
echo "- API hosted on http://localhost:8000"
