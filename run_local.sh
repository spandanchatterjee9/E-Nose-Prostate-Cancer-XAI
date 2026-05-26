#!/bin/bash

echo "=== Starting local servers for E-Nose Prostate Cancer Predictor ==="

# Function to clean up background processes on script exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start FastAPI backend
if [ -f "venv/bin/python" ]; then
    echo "Launching FastAPI backend server on http://localhost:8000..."
    source venv/bin/activate
    cd backend
    python -m uvicorn app.main:app --reload --port 8000 &
    BACKEND_PID=$!
    cd ..
else
    echo "Error: Virtual environment not found. Please run setup_local.sh first."
    exit 1
fi

# 2. Start Next.js frontend
if command -v npm &> /dev/null; then
    echo "Launching Next.js frontend server on http://localhost:3000..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
else
    echo "Warning: Node/npm is not installed. To run Next.js frontend locally, please install Node.js."
    echo "Alternatively, run 'docker-compose up --build' to run the full application in Docker."
fi

# Keep script running to show logs and catch Ctrl+C
wait
