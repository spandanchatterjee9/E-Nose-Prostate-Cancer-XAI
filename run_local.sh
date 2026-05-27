#!/bin/bash

echo "=== Starting local servers for E-Nose Prostate Cancer Predictor ==="

# 1. Activate virtual environment
if [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run ./setup_local.sh first."
    exit 1
fi

# 2. Run Pre-flight Verification Check
echo "Running environment health verification check..."
if python verify_env.py; then
    echo "Pre-flight checks passed."
else
    echo "Warning: Pre-flight checks detected issues."
    read -p "Do you want to attempt starting the servers anyway? (y/n) " choice
    if [ "$choice" != "y" ] && [ "$choice" != "Y" ]; then
        echo "Exiting startup script."
        exit 1
    fi
fi

# 3. Double Check Port Conflicts
if command -v netstat.exe &> /dev/null; then
    if netstat.exe -ano | grep -q "0.0.0.0:8000" || netstat.exe -ano | grep -q "127.0.0.1:8000"; then
        echo "Error: Port 8000 is already in use. Please free the port before starting."
        exit 1
    fi
    if netstat.exe -ano | grep -q "0.0.0.0:3000" || netstat.exe -ano | grep -q "127.0.0.1:3000"; then
        echo "Error: Port 3000 is already in use. Please free the port before starting."
        exit 1
    fi
fi

# 4. Function to clean up background processes on script exit
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 5. Start FastAPI Backend
echo "Launching FastAPI backend server on http://localhost:8000..."
python -m uvicorn app.main:app --app-dir backend --reload --port 8000 &
BACKEND_PID=$!

# 6. Start Next.js Frontend
if command -v npm &> /dev/null; then
    echo "Launching Next.js frontend server on http://localhost:3000..."
    npm run dev --prefix frontend &
    FRONTEND_PID=$!
else
    echo "Warning: npm is missing. Next.js frontend will not start."
fi

# 7. Poll Backend until responsive
echo "Polling FastAPI backend until responsive..."
for i in {1..15}; do
    if curl -s -m 1 http://localhost:8000/ > /dev/null; then
        echo "FastAPI backend is ONLINE."
        break
    fi
    sleep 1
done

# 8. Poll Frontend until responsive
if command -v npm &> /dev/null; then
    echo "Polling Next.js frontend until responsive..."
    for i in {1..15}; do
        if curl -s -m 1 http://localhost:3000/ > /dev/null; then
            echo "Next.js frontend is ONLINE."
            break
        fi
        sleep 1
    done
fi

# 9. Open Browser
echo "Opening Clinical Dashboard..."
if command -v cmd.exe &> /dev/null; then
    cmd.exe /c start http://localhost:3000
elif command -v open &> /dev/null; then
    open http://localhost:3000
elif command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:3000
fi

# Keep script running to show logs and catch Ctrl+C
wait
