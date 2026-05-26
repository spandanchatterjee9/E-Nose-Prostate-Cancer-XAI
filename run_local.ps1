Write-Host "=== Starting local servers for E-Nose Prostate Cancer Predictor ===" -ForegroundColor Cyan

# 1. Start backend service in new terminal window
if (Test-Path "venv/Scripts/python.exe") {
    Write-Host "Launching FastAPI backend server on http://localhost:8000..." -ForegroundColor Blue
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd backend; ..\venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000"
} else {
    Write-Error "Virtual environment not found. Please run setup_local.ps1 first."
    exit 1
}

# 2. Check node/npm and start frontend in new terminal window
if (Get-Command "npm" -ErrorAction SilentlyContinue) {
    Write-Host "Launching Next.js frontend server on http://localhost:3000..." -ForegroundColor Blue
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev"
} else {
    Write-Host "Warning: Node/npm is not installed. To run Next.js frontend locally, please install Node.js." -ForegroundColor Yellow
    Write-Host "Alternatively, run 'docker-compose up --build' to run the full application in Docker." -ForegroundColor Yellow
}

Write-Host "=== Launcher completed. Inspect separate terminal windows for server logs. ===" -ForegroundColor Green
