Write-Host "=== Starting local servers for E-Nose Prostate Cancer Predictor ===" -ForegroundColor Cyan

# 1. Run Pre-flight Verification Check
if (Test-Path "venv/Scripts/python.exe") {
    Write-Host "Running environment health verification check..." -ForegroundColor Blue
    & venv/Scripts/python.exe verify_env.py
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Pre-flight checks returned errors. Please ensure setup completed successfully."
        # Ask user for permission to proceed or exit
        $choice = Read-Host "Do you want to attempt starting the servers anyway? (Y/N)"
        if ($choice -ne "Y" -and $choice -ne "y") {
            Write-Host "Exiting startup script." -ForegroundColor Yellow
            exit 1
        }
    }
} else {
    Write-Error "Virtual environment not found. Please run 'setup_local.ps1' first."
    exit 1
}

# 2. Port Conflict Checks
$port8000Used = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($port8000Used) {
    Write-Error "Port 8000 is already in use. Please terminate the process using this port before starting."
    exit 1
}

$port3000Used = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($port3000Used) {
    Write-Error "Port 3000 is already in use. Please terminate the process using this port before starting."
    exit 1
}

# 3. Start FastAPI Backend Service
Write-Host "Launching FastAPI backend server on http://localhost:8000..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000"

# 4. Start Next.js Frontend Service
if (Get-Command "npm" -ErrorAction SilentlyContinue) {
    Write-Host "Launching Next.js frontend server on http://localhost:3000..." -ForegroundColor Blue
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev --prefix frontend"
} else {
    Write-Warning "npm not found. Skipping Next.js frontend launch. Please run the frontend manually."
}

# 5. Wait and Poll Backend
Write-Host "Polling FastAPI backend until responsive..." -ForegroundColor Blue
$backendReady = $false
for ($i=0; $i -lt 15; $i++) {
    try {
        $res = Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get -TimeoutSec 1 -ErrorAction Stop
        if ($res.docs_url -eq "/docs") {
            $backendReady = $true
            Write-Host "FastAPI backend is ONLINE." -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

# 6. Wait and Poll Frontend
Write-Host "Polling Next.js frontend until responsive..." -ForegroundColor Blue
$frontendReady = $false
for ($i=0; $i -lt 15; $i++) {
    try {
        $res = Invoke-WebRequest -Uri "http://localhost:3000" -Method Get -TimeoutSec 1 -ErrorAction Stop
        if ($res.StatusCode -eq 200) {
            $frontendReady = $true
            Write-Host "Next.js frontend is ONLINE." -ForegroundColor Green
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}

# 7. Open Browser Dashboard
if ($frontendReady) {
    Write-Host "Opening Clinical Dashboard at http://localhost:3000..." -ForegroundColor Green
    Start-Process "http://localhost:3000"
} else {
    Write-Warning "Next.js server taking longer to start. You can open http://localhost:3000 manually once initialized."
}

Write-Host "=== Startup completed successfully! ===" -ForegroundColor Green
Write-Host "Separate terminal windows are running backend (8000) and frontend (3000) logs." -ForegroundColor Yellow
