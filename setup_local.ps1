Write-Host "=== Setting up local environment for E-Nose Prostate Cancer Predictor ===" -ForegroundColor Cyan

# 1. Check Python installation
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not added to PATH. Please install Python 3.9+."
    exit 1
}

# 2. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Blue
    python -m venv venv
}

# 3. Activate virtual environment and install backend requirements
Write-Host "Installing backend python dependencies..." -ForegroundColor Blue
& venv/Scripts/pip install -r backend/requirements.txt

# 4. Initialize local .env file
if (-not (Test-Path "backend/.env")) {
    Write-Host "Seeding default backend .env configuration..." -ForegroundColor Blue
    Copy-Item "backend/.env.example" "backend/.env"
}

# 5. Initialize local frontend .env.local file
if (-not (Test-Path "frontend/.env.local")) {
    Write-Host "Seeding default frontend .env.local configuration..." -ForegroundColor Blue
    Copy-Item "frontend/.env.local.example" "frontend/.env.local"
}

Write-Host "=== Setup completed successfully! ===" -ForegroundColor Green
Write-Host "Run 'run_local.ps1' or start uvicorn backend and npm dev frontend." -ForegroundColor Yellow
