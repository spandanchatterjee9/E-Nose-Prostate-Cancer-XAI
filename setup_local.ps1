Write-Host "=== Setting up local environment for E-Nose Prostate Cancer Predictor ===" -ForegroundColor Cyan

# 1. Check Python installation
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not added to PATH. Please install Python 3.9+."
    exit 1
}

# 2. Check and install Node.js/npm if missing
if (-not (Get-Command "node" -ErrorAction SilentlyContinue)) {
    Write-Host "Node.js is not found. Attempting to install Node.js (LTS) via winget..." -ForegroundColor Blue
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        Write-Host "Running winget to install Node.js (LTS)..." -ForegroundColor Blue
        # Launch winget to install NodeJS
        Start-Process winget -ArgumentList "install --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements" -Wait -NoNewWindow
        
        # Reload Environment variables to pick up Node.js path
        Write-Host "Reloading registry environment PATH..." -ForegroundColor Blue
        $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        # Proactive PATH fallback check (default Node.js install directory)
        if (-not (Get-Command "node" -ErrorAction SilentlyContinue) -and (Test-Path "C:\Program Files\nodejs")) {
            Write-Host "Manually adding default Node.js path to current session PATH..." -ForegroundColor Yellow
            $env:PATH += ";C:\Program Files\nodejs"
        }
        
        if (-not (Get-Command "node" -ErrorAction SilentlyContinue)) {
            Write-Warning "Node.js installation finished but 'node' is still not recognized. You may need to restart your terminal or computer."
        } else {
            Write-Host "Node.js and npm installed successfully and loaded in current session!" -ForegroundColor Green
        }
    } else {
        Write-Error "Node.js is missing and 'winget' is not available. Please install Node.js manually: https://nodejs.org/"
    }
} else {
    Write-Host "Node.js is already installed: $(node -v)" -ForegroundColor Green
}

# 3. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Blue
    python -m venv venv
}

# 4. Upgrade pip and install backend requirements
Write-Host "Installing backend python dependencies..." -ForegroundColor Blue
& venv/Scripts/python.exe -m pip install --upgrade pip
& venv/Scripts/pip install -r backend/requirements.txt

# 5. Install frontend dependencies (via root npm workspaces)
if (Get-Command "npm" -ErrorAction SilentlyContinue) {
    Write-Host "Installing frontend dependencies using npm workspaces..." -ForegroundColor Blue
    npm install
} else {
    Write-Warning "npm is not available. Frontend dependencies will not be installed automatically."
}

# 6. Initialize environment variables
if (-not (Test-Path ".env")) {
    Write-Host "Seeding root .env configuration..." -ForegroundColor Blue
    Copy-Item ".env.example" ".env"
}
if (-not (Test-Path "backend/.env")) {
    Write-Host "Seeding default backend .env configuration..." -ForegroundColor Blue
    Copy-Item "backend/.env.example" "backend/.env"
}
if (-not (Test-Path "frontend/.env.local")) {
    Write-Host "Seeding default frontend .env.local configuration..." -ForegroundColor Blue
    Copy-Item "frontend/.env.local.example" "frontend/.env.local"
}

# 7. Run Verification Script
Write-Host "Running automated environment checks..." -ForegroundColor Blue
& venv/Scripts/python.exe verify_env.py

Write-Host "=== Setup process completed! ===" -ForegroundColor Green
Write-Host "Run 'run_local.ps1' to start uvicorn backend and next dev frontend." -ForegroundColor Yellow
