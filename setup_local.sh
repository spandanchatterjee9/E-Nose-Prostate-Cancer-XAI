#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Setting up local environment for E-Nose Prostate Cancer Predictor ==="

# 1. Detect Python command
PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python is not installed or not in PATH. Please install Python 3.9+."
        exit 1
    fi
fi
echo "Using Python: $($PYTHON_CMD --version)"

# 2. Check and install Node.js/npm if missing (Windows winget fallback)
if ! command -v node &> /dev/null; then
    echo "Node.js is not found. Checking if winget.exe is available to install it..."
    if command -v winget.exe &> /dev/null; then
        echo "Running winget.exe to install Node.js (LTS)..."
        winget.exe install --id OpenJS.NodeJS.LTS --silent --accept-package-agreements --accept-source-agreements
        
        # Try to reload PATH from Windows Registry for Git Bash
        if command -v powershell.exe &> /dev/null; then
            echo "Reloading environment path..."
            REG_PATH=$(powershell.exe -Command "[System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path', 'User')")
            # Convert Windows semi-colon path to Git Bash colon path
            export PATH=$(echo "$REG_PATH" | sed 's/\\/\//g' | sed 's/;\([a-zA-Z]\):/:\/\1/g' | sed 's/;/:/g' | tr -d '\r')
        fi
        
        # Proactive check for default Git Bash path mount
        if [ -d "/c/Program Files/nodejs" ]; then
            echo "Appending default Node.js mount to session PATH..."
            export PATH="$PATH:/c/Program Files/nodejs"
        fi
    fi
fi

if command -v node &> /dev/null; then
    echo "Node.js is installed: $(node -v)"
else
    echo "Warning: Node.js/npm is missing. Next.js frontend will not function locally."
    echo "Please install Node.js manually: https://nodejs.org/"
fi

# 3. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# 4. Activate virtual environment (Windows-aware activate script location)
if [ -f "venv/Scripts/activate" ]; then
    echo "Activating Windows virtual environment..."
    source venv/Scripts/activate
elif [ -f "venv/bin/activate" ]; then
    echo "Activating Unix virtual environment..."
    source venv/bin/activate
else
    echo "Error: Virtual environment activation script not found."
    exit 1
fi

# 5. Install backend dependencies
echo "Installing backend python dependencies..."
pip install --upgrade pip
pip install -r backend/requirements.txt

# 6. Install frontend dependencies
if command -v npm &> /dev/null; then
    echo "Installing frontend dependencies using npm workspaces..."
    npm install
else
    echo "Warning: npm not found. Skipping frontend dependencies installation."
fi

# 7. Initialize environment configurations
if [ ! -f ".env" ]; then
    echo "Seeding root .env configuration..."
    cp .env.example .env
fi
if [ ! -f "backend/.env" ]; then
    echo "Seeding default backend .env configuration..."
    cp backend/.env.example backend/.env
fi
if [ ! -f "frontend/.env.local" ]; then
    echo "Seeding default frontend .env.local configuration..."
    cp frontend/.env.local.example frontend/.env.local
fi

# 8. Run Verification Utility
echo "Running automated environment checks..."
python verify_env.py

echo "=== Setup completed successfully! ==="
echo "Run 'run_local.sh' to start the servers."
