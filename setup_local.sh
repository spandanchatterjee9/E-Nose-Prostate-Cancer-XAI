#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Setting up local environment for E-Nose Prostate Cancer Predictor ==="

# 1. Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH. Please install Python 3.9+."
    exit 1
fi

# 2. Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# 3. Activate virtual environment and install backend requirements
echo "Installing backend python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Initialize local .env file
if [ ! -f "backend/.env" ]; then
    echo "Seeding default backend .env configuration..."
    cp backend/.env.example backend/.env
fi

# 5. Initialize local frontend .env.local file
if [ ! -f "frontend/.env.local" ]; then
    echo "Seeding default frontend .env.local configuration..."
    cp frontend/.env.local.example frontend/.env.local
fi

echo "=== Setup completed successfully! ==="
echo "Run 'run_local.sh' or start uvicorn backend and npm dev frontend."
