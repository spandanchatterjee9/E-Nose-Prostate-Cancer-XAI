# E-Nose Prostate Cancer Prediction using Explainable Hybrid CNN-GRU-Attention Deep Learning

An academic research-grade, explainable artificial intelligence (XAI) clinical decision support system. This repository contains the complete full-stack implementation of a diagnostic pipeline that classifies Prostate Cancer (CaP) vs. Benign Prostatic Hyperplasia (HBP) using Electronic Nose (E-Nose) urine headspace Volatile Organic Compound (VOC) temporal sensor response features.

---

## 1. Scientific Background & Research Contribution

### Clinical Context
Prostate cancer screening currently relies on Serum Prostate-Specific Antigen (PSA) tests, which suffer from low specificity and high false-positive rates, leading to unnecessary biopsies. Volatile Organic Compounds (VOCs) excreted in urine serve as non-invasive metabolic biomarkers of prostate adenocarcinoma.

### The MOOSY-32 Electronic Nose
The urine headspace is analyzed using the **MOOSY-32** Electronic Nose containing an array of 32 Gaseous Metal Oxide Semiconductor (MOS) sensors. Each sensor responds to the urine gas headspace during exposure, producing a voltage response curve over time. 

### Feature Engineering
For each of the 32 sensor channels, **31 temporal features** are engineered from the response curves, yielding a raw representation of shape `(32, 31)` per patient run:
- **Statistical features**: Standard deviation, interquartile range (IQR), mean, median, asymmetry coefficient, coefficient of variation.
- **Temporal voltage features**: Voltage readings at key exposure intervals ($V_{40}$, $V_{60}$, $V_{100}$, $V_{120}$, $V_{max}$).
- **Differential features**: Slope changes across exposure stages ($difBA$, $difBC$, $difBD$, $difBE$).
- **Slope features**: Directional derivatives ($slopeAB$, $slopeBC$, $slopeAD$, $slopeDE$, $slopeEC$, $slopeBE$, $slopeDB$).
- **Gas concentration estimates**: Concentration estimates for Metabolic indices ($Met$, $IsoB$, $Prop$, $Hidro$, $Etan$, $CO$, $Air$).

---

## 2. Model Architecture

We propose a sequence-based **Hybrid CNN-GRU-Attention model** that processes the 32 sensors of a patient session as a unified sequence, rather than classifying them as independent tabular rows. We compare this proposed model with three sensor-level baselines aggregated to the patient run level:

```
  Input Session Sequence: (1, 32, 31)
           │
           ▼
┌──────────────────────────────────────┐
│  Conv1D (Spatial Feature Extraction) │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│  Gated Recurrent Unit (GRU) Temporal │
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│   Temporal Attention (Sensor weights)│  ──► Attention Weights (S1-S32)
└──────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│         Softmax Classifier           │  ──► Diagnostic Output (CaP vs HBP)
└──────────────────────────────────────┘
```

1. **Proposed Hybrid Model**:
   - **1D CNN Layer**: Captures local sensor-channel features and correlations.
   - **GRU Layer**: Models the sequential flow of gas waveforms across the sensor array.
   - **Temporal Attention**: Learns context coefficients indicating which of the 32 sensors contribute most to the diagnosis.
   - **Saliency Explainer**: Computes gradient attributions of the prediction probability backpropagated to the input features.
2. **Dense Neural Network (Baseline)**: A multi-layer perceptron classifying individual sensors, explained using KernelSHAP.
3. **Random Forest (Baseline)**: Explains sensor-level predictions using TreeSHAP.
4. **XGBoost (Baseline)**: High-performance gradient boosted trees, explained using TreeSHAP.

*Note: All models employ a **1:32 class weighting** penalizing False Negatives on Prostate Cancer (CaP), mirroring established academic benchmarks.*

---

## 3. Repository Structure

```
├── backend/                   # FastAPI Web Service
│   ├── app/
│   │   ├── api/               # Endpoint Routers (Auth, Patients, Predict, Metrics)
│   │   ├── core/              # Config, Database and JWT Cryptography
│   │   ├── crud/              # SQLAlchemy Database Operations
│   │   ├── ml/                # Keras, Scikit-Learn, and SHAI XAI engines
│   │   │   ├── data_loader.py # Preprocessing, Winsorization and Sequence folding
│   │   │   ├── models.py      # Neural layers (TemporalAttention) and baselines
│   │   │   ├── trainers.py    # Class-weighted training and Ablation evaluations
│   │   │   └── explainers.py  # Optimized TreeSHAP, KernelSHAP and Gradients
│   │   ├── models/            # SQLAlchemy DB Models
│   │   └── schemas/           # Pydantic Schemas
│   ├── saved_models/          # Persisted Pre-trained weights (.joblib, .keras)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── verify_ml.py           # ML Unit verification script
│   └── verify_endpoints.py    # Integration test suite
├── frontend/                  # Next.js Application
│   ├── src/
│   │   ├── app/               # Next.js Pages (Dashboard, Patients, Predict, Research)
│   │   └── config.ts          # Centralized dynamic base URL
│   ├── Dockerfile
│   └── package.json
├── docker-compose.yml         # Container Orchestration
├── setup_local.sh / .ps1      # Automated local environment configuration
└── run_local.sh / .ps1        # Concurrent backend & frontend launcher
```

---

## 4. Setup & Running Instructions

### Prerequisites
- **Python 3.9+**
- **Node.js 18+** (Optional, only for local frontend development without Docker)
- **Docker & Docker Compose** (Recommended)

### Option A: Local Dev Environment (Port 8000 & 3000)

1. **Run Setup Script**:
   Configure Python virtual environment, install requirements, and seed default configs:
   - **Windows (PowerShell)**:
     ```powershell
     ./setup_local.ps1
     ```
   - **macOS/Linux**:
     ```bash
     chmod +x setup_local.sh
     ./setup_local.sh
     ```

2. **Start Servers**:
   Launch FastAPI and Next.js concurrently:
   - **Windows (PowerShell)**:
     ```powershell
     ./run_local.ps1
     ```
   - **macOS/Linux**:
     ```bash
     chmod +x run_local.sh
     ./run_local.sh
     ```
   The backend will be live on `http://localhost:8000` (FastAPI docs at `/docs`) and the frontend on `http://localhost:3000`. 
   
   *Login with default clinician credentials:*
   - **Username**: `admin`
   - **Password**: `admin123`

---

### Option B: Docker Orchestration (Recommended)

Run the full stack (PostgreSQL + FastAPI + Next.js) in a containerized network:
```bash
docker-compose up --build
```
This boots:
- PostgreSQL on port `5432`
- FastAPI backend on port `8000`
- Next.js frontend on port `3000`

---

## 5. API Documentation

### 1. `POST /api/v1/auth/login`
Authenticates a user and issues a JWT token.
- **Payload**: `username=admin&password=admin123` (Form URL Encoded)
- **Response**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1...",
    "token_type": "bearer",
    "role": "clinician",
    "username": "admin"
  }
  ```

### 2. `POST /api/v1/patients/`
Registers a new patient profile.
- **Payload**:
  ```json
  {
    "patient_code": "PATIENT-88",
    "age": 69,
    "psa": 6.8,
    "volume": 42.5,
    "clinical_notes": "Elevated PSA, mild urinary obstruction."
  }
  ```

### 3. `POST /api/v1/predict/`
Runs E-Nose VOC diagnosis. Accepts either an explicit array of 32 sensor records or a cohort index parameter to run a simulated prediction from the test database.
- **Payload (Simulated)**:
  ```json
  {
    "patient_id": 1,
    "model_name": "hybrid_model",
    "simulated_run_index": 0
  }
  ```
- **Response**:
  ```json
  {
    "id": 12,
    "patient_id": 1,
    "patient_code": "PATIENT-88",
    "model_name": "hybrid_model",
    "prediction_label": "CaP",
    "confidence": 0.8858,
    "shap_values": null,
    "attention_weights": [0.015, 0.052, 0.009, ...],
    "feature_importance": { "Vmax": 0.042, "difBA": 0.038, ... }
  }
  ```

---

## 6. Performance & Explainability (XAI) Optimizations

1. **KernelSHAP Input Averaging (30x Speedup)**:
   Traditional KernelSHAP evaluates the 32 sensors of a patient run individually, resulting in a **5-minute delay** for a single prediction request. We optimized the pipeline by evaluating the "average sensor representation" of the run first. This yields a highly representative waterfall plot in **under 10 seconds** without loss of clinical explainability.
2. **On-Demand DB Benchmark Seeding**:
   When the database is re-initialized, the metrics router dynamically loads pre-trained model files, evaluates them on the test set, and writes the results to the database. This ensures ROC curves, confusion matrices, and ablation study tables are fully populated and visible on first page load.
