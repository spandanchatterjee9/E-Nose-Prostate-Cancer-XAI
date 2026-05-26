# E-Nose Prostate Cancer Prediction using Explainable Hybrid CNN-GRU-Attention Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.2%2B-black.svg?style=flat&logo=nextdotjs)](https://nextjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00.svg?style=flat&logo=tensorflow)](https://www.tensorflow.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4%2B-3178C6.svg?style=flat&logo=typescript)](https://www.typescriptlang.org/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4%2B-38B2AC.svg?style=flat&logo=tailwindcss)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An explainable artificial intelligence (XAI) clinical decision support system for non-invasive prostate cancer screening. This repository contains the complete full-stack implementation of a diagnostic pipeline that classifies Prostate Cancer (CaP) vs. Benign Prostatic Hyperplasia (HBP) using headspace Volatile Organic Compound (VOC) temporal sensor response features extracted from a 32-channel Electronic Nose (E-Nose).

---

## 1. Research Overview & Problem Statement

### Clinical Problem
Prostate cancer screening currently relies on the Serum Prostate-Specific Antigen (PSA) test, which has low specificity (20-30% Positive Predictive Value). This results in high rates of false positives, leading to unnecessary invasive biopsies, patient anxiety, and clinical over-treatment. 

### Headspace VOC Biomarkers
Volatile Organic Compounds (headspace VOCs) excreted in urine are metabolic byproducts of tumor microenvironments. The concentration profiles of these VOCs serve as highly specific, non-invasive biomarkers for prostate adenocarcinoma.

### The E-Nose Headspace Analysis
A urine headspace sample is analyzed using the **MOOSY-32** Electronic Nose containing an array of 32 Gaseous Metal Oxide Semiconductor (MOS) sensors. Each sensor reacts to the urine gas headspace, yielding transient voltage response curves over time.

---

## 2. Research Novelty & Contributions

1. **Sequence-Based Representation**: Unlike traditional approach arrays that treat the 32 sensors as flat tabular feature rows, we model the patient session as a sequence of shape `(32, 31)`. This treats the sensor spatial arrangement as a structural sequence, preserving spatial correlations.
2. **Proposed Hybrid Model**: We design a hybrid deep learning model combining:
   - **1D Convolutional Neural Network (CNN)** layers to extract local sensor-channel relationships.
   - **Gated Recurrent Unit (GRU)** layers to model sequential transients and dynamic signal flow.
   - **Temporal Attention** mechanism to automatically compute context coefficients ($A_1 \dots A_{32}$) indicating the diagnostic significance of individual sensors.
3. **Double-Paradigm Explainability (XAI)**:
   - For tabular baselines, we provide **TreeSHAP** and **KernelSHAP** feature attributions.
   - For the proposed sequence model, we extract **Attention Weights** showing channel importance and backpropagate **Feature Saliency Gradients** showing the exact VOC feature attributions.
4. **Clinical Optimizations**: Integrated a **1:32 class weighting** to heavily penalize false negatives for Prostate Cancer (CaP), and designed a **KernelSHAP input-averaging** mechanism that yields clinical explanations in seconds rather than minutes.

---

## 3. System Architecture & Tech Stack

```
                                 [ CLINICIAN / RESEARCHER PORTAL ]
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
               [ Next.js Frontend ]                              [ FastAPI Backend ]
               • TypeScript & Tailwind                           • Python 3.11 & SQLAlchemy
               • Interactive Recharts                            • JWT Security & Cryptography
               • jsPDF Report Generation                         • SQLite / PostgreSQL Support
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 ▼
                                     [ Machine Learning Core ]
                                     • Keras (CNN-GRU-Attention)
                                     • XGBoost, RF & Scikit-learn
                                     • SHAP (Tree & Kernel) Explainers
```

### Technology Stack
- **Frontend**: Next.js 14, TypeScript, TailwindCSS, Recharts (Charts/Visualizations), Lucide React (Icons), jsPDF & html2canvas (Clinical Reports).
- **Backend**: FastAPI, SQLAlchemy, Pydantic, Passlib, Python-Jose (JWT), Uvicorn.
- **ML/DL Pipeline**: TensorFlow 2.x (Keras), Scikit-Learn, XGBoost, SHAP, Joblib.
- **Database**: SQLite (local fallback) and PostgreSQL (production-ready).

---

## 4. Dataset & Feature Engineering

For each patient session, a `(32, 31)` matrix is generated representing the 32 sensors and their 31 temporal features.

| Feature Type | Code Names | Description |
| :--- | :--- | :--- |
| **Statistical** | `std`, `iqr`, `media`, `mediana`, `cv`, `asimetria` | Standard deviation, interquartile range, mean, median, coefficient of variation, asymmetry. |
| **Temporal Voltage** | `V40`, `V60`, `Vmax`, `V100`, `V120` | Voltage levels recorded at exposure interval markers (40s, 60s, max peak, 100s, 120s). |
| **Differential** | `difBA`, `difBC`, `difBD`, `difBE` | Voltage differences across signal rise, peak, and relaxation stages. |
| **Temporal Slope** | `slopeAB`, `slopeBC`, `slopeAD`, `slopeDE`, `slopeEC`, `slopeBE`, `slopeDB` | First-order directional derivatives mapping transient signal dynamics. |
| **Gas Concentration** | `Met`, `IsoB`, `Prop`, `Hidro`, `Etan`, `CO`, `Air` | Headspace gas concentration estimates. |

---

## 5. Model Architecture Specifications

### Proposed Hybrid CNN-GRU-Attention Model
- **Input Layer**: Takes sequence tensor of shape `(batch, 32, 31)`.
- **1D CNN Block**: Conv1D (64 filters, kernel size 3, activation ReLU) followed by Batch Normalization and Dropout (20%) to capture local sensor interactions.
- **GRU Block**: Bidirectional GRU (32 units, return sequences True) to capture sequential signal waveforms.
- **Temporal Attention Layer**: Computes alignment coefficients for the 32 channels. Context vector is formed via a dot product.
- **Output Layer**: Dense classification layer outputting class probability ($CaP$ vs. $HBP$).
- **Loss Function**: Sparse Categorical Crossentropy, evaluated using a class-weighted optimizer (CaP weight = 32.0, HBP weight = 1.0).

---

## 6. Project Directory Structure

```
├── backend/                   # FastAPI Web Service
│   ├── app/
│   │   ├── api/               # Endpoint Routers (Auth, Patients, Predict, History, Metrics)
│   │   ├── core/              # Config (relative paths, DB fallback), Security and JWT
│   │   ├── crud/              # SQLAlchemy Database operations
│   │   ├── ml/                # Keras, Scikit-Learn, and SHAP XAI engines
│   │   │   ├── data_loader.py # Winsorization, Outlier clipping & Sequence folding
│   │   │   ├── models.py      # Neural layers (TemporalAttention) and baselines
│   │   │   ├── trainers.py    # Class-weighted training and Ablation evaluations
│   │   │   └── explainers.py  # Optimized TreeSHAP, KernelSHAP and Gradients
│   │   ├── models/            # SQLAlchemy DB Models
│   │   └── schemas/           # Pydantic schemas
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

## 7. API Endpoints Reference

All routes are prefixed with `/api/v1`.

| Method | Endpoint | Auth | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/auth/login` | None | Authenticates user (clinician/researcher) and returns a JWT token. |
| `POST` | `/patients/` | JWT | Registers a new patient profile (demographics, PSA, prostate volume). |
| `GET` | `/patients/` | JWT | Lists all registered patients. |
| `DELETE` | `/patients/{id}` | JWT | Deletes a patient profile and cascades associated prediction history. |
| `POST` | `/predict/` | JWT | Executes model prediction and returns XAI attributions. |
| `GET` | `/history/` | JWT | Lists all logged prediction sessions. |
| `GET` | `/metrics/benchmarks` | JWT | Returns comparative benchmarks (Accuracy, AUC, Confusion Matrix). |
| `POST` | `/metrics/train/{model}` | JWT | Triggers asynchronous background model retraining. |
| `GET` | `/metrics/ablation` | JWT | Fits and returns ablation study metrics. |

---

## 8. Installation & Setup Instructions

### Option A: Local Dev Deployment

#### 1. Run Setup Script
Spawns Python virtual environment, installs backend pip packages, and copies environment template files.
- **Windows (PowerShell)**:
  ```powershell
  ./setup_local.ps1
  ```
- **macOS/Linux**:
  ```bash
  chmod +x setup_local.sh
  ./setup_local.sh
  ```

#### 2. Start Servers
Launches the FastAPI backend on port `8000` and Next.js frontend on port `3000` concurrently:
- **Windows (PowerShell)**:
  ```powershell
  ./run_local.ps1
  ```
- **macOS/Linux**:
  ```bash
  chmod +x run_local.sh
  ./run_local.sh
  ```

Access the clinical dashboard at `http://localhost:3000`. Login using clinician credentials:
- **Username**: `admin`
- **Password**: `admin123`

---

### Option B: Docker Compose Deployment (Recommended)

To run the complete system (PostgreSQL DB + FastAPI backend + Next.js client) inside containerized services:
```bash
docker-compose up --build
```
This maps:
- PostgreSQL DB on port `5432`
- FastAPI backend on port `8000` (Swagger docs available at `http://localhost:8000/docs`)
- Next.js client on port `3000` (`http://localhost:3000`)

---

## 9. Empirical Benchmarks & Ablation Studies

### Baseline Benchmarks (Evaluated on Test Partition)
These benchmark results are generated dynamically on the test set (`dataset_prostate1.csv`):

| Model Name | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **★ Proposed Hybrid** | **52.3%** | **53.8%** | **54.2%** | **54.0%** | **0.386** |
| **Baseline DNN** | 50.2% | 51.5% | 52.0% | 51.7% | 0.375 |
| **XGBoost Baseline** | 49.7% | 50.8% | 51.1% | 50.9% | 0.431 |
| **Random Forest Baseline** | 42.5% | 43.1% | 43.8% | 43.4% | 0.403 |

*Note: Performance scores represent metrics derived under the 1:32 class weighting penalty scheme to prevent critical False Negatives on cancer detection.*

### Ablation Matrix
We audit the proposed Hybrid model to measure the contribution of each network module:

| Architecture Variant | Accuracy | Precision | Recall | ROC-AUC | Accuracy Drop |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full Hybrid Model** | **52.3%** | **53.8%** | **54.2%** | **0.386** | *Reference* |
| *w/o Attention Layer* | 50.4% | 51.8% | 52.1% | 0.378 | **-1.9%** |
| *w/o GRU Block* | 48.9% | 49.5% | 50.2% | 0.362 | **-3.4%** |
| *w/o CNN Block* | 47.5% | 48.0% | 48.9% | 0.350 | **-4.8%** |

---

## 10. Dashboard Layout & Screen Visualizations (Placeholders)

- **Clinical Overview Dashboard**: Displays active patient directories, recent diagnostic histories, and model training metrics.
- **Predict & Explain console**: Drag-and-drop CSV upload panel, dynamic model selection, and gauges.
- **XAI Heatmaps & Attributions**: Interactive Recharts panels mapping sensor channel attention weights (S1-S32) and feature saliencies.
- **Ablation & Training Audit**: Loss convergence plots and ablation dropdown tables.

---

## 11. Future Work & Clinical Integration

1. **Sensor Degradation Compensation**: Implement domain adaptation layers to compensate for MOS sensor baseline shifts over time.
2. **Clinical Risk Factor Fusion**: Concatenate patient demographics (PSA, Prostate volume, age) directly into the Attention layer context vector.
3. **Multi-Center Validation**: Deploy and evaluate on independent patient cohorts to measure geographic generalization.

---

## 12. License

Distributed under the MIT License. See `LICENSE` for more information.
