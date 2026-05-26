# E-Nose Prostate Cancer Prediction Clinical Dashboard

This project implements an explainable AI (XAI) clinical dashboard for classifying Prostate Cancer (CaP) vs. Benign Prostatic Hyperplasia (HBP) using Electronic Nose (E-Nose) temporal Volatile Organic Compound (VOC) features. It reproduces the dense neural network baseline from the reference paper, implements Random Forest and XGBoost benchmarks, and proposes a hybrid CNN-GRU-Attention deep learning model.

---

## User Review Required

> [!IMPORTANT]
> **Data Redundancy Strategy & Sequence Grouping**: 
> The dataset has 12,800 rows per split. Each patient's clinical session is repeated across 4 containers, 5 acquisitions, and 32 sensors (`4 * 5 * 32 = 640` rows per patient). 
> - **Baselines** (Dense NN, Random Forest, XGBoost) will be trained at the **sensor level** (12,800 rows of shape `(32,)` including sensor index).
> - The **Proposed Hybrid CNN-GRU-Attention** model will group the 32 sensors of each acquisition run into a sequence of shape `(32, 31)` to learn spatial/channel patterns across the sensor array, yielding `400` training sequences.
> - **Ablation Studies**: We will implement automated tests comparing models with subsets of features (e.g. without differential features, without gas estimates) to validate our model design for academic publication.

---

## Open Questions

> [!NOTE]
> **PostgreSQL Connectivity**:
> The FastAPI backend will support PostgreSQL as the production database, but we will configure an automatic fallback to SQLite (`sqlite:///./enose.db`) to allow seamless local execution without requiring a local PostgreSQL service running. We will provide a `.env` template to easily hook into your PostgreSQL instance.

---

## Proposed Changes

### Backend Component

We will create a modular FastAPI backend in `backend/` utilizing SQLAlchemy, Scikit-learn, XGBoost, TensorFlow, and SHAP.

#### [NEW] [config.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/core/config.py)
Core configuration, environment variables, security constants, and path managers.

#### [NEW] [database.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/core/database.py)
SQLAlchemy engine setup and database session dependencies.

#### [NEW] [security.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/core/security.py)
Password hashing (bcrypt) and JWT token generation/validation.

#### [NEW] [models.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/models/models.py)
Database schema definitions matching SQLite/PostgreSQL:
- `users`: Clinician credentials and roles.
- `patients`: ID, patient_code, age, PSA level, prostate volume, clinical notes.
- `predictions`: Predicted label (CaP/HBP), confidence, inputs (JSON), SHAP values (JSON), clinician override.
- `model_results`: Evaluation metrics (Accuracy, Precision, Recall, F1, AUC, Confusion Matrix) for model tracking.
- `experiment_logs`: Hyperparameter logging, training time, training histories (loss, validation accuracy).

#### [NEW] [data_loader.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/ml/data_loader.py)
Robust data pre-processing pipeline:
- Loads train (`dataset_prostate.csv`) and test (`dataset_prostate1.csv`) files.
- Fits standard scaler on training features.
- Maps `Sensor` identifiers (S1-S32) to categorical integers (0-31).
- Generates 1D sensor-level datasets (shape `(N, 32)`) and sequence run-level datasets (shape `(N/32, 32, 31)`).

#### [NEW] [models.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/ml/models.py)
Model definitions:
- **Baseline FFNN**: Input(32) -> Batch Normalization -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Softmax(2).
- **Proposed CNN-GRU-Attention**:
  - Input: `(32, 31)` representing sequence of 32 sensors.
  - Conv1D: Filters=64, Kernel_size=3, padding='same', ReLU activation.
  - Batch Normalization & Dropout (0.2).
  - GRU: 64 units, returning sequences.
  - Temporal Attention Layer: Computes weights over the 32 step dimensions, reduces sequences to shape `(batch, 64)`.
  - Dense: 32 units, ReLU activation.
  - Dropout (0.2).
  - Output: Dense(2) with Softmax.

#### [NEW] [trainers.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/ml/trainers.py)
Training logic for all four models. Implements:
- Class weight support (1:32 penalty for CaP as per reference paper).
- Automated metric logging (accuracy, precision, recall, f1, roc-auc, confusion matrix).
- Registry integration to save models under version control.
- Hyperparameter logging into `experiment_logs`.

#### [NEW] [explainers.py](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/ml/explainers.py)
XAI engine:
- Computes SHAP values using `shap.TreeExplainer` for Random Forest/XGBoost, `shap.DeepExplainer` for Dense NN.
- Extracts Attention weights for the Hybrid model to display which sensor nodes (S1-S32) were most influential.
- Formats local and global feature importance scores for API consumption.

#### [NEW] [endpoints](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/app/api/endpoints)
FastAPI endpoints to drive the user interface:
- `/auth/register` & `/auth/login`: User creation and JWT token generation.
- `/patients`: Patient listing, details, creation.
- `/predict`: Accepts E-Nose VOC inputs, runs inference on selected model, saves to `predictions`, and triggers explainability.
- `/explain`: Returns SHAP and Attention values for a prediction.
- `/train`: Triggers model re-training or runs baseline training on loading the dashboard.
- `/history`: Retrieves prediction logs.
- `/metrics`: Returns model benchmark reports, ablation studies, and training histories.

---

### Frontend Component

We will create a Next.js (TypeScript + TailwindCSS + Recharts) frontend in `frontend/`.

#### [NEW] [Dashboard Layout](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/frontend/src/app/dashboard)
A modern, glassmorphic dark-themed layout featuring sidebar navigation.

#### [NEW] [Patient Management UI](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/frontend/src/app/dashboard/patients/page.tsx)
Interactive CRUD list of clinical patients. Shows demographics, clinical features (PSA, volume), and prediction status.

#### [NEW] [Prediction Interface](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/frontend/src/app/dashboard/predict/page.tsx)
Enables prediction run selection:
- Select a patient.
- Input VOC sensor data (either choose pre-existing sample runs or upload a CSV file containing 32 sensor rows).
- Choose classification model.
- Visual display of prediction label (CaP vs HBP), model confidence, and prediction explanation panel.

#### [NEW] [Explainable AI Panel](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/frontend/src/components/ExplainabilityPanel.tsx)
Displays XAI visualizations using SVG and Recharts:
- **SHAP Waterfall/Bar chart**: Shows positive and negative contributions of top features (e.g. `Vmax`, `slopeBC`, `difBA`, `Met`).
- **Attention heatmap/bar**: For the Hybrid model, highlights which sensors (S1-S32) the attention layer prioritized.

#### [NEW] [Research & Benchmarking Dashboard](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/frontend/src/app/dashboard/research/page.tsx)
Dashboard with academic visualizations:
- ROC Curve comparison (plotting all four models).
- Confusion Matrix grid (visualizing TP/FP/FN/TN metrics).
- Ablation study comparison matrix (checking performance under different feature configurations).
- Hyperparameter logging logs and training convergence plots (epochs vs. loss/accuracy).

---

### Deployment & Config Files

#### [NEW] [docker-compose.yml](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/docker-compose.yml)
Multi-container configuration orchestrating PostgreSQL, Next.js, and FastAPI.

#### [NEW] [Dockerfiles](file:///c:/Users/spand/college/research_ml_E-nose/E-nose_prostate/backend/Dockerfile)
Docker configurations for the Next.js and FastAPI modules to run standalone.

---

## Verification Plan

### Automated Tests
1. **Model Testing Script**: A verification script `verify_ml.py` that trains models on the training dataset, evaluates on the test dataset, and logs the metrics.
2. **API Endpoint Tests**: Python tests using FastAPI's `TestClient` to verify auth, prediction, patient, and training endpoints.
3. **Frontend Integration Test**: Build verification of the Next.js application.

### Manual Verification
1. **Interactive Demo**: Booting both servers and simulating the clinician user flow:
   - Login.
   - Register a patient.
   - Upload and predict using standard patient test cases.
   - Inspect the SHAP chart, Attention map, ROC curve, and confusion matrix.
   - Download the PDF report.
