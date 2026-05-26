from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# ================= USER SCHEMAS =================
class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    password: str
    role: Optional[str] = "clinician"

class UserResponse(UserBase):
    id: int
    role: str
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True

# ================= AUTH TOKEN SCHEMAS =================
class Token(BaseModel):
    access_token: str
    token_type: str
    role: str
    username: str

class TokenPayload(BaseModel):
    sub: Optional[str] = None

# ================= PATIENT SCHEMAS =================
class PatientBase(BaseModel):
    patient_code: str
    age: int
    psa: float = Field(..., description="Prostate-Specific Antigen level in ng/mL")
    volume: float = Field(..., description="Prostate volume in cc")
    clinical_notes: Optional[str] = None

class PatientCreate(PatientBase):
    pass

class PatientUpdate(PatientBase):
    patient_code: Optional[str] = None
    age: Optional[int] = None
    psa: Optional[float] = None
    volume: Optional[float] = None

class PatientResponse(PatientBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# ================= PREDICTION SCHEMAS =================
class SensorRecord(BaseModel):
    Sensor: str
    el75: float
    std: float
    moda: float
    media: float
    mediana: float
    iqr: float
    cv: float
    V40: float
    V60: float
    Vmax: float
    V100: float
    V120: float
    difBA: float
    difBC: float
    difBD: float
    difBE: float
    slopeAB: float
    slopeBC: float
    slopeAD: float
    slopeDE: float
    slopeEC: float
    slopeBE: float
    slopeDB: float
    asimetria: float
    Met: float
    IsoB: float
    Prop: float
    Hidro: float
    Etan: float
    CO: float
    Air: float

class PredictionRequest(BaseModel):
    patient_id: int
    model_name: str = Field("hybrid_model", description="baseline_dnn, random_forest, xgboost, hybrid_model")
    # Accept either explicit list of 32 sensor records OR a dataset index to simulate prediction
    sensor_data: Optional[List[SensorRecord]] = None
    simulated_run_index: Optional[int] = None  # To mock prediction by reading from test dataset

class PredictionResponse(BaseModel):
    id: int
    patient_id: int
    patient_code: str
    model_name: str
    prediction_label: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionDetailResponse(PredictionResponse):
    features: List[Dict[str, Any]]
    shap_values: Optional[Dict[str, Any]] = None
    attention_weights: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None

# ================= MODEL RESULTS SCHEMAS =================
class ModelResultCreate(BaseModel):
    model_name: str
    run_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    classification_report: Optional[Dict[str, Any]] = None

class ModelResultResponse(ModelResultCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True

# ================= EXPERIMENT LOG SCHEMAS =================
class ExperimentLogResponse(BaseModel):
    id: int
    run_id: str
    model_name: str
    hyperparameters: Dict[str, Any]
    training_metrics: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

# ================= RESEARCH & BENCHMARK SCHEMAS =================
class BenchmarkResponse(BaseModel):
    metrics: Dict[str, ModelResultResponse]
    ablation: Dict[str, Dict[str, float]]
