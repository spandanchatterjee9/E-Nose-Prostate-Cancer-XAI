import datetime
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Boolean, JSON, Text
from sqlalchemy.orm import relationship
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="clinician") # admin, clinician, researcher
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Patient(Base):
    __tablename__ = "patients"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_code = Column(String(50), unique=True, index=True, nullable=False)
    age = Column(Integer, nullable=False)
    psa = Column(Float, nullable=False)           # Prostate-Specific Antigen (ng/mL)
    volume = Column(Float, nullable=False)        # Prostate Volume (cc)
    clinical_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    model_name = Column(String(50), nullable=False)  # baseline_dnn, random_forest, xgboost, hybrid_model
    features = Column(JSON, nullable=False)          # Stores raw or processed inputs
    prediction_label = Column(String(10), nullable=False) # CaP or HBP
    confidence = Column(Float, nullable=False)       # Output probability (0 to 1)
    shap_values = Column(JSON, nullable=True)        # Dictionary of attributions or list
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="predictions")
    
    @property
    def patient_code(self) -> str:
        return self.patient.patient_code if self.patient else "UNKNOWN"

class ModelResult(Base):
    __tablename__ = "model_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(50), nullable=False)
    run_id = Column(String(100), nullable=False)     # Unique key for the training run
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    roc_auc = Column(Float, nullable=False)
    confusion_matrix = Column(JSON, nullable=False)  # 2x2 nested list
    classification_report = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ExperimentLog(Base):
    __tablename__ = "experiment_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(100), nullable=False, index=True)
    model_name = Column(String(50), nullable=False)
    hyperparameters = Column(JSON, nullable=False)   # Epochs, learning rate, weights, etc.
    training_metrics = Column(JSON, nullable=False)  # History dict: loss, accuracy, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
