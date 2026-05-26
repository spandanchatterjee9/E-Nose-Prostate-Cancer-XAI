from sqlalchemy.orm import Session
from app.models import models
from app.schemas import schemas
from app.core.security import get_password_hash

# ================= USER CRUD =================
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        hashed_password=hashed_password,
        role=user.role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# ================= PATIENT CRUD =================
def get_patient(db: Session, patient_id: int):
    return db.query(models.Patient).filter(models.Patient.id == patient_id).first()

def get_patient_by_code(db: Session, patient_code: str):
    return db.query(models.Patient).filter(models.Patient.patient_code == patient_code).first()

def get_patients(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Patient).offset(skip).limit(limit).all()

def create_patient(db: Session, patient: schemas.PatientCreate):
    db_patient = models.Patient(**patient.dict())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def update_patient(db: Session, patient_id: int, patient: schemas.PatientUpdate):
    db_patient = get_patient(db, patient_id)
    if not db_patient:
        return None
    for var, value in vars(patient).items():
        if value is not None:
            setattr(db_patient, var, value)
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient

def delete_patient(db: Session, patient_id: int):
    db_patient = get_patient(db, patient_id)
    if not db_patient:
        return None
    db.delete(db_patient)
    db.commit()
    return db_patient

# ================= PREDICTION CRUD =================
def get_prediction(db: Session, prediction_id: int):
    return db.query(models.Prediction).filter(models.Prediction.id == prediction_id).first()

def get_predictions(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Prediction).order_by(models.Prediction.created_at.desc()).offset(skip).limit(limit).all()

def get_predictions_by_patient(db: Session, patient_id: int):
    return db.query(models.Prediction).filter(models.Prediction.patient_id == patient_id).order_by(models.Prediction.created_at.desc()).all()

def create_prediction(db: Session, prediction: dict):
    db_prediction = models.Prediction(**prediction)
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

# ================= MODEL RESULTS CRUD =================
def create_model_result(db: Session, result: schemas.ModelResultCreate):
    db_result = models.ModelResult(**result.dict())
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    return db_result

def get_latest_model_results(db: Session):
    # Returns a list of the latest evaluation result for each unique model_name
    latest_ids = db.query(
        models.ModelResult.model_name,
        models.ModelResult.id
    ).order_by(models.ModelResult.created_at.desc()).all()
    
    # Filter to get only the unique, most recent runs
    seen = set()
    unique_ids = []
    for model_name, result_id in latest_ids:
        if model_name not in seen:
            seen.add(model_name)
            unique_ids.append(result_id)
            
    return db.query(models.ModelResult).filter(models.ModelResult.id.in_(unique_ids)).all()

def get_model_history(db: Session, model_name: str):
    return db.query(models.ModelResult).filter(models.ModelResult.model_name == model_name).order_by(models.ModelResult.created_at.desc()).all()

# ================= EXPERIMENT LOG CRUD =================
def create_experiment_log(db: Session, model_name: str, run_id: str, hyperparameters: dict, training_metrics: dict):
    db_log = models.ExperimentLog(
        model_name=model_name,
        run_id=run_id,
        hyperparameters=hyperparameters,
        training_metrics=training_metrics
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_experiment_logs(db: Session, limit: int = 10):
    return db.query(models.ExperimentLog).order_by(models.ExperimentLog.created_at.desc()).limit(limit).all()
