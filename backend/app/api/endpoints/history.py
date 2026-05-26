from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.deps import get_db, get_current_user
from app.crud import crud
from app.schemas import schemas
from app.models import models

router = APIRouter()

@router.get("/", response_model=List[schemas.PredictionResponse])
def get_prediction_history(
    patient_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves all past predictions, optionally filtered by patient database ID."""
    if patient_id:
        predictions = crud.get_predictions_by_patient(db, patient_id)
        # Apply skip/limit slice manually on returned list
        return predictions[skip : skip + limit]
    return crud.get_predictions(db, skip=skip, limit=limit)

@router.get("/{prediction_id}", response_model=schemas.PredictionDetailResponse)
def get_prediction_details(
    prediction_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves the full inputs, outputs, SHAP, and Attention attributions of a past prediction."""
    db_pred = crud.get_prediction(db, prediction_id)
    if not db_pred:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Prediction record not found"
        )
        
    patient = crud.get_patient(db, db_pred.patient_id)
    patient_code = patient.patient_code if patient else "UNKNOWN"
    
    # Extract saved SHAP and attention fields from JSON column
    saved_xai = db_pred.shap_values or {}
    shap_vals = saved_xai.get("shap_attributions")
    attn_weights = saved_xai.get("attention_weights")
    feat_imp = saved_xai.get("feature_importance")
    
    return schemas.PredictionDetailResponse(
        id=db_pred.id,
        patient_id=db_pred.patient_id,
        patient_code=patient_code,
        model_name=db_pred.model_name,
        prediction_label=db_pred.prediction_label,
        confidence=db_pred.confidence,
        features=db_pred.features,
        shap_values=shap_vals,
        attention_weights=attn_weights,
        feature_importance=feat_imp,
        created_at=db_pred.created_at
    )
