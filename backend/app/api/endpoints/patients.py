from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from app.api.deps import get_db, get_current_user
from app.crud import crud
from app.schemas import schemas
from app.models import models

router = APIRouter()

@router.get("/", response_model=List[schemas.PatientResponse])
def get_patients_list(
    skip: int = 0, 
    limit: int = 100, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves all registered patient records."""
    return crud.get_patients(db, skip=skip, limit=limit)

@router.get("/{patient_id}", response_model=schemas.PatientResponse)
def get_patient_by_id(
    patient_id: int, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves a specific patient record by their database ID."""
    db_patient = crud.get_patient(db, patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Patient not found"
        )
    return db_patient

@router.post("/", response_model=schemas.PatientResponse, status_code=status.HTTP_201_CREATED)
def add_patient(
    patient_in: schemas.PatientCreate, 
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Registers a new patient profile in the system."""
    db_patient = crud.get_patient_by_code(db, patient_code=patient_in.patient_code)
    if db_patient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Patient code already registered."
        )
    return crud.create_patient(db, patient_in)

@router.put("/{patient_id}", response_model=schemas.PatientResponse)
def update_patient_profile(
    patient_id: int,
    patient_in: schemas.PatientUpdate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Updates an existing patient record's clinical parameters."""
    db_patient = crud.get_patient(db, patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Patient not found"
        )
    return crud.update_patient(db, patient_id, patient_in)

@router.delete("/{patient_id}", response_model=schemas.PatientResponse)
def delete_patient_record(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Deletes a patient record and all their historical predictions."""
    db_patient = crud.get_patient(db, patient_id)
    if not db_patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Patient not found"
        )
    return crud.delete_patient(db, patient_id)
