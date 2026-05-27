import uuid
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.api.deps import get_db, get_current_user
from app.crud import crud
from app.schemas import schemas
from app.models import models
from app.ml.trainers import ModelTrainer
from app.ml.data_loader import E_Nose_DataLoader

router = APIRouter()

def run_background_training(model_name: str, weighted: bool, run_id: str, db_session_maker):
    """
    Background task to train models without blocking the API response.
    Saves metrics to model_results and histories to experiment_logs.
    """
    db = db_session_maker()
    try:
        loader = E_Nose_DataLoader()
        trainer = ModelTrainer(loader)
        
        print(f"Background training started for: {model_name} (run_id: {run_id})")
        
        if model_name == 'random_forest':
            model, train_time = trainer.train_random_forest(weighted=weighted)
            metrics = trainer.evaluate_model('random_forest', model, is_sequence=False)
            hist = {'train_time': train_time}
            
        elif model_name == 'xgboost':
            model, train_time = trainer.train_xgboost(weighted=weighted)
            metrics = trainer.evaluate_model('xgboost', model, is_sequence=False)
            hist = {'train_time': train_time}
            
        elif model_name == 'baseline_dnn':
            # Train for 150 epochs in background (fast enough but yields decent convergence)
            model, hist, train_time = trainer.train_baseline_dnn(weighted=weighted, epochs=150, batch_size=32)
            metrics = trainer.evaluate_model('baseline_dnn', model, is_sequence=False)
            hist['train_time'] = train_time
            
        elif model_name == 'hybrid_model':
            # Train for 80 epochs in background
            model, hist, train_time = trainer.train_hybrid_model(weighted=weighted, epochs=80, batch_size=16)
            metrics = trainer.evaluate_model('hybrid_model', model, is_sequence=True)
            hist['train_time'] = train_time
            
        elif model_name == 'cnn':
            # Train for 80 epochs in background
            model, hist, train_time = trainer.train_cnn(weighted=weighted, epochs=80, batch_size=16)
            metrics = trainer.evaluate_model('cnn', model, is_sequence=True)
            hist['train_time'] = train_time
            
        elif model_name == 'gru':
            # Train for 80 epochs in background
            model, hist, train_time = trainer.train_gru(weighted=weighted, epochs=80, batch_size=16)
            metrics = trainer.evaluate_model('gru', model, is_sequence=True)
            hist['train_time'] = train_time
            
        elif model_name == 'cnn_gru':
            # Train for 80 epochs in background
            model, hist, train_time = trainer.train_cnn_gru(weighted=weighted, epochs=80, batch_size=16)
            metrics = trainer.evaluate_model('cnn_gru', model, is_sequence=True)
            hist['train_time'] = train_time
            
        else:
            print(f"Unknown model name for training: {model_name}")
            return
            
        # Log results to database
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name=model_name,
            run_id=run_id,
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            roc_auc=metrics['roc_auc'],
            confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        
        # Log hyperparameters and training histories
        hyperparams = {
            'weighted': weighted,
            'epochs': 150 if model_name == 'baseline_dnn' else 80,
            'batch_size': 32 if model_name == 'baseline_dnn' else 16
        }
        crud.create_experiment_log(db, model_name, run_id, hyperparams, hist)
        print(f"Background training finished successfully for: {model_name} (run_id: {run_id})")
        
    except Exception as e:
        print(f"Error during background training of {model_name}: {str(e)}")
    finally:
        db.close()


@router.post("/train/{model_name}", status_code=status.HTTP_202_ACCEPTED)
def trigger_model_training(
    model_name: str,
    background_tasks: BackgroundTasks,
    weighted: bool = True,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Asynchronously triggers re-training of a specific model (dnn, rf, xgb, cnn, gru, cnn_gru, hybrid).
    Returns immediately with a run_id while training progresses in the background.
    """
    if model_name not in ['baseline_dnn', 'random_forest', 'xgboost', 'hybrid_model', 'cnn', 'gru', 'cnn_gru']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Invalid model name: {model_name}"
        )
        
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    # We pass the SessionLocal generator to open a new DB connection inside the background thread
    from app.core.database import SessionLocal
    background_tasks.add_task(
        run_background_training, 
        model_name, 
        weighted, 
        run_id, 
        SessionLocal
    )
    
    return {
        "message": f"Training session initiated for {model_name} in background.",
        "run_id": run_id,
        "status": "training"
    }


@router.get("/benchmarks", response_model=List[schemas.ModelResultResponse])
def get_model_benchmarks(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves the latest evaluation metrics (accuracy, recall, AUC, confusion matrix) for all models."""
    # Ensure standard models are preloaded in saved_models directory and mapped in DB
    from app.api.endpoints.predict import ensure_models_trained
    ensure_models_trained(db)
    
    results = crud.get_latest_model_results(db)
    
    # Generate and save comparative plot dynamically based on latest metrics
    try:
        trainer = ModelTrainer()
        trainer.save_model_comparison_plot(results)
    except Exception as e:
        print(f"Error generating comparison plot: {e}")
        
    return results


@router.get("/ablation")
def get_ablation_metrics(
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Runs a fast ablation study (20 epochs) and returns performance differences."""
    loader = E_Nose_DataLoader()
    trainer = ModelTrainer(loader)
    
    print("Running on-demand ablation study...")
    # Run a fast evaluation ablation (20 epochs) to keep response time reasonable
    results = trainer.run_ablation_study(epochs=20)
    return results


@router.get("/experiment-logs", response_model=List[schemas.ExperimentLogResponse])
def get_training_logs(
    limit: int = 10,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Retrieves hyperparameter logs and training histories (epoch-by-epoch losses) for visualizations."""
    return crud.get_experiment_logs(db, limit=limit)
