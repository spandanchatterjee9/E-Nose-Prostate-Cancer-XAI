import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.api.deps import get_db, get_current_user
from app.crud import crud
from app.schemas import schemas
from app.models import models
from app.ml.data_loader import E_Nose_DataLoader, VOC_FEATURES, get_sensor_index
from app.ml.explainers import ExplainabilityEngine
from app.ml.trainers import ModelTrainer

router = APIRouter()

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
    'saved_models'
)

def ensure_models_trained(db: Session):
    """Ensures that all models are trained and present in the saved_models folder on request."""
    loader = E_Nose_DataLoader()
    trainer = ModelTrainer(loader)
    
    # 1. Random Forest
    rf_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
    db_rf = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'random_forest').first()
    
    if not os.path.exists(rf_path):
        print("Pre-training Random Forest for active inference...")
        model, _ = trainer.train_random_forest(weighted=True)
        metrics = trainer.evaluate_model('random_forest', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='random_forest', run_id='init_rf',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
    elif not db_rf:
        print("Model file exists but DB result missing. Seeding Random Forest benchmark metrics...")
        model = joblib.load(rf_path)
        metrics = trainer.evaluate_model('random_forest', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='random_forest', run_id='init_rf',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 2. XGBoost
    xgb_path = os.path.join(MODELS_DIR, 'xgboost.joblib')
    db_xgb = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'xgboost').first()
    
    if not os.path.exists(xgb_path):
        print("Pre-training XGBoost for active inference...")
        model, _ = trainer.train_xgboost(weighted=True)
        metrics = trainer.evaluate_model('xgboost', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='xgboost', run_id='init_xgb',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
    elif not db_xgb:
        print("Model file exists but DB result missing. Seeding XGBoost benchmark metrics...")
        model = joblib.load(xgb_path)
        metrics = trainer.evaluate_model('xgboost', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='xgboost', run_id='init_xgb',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 3. Baseline DNN
    dnn_path = os.path.join(MODELS_DIR, 'baseline_dnn.keras')
    db_dnn = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'baseline_dnn').first()
    
    if not os.path.exists(dnn_path):
        print("Pre-training Baseline DNN (fast 10 epochs) for active inference...")
        model, hist, _ = trainer.train_baseline_dnn(weighted=True, epochs=10, batch_size=32)
        metrics = trainer.evaluate_model('baseline_dnn', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='baseline_dnn', run_id='init_dnn',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        crud.create_experiment_log(db, 'baseline_dnn', 'init_dnn', {'epochs': 10, 'batch_size': 32}, hist)
    elif not db_dnn:
        print("Model file exists but DB result missing. Seeding Baseline DNN benchmark metrics...")
        model = tf.keras.models.load_model(dnn_path)
        metrics = trainer.evaluate_model('baseline_dnn', model, is_sequence=False)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='baseline_dnn', run_id='init_dnn',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 4. Hybrid CNN-GRU-Attention
    hybrid_path = os.path.join(MODELS_DIR, 'hybrid_model.keras')
    db_hybrid = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'hybrid_model').first()
    
    if not os.path.exists(hybrid_path):
        print("Pre-training Hybrid model (fast 10 epochs) for active inference...")
        model, hist, _ = trainer.train_hybrid_model(weighted=True, epochs=10, batch_size=16)
        metrics = trainer.evaluate_model('hybrid_model', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='hybrid_model', run_id='init_hybrid',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        crud.create_experiment_log(db, 'hybrid_model', 'init_hybrid', {'epochs': 10, 'batch_size': 16}, hist)
    elif not db_hybrid:
        print("Model file exists but DB result missing. Seeding Hybrid Model benchmark metrics...")
        from app.ml.models import TemporalAttention
        model = tf.keras.models.load_model(hybrid_path, custom_objects={'TemporalAttention': TemporalAttention})
        metrics = trainer.evaluate_model('hybrid_model', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='hybrid_model', run_id='init_hybrid',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 5. CNN Baseline
    cnn_path = os.path.join(MODELS_DIR, 'cnn.keras')
    db_cnn = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'cnn').first()
    
    if not os.path.exists(cnn_path):
        print("Pre-training CNN model (fast 10 epochs) for active inference...")
        model, hist, _ = trainer.train_cnn(weighted=True, epochs=10, batch_size=16)
        metrics = trainer.evaluate_model('cnn', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='cnn', run_id='init_cnn',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        crud.create_experiment_log(db, 'cnn', 'init_cnn', {'epochs': 10, 'batch_size': 16}, hist)
    elif not db_cnn:
        print("Model file exists but DB result missing. Seeding CNN benchmark metrics...")
        model = tf.keras.models.load_model(cnn_path)
        metrics = trainer.evaluate_model('cnn', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='cnn', run_id='init_cnn',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 6. GRU Baseline
    gru_path = os.path.join(MODELS_DIR, 'gru.keras')
    db_gru = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'gru').first()
    
    if not os.path.exists(gru_path):
        print("Pre-training GRU model (fast 10 epochs) for active inference...")
        model, hist, _ = trainer.train_gru(weighted=True, epochs=10, batch_size=16)
        metrics = trainer.evaluate_model('gru', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='gru', run_id='init_gru',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        crud.create_experiment_log(db, 'gru', 'init_gru', {'epochs': 10, 'batch_size': 16}, hist)
    elif not db_gru:
        print("Model file exists but DB result missing. Seeding GRU benchmark metrics...")
        model = tf.keras.models.load_model(gru_path)
        metrics = trainer.evaluate_model('gru', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='gru', run_id='init_gru',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))

    # 7. CNN-GRU Baseline
    cnn_gru_path = os.path.join(MODELS_DIR, 'cnn_gru.keras')
    db_cnn_gru = db.query(models.ModelResult).filter(models.ModelResult.model_name == 'cnn_gru').first()
    
    if not os.path.exists(cnn_gru_path):
        print("Pre-training CNN-GRU model (fast 10 epochs) for active inference...")
        model, hist, _ = trainer.train_cnn_gru(weighted=True, epochs=10, batch_size=16)
        metrics = trainer.evaluate_model('cnn_gru', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='cnn_gru', run_id='init_cnn_gru',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))
        crud.create_experiment_log(db, 'cnn_gru', 'init_cnn_gru', {'epochs': 10, 'batch_size': 16}, hist)
    elif not db_cnn_gru:
        print("Model file exists but DB result missing. Seeding CNN-GRU benchmark metrics...")
        model = tf.keras.models.load_model(cnn_gru_path)
        metrics = trainer.evaluate_model('cnn_gru', model, is_sequence=True)
        crud.create_model_result(db, schemas.ModelResultCreate(
            model_name='cnn_gru', run_id='init_cnn_gru',
            accuracy=metrics['accuracy'], precision=metrics['precision'], recall=metrics['recall'],
            f1_score=metrics['f1_score'], roc_auc=metrics['roc_auc'], confusion_matrix=metrics['confusion_matrix'],
            classification_report={'roc_points': metrics['roc_points']}
        ))


@router.post("/", response_model=schemas.PredictionDetailResponse)
def run_model_inference(
    req: schemas.PredictionRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Runs clinical E-Nose VOC diagnosis using the chosen model, computes explainability metrics
    and logs the patient prediction in the history database.
    """
    # 1. Fetch Patient
    patient = crud.get_patient(db, req.patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient record not found")
        
    # 2. Make sure models exist
    ensure_models_trained(db)
    
    loader = E_Nose_DataLoader()
    scaler = loader.scaler
    # Fit scaler using training set
    df_train, df_test = loader.load_raw_data()
    scaler.fit(df_train[VOC_FEATURES])
    
    # 3. Retrieve VOC data (from request payload or simulated dataset run)
    raw_sensor_data = []
    if req.simulated_run_index is not None:
        # Load run index from test set (dataset_prostate1.csv)
        idx = req.simulated_run_index
        # Validate index bounds
        max_idx = len(df_test) // 32
        if idx < 0 or idx >= max_idx:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"Simulated run index out of bounds. Must be 0 to {max_idx-1}"
            )
        # Fetch 32 rows representing this run
        df_run = df_test.iloc[idx * 32 : (idx + 1) * 32]
        for _, row in df_run.iterrows():
            rec_dict = row[VOC_FEATURES].to_dict()
            rec_dict['Sensor'] = row['Sensor']
            raw_sensor_data.append(schemas.SensorRecord(**rec_dict))
    elif req.sensor_data is not None:
        if len(req.sensor_data) != 32:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="A clinical E-Nose VOC session requires exactly 32 sensor readings."
            )
        raw_sensor_data = req.sensor_data
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Must provide either sensor_data array or simulated_run_index."
        )

    # 4. Preprocess input for models
    # Convert SensorRecord schema list to pandas DataFrame
    input_records = [r.dict() for r in raw_sensor_data]
    df_input = pd.DataFrame(input_records)
    
    # Scale VOC features
    scaled_voc = scaler.transform(df_input[VOC_FEATURES])
    
    # Encode sensor identifier index
    sensor_indices = df_input['Sensor'].apply(get_sensor_index).values.reshape(-1, 1)
    
    # Combined tabular features (32,) for RF/XGB/DNN
    X_tab = np.hstack([scaled_voc, sensor_indices])
    
    # Combined sequence feature (1, 32, 31) for Hybrid
    X_seq = scaled_voc.reshape(1, 32, 31)
    
    # 5. Initialize Explainers
    explainer_engine = ExplainabilityEngine(loader)
    
    prediction_label = "HBP"
    confidence = 0.5
    shap_val_dict = None
    attention_weights = None
    feature_importance = None
    
    # 6. Run Inference and Attributions
    if req.model_name in ['random_forest', 'xgboost', 'baseline_dnn']:
        # Load tabular model
        if req.model_name == 'random_forest':
            model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
            probs = model.predict_proba(X_tab)[:, 1]  # Prob of CaP for each of the 32 rows
        elif req.model_name == 'xgboost':
            model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
            probs = model.predict_proba(X_tab)[:, 1]
        else:  # baseline_dnn
            model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'baseline_dnn.keras'))
            probs = model.predict(X_tab, verbose=0)[:, 1]
            
        # Aggregate sensor predictions to get run-level prediction
        avg_prob_cap = float(np.mean(probs))
        if avg_prob_cap >= 0.5:
            prediction_label = "CaP"
            confidence = avg_prob_cap
        else:
            prediction_label = "HBP"
            confidence = 1.0 - avg_prob_cap
            
        # Compute SHAP values for the full sensor array in one optimized batch
        shap_val_dict = explainer_engine.explain_tabular_batch(req.model_name, X_tab)
            
    elif req.model_name in ['hybrid_model', 'cnn', 'gru', 'cnn_gru']:
        # Explain and predict Sequence model
        explanation = explainer_engine.explain_sequence_instance(req.model_name, X_seq)
        
        prob_cap = explanation['prediction']['CaP']
        prob_hbp = explanation['prediction']['HBP']
        
        if prob_cap >= 0.5:
            prediction_label = "CaP"
            confidence = prob_cap
        else:
            prediction_label = "HBP"
            confidence = prob_hbp
            
        attention_weights = explanation['attention_weights']
        feature_importance = explanation['feature_importance']
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Unknown model name: {req.model_name}"
        )
        
    # 7. Write to database
    db_pred_dict = {
        "patient_id": req.patient_id,
        "model_name": req.model_name,
        "features": input_records,
        "prediction_label": prediction_label,
        "confidence": confidence,
        "shap_values": {
            "shap_attributions": shap_val_dict,
            "attention_weights": attention_weights,
            "feature_importance": feature_importance
        }
    }
    
    db_pred = crud.create_prediction(db, db_pred_dict)
    
    # 8. Return response
    return schemas.PredictionDetailResponse(
        id=db_pred.id,
        patient_id=db_pred.patient_id,
        patient_code=patient.patient_code,
        model_name=db_pred.model_name,
        prediction_label=db_pred.prediction_label,
        confidence=db_pred.confidence,
        features=db_pred.features,
        shap_values=shap_val_dict,
        attention_weights=attention_weights,
        feature_importance=feature_importance,
        created_at=db_pred.created_at
    )
