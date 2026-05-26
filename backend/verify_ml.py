import os
import sys

# Add backend dir to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
sys.path.append(os.path.dirname(__file__))

from app.ml.data_loader import E_Nose_DataLoader
from app.ml.trainers import ModelTrainer
from app.ml.explainers import ExplainabilityEngine

def main():
    print("=== STARTING ML/DL ENGINE VERIFICATION ===")
    
    print("\n1. Initializing Data Loader...")
    loader = E_Nose_DataLoader()
    X_train, y_train, X_test, y_test = loader.get_processed_tabular_data()
    X_train_seq, y_train_seq, X_test_seq, y_test_seq = loader.get_processed_sequence_data()
    print(f"Tabular - Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Sequence - Train shape: {X_train_seq.shape}, Test shape: {X_test_seq.shape}")

    trainer = ModelTrainer(loader)
    
    print("\n2. Training Random Forest Baseline...")
    rf_model, rf_time = trainer.train_random_forest(weighted=True)
    rf_metrics = trainer.evaluate_model('random_forest', rf_model, is_sequence=False)
    print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}, Recall: {rf_metrics['recall']:.4f}, AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"Random Forest Run Level - Accuracy: {rf_metrics['run_level']['accuracy']:.4f}, Recall: {rf_metrics['run_level']['recall']:.4f}")
    
    print("\n3. Training XGBoost Baseline...")
    xgb_model, xgb_time = trainer.train_xgboost(weighted=True)
    xgb_metrics = trainer.evaluate_model('xgboost', xgb_model, is_sequence=False)
    print(f"XGBoost - Accuracy: {xgb_metrics['accuracy']:.4f}, Recall: {xgb_metrics['recall']:.4f}, AUC: {xgb_metrics['roc_auc']:.4f}")
    print(f"XGBoost Run Level - Accuracy: {xgb_metrics['run_level']['accuracy']:.4f}, Recall: {xgb_metrics['run_level']['recall']:.4f}")
    
    print("\n4. Training Baseline DNN (Fast 2 epochs check)...")
    dnn_model, dnn_hist, dnn_time = trainer.train_baseline_dnn(weighted=True, epochs=2, batch_size=32)
    dnn_metrics = trainer.evaluate_model('baseline_dnn', dnn_model, is_sequence=False)
    print(f"Baseline DNN - Accuracy: {dnn_metrics['accuracy']:.4f}, Recall: {dnn_metrics['recall']:.4f}, AUC: {dnn_metrics['roc_auc']:.4f}")
    print(f"Baseline DNN Run Level - Accuracy: {dnn_metrics['run_level']['accuracy']:.4f}, Recall: {dnn_metrics['run_level']['recall']:.4f}")

    print("\n5. Training Proposed Hybrid CNN-GRU-Attention Model (Fast 2 epochs check)...")
    hybrid_model, hybrid_hist, hybrid_time = trainer.train_hybrid_model(weighted=True, epochs=2, batch_size=16)
    hybrid_metrics = trainer.evaluate_model('hybrid_model', hybrid_model, is_sequence=True)
    print(f"Hybrid Model - Accuracy: {hybrid_metrics['accuracy']:.4f}, Recall: {hybrid_metrics['recall']:.4f}, AUC: {hybrid_metrics['roc_auc']:.4f}")

    print("\n6. Running Ablation Study (Fast 1 epoch check)...")
    ablation_results = trainer.run_ablation_study(epochs=1)
    for model_variant, metrics in ablation_results.items():
        print(f"  Ablation [{model_variant}] - Acc: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, AUC: {metrics['roc_auc']:.4f}")

    print("\n7. Initializing Explainability Engine...")
    explainer = ExplainabilityEngine(loader)
    
    print("\n8. Explaining Tabular Instance (Random Forest)...")
    instance = X_test[0]
    rf_explanation = explainer.explain_tabular_instance('random_forest', instance)
    print("Random Forest Top Attributions:")
    # Print top 5 attributions
    sorted_rf_att = sorted(rf_explanation.items(), key=lambda item: abs(item[1]), reverse=True)
    for name, val in sorted_rf_att[:5]:
        print(f"  {name}: {val:+.6f}")
        
    print("\n9. Explaining Sequence Instance (Hybrid)...")
    seq_instance = X_test_seq[0]
    hybrid_explanation = explainer.explain_hybrid_sequence(seq_instance)
    print("Hybrid Model Predictions:")
    print(f"  HBP Probability: {hybrid_explanation['prediction']['HBP']:.4f}")
    print(f"  CaP Probability: {hybrid_explanation['prediction']['CaP']:.4f}")
    print("Hybrid Sensor Attention Weights (Top 5):")
    sorted_attn = sorted(list(enumerate(hybrid_explanation['attention_weights'])), key=lambda item: item[1], reverse=True)
    for idx, weight in sorted_attn[:5]:
        print(f"  Sensor S{idx+1}: {weight:.4f}")
    print("Hybrid Feature Importance (Top 5):")
    sorted_feat_imp = sorted(hybrid_explanation['feature_importance'].items(), key=lambda item: item[1], reverse=True)
    for name, val in sorted_feat_imp[:5]:
        print(f"  {name}: {val:.6f}")
        
    print("\n=== ALL ML/DL COMPONENT VERIFICATIONS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
