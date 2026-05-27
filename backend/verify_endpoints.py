import os
import sys
from fastapi.testclient import TestClient

# Add backend dir to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))
sys.path.append(os.path.dirname(__file__))

from app.main import app
from app.core.database import Base, engine

client = TestClient(app)

def main():
    print("=== STARTING API ENDPOINT INTEGRATION VERIFICATION ===")
    
    # 1. Base check
    print("\n1. Querying Root Endpoint...")
    res = client.get("/")
    assert res.status_code == 200
    print("  Root returns:", res.json())

    # 2. Login check with default seeded user
    print("\n2. Testing Authenticated Login (Default clinician)...")
    res = client.post(
        "/api/v1/auth/login",
        data={"username": "admin", "password": "admin123"}
    )
    assert res.status_code == 200
    login_data = res.json()
    token = login_data["access_token"]
    print("  Login Successful! Token issued.")
    headers = {"Authorization": f"Bearer {token}"}

    # 3. Create Patient
    print("\n3. Testing Patient Registration...")
    patient_payload = {
        "patient_code": "TEST-PATIENT-99",
        "age": 70,
        "psa": 6.8,
        "volume": 45.0,
        "clinical_notes": "Integration test patient profile."
    }
    # Check if patient exists, if so delete first to ensure idempotency
    get_res = client.get("/api/v1/patients/", headers=headers)
    for p in get_res.json():
        if p["patient_code"] == "TEST-PATIENT-99":
            client.delete(f"/api/v1/patients/{p['id']}", headers=headers)
            print("  Cleaned up old test patient record.")
            
    res = client.post("/api/v1/patients/", json=patient_payload, headers=headers)
    assert res.status_code == 201
    patient_data = res.json()
    patient_id = patient_data["id"]
    print(f"  Patient Registered successfully! ID: {patient_id}")

    # 4. Predict using simulated test run
    print("\n4. Testing Prediction Inference & XAI Attributions (DNN Model)...")
    predict_payload = {
        "patient_id": patient_id,
        "model_name": "baseline_dnn",
        "simulated_run_index": 0
    }
    res = client.post("/api/v1/predict/", json=predict_payload, headers=headers)
    assert res.status_code == 200
    pred_data = res.json()
    print(f"  Prediction Completed!")
    print(f"    Diagnosis: {pred_data['prediction_label']}")
    print(f"    Confidence: {pred_data['confidence'] * 100:.2f}%")
    print(f"    SHAP Attributions Count: {len(pred_data['shap_values'] or {})}")

    # 5. Predict using Hybrid model
    print("\n5. Testing Proposed Hybrid CNN-GRU-Attention Prediction...")
    predict_payload_hybrid = {
        "patient_id": patient_id,
        "model_name": "hybrid_model",
        "simulated_run_index": 0
      }
    res = client.post("/api/v1/predict/", json=predict_payload_hybrid, headers=headers)
    assert res.status_code == 200
    pred_data_hybrid = res.json()
    print(f"  Hybrid Prediction Completed!")
    print(f"    Diagnosis: {pred_data_hybrid['prediction_label']}")
    print(f"    Confidence: {pred_data_hybrid['confidence'] * 100:.2f}%")
    print(f"    Attention Weights Count: {len(pred_data_hybrid['attention_weights'] or [])}")
    print(f"    Saliency Feature Importance Count: {len(pred_data_hybrid['feature_importance'] or {})}")

    # 5b. Predict using CNN Sequence Model
    print("\n5b. Testing CNN Sequence Prediction...")
    predict_payload_cnn = {
        "patient_id": patient_id,
        "model_name": "cnn",
        "simulated_run_index": 0
    }
    res = client.post("/api/v1/predict/", json=predict_payload_cnn, headers=headers)
    assert res.status_code == 200
    pred_data_cnn = res.json()
    print(f"  CNN Prediction Completed!")
    print(f"    Diagnosis: {pred_data_cnn['prediction_label']}")
    print(f"    Confidence: {pred_data_cnn['confidence'] * 100:.2f}%")
    print(f"    Saliency Feature Importance Count: {len(pred_data_cnn['feature_importance'] or {})}")

    # 6. Fetch Prediction History
    print("\n6. Testing Prediction History Query...")
    res = client.get("/api/v1/history/", headers=headers)
    assert res.status_code == 200
    history_list = res.json()
    print(f"  History items count: {len(history_list)}")

    # 7. Fetch Benchmarks
    print("\n7. Testing Benchmarking Metrics...")
    res = client.get("/api/v1/metrics/benchmarks", headers=headers)
    assert res.status_code == 200
    benchmarks_list = res.json()
    print(f"  Benchmark metrics calculated for {len(benchmarks_list)} models:")
    for b in benchmarks_list:
      print(f"    {b['model_name']}: Acc={b['accuracy']:.4f}, AUC={b['roc_auc']:.4f}")

    # 8. Clean up
    print("\n8. Cleaning up database records...")
    del_res = client.delete(f"/api/v1/patients/{patient_id}", headers=headers)
    assert del_res.status_code == 200
    print("  Test patient record and cascade predictions cleaned successfully.")

    print("\n=== ALL ENDPOINT INTEGRATION VERIFICATIONS COMPLETED SUCCESSFULLY ===")

if __name__ == '__main__':
    main()
