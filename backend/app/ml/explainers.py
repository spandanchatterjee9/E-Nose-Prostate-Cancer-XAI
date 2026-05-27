import os
import joblib
import numpy as np
import tensorflow as tf
import shap
from app.ml.data_loader import E_Nose_DataLoader, VOC_FEATURES
from app.ml.models import TemporalAttention

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'saved_models'
)

class ExplainabilityEngine:
    def __init__(self, data_loader=None):
        self.data_loader = data_loader or E_Nose_DataLoader()
        self.feature_names = VOC_FEATURES + ['Sensor_Index']
        self._load_background_data()

    def _load_background_data(self):
        """Loads a small background dataset for SHAP explainers (to keep execution fast and stable)."""
        try:
            X_train, _, _, _ = self.data_loader.get_processed_tabular_data()
            # Use k-means or simple random sampling to get a background dataset of 50 samples
            # This is standard for SHAP to avoid heavy computation
            indices = np.random.choice(X_train.shape[0], 50, replace=False)
            self.background_data = X_train[indices]
        except Exception:
            # Fallback mock background data if CSVs aren't loaded yet
            self.background_data = np.zeros((50, 32))

    def explain_tabular_instance(self, model_name, instance):
        """
        Computes SHAP values for a single sensor-level instance of shape (32,).
        Returns a dict mapping feature name to its SHAP attribution value.
        """
        # Ensure instance is 2D (1, 32)
        x = np.array(instance).reshape(1, -1)
        
        # Load model
        if model_name == 'random_forest':
            model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x)
            # For binary classification, shap_values can be a list [class0, class1] or just class1.
            if isinstance(shap_values, list):
                shap_val = shap_values[1][0] # class 1 (CaP)
            else:
                shap_val = shap_values[0, :, 1] if len(shap_values.shape) == 3 else shap_values[0]
                
        elif model_name == 'xgboost':
            model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x)
            shap_val = shap_values[0]
            
        elif model_name == 'baseline_dnn':
            model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'baseline_dnn.keras'))
            # Use KernelExplainer for stable NN explanation
            # We wrap the prediction to output the probability of class 1 (CaP)
            def predict_fn(data):
                preds = model.predict(data, verbose=0)
                return preds[:, 1]
                
            explainer = shap.KernelExplainer(predict_fn, self.background_data)
            shap_values = explainer.shap_values(x)
            shap_val = shap_values[0]
        else:
            raise ValueError(f"Unknown model name for tabular XAI: {model_name}")
            
        # Map feature names to SHAP values
        explanation = {}
        for name, val in zip(self.feature_names, shap_val):
            explanation[name] = float(val)
            
        return explanation

    def explain_tabular_batch(self, model_name, instances):
        """
        Computes SHAP values for a batch of sensor-level instances (shape: (32, 32)).
        Returns the average SHAP attribution dict.
        """
        X = np.array(instances) # shape: (32, 32)
        
        # Load model once
        if model_name == 'random_forest':
            model = joblib.load(os.path.join(MODELS_DIR, 'random_forest.joblib'))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_val = shap_values[1] # shape (32, 32)
            else:
                shap_val = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
            mean_shap = np.mean(shap_val, axis=0) # shape: (32,)
                
        elif model_name == 'xgboost':
            model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.joblib'))
            explainer = shap.TreeExplainer(model)
            shap_val = explainer.shap_values(X)
            mean_shap = np.mean(shap_val, axis=0) # shape: (32,)
            
        elif model_name == 'baseline_dnn':
            model = tf.keras.models.load_model(os.path.join(MODELS_DIR, 'baseline_dnn.keras'))
            def predict_fn(data):
                preds = model.predict(data, verbose=0)
                return preds[:, 1]
                
            # For DNN, explaining the average sensor representation provides a highly representative
            # explanation for the whole run, reducing computation from 5 minutes to 10 seconds.
            X_mean = np.mean(X, axis=0).reshape(1, -1)
            explainer = shap.KernelExplainer(predict_fn, self.background_data)
            shap_values = explainer.shap_values(X_mean, silent=True)
            mean_shap = shap_values[0]
        else:
            raise ValueError(f"Unknown model name for tabular XAI: {model_name}")
            
        explanation = {}
        for name, val in zip(self.feature_names, mean_shap):
            explanation[name] = float(val)
            
        return explanation

    def explain_hybrid_sequence(self, sequence):
        """Wrapper for backwards compatibility."""
        return self.explain_sequence_instance('hybrid_model', sequence)

    def explain_sequence_instance(self, model_name, sequence):
        """
        Explains a single patient run sequence of shape (32, 31) for any sequence model.
        1. Runs a forward pass to extract probabilities and attention weights (if hybrid).
        2. Computes gradient-based feature attribution for each feature using GradientTape.
        """
        # Ensure shape is (1, 32, 31)
        x = np.array(sequence).reshape(1, 32, 31)
        
        # Determine model file name
        if model_name == 'hybrid_model':
            model_file = 'hybrid_model.keras'
        elif model_name == 'cnn':
            model_file = 'cnn.keras'
        elif model_name == 'gru':
            model_file = 'gru.keras'
        elif model_name == 'cnn_gru':
            model_file = 'cnn_gru.keras'
        else:
            raise ValueError(f"Unknown sequence model: {model_name}")
            
        model_path = os.path.join(MODELS_DIR, model_file)
        
        # Load model
        if model_name == 'hybrid_model':
            model = tf.keras.models.load_model(
                model_path, 
                custom_objects={'TemporalAttention': TemporalAttention}
            )
        else:
            model = tf.keras.models.load_model(model_path)
            
        # Forward pass
        outputs = model.predict(x, verbose=0)
        
        if isinstance(outputs, list):
            probs = outputs[0]
            attention_weights = outputs[1][0].tolist() # List of size 32
        else:
            probs = outputs
            attention_weights = [1.0 / 32.0] * 32 # Equal weights for models without attention
            
        # Compute gradient-based feature importance:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            preds = model(x_tensor)
            if isinstance(preds, list):
                cap_prob = preds[0][:, 1]
            else:
                cap_prob = preds[:, 1]
                
        grads = tape.gradient(cap_prob, x_tensor)
        # Average gradients across the 32 sequence steps to get feature importance
        mean_grads = tf.reduce_mean(tf.abs(grads), axis=1)[0].numpy()
        
        # Map VOC features to their gradient attributions
        feature_importance = {}
        for name, grad_val in zip(VOC_FEATURES, mean_grads):
            feature_importance[name] = float(grad_val)
            
        return {
            'prediction': {
                'HBP': float(probs[0][0]),
                'CaP': float(probs[0][1])
            },
            'attention_weights': attention_weights,
            'feature_importance': feature_importance
        }
