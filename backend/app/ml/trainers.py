import os
import time
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D

# Import our loaders and models
from app.ml.data_loader import E_Nose_DataLoader
from app.ml.models import build_baseline_dnn, build_hybrid_model, TemporalAttention

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'saved_models'
)
os.makedirs(MODELS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self, data_loader=None):
        self.data_loader = data_loader or E_Nose_DataLoader()
        
    def get_class_weights(self, weighted=True):
        """Returns 1:32 class weights for CaP as in reference paper, or uniform weights."""
        if weighted:
            return {0: 1.0, 1: 32.0}
        return {0: 1.0, 1: 1.0}

    def train_baseline_dnn(self, weighted=True, epochs=1280, batch_size=32):
        """Trains the reference paper baseline Dense Neural Network (FFNN)."""
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_tabular_data()
        
        model = build_baseline_dnn(input_dim=32)
        # Use SGD with momentum as noted in literature review
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(
            optimizer=opt, 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        
        class_weights = self.get_class_weights(weighted)
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'baseline_dnn.keras')
        model.save(model_path)
        
        # Save history
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        
        return model, history_dict, train_time

    def train_random_forest(self, weighted=True):
        """Trains Random Forest baseline."""
        X_train, y_train, _, _ = self.data_loader.get_processed_tabular_data()
        
        class_weight = 'balanced' if weighted else None
        rf = RandomForestClassifier(n_estimators=100, class_weight=class_weight, random_state=42, n_jobs=-1)
        
        start_time = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Save model
        rf_path = os.path.join(MODELS_DIR, 'random_forest.joblib')
        joblib.dump(rf, rf_path)
        
        return rf, train_time

    def train_xgboost(self, weighted=True):
        """Trains XGBoost baseline."""
        X_train, y_train, _, _ = self.data_loader.get_processed_tabular_data()
        
        # In XGBoost, scale_pos_weight is sum(negative instances) / sum(positive instances)
        # To simulate 1:32 weight, we set scale_pos_weight = 32.0 or balanced
        scale_pos_weight = 32.0 if weighted else 1.0
        
        xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight, 
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        xgb.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Save model
        xgb_path = os.path.join(MODELS_DIR, 'xgboost.joblib')
        joblib.dump(xgb, xgb_path)
        
        return xgb, train_time

    def train_hybrid_model(self, weighted=True, epochs=200, batch_size=16):
        """Trains the proposed Hybrid CNN-GRU-Attention model at the patient run level."""
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_sequence_data()
        
        model = build_hybrid_model(seq_len=32, feature_dim=31)
        # Use Adam for the Hybrid sequence model and pass None loss for attention weights output
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=['sparse_categorical_crossentropy', None],
            metrics=[['accuracy'], []]
        )
        
        class_weights = self.get_class_weights(weighted)
        
        start_time = time.time()
        # Train model (note: Keras fit handles class_weights on the label outputs)
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'hybrid_model.keras')
        model.save(model_path)
        
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['class_output_accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_class_output_accuracy']],
        }
        
        return model, history_dict, train_time

    def evaluate_model(self, model_name, model, is_sequence=False):
        """
        Computes clinical performance metrics.
        For sensor-level models (DNN, RF, XGB), evaluates on sensor-level test rows, 
        and also aggregates predictions to compute patient run-level metrics.
        For run-level model (Hybrid), evaluates on sequences.
        """
        if is_sequence:
            _, _, X_test, y_test = self.data_loader.get_processed_sequence_data()
            preds_probs, _ = model.predict(X_test, verbose=0)
            preds = np.argmax(preds_probs, axis=1)
            probs = preds_probs[:, 1]
            y_true = y_test
        else:
            _, _, X_test, y_test = self.data_loader.get_processed_tabular_data()
            
            # Get predictions
            if hasattr(model, 'predict_proba'):
                preds_probs = model.predict_proba(X_test)
                probs = preds_probs[:, 1]
                preds = model.predict(X_test)
            else:  # Keras DNN
                preds_probs = model.predict(X_test, verbose=0)
                probs = preds_probs[:, 1]
                preds = np.argmax(preds_probs, axis=1)
            y_true = y_test

        # Compute standard metrics
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5
            
        from sklearn.metrics import roc_curve
        try:
            fpr_arr, tpr_arr, _ = roc_curve(y_true, probs)
            # Downsample to 20 points
            indices = np.linspace(0, len(fpr_arr) - 1, 20, dtype=int)
            roc_points = [{"fpr": float(fpr_arr[i]), "tpr": float(tpr_arr[i])} for i in indices]
        except Exception:
            roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
            
        cm = confusion_matrix(y_true, preds).tolist()
        
        metrics = {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1),
            'roc_auc': float(auc),
            'confusion_matrix': cm,
            'roc_points': roc_points
        }
        
        # If it is a sensor-level model, aggregate predictions to run level
        if not is_sequence:
            # We have 12,800 sensor rows representing 400 runs (each 32 rows)
            n_runs = len(y_test) // 32
            run_y_true = y_test[::32]
            run_probs = np.mean(probs.reshape(n_runs, 32), axis=1)
            run_preds = (run_probs >= 0.5).astype(int)
            
            run_acc = accuracy_score(run_y_true, run_preds)
            run_prec = precision_score(run_y_true, run_preds, zero_division=0)
            run_rec = recall_score(run_y_true, run_preds, zero_division=0)
            run_f1 = f1_score(run_y_true, run_preds, zero_division=0)
            
            try:
                run_auc = roc_auc_score(run_y_true, run_probs)
            except Exception:
                run_auc = 0.5
                
            run_cm = confusion_matrix(run_y_true, run_preds).tolist()
            
            metrics['run_level'] = {
                'accuracy': float(run_acc),
                'precision': float(run_prec),
                'recall': float(run_rec),
                'f1_score': float(run_f1),
                'roc_auc': float(run_auc),
                'confusion_matrix': run_cm
            }
        else:
            # For native sequence models, sensor and run level are identical since they classify the run
            metrics['run_level'] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(auc),
                'confusion_matrix': cm
            }
            
        return metrics

    def run_ablation_study(self, epochs=50):
        """
        Runs an ablation study for the Proposed Hybrid model:
        1. Full Proposed Model (CNN + GRU + Attention)
        2. No Attention (CNN + GRU + Global Average Pooling)
        3. No GRU (CNN + Attention)
        4. No CNN (GRU + Attention)
        """
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_sequence_data()
        class_weights = self.get_class_weights(weighted=True)
        results = {}
        
        # Helper to compile and train a Keras model
        def train_ablation(model, name):
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=16,
                class_weight=class_weights,
                verbose=0
            )
            # Evaluate
            preds_probs = model.predict(X_test, verbose=0)
            preds = np.argmax(preds_probs, axis=1)
            probs = preds_probs[:, 1]
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            
            return {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(auc)
            }

        # 1. Full Model (We can build it with single output for simpler ablation fitting)
        inputs = Input(shape=(32, 31))
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=True)(x)
        context, _ = TemporalAttention()(x)
        x = Dense(32, activation='relu')(context)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        full_model = Model(inputs, out)
        results['Full Model'] = train_ablation(full_model, 'Full Model')
        
        # 2. No Attention (Global Average Pooling)
        inputs = Input(shape=(32, 31))
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = GRU(64, return_sequences=False)(x) # return_sequences=False gets the last state (effectively pooling)
        x = Dense(32, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        no_attn_model = Model(inputs, out)
        results['No Attention'] = train_ablation(no_attn_model, 'No Attention')

        # 3. No GRU (CNN + Attention)
        inputs = Input(shape=(32, 31))
        x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # Apply attention directly to Conv1D outputs
        context, _ = TemporalAttention()(x)
        x = Dense(32, activation='relu')(context)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        no_gru_model = Model(inputs, out)
        results['No GRU'] = train_ablation(no_gru_model, 'No GRU')

        # 4. No CNN (GRU + Attention)
        inputs = Input(shape=(32, 31))
        x = GRU(64, return_sequences=True)(inputs)
        context, _ = TemporalAttention()(x)
        x = Dense(32, activation='relu')(context)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        out = Dense(2, activation='softmax')(x)
        no_cnn_model = Model(inputs, out)
        results['No CNN'] = train_ablation(no_cnn_model, 'No CNN')
        
        return results
