import os
import time
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, BatchNormalization, Dropout, GlobalAveragePooling1D

# Import our loaders and models
from app.ml.data_loader import E_Nose_DataLoader
from app.ml.models import (
    build_baseline_dnn, 
    build_hybrid_model, 
    build_cnn_model, 
    build_gru_model, 
    build_cnn_gru_model, 
    TemporalAttention
)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'saved_models'
)
os.makedirs(MODELS_DIR, exist_ok=True)

PLOTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
    'static', 'plots'
)
os.makedirs(PLOTS_DIR, exist_ok=True)

class ModelTrainer:
    def __init__(self, data_loader=None):
        self.data_loader = data_loader or E_Nose_DataLoader()
        
    def get_class_weights(self, weighted=True):
        """Returns 1:32 class weights for CaP as in reference paper, or uniform weights."""
        if weighted:
            return {0: 1.0, 1: 32.0}
        return {0: 1.0, 1: 1.0}

    def train_baseline_dnn(self, weighted=True, epochs=150, batch_size=32):
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
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'baseline_dnn.keras')
        model.save(model_path)
        
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        
        self.save_loss_accuracy_plots(history_dict, 'baseline_dnn')
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

    def train_hybrid_model(self, weighted=True, epochs=80, batch_size=16):
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
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
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
        
        self.save_loss_accuracy_plots(history_dict, 'hybrid_model')
        return model, history_dict, train_time

    def train_cnn(self, weighted=True, epochs=80, batch_size=16):
        """Trains the baseline CNN model at the patient run level."""
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_sequence_data()
        
        model = build_cnn_model(seq_len=32, feature_dim=31)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        class_weights = self.get_class_weights(weighted)
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'cnn.keras')
        model.save(model_path)
        
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        
        self.save_loss_accuracy_plots(history_dict, 'cnn')
        return model, history_dict, train_time

    def train_gru(self, weighted=True, epochs=80, batch_size=16):
        """Trains the baseline GRU model at the patient run level."""
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_sequence_data()
        
        model = build_gru_model(seq_len=32, feature_dim=31)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        class_weights = self.get_class_weights(weighted)
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'gru.keras')
        model.save(model_path)
        
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        
        self.save_loss_accuracy_plots(history_dict, 'gru')
        return model, history_dict, train_time

    def train_cnn_gru(self, weighted=True, epochs=80, batch_size=16):
        """Trains the baseline CNN-GRU model at the patient run level."""
        X_train, y_train, X_test, y_test = self.data_loader.get_processed_sequence_data()
        
        model = build_cnn_gru_model(seq_len=32, feature_dim=31)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        class_weights = self.get_class_weights(weighted)
        
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=0)
        ]
        
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        train_time = time.time() - start_time
        
        # Save model
        model_path = os.path.join(MODELS_DIR, 'cnn_gru.keras')
        model.save(model_path)
        
        history_dict = {
            'loss': [float(x) for x in history.history['loss']],
            'accuracy': [float(x) for x in history.history['accuracy']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        }
        
        self.save_loss_accuracy_plots(history_dict, 'cnn_gru')
        return model, history_dict, train_time

    def evaluate_model(self, model_name, model, is_sequence=False):
        """
        Computes clinical performance metrics.
        For sensor-level models (DNN, RF, XGB), evaluates on sensor-level test rows, 
        and also aggregates predictions to compute patient run-level metrics.
        For run-level models (Hybrid, CNN, GRU, CNN-GRU), evaluates on sequences directly.
        """
        if is_sequence:
            _, _, X_test, y_test = self.data_loader.get_processed_sequence_data()
            outputs = model.predict(X_test, verbose=0)
            if isinstance(outputs, list):
                preds_probs = outputs[0]
                attn_weights = outputs[1]
                # Save global attention plot
                mean_attn = np.mean(attn_weights, axis=0).tolist()
                self.save_attention_plot(mean_attn, model_name)
            else:
                preds_probs = outputs
                attn_weights = None
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

            # If it has feature_importances_ (RF/XGB), generate feature importance plot
            if hasattr(model, 'feature_importances_'):
                from app.ml.data_loader import VOC_FEATURES
                feat_names = VOC_FEATURES + ['Sensor_Index']
                self.save_feature_importance_plot(model.feature_importances_, feat_names, model_name)

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
            self.save_roc_curve_plot(fpr_arr, tpr_arr, auc, model_name)
            # Downsample to 20 points
            indices = np.linspace(0, len(fpr_arr) - 1, 20, dtype=int)
            roc_points = [{"fpr": float(fpr_arr[i]), "tpr": float(tpr_arr[i])} for i in indices]
        except Exception:
            roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]
            
        cm = confusion_matrix(y_true, preds).tolist()
        self.save_confusion_matrix_plot(cm, model_name)
        
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

    def run_ablation_study(self, epochs=20):
        """
        Runs/loads evaluations for the 5 comparative architectures:
        1. Baseline DNN (Tabular)
        2. CNN (Sequence)
        3. GRU (Sequence)
        4. CNN-GRU (Sequence)
        5. CNN-GRU-Attention (Proposed Hybrid Model)
        
        Returns accuracy, precision, recall, f1, and roc_auc for each.
        """
        results = {}
        
        # Helper to load and evaluate a model
        def get_or_train_evaluation(model_name):
            if model_name == 'baseline_dnn':
                model_path = os.path.join(MODELS_DIR, 'baseline_dnn.keras')
                is_seq = False
                if not os.path.exists(model_path):
                    self.train_baseline_dnn(weighted=True, epochs=epochs)
                model = tf.keras.models.load_model(model_path)
            elif model_name == 'cnn':
                model_path = os.path.join(MODELS_DIR, 'cnn.keras')
                is_seq = True
                if not os.path.exists(model_path):
                    self.train_cnn(weighted=True, epochs=epochs)
                model = tf.keras.models.load_model(model_path)
            elif model_name == 'gru':
                model_path = os.path.join(MODELS_DIR, 'gru.keras')
                is_seq = True
                if not os.path.exists(model_path):
                    self.train_gru(weighted=True, epochs=epochs)
                model = tf.keras.models.load_model(model_path)
            elif model_name == 'cnn_gru':
                model_path = os.path.join(MODELS_DIR, 'cnn_gru.keras')
                is_seq = True
                if not os.path.exists(model_path):
                    self.train_cnn_gru(weighted=True, epochs=epochs)
                model = tf.keras.models.load_model(model_path)
            elif model_name == 'hybrid_model':
                model_path = os.path.join(MODELS_DIR, 'hybrid_model.keras')
                is_seq = True
                if not os.path.exists(model_path):
                    self.train_hybrid_model(weighted=True, epochs=epochs)
                model = tf.keras.models.load_model(
                    model_path, 
                    custom_objects={'TemporalAttention': TemporalAttention}
                )
            else:
                raise ValueError(f"Unknown ablation model: {model_name}")
                
            metrics = self.evaluate_model(model_name, model, is_sequence=is_seq)
            rl = metrics.get('run_level', metrics)
            return {
                'accuracy': float(rl['accuracy']),
                'precision': float(rl['precision']),
                'recall': float(rl['recall']),
                'f1_score': float(rl['f1_score']),
                'roc_auc': float(rl['roc_auc'])
            }

        # Names mapping for table
        results['Full Model'] = get_or_train_evaluation('hybrid_model')
        results['No Attention'] = get_or_train_evaluation('cnn_gru')
        results['No GRU'] = get_or_train_evaluation('cnn')
        results['No CNN'] = get_or_train_evaluation('gru')
        results['Baseline DNN'] = get_or_train_evaluation('baseline_dnn')
        
        return results

    # ================= PLOT SAVING FUNCTIONS =================

    def save_loss_accuracy_plots(self, history, model_name):
        """Generates and saves loss and accuracy curves for a model."""
        if not history or 'loss' not in history:
            return
        
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss', color='#22d3ee', linewidth=2)
        if 'val_loss' in history:
            plt.plot(history['val_loss'], label='Val Loss', color='#a78bfa', linewidth=1.5)
        plt.title(f'{model_name.replace("_", " ").upper()} Loss Convergence')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        if 'accuracy' in history:
            plt.plot(history['accuracy'], label='Train Acc', color='#34d399', linewidth=2)
        if 'val_accuracy' in history:
            plt.plot(history['val_accuracy'], label='Val Acc', color='#fb7185', linewidth=1.5)
        plt.title(f'{model_name.replace("_", " ").upper()} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(PLOTS_DIR, f'{model_name}_learning_curves.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()

    def save_confusion_matrix_plot(self, cm, model_name):
        """Saves confusion matrix heatmap plot."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['HBP (Benign)', 'CaP (Cancer)'], 
                    yticklabels=['HBP (Benign)', 'CaP (Cancer)'],
                    cbar=False, annot_kws={"size": 14, "weight": "bold"})
        plt.title(f'{model_name.replace("_", " ").upper()} Confusion Matrix')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'{model_name}_confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()

    def save_roc_curve_plot(self, fpr, tpr, auc_val, model_name):
        """Saves ROC curve plot."""
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#a78bfa', label=f'ROC (AUC = {auc_val:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], color='#64748b', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'{model_name.replace("_", " ").upper()} ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'{model_name}_roc.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()

    def save_feature_importance_plot(self, importances, feature_names, model_name):
        """Saves feature importance horizontal bar chart."""
        plt.figure(figsize=(8, 6))
        indices = np.argsort(importances)[::-1][:15] # Top 15 features
        top_importances = importances[indices]
        top_names = [feature_names[i] for i in indices]
        
        sns.barplot(x=top_importances, y=top_names, palette='viridis')
        plt.title(f'{model_name.replace("_", " ").upper()} Feature Importance (Top 15)')
        plt.xlabel('Relative Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'{model_name}_feature_importance.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()

    def save_attention_plot(self, attention_weights, model_name='hybrid_model'):
        """Saves E-Nose Attention weights bar chart."""
        plt.figure(figsize=(10, 4))
        sns.barplot(x=[f'S{i+1}' for i in range(len(attention_weights))], y=attention_weights, color='#a78bfa')
        plt.title('MOOSY-32 Sensor Array Attention Coefficients')
        plt.xlabel('Sensors')
        plt.ylabel('Attention Score')
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, f'{model_name}_attention.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()

    def save_model_comparison_plot(self, metrics_list):
        """Saves a bar chart comparing performance across all models."""
        plt.figure(figsize=(12, 6))
        
        # Prepare data for plotting
        models_names = []
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        aucs = []
        
        for m in metrics_list:
            models_names.append(m.model_name.replace('_', ' ').capitalize())
            accuracies.append(m.accuracy)
            precisions.append(m.precision)
            recalls.append(m.recall)
            f1_scores.append(m.f1_score)
            aucs.append(m.roc_auc)
            
        x = np.arange(len(models_names))
        width = 0.15
        
        plt.bar(x - 2*width, accuracies, width, label='Accuracy', color='#3b82f6')
        plt.bar(x - width, precisions, width, label='Precision', color='#10b981')
        plt.bar(x, recalls, width, label='Recall (Sens.)', color='#ec4899')
        plt.bar(x + width, f1_scores, width, label='F1-Score', color='#8b5cf6')
        plt.bar(x + 2*width, aucs, width, label='ROC-AUC', color='#f59e0b')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Empirical Model Performance Comparison')
        plt.xticks(x, models_names, rotation=15)
        plt.legend(loc='lower left')
        plt.grid(True, linestyle='--', alpha=0.3, axis='y')
        plt.ylim(0, 1.1)
        plt.tight_layout()
        
        plot_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
        plt.savefig(plot_path, dpi=150, facecolor='white', bbox_inches='tight')
        plt.close()
