import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Define constant feature names
VOC_FEATURES = [
    'el75', 'std', 'moda', 'media', 'mediana', 'iqr', 'cv', 
    'V40', 'V60', 'Vmax', 'V100', 'V120', 
    'difBA', 'difBC', 'difBD', 'difBE', 
    'slopeAB', 'slopeBC', 'slopeAD', 'slopeDE', 'slopeEC', 'slopeBE', 'slopeDB', 
    'asimetria', 
    'Met', 'IsoB', 'Prop', 'Hidro', 'Etan', 'CO', 'Air'
]

# Map sensor strings to integer indexes (0-31)
SENSOR_MAP = {
    f'S{i}': i - 1 for i in range(1, 33)
}

def get_sensor_index(sensor_name):
    # Sensor names are like 'S1-TGS2611-c00'
    prefix = sensor_name.split('-')[0]
    return SENSOR_MAP.get(prefix, 0)

class E_Nose_DataLoader:
    def __init__(self, data_dir=None):
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = data_dir or os.getenv("DATA_DIR", repo_root)
        self.train_path = os.path.join(self.data_dir, 'dataset_prostate.csv')
        self.test_path = os.path.join(self.data_dir, 'dataset_prostate1.csv')
        
        self.scaler = StandardScaler()
        self.sensor_mapping = {}
        
    def load_raw_data(self):
        """Loads train and test CSVs raw."""
        if not os.path.exists(self.train_path) or not os.path.exists(self.test_path):
            raise FileNotFoundError("Training or testing dataset CSV not found in data directory.")
            
        df_train = pd.read_csv(self.train_path)
        df_test = pd.read_csv(self.test_path)
        return df_train, df_test

    def get_processed_tabular_data(self):
        """
        Prepares 2D data for sensor-level models (DNN, RF, XGBoost).
        X contains 31 scaled VOC features + 1 sensor index (total 32 columns).
        y contains the target labels.
        """
        df_train, df_test = self.load_raw_data()
        
        # Clean nulls/infs and outliers in training and testing data
        for col in VOC_FEATURES:
            # Replace infs with NaN
            df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
            df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
            
            # Impute NaNs with training median
            median_val = df_train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_train[col] = df_train[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
            
            # Pre-clip extreme outliers to prevent floating point overflow in scaling
            df_train[col] = df_train[col].clip(-10000.0, 10000.0)
            df_test[col] = df_test[col].clip(-10000.0, 10000.0)
            
            # Winsorize: Clip to 1st and 99th percentiles of training set
            q_low = df_train[col].quantile(0.01)
            q_high = df_train[col].quantile(0.99)
            df_train[col] = df_train[col].clip(q_low, q_high)
            df_test[col] = df_test[col].clip(q_low, q_high)

        # Fit scaler on training set VOC features only
        self.scaler.fit(df_train[VOC_FEATURES])
        
        # Transform VOC features
        X_train_voc = self.scaler.transform(df_train[VOC_FEATURES])
        X_test_voc = self.scaler.transform(df_test[VOC_FEATURES])
        
        # Encode sensor labels as numeric index
        train_sensor_idx = df_train['Sensor'].apply(get_sensor_index).values.reshape(-1, 1)
        test_sensor_idx = df_test['Sensor'].apply(get_sensor_index).values.reshape(-1, 1)
        
        # Combine scaled VOC features and sensor index to create 32-dim inputs
        X_train = np.hstack([X_train_voc, train_sensor_idx])
        X_test = np.hstack([X_test_voc, test_sensor_idx])
        
        y_train = df_train['target'].values
        y_test = df_test['target'].values
        
        return X_train, y_train, X_test, y_test

    def get_processed_sequence_data(self):
        """
        Prepares 3D sequence data for patient run-level models (CNN-GRU-Attention).
        Each clinical acquisition run has 32 sensors. We reshape the data to:
        X shape: (N_runs, 32_sensors, 31_features)
        y shape: (N_runs,)
        """
        df_train, df_test = self.load_raw_data()
        
        # Clean nulls/infs and outliers in training and testing data
        for col in VOC_FEATURES:
            # Replace infs with NaN
            df_train[col] = df_train[col].replace([np.inf, -np.inf], np.nan)
            df_test[col] = df_test[col].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaNs with training median
            # Fill NaNs with training median
            median_val = df_train[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df_train[col] = df_train[col].fillna(median_val)
            df_test[col] = df_test[col].fillna(median_val)
            
            # Pre-clip extreme outliers to prevent floating point overflow in scaling
            df_train[col] = df_train[col].clip(-10000.0, 10000.0)
            df_test[col] = df_test[col].clip(-10000.0, 10000.0)
            
            # Winsorize: Clip to 1st and 99th percentiles of training set
            q_low = df_train[col].quantile(0.01)
            q_high = df_train[col].quantile(0.99)
            df_train[col] = df_train[col].clip(q_low, q_high)
            df_test[col] = df_test[col].clip(q_low, q_high)

        # Scale features
        self.scaler.fit(df_train[VOC_FEATURES])
        X_train_voc = self.scaler.transform(df_train[VOC_FEATURES])
        X_test_voc = self.scaler.transform(df_test[VOC_FEATURES])
        
        # Ensure data is sorted by patient run blocks
        # (each block of 32 rows has the same target and sequential sensors S1-S32)
        n_train_runs = len(df_train) // 32
        n_test_runs = len(df_test) // 32
        
        X_train_seq = X_train_voc.reshape(n_train_runs, 32, 31)
        X_test_seq = X_test_voc.reshape(n_test_runs, 32, 31)
        
        # For target, take every 32nd label (since it is constant for each sequence run block)
        y_train_seq = df_train['target'].values[::32]
        y_test_seq = df_test['target'].values[::32]
        
        return X_train_seq, y_train_seq, X_test_seq, y_test_seq
