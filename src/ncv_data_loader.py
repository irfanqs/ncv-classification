"""
NCV Data Loader - Load and parse NCV data files
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class NCVDataLoader:
    """Load NCV data dari CSV/Excel files"""
    
    def __init__(self, data_directory: str, config: dict):
        self.data_directory = data_directory
        self.config = config
        self.class_mapping = config['data']['class_mapping']
        self.classes = config['data']['classes']
        
    def scan_ncv_files(self) -> Dict[str, List[str]]:
        """Scan semua NCV files per class"""
        file_paths = {}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_directory, class_name)
            if os.path.exists(class_dir):
                files = []
                for ext in self.config['data']['file_extensions']:
                    files.extend([
                        os.path.join(class_dir, f) 
                        for f in os.listdir(class_dir) 
                        if f.endswith(ext)
                    ])
                file_paths[class_name] = files
            else:
                file_paths[class_name] = []
                print(f"Warning: Directory not found: {class_dir}")
        
        return file_paths
    
    def load_single_file(self, filepath: str) -> Optional[pd.DataFrame]:
        """Load single NCV file (CSV or Excel)"""
        try:
            if filepath.endswith('.csv') or filepath.endswith('.txt'):
                df = pd.read_csv(filepath)
            elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                print(f"Unsupported file format: {filepath}")
                return None
            
            return df
        
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract NCV features dari dataframe"""
        features_config = self.config['features']
        
        feature_names = []
        if features_config.get('use_motor', True):
            feature_names.extend(features_config.get('motor_features', []))
        if features_config.get('use_sensory', True):
            feature_names.extend(features_config.get('sensory_features', []))
        
        # Cari kolom yang match dengan feature names (case-insensitive)
        available_features = []
        df_columns_lower = {col.lower(): col for col in df.columns}
        
        for feature in feature_names:
            feature_lower = feature.lower()
            if feature_lower in df_columns_lower:
                available_features.append(df_columns_lower[feature_lower])
        
        if not available_features:
            # Fallback: ambil semua kolom numerik
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not available_features:
            return None
        
        # Extract features
        feature_values = df[available_features].values
        
        # Handle multiple rows (ambil mean jika ada multiple measurements)
        if len(feature_values) > 1:
            feature_values = np.mean(feature_values, axis=0, keepdims=True)
        
        return feature_values.flatten()
    
    def load_all_data(self, max_files_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load semua NCV data"""
        file_paths = self.scan_ncv_files()
        
        all_features = []
        all_labels = []
        data_info = []
        
        print("\nLoading NCV data...")
        for class_name, files in file_paths.items():
            if max_files_per_class:
                files = files[:max_files_per_class]
            
            class_label = self.class_mapping[class_name]
            
            for filepath in files:
                df = self.load_single_file(filepath)
                if df is not None:
                    features = self.extract_features(df)
                    if features is not None and len(features) > 0:
                        all_features.append(features)
                        all_labels.append(class_label)
                        data_info.append({
                            'filepath': filepath,
                            'class': class_name,
                            'label': class_label,
                            'n_features': len(features)
                        })
        
        if not all_features:
            raise ValueError("No valid NCV data loaded!")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"Loaded {len(X)} samples with {X.shape[1]} features")
        print(f"Classes: {self.classes}")
        for class_name in self.classes:
            count = np.sum(y == self.class_mapping[class_name])
            print(f"  {class_name}: {count} samples")
        
        return X, y, data_info
