"""
NCV Data Preprocessing - Normalization, feature engineering
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional


class NCVPreprocessor:
    """Preprocessing untuk NCV data"""
    
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.normalization_method = config['features'].get('normalization', 'standard')
        
    def fit_scaler(self, X: np.ndarray):
        """Fit scaler pada training data"""
        if self.normalization_method == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.normalization_method == 'robust':
            self.scaler = RobustScaler()
        else:
            print(f"Unknown normalization: {self.normalization_method}, using standard")
            self.scaler = StandardScaler()
        
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data menggunakan fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit dan transform sekaligus"""
        self.fit_scaler(X)
        return self.transform(X)
    
    def handle_missing_values(self, X: np.ndarray) -> np.ndarray:
        """Handle missing values (NaN, inf)"""
        # Replace inf with nan
        X = np.where(np.isinf(X), np.nan, X)
        
        # Replace nan with column mean
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        
        return X
    
    def augment_data(self, X: np.ndarray, y: np.ndarray, 
                     noise_level: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Data augmentation dengan Gaussian noise"""
        X_augmented = []
        y_augmented = []
        
        for i in range(len(X)):
            # Original
            X_augmented.append(X[i])
            y_augmented.append(y[i])
            
            # Augmented dengan noise
            noise = np.random.normal(0, noise_level, X[i].shape)
            X_aug = X[i] + noise
            X_augmented.append(X_aug)
            y_augmented.append(y[i])
        
        return np.array(X_augmented), np.array(y_augmented)
    
    def preprocess_pipeline(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                           fit: bool = True, augment: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Complete preprocessing pipeline"""
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Normalize
        if fit:
            X = self.fit_transform(X)
        else:
            X = self.transform(X)
        
        # Augment (only for training)
        if augment and y is not None:
            X, y = self.augment_data(X, y)
        
        return X, y
