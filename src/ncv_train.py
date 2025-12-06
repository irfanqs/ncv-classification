"""
NCV Training Module
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from typing import Dict, List, Tuple, Optional
import os


class NCVTrainer:
    """Train NCV classification models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.history = None
        self.model = None
        
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data ke train/val/test"""
        split_config = self.config['training']['data_split']
        
        train_size = split_config['train_size']
        val_size = split_config['validation_size']
        test_size = split_config['test_size']
        
        # Split train and temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(val_size + test_size),
            stratify=y if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
        
        # Split temp ke val and test
        val_ratio = val_size / (val_size + test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_ratio),
            stratify=y_temp if split_config['stratify'] else None,
            random_state=split_config['random_state']
        )
        
        print(f"\nData split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Val:   {len(X_val)} samples")
        print(f"  Test:  {len(X_test)} samples")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute class weights untuk imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def create_callbacks(self, model_name: str, save_dir: str) -> List[keras.callbacks.Callback]:
        """Create training callbacks"""
        callbacks = []
        
        callbacks_config = self.config['training']['callbacks']
        
        # Early stopping
        if callbacks_config['early_stopping']['enabled']:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=callbacks_config['early_stopping']['monitor'],
                patience=callbacks_config['early_stopping']['patience'],
                restore_best_weights=True,
                verbose=1
            ))
        
        # Reduce LR on plateau
        if callbacks_config['reduce_lr']['enabled']:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=callbacks_config['reduce_lr']['factor'],
                patience=callbacks_config['reduce_lr']['patience'],
                min_lr=1e-6,
                verbose=1
            ))
        
        # Model checkpoint
        checkpoint_path = os.path.join(save_dir, f'{model_name}_best.h5')
        callbacks.append(keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ))
        
        return callbacks
    
    def train_model(self, model: keras.Model, split_data: Dict[str, np.ndarray],
                   model_name: str, save_dir: str) -> keras.callbacks.History:
        """Train single model"""
        train_config = self.config['training']['hyperparameters']
        
        # Reshape untuk CNN/LSTM (tambah dimension)
        X_train = split_data['X_train'].reshape(split_data['X_train'].shape[0], -1, 1)
        X_val = split_data['X_val'].reshape(split_data['X_val'].shape[0], -1, 1)
        
        # Class weights
        class_weights = self.compute_class_weights(split_data['y_train'])
        
        # Callbacks
        callbacks = self.create_callbacks(model_name, save_dir)
        
        print(f"\nTraining {model_name}...")
        print(f"  Epochs: {train_config['epochs']}")
        print(f"  Batch size: {train_config['batch_size']}")
        print(f"  Learning rate: {train_config['learning_rate']}")
        
        # Train
        history = model.fit(
            X_train, split_data['y_train'],
            validation_data=(X_val, split_data['y_val']),
            epochs=train_config['epochs'],
            batch_size=train_config['batch_size'],
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        self.model = model
        self.history = history
        
        return history


class NCVCrossValidator:
    """Cross validation untuk NCV models"""
    
    def __init__(self, config: dict):
        self.config = config
        self.cv_results = {}
        
    def cross_validate(self, model_builder, X: np.ndarray, y: np.ndarray,
                      model_name: str, model_type: str) -> Dict:
        """Perform k-fold cross validation"""
        cv_config = self.config['cross_validation']
        n_splits = cv_config['n_splits']
        
        print(f"\n{'='*60}")
        print(f"Cross Validation: {model_name} ({n_splits}-Fold)")
        print(f"{'='*60}")
        
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Reshape
            X_train = X_train.reshape(X_train.shape[0], -1, 1)
            X_val = X_val.reshape(X_val.shape[0], -1, 1)
            
            # Build fresh model
            from src.ncv_models import NCVModelBuilder, compile_model
            builder = NCVModelBuilder(input_shape=(X.shape[1],), num_classes=len(np.unique(y)))
            model = builder.build_model(model_type)
            model = compile_model(model, learning_rate=0.001)
            
            # Train
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=16,
                verbose=0
            )
            
            # Evaluate
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            fold_scores.append(val_acc)
            
            print(f"  Fold {fold} accuracy: {val_acc:.4f}")
        
        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        
        print(f"\n{model_name} CV Results:")
        print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        
        return {
            'model_name': model_name,
            'fold_scores': fold_scores,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc
        }
