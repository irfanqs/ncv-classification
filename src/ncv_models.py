"""
NCV Models - CNN, LSTM, CNN-LSTM untuk data NCV
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np


class NCVModelBuilder:
    """Build models untuk NCV classification"""
    
    def __init__(self, input_shape: tuple, num_classes: int = 4):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_cnn_1d(self, filters: int = 64, dropout: float = 0.4) -> keras.Model:
        """
        1D CNN untuk NCV features
        Input: (n_features,) → reshape ke (n_features, 1)
        """
        model = models.Sequential(name='NCV_CNN')
        
        # Input layer
        model.add(layers.Input(shape=(self.input_shape[0], 1)))
        
        # Conv Block 1
        model.add(layers.Conv1D(filters, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout * 0.5))
        
        # Conv Block 2
        model.add(layers.Conv1D(filters * 2, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout * 0.75))
        
        # Conv Block 3
        model.add(layers.Conv1D(filters * 4, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout * 0.5))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_lstm(self, lstm_units: int = 64, dropout: float = 0.3) -> keras.Model:
        """
        LSTM untuk NCV temporal patterns
        Input: (n_features,) → reshape ke (n_features, 1)
        """
        model = models.Sequential(name='NCV_LSTM')
        
        # Input
        model.add(layers.Input(shape=(self.input_shape[0], 1)))
        
        # LSTM layers
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
        ))
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units // 2, dropout=dropout)
        ))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout * 0.5))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_cnn_lstm_hybrid(self, filters: int = 64, lstm_units: int = 64,
                             dropout: float = 0.35) -> keras.Model:
        """
        Hybrid CNN-LSTM untuk NCV
        CNN untuk feature extraction, LSTM untuk temporal modeling
        """
        model = models.Sequential(name='NCV_CNN_LSTM')
        
        # Input
        model.add(layers.Input(shape=(self.input_shape[0], 1)))
        
        # CNN feature extraction
        model.add(layers.Conv1D(filters, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout * 0.5))
        
        model.add(layers.Conv1D(filters * 2, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout * 0.75))
        
        # LSTM temporal modeling
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
        ))
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units // 2, dropout=dropout)
        ))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizers.l2(0.001)))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_model(self, model_type: str, **kwargs) -> keras.Model:
        """Build model berdasarkan type"""
        if model_type == 'cnn_1d':
            return self.build_cnn_1d(**kwargs)
        elif model_type == 'lstm':
            return self.build_lstm(**kwargs)
        elif model_type == 'cnn_lstm_hybrid':
            return self.build_cnn_lstm_hybrid(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def compile_model(model: keras.Model, learning_rate: float = 0.001,
                 optimizer: str = 'adam') -> keras.Model:
    """Compile model dengan optimizer dan loss"""
    if optimizer == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'adamw':
        opt = keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=opt,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
