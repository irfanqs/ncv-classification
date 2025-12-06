"""
NCV Signal Models
Models untuk NCV signal classification (dari extracted features)
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np


class NCVSignalModelBuilder:
    """Build models untuk NCV signal classification"""
    
    def __init__(self, input_shape: tuple, num_classes: int = 4):
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def build_dense_nn(self, hidden_units: list = [128, 64, 32], 
                      dropout: float = 0.4) -> keras.Model:
        """
        Dense Neural Network untuk extracted features
        """
        model = models.Sequential(name='NCV_Signal_DenseNN')
        
        # Input
        model.add(layers.Input(shape=self.input_shape))
        
        # Hidden layers
        for i, units in enumerate(hidden_units):
            model.add(layers.Dense(units, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.001)))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_cnn_1d(self, filters: int = 64, dropout: float = 0.4) -> keras.Model:
        """
        1D CNN untuk feature sequence
        """
        model = models.Sequential(name='NCV_Signal_CNN')
        
        # Reshape input untuk CNN
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Reshape((self.input_shape[0], 1)))
        
        # Conv blocks
        model.add(layers.Conv1D(filters, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2))
        model.add(layers.Dropout(dropout * 0.5))
        
        model.add(layers.Conv1D(filters * 2, 3, padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.GlobalAveragePooling1D())
        model.add(layers.Dropout(dropout))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_lstm(self, lstm_units: int = 64, dropout: float = 0.3) -> keras.Model:
        """
        LSTM untuk temporal patterns
        """
        model = models.Sequential(name='NCV_Signal_LSTM')
        
        # Reshape input
        model.add(layers.Input(shape=self.input_shape))
        model.add(layers.Reshape((self.input_shape[0], 1)))
        
        # LSTM layers
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout)
        ))
        model.add(layers.Bidirectional(
            layers.LSTM(lstm_units // 2, dropout=dropout)
        ))
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        
        # Output
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        
        return model
    
    def build_model(self, model_type: str, **kwargs) -> keras.Model:
        """Build model berdasarkan type"""
        if model_type == 'dense_nn':
            return self.build_dense_nn(**kwargs)
        elif model_type == 'cnn_1d':
            return self.build_cnn_1d(**kwargs)
        elif model_type == 'lstm':
            return self.build_lstm(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


def compile_model(model: keras.Model, learning_rate: float = 0.001,
                 optimizer: str = 'adam') -> keras.Model:
    """Compile model"""
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
