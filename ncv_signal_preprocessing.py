"""
NCV Signal Preprocessing
Process NCV waveforms and extract features
"""
import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


class NCVSignalPreprocessor:
    """Preprocess NCV signal waveforms"""
    
    def __init__(self, config: dict):
        self.config = config
        self.sampling_rate = config.get('signal_processing', {}).get('sampling_rate', 12804)
        self.scaler = None
        
    def normalize_signal(self, waveform: np.ndarray) -> np.ndarray:
        """Normalize waveform"""
        # Remove DC offset
        waveform = waveform - np.mean(waveform)
        
        # Z-score normalization
        std = np.std(waveform)
        if std > 0:
            waveform = waveform / std
        
        return waveform
    
    def bandpass_filter(self, waveform: np.ndarray, lowcut: float = 10, 
                       highcut: float = 2500, order: int = 4) -> np.ndarray:
        """Apply bandpass filter"""
        try:
            nyquist = 0.5 * self.sampling_rate
            low = lowcut / nyquist
            high = min(highcut / nyquist, 0.99)
            
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, waveform)
            
            return filtered
        except Exception as e:
            print(f"Warning: Bandpass filter failed - {e}")
            return waveform
    
    def extract_latency(self, waveform: np.ndarray, threshold_percent: float = 10) -> float:
        """
        Extract onset latency from waveform
        
        Returns:
            Latency in milliseconds
        """
        # Find peak
        peak_idx = np.argmax(np.abs(waveform))
        peak_value = np.abs(waveform[peak_idx])
        
        # Find onset (threshold crossing before peak)
        threshold = peak_value * (threshold_percent / 100.0)
        
        onset_idx = 0
        for i in range(peak_idx):
            if np.abs(waveform[i]) >= threshold:
                onset_idx = i
                break
        
        # Convert to milliseconds
        ms_per_sample = 1000.0 / self.sampling_rate
        latency_ms = onset_idx * ms_per_sample
        
        return latency_ms
    
    def extract_amplitude(self, waveform: np.ndarray) -> float:
        """Extract peak-to-peak amplitude"""
        peak_positive = np.max(waveform)
        peak_negative = np.min(waveform)
        amplitude = abs(peak_positive - peak_negative)
        
        return amplitude
    
    def extract_features_from_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract multiple features from waveform
        
        Returns:
            Feature vector
        """
        # Normalize first
        waveform_norm = self.normalize_signal(waveform)
        
        # Extract features
        latency = self.extract_latency(waveform_norm)
        amplitude = self.extract_amplitude(waveform_norm)
        
        # Additional features
        rms = np.sqrt(np.mean(waveform_norm**2))
        peak_value = np.max(np.abs(waveform_norm))
        mean_value = np.mean(np.abs(waveform_norm))
        std_value = np.std(waveform_norm)
        
        # Statistical features
        skewness = self._calculate_skewness(waveform_norm)
        kurtosis = self._calculate_kurtosis(waveform_norm)
        
        # Frequency domain features
        freq_features = self._extract_frequency_features(waveform_norm)
        
        # Combine all features
        features = np.array([
            latency,
            amplitude,
            rms,
            peak_value,
            mean_value,
            std_value,
            skewness,
            kurtosis,
            *freq_features
        ])
        
        return features
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _extract_frequency_features(self, waveform: np.ndarray) -> List[float]:
        """Extract frequency domain features using FFT"""
        # FFT
        fft_values = np.fft.fft(waveform)
        fft_magnitude = np.abs(fft_values[:len(fft_values)//2])
        
        # Frequency bins
        freqs = np.fft.fftfreq(len(waveform), 1/self.sampling_rate)[:len(fft_values)//2]
        
        # Dominant frequency
        dominant_freq_idx = np.argmax(fft_magnitude)
        dominant_freq = freqs[dominant_freq_idx] if dominant_freq_idx < len(freqs) else 0
        
        # Mean frequency
        mean_freq = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0
        
        # Spectral energy
        spectral_energy = np.sum(fft_magnitude ** 2)
        
        return [dominant_freq, mean_freq, spectral_energy]
    
    def process_signals_batch(self, signals: List[np.ndarray], 
                              apply_filter: bool = True) -> np.ndarray:
        """
        Process batch of signals and extract features
        
        Args:
            signals: List of waveform arrays
            apply_filter: Whether to apply bandpass filter
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        all_features = []
        
        for waveform in signals:
            # Apply filter if requested
            if apply_filter:
                waveform = self.bandpass_filter(waveform)
            
            # Extract features
            features = self.extract_features_from_waveform(waveform)
            all_features.append(features)
        
        X = np.array(all_features)
        
        return X
    
    def fit_scaler(self, X: np.ndarray):
        """Fit scaler on training data"""
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted! Call fit_scaler first.")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit_scaler(X)
        return self.transform(X)
