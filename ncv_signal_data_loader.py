"""
NCV Signal Data Loader
Load and parse NCV signal waveform files
"""
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class NCVSignalDataLoader:
    """Load NCV signal data dari CSV files dengan waveforms"""
    
    def __init__(self, data_directory: str, config: dict):
        self.data_directory = data_directory
        self.config = config
        self.class_mapping = config['data']['class_mapping']
        self.classes = config['data']['classes']
        
    def parse_ncv_signal_file(self, filepath: str) -> Optional[Dict]:
        """
        Parse NCV signal CSV file
        
        Args:
            filepath: Path to NCV signal CSV file
            
        Returns:
            dict: Parsed data with metadata and waveforms
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Parse metadata
            metadata = {}
            trace_start_idx = None
            
            for idx, line in enumerate(lines):
                if 'Trace Data' in line:
                    trace_start_idx = idx + 1
                    break
                
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ''
                    metadata[key] = value
            
            if trace_start_idx is None:
                return None
            
            # Parse trace data
            traces = []
            for line in lines[trace_start_idx:]:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        # Skip first empty column, get waveform values
                        values = [float(p) for p in parts[1:] if p.strip()]
                        if values:
                            traces.append(values)
                    except:
                        continue
            
            if not traces:
                return None
            
            # Convert to numpy array
            traces_array = np.array(traces)
            
            # Flatten if needed (take first trace if multiple)
            if len(traces_array.shape) > 1:
                # Take average of all traces or first trace
                waveform = np.mean(traces_array, axis=1) if traces_array.shape[1] > 1 else traces_array[:, 0]
            else:
                waveform = traces_array
            
            return {
                'metadata': metadata,
                'waveform': waveform,
                'filepath': filepath,
                'patient_id': metadata.get('Patient ID', 'unknown'),
                'test_item': metadata.get('Test Item', 'unknown')
            }
            
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return None
    
    def scan_signal_files(self) -> Dict[str, List[str]]:
        """Scan semua NCV signal files per class"""
        file_paths = {}
        
        for class_name in self.classes:
            class_dir = os.path.join(self.data_directory, class_name)
            if os.path.exists(class_dir):
                files = [
                    os.path.join(class_dir, f) 
                    for f in os.listdir(class_dir) 
                    if f.endswith('.csv')
                ]
                file_paths[class_name] = files
            else:
                file_paths[class_name] = []
                print(f"Warning: Directory not found: {class_dir}")
        
        return file_paths
    
    def load_all_signals(self, max_files_per_class: Optional[int] = None) -> Tuple[List[np.ndarray], np.ndarray, List[Dict]]:
        """
        Load semua NCV signal data
        
        Returns:
            signals: List of waveform arrays
            labels: Array of class labels
            signal_info: List of metadata dicts
        """
        file_paths = self.scan_signal_files()
        
        all_signals = []
        all_labels = []
        signal_info = []
        
        print("\nLoading NCV signal data...")
        for class_name, files in file_paths.items():
            if max_files_per_class:
                files = files[:max_files_per_class]
            
            class_label = self.class_mapping[class_name]
            
            loaded = 0
            for filepath in files:
                parsed = self.parse_ncv_signal_file(filepath)
                if parsed is not None and len(parsed['waveform']) > 0:
                    all_signals.append(parsed['waveform'])
                    all_labels.append(class_label)
                    signal_info.append({
                        'filepath': filepath,
                        'class': class_name,
                        'label': class_label,
                        'length': len(parsed['waveform']),
                        'patient_id': parsed['patient_id'],
                        'test_item': parsed['test_item']
                    })
                    loaded += 1
            
            print(f"  {class_name}: {loaded} signals loaded")
        
        if not all_signals:
            raise ValueError("No valid NCV signal data loaded!")
        
        y = np.array(all_labels)
        
        print(f"\nTotal loaded: {len(all_signals)} signals")
        print(f"Classes: {self.classes}")
        for class_name in self.classes:
            count = np.sum(y == self.class_mapping[class_name])
            print(f"  {class_name}: {count} samples")
        
        return all_signals, y, signal_info
