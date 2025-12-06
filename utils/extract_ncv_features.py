"""
Extract NCV features from raw signal waveforms
Convert NCV signal CSV files to tabular feature CSV
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path


def parse_ncv_signal_file(filepath):
    """
    Parse NCV signal CSV file
    
    Args:
        filepath: Path to NCV signal CSV file
        
    Returns:
        dict: Parsed data with metadata and waveforms
    """
    try:
        # Read file
        with open(filepath, 'r') as f:
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
                    traces.append(values)
                except:
                    continue
        
        if not traces:
            return None
        
        # Convert to numpy array
        traces_array = np.array(traces)
        
        return {
            'metadata': metadata,
            'traces': traces_array,
            'filepath': filepath
        }
        
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return None


def extract_latency(waveform, sampling_rate=12804, threshold_percent=10):
    """
    Extract onset latency from waveform
    
    Args:
        waveform: 1D array of signal values
        sampling_rate: Sampling rate in Hz
        threshold_percent: Threshold as percentage of peak
        
    Returns:
        float: Latency in milliseconds
    """
    # Find peak amplitude
    peak_idx = np.argmax(np.abs(waveform))
    peak_value = np.abs(waveform[peak_idx])
    
    # Find onset (first point above threshold before peak)
    threshold = peak_value * (threshold_percent / 100.0)
    
    onset_idx = 0
    for i in range(peak_idx):
        if np.abs(waveform[i]) >= threshold:
            onset_idx = i
            break
    
    # Convert to milliseconds
    ms_per_sample = 1000.0 / sampling_rate
    latency_ms = onset_idx * ms_per_sample
    
    return latency_ms


def extract_amplitude(waveform):
    """
    Extract peak-to-peak amplitude
    
    Args:
        waveform: 1D array of signal values
        
    Returns:
        float: Amplitude in microvolts
    """
    peak_positive = np.max(waveform)
    peak_negative = np.min(waveform)
    amplitude = peak_positive - peak_negative
    
    return abs(amplitude)


def calculate_conduction_velocity(latency_proximal, latency_distal, distance_cm=10):
    """
    Calculate nerve conduction velocity
    
    Args:
        latency_proximal: Latency at proximal stimulation (ms)
        latency_distal: Latency at distal stimulation (ms)
        distance_cm: Distance between stimulation points (cm)
        
    Returns:
        float: Conduction velocity in m/s
    """
    if latency_proximal <= latency_distal:
        return 0.0
    
    latency_diff_ms = latency_proximal - latency_distal
    latency_diff_s = latency_diff_ms / 1000.0
    
    distance_m = distance_cm / 100.0
    velocity_ms = distance_m / latency_diff_s
    
    return velocity_ms


def extract_features_from_signal(parsed_data):
    """
    Extract NCV features from parsed signal data
    
    Args:
        parsed_data: Dict from parse_ncv_signal_file()
        
    Returns:
        dict: Extracted features
    """
    traces = parsed_data['traces']
    
    if traces.shape[1] < 2:
        # Single trace
        waveform = traces[:, 0]
        latency = extract_latency(waveform)
        amplitude = extract_amplitude(waveform)
        
        return {
            'distal_latency': latency,
            'amplitude': amplitude,
            'conduction_velocity': 0.0,  # Cannot calculate without two points
            'f_wave_latency': 0.0
        }
    else:
        # Two traces (distal and proximal)
        waveform_distal = traces[:, 0]
        waveform_proximal = traces[:, 1]
        
        latency_distal = extract_latency(waveform_distal)
        latency_proximal = extract_latency(waveform_proximal)
        amplitude = extract_amplitude(waveform_distal)
        
        # Calculate conduction velocity
        velocity = calculate_conduction_velocity(
            latency_proximal, 
            latency_distal,
            distance_cm=10  # Typical wrist-elbow distance
        )
        
        return {
            'distal_latency': latency_distal,
            'amplitude': amplitude,
            'conduction_velocity': velocity,
            'f_wave_latency': 0.0  # Would need F-wave analysis
        }


def process_folder(input_folder, output_folder, class_name):
    """
    Process all NCV signal files in a folder
    
    Args:
        input_folder: Folder with NCV signal CSV files
        output_folder: Output folder for feature CSV files
        class_name: Class name (non_cts, mild, moderate, severe)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
    print(f"\nProcessing {class_name}: {len(files)} files")
    
    for filename in files:
        filepath = os.path.join(input_folder, filename)
        
        # Parse signal file
        parsed = parse_ncv_signal_file(filepath)
        if parsed is None:
            print(f"  Skipped: {filename}")
            continue
        
        # Extract features
        features = extract_features_from_signal(parsed)
        
        # Save as CSV
        output_filename = filename.replace('.csv', '_features.csv')
        output_path = os.path.join(output_folder, output_filename)
        
        df = pd.DataFrame([features])
        df.to_csv(output_path, index=False)
    
    print(f"  Saved {len(files)} feature files to {output_folder}")


def main():
    """
    Main function to extract features from all NCV signal data
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract NCV features from signal files')
    parser.add_argument('--input-dir', type=str, default='data/Full_Data',
                       help='Input directory with signal files')
    parser.add_argument('--output-dir', type=str, default='data/NCV',
                       help='Output directory for feature files')
    
    args = parser.parse_args()
    
    classes = ['non_cts', 'mild', 'moderate', 'severe']
    
    print("="*80)
    print("NCV FEATURE EXTRACTION")
    print("="*80)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    
    for class_name in classes:
        input_folder = os.path.join(args.input_dir, class_name)
        output_folder = os.path.join(args.output_dir, class_name)
        
        if os.path.exists(input_folder):
            process_folder(input_folder, output_folder, class_name)
        else:
            print(f"\nSkipping {class_name}: folder not found")
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nExtracted features saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Check extracted features in data/NCV/")
    print("  2. Run: python ncv_main.py")


if __name__ == "__main__":
    main()
