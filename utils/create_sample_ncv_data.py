"""
Script untuk generate sample NCV data untuk testing
"""
import os
import numpy as np
import pandas as pd

def create_sample_ncv_data(base_dir='data/NCV', samples_per_class=10):
    """
    Generate sample NCV data untuk testing pipeline
    
    Features:
    - Motor: distal_latency, amplitude, conduction_velocity, f_wave_latency
    - Sensory: peak_latency, amplitude, conduction_velocity
    """
    
    classes = ['non_cts', 'mild', 'moderate', 'severe']
    
    # Create directories
    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    print("Generating sample NCV data...")
    
    # Define realistic ranges untuk setiap class
    # Based on clinical NCV values
    
    ranges = {
        'non_cts': {
            'motor': {
                'distal_latency': (2.5, 3.5),      # ms (normal)
                'amplitude': (8.0, 15.0),          # mV (normal)
                'conduction_velocity': (50.0, 65.0), # m/s (normal)
                'f_wave_latency': (25.0, 30.0)     # ms (normal)
            },
            'sensory': {
                'peak_latency': (2.0, 3.0),        # ms (normal)
                'amplitude': (15.0, 50.0),         # μV (normal)
                'conduction_velocity': (50.0, 65.0) # m/s (normal)
            }
        },
        'mild': {
            'motor': {
                'distal_latency': (3.5, 4.5),      # slightly prolonged
                'amplitude': (6.0, 10.0),          # slightly reduced
                'conduction_velocity': (45.0, 52.0), # slightly slow
                'f_wave_latency': (30.0, 33.0)
            },
            'sensory': {
                'peak_latency': (3.0, 3.8),
                'amplitude': (10.0, 20.0),         # reduced
                'conduction_velocity': (45.0, 52.0)
            }
        },
        'moderate': {
            'motor': {
                'distal_latency': (4.5, 6.0),      # prolonged
                'amplitude': (4.0, 7.0),           # reduced
                'conduction_velocity': (38.0, 47.0), # slow
                'f_wave_latency': (33.0, 37.0)
            },
            'sensory': {
                'peak_latency': (3.8, 4.8),
                'amplitude': (5.0, 12.0),          # significantly reduced
                'conduction_velocity': (38.0, 47.0)
            }
        },
        'severe': {
            'motor': {
                'distal_latency': (6.0, 8.0),      # very prolonged
                'amplitude': (2.0, 5.0),           # very reduced
                'conduction_velocity': (30.0, 40.0), # very slow
                'f_wave_latency': (37.0, 42.0)
            },
            'sensory': {
                'peak_latency': (4.8, 6.5),
                'amplitude': (2.0, 8.0),           # very reduced or absent
                'conduction_velocity': (30.0, 40.0)
            }
        }
    }
    
    for class_name in classes:
        print(f"\nGenerating {class_name} samples...")
        
        for i in range(samples_per_class):
            # Generate motor data
            motor_data = {}
            for feature, (min_val, max_val) in ranges[class_name]['motor'].items():
                motor_data[feature] = np.random.uniform(min_val, max_val)
            
            # Generate sensory data
            sensory_data = {}
            for feature, (min_val, max_val) in ranges[class_name]['sensory'].items():
                sensory_data[feature] = np.random.uniform(min_val, max_val)
            
            # Save motor file
            motor_df = pd.DataFrame([motor_data])
            motor_filename = f"patient_{class_name}_{i+1:03d}_motor.csv"
            motor_path = os.path.join(base_dir, class_name, motor_filename)
            motor_df.to_csv(motor_path, index=False)
            
            # Save sensory file
            sensory_df = pd.DataFrame([sensory_data])
            sensory_filename = f"patient_{class_name}_{i+1:03d}_sensory.csv"
            sensory_path = os.path.join(base_dir, class_name, sensory_filename)
            sensory_df.to_csv(sensory_path, index=False)
            
            # Save combined file (motor + sensory)
            combined_data = {**motor_data, **sensory_data}
            # Rename sensory features to avoid conflict
            combined_data['sensory_peak_latency'] = combined_data.pop('peak_latency')
            combined_data['sensory_amplitude'] = combined_data.pop('amplitude')
            combined_data['sensory_conduction_velocity'] = combined_data.pop('conduction_velocity')
            
            combined_df = pd.DataFrame([combined_data])
            combined_filename = f"patient_{class_name}_{i+1:03d}_combined.csv"
            combined_path = os.path.join(base_dir, class_name, combined_filename)
            combined_df.to_csv(combined_path, index=False)
        
        print(f"  Created {samples_per_class * 3} files for {class_name}")
    
    print(f"\n✓ Sample data created in: {base_dir}")
    print(f"  Total files: {len(classes) * samples_per_class * 3}")
    print(f"  Classes: {classes}")
    print(f"  Samples per class: {samples_per_class * 3} (motor + sensory + combined)")
    
    # Create summary
    summary_path = os.path.join(base_dir, 'DATA_SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("NCV Sample Data Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {len(classes) * samples_per_class * 3}\n")
        f.write(f"Classes: {', '.join(classes)}\n")
        f.write(f"Samples per class: {samples_per_class * 3}\n\n")
        
        f.write("File types:\n")
        f.write("  - *_motor.csv: Motor nerve conduction data\n")
        f.write("  - *_sensory.csv: Sensory nerve conduction data\n")
        f.write("  - *_combined.csv: Combined motor + sensory data\n\n")
        
        f.write("Features:\n")
        f.write("  Motor:\n")
        f.write("    - distal_latency (ms)\n")
        f.write("    - amplitude (mV)\n")
        f.write("    - conduction_velocity (m/s)\n")
        f.write("    - f_wave_latency (ms)\n\n")
        f.write("  Sensory:\n")
        f.write("    - peak_latency (ms)\n")
        f.write("    - amplitude (μV)\n")
        f.write("    - conduction_velocity (m/s)\n\n")
        
        f.write("Clinical ranges:\n")
        for class_name in classes:
            f.write(f"\n  {class_name.upper()}:\n")
            f.write(f"    Motor distal latency: {ranges[class_name]['motor']['distal_latency']}\n")
            f.write(f"    Motor amplitude: {ranges[class_name]['motor']['amplitude']}\n")
            f.write(f"    Conduction velocity: {ranges[class_name]['motor']['conduction_velocity']}\n")
    
    print(f"\n✓ Summary saved: {summary_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample NCV data')
    parser.add_argument('--output-dir', type=str, default='data/NCV',
                       help='Output directory')
    parser.add_argument('--samples', type=int, default=10,
                       help='Samples per class')
    
    args = parser.parse_args()
    
    create_sample_ncv_data(args.output_dir, args.samples)
    
    print("\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Update ncv_config.json:")
    print(f'   "base_directory": "{args.output_dir}"')
    print("\n2. Run pipeline:")
    print("   python ncv_main.py")
    print("="*60)
