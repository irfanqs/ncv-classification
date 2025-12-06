"""
Interactive Menu untuk NCV Signal Classification V2
Pilih dataset dan konfigurasi sebelum training
"""
import os
import sys
import json


def check_data_availability():
    """Check available datasets"""
    datasets = {}
    
    data_folders = {
        'Full_Data': 'data/Full_Data',
        'Motorik': 'data/Motorik',
        'Sensorik': 'data/Sensorik'
    }
    
    for name, path in data_folders.items():
        if os.path.exists(path):
            # Count files in mild folder as sample
            mild_path = os.path.join(path, 'mild')
            if os.path.exists(mild_path):
                files = [f for f in os.listdir(mild_path) if f.endswith('.csv')]
                if len(files) > 0:
                    # Count all classes
                    total_files = 0
                    for class_name in ['non_cts', 'mild', 'moderate', 'severe']:
                        class_path = os.path.join(path, class_name)
                        if os.path.exists(class_path):
                            class_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
                            total_files += len(class_files)
                    
                    datasets[name] = {
                        'path': path,
                        'files': total_files,
                        'available': True
                    }
    
    return datasets


def print_header():
    """Print header"""
    print("\n" + "="*80)
    print("NCV SIGNAL CLASSIFICATION V2 - INTERACTIVE MENU")
    print("="*80)
    print("\nUsing Spectrogram Approach (Expected Accuracy: 60-75%)")
    print("Much better than V1's feature-based approach (36%)")


def select_dataset(datasets):
    """Select dataset"""
    print("\n" + "="*80)
    print("STEP 1: SELECT DATASET")
    print("="*80)
    
    if not datasets:
        print("\n✗ No datasets found!")
        print("\nPlease ensure data exists in:")
        print("  - data/Full_Data/")
        print("  - data/Motorik/")
        print("  - data/Sensorik/")
        return None
    
    print("\nAvailable datasets:\n")
    
    dataset_list = list(datasets.keys())
    for idx, name in enumerate(dataset_list, 1):
        info = datasets[name]
        print(f"  [{idx}] {name}")
        print(f"      Path: {info['path']}")
        print(f"      Files: {info['files']} total")
        
        if name == 'Full_Data':
            print(f"      Type: Motor + Sensory combined")
            print(f"      Best for: Complete analysis")
        elif name == 'Motorik':
            print(f"      Type: Motor nerve only")
            print(f"      Best for: Motor pathway analysis")
        elif name == 'Sensorik':
            print(f"      Type: Sensory nerve only")
            print(f"      Best for: Sensory pathway analysis")
        print()
    
    while True:
        try:
            choice = input(f"Select dataset [1-{len(dataset_list)}]: ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(dataset_list):
                selected = dataset_list[choice_idx]
                print(f"\n✓ Selected: {selected}")
                return datasets[selected]['path']
            else:
                print(f"Please enter a number between 1 and {len(dataset_list)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            return None


def select_mode():
    """Select training mode"""
    print("\n" + "="*80)
    print("STEP 2: SELECT TRAINING MODE")
    print("="*80)
    
    print("\nAvailable modes:\n")
    print("  [1] Quick Test")
    print("      - Use 15 files per class")
    print("      - Train 1 model (CNN)")
    print("      - 30 epochs")
    print("      - Time: ~15-20 minutes")
    print("      - Good for: Testing & validation")
    print()
    print("  [2] Full Training")
    print("      - Use all available files")
    print("      - Train 3 models (CNN, LSTM, Hybrid)")
    print("      - 100 epochs")
    print("      - Time: ~60-90 minutes")
    print("      - Good for: Production & best results")
    print()
    
    while True:
        try:
            choice = input("Select mode [1-2]: ").strip()
            
            if choice == '1':
                print("\n✓ Selected: Quick Test")
                return 'quick'
            elif choice == '2':
                print("\n✓ Selected: Full Training")
                return 'full'
            else:
                print("Please enter 1 or 2")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            return None


def create_config(data_dir, mode):
    """Create configuration"""
    print("\n" + "="*80)
    print("STEP 3: CREATING CONFIGURATION")
    print("="*80)
    
    if mode == 'quick':
        config = {
            "project_info": {
                "name": "NCV Signal V2 - Quick Test",
                "version": "2.0.0"
            },
            "data": {
                "base_directory": data_dir,
                "classes": ["non_cts", "mild", "moderate", "severe"],
                "class_mapping": {
                    "non_cts": 0,
                    "mild": 1,
                    "moderate": 2,
                    "severe": 3
                },
                "max_files_per_class": 15
            },
            "signal_processing": {
                "sampling_rate": 12804,
                "apply_bandpass_filter": True
            },
            "training": {
                "data_split": {
                    "train_size": 0.7,
                    "validation_size": 0.15,
                    "test_size": 0.15,
                    "stratify": True,
                    "random_state": 42
                },
                "hyperparameters": {
                    "epochs": 30,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                "callbacks": {
                    "early_stopping": {
                        "enabled": True,
                        "patience": 10,
                        "monitor": "val_accuracy"
                    },
                    "reduce_lr": {
                        "enabled": True,
                        "patience": 5,
                        "factor": 0.5
                    }
                }
            },
            "cross_validation": {
                "enabled": False
            },
            "output": {
                "experiments_dir": "experiments_ncv_signal_v2",
                "save_models": True,
                "save_plots": True,
                "generate_report": True
            }
        }
    else:  # full
        config = {
            "project_info": {
                "name": "NCV Signal V2 - Full Training",
                "version": "2.0.0"
            },
            "data": {
                "base_directory": data_dir,
                "classes": ["non_cts", "mild", "moderate", "severe"],
                "class_mapping": {
                    "non_cts": 0,
                    "mild": 1,
                    "moderate": 2,
                    "severe": 3
                },
                "max_files_per_class": None
            },
            "signal_processing": {
                "sampling_rate": 12804,
                "apply_bandpass_filter": True
            },
            "training": {
                "data_split": {
                    "train_size": 0.7,
                    "validation_size": 0.15,
                    "test_size": 0.15,
                    "stratify": True,
                    "random_state": 42
                },
                "hyperparameters": {
                    "epochs": 100,
                    "batch_size": 16,
                    "learning_rate": 0.001,
                    "optimizer": "adam"
                },
                "callbacks": {
                    "early_stopping": {
                        "enabled": True,
                        "patience": 20,
                        "monitor": "val_accuracy"
                    },
                    "reduce_lr": {
                        "enabled": True,
                        "patience": 10,
                        "factor": 0.5
                    }
                }
            },
            "cross_validation": {
                "enabled": False
            },
            "output": {
                "experiments_dir": "experiments_ncv_signal_v2",
                "save_models": True,
                "save_plots": True,
                "generate_report": True
            }
        }
    
    config_path = 'ncv_signal_config_interactive.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Configuration created: {config_path}")
    print(f"\nSettings:")
    print(f"  Dataset: {data_dir}")
    print(f"  Max files per class: {config['data']['max_files_per_class'] or 'All'}")
    print(f"  Epochs: {config['training']['hyperparameters']['epochs']}")
    print(f"  Batch size: {config['training']['hyperparameters']['batch_size']}")
    
    return config_path


def run_pipeline(config_path):
    """Run the pipeline"""
    print("\n" + "="*80)
    print("STEP 4: RUNNING PIPELINE")
    print("="*80)
    
    print("\nStarting NCV Signal V2 pipeline...")
    print("This will:")
    print("  1. Load NCV signal waveforms")
    print("  2. Generate spectrograms (STFT)")
    print("  3. Train models (CNN, LSTM, Hybrid)")
    print("  4. Evaluate and compare results")
    print("\nPlease wait...\n")
    
    try:
        from ncv_signal_main_v2 import NCVSignalPipelineV2
        
        pipeline = NCVSignalPipelineV2(config_path)
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {pipeline.experiment_dir}")
        print("\nCheck:")
        print("  - FINAL_REPORT.md for detailed results")
        print("  - plots/ for confusion matrices")
        print("  - results/model_comparison.csv for metrics")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Print header
    print_header()
    
    # Check dependencies
    print("\nChecking dependencies...")
    try:
        import tensorflow
        import numpy
        import scipy
        print("✓ All dependencies OK")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install:")
        print("  pip install tensorflow numpy scipy scikit-learn matplotlib seaborn")
        sys.exit(1)
    
    # Check data availability
    datasets = check_data_availability()
    
    # Select dataset
    data_dir = select_dataset(datasets)
    if data_dir is None:
        sys.exit(1)
    
    # Select mode
    mode = select_mode()
    if mode is None:
        sys.exit(1)
    
    # Create config
    config_path = create_config(data_dir, mode)
    
    # Confirm
    print("\n" + "="*80)
    print("READY TO START")
    print("="*80)
    
    confirm = input("\nProceed with training? [y/n]: ").strip().lower()
    
    if confirm != 'y':
        print("\nCancelled by user")
        sys.exit(0)
    
    # Run pipeline
    success = run_pipeline(config_path)
    
    if success:
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print("\nExpected accuracy: 60-75%")
        print("Much better than V1's 36%!")
        sys.exit(0)
    else:
        print("\n✗ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
