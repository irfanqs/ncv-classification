"""
Quick test script untuk NCV Signal pipeline
"""
import os
import sys
import json


def test_ncv_signal_pipeline():
    """Test NCV signal pipeline dengan data yang ada"""
    
    print("="*80)
    print("NCV SIGNAL PIPELINE QUICK TEST")
    print("="*80)
    
    # Check if data exists
    data_dirs = ['data/Full_Data', 'data/Motorik', 'data/Sensorik']
    available_dir = None
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            # Check if has data
            mild_dir = os.path.join(data_dir, 'mild')
            if os.path.exists(mild_dir):
                files = [f for f in os.listdir(mild_dir) if f.endswith('.csv')]
                if len(files) > 0:
                    available_dir = data_dir
                    print(f"\n✓ Found data in: {data_dir}")
                    print(f"  Sample files: {len(files)} in mild/")
                    break
    
    if available_dir is None:
        print("\n✗ No NCV signal data found!")
        print("\nPlease ensure data exists in:")
        print("  - data/Full_Data/")
        print("  - data/Motorik/")
        print("  - data/Sensorik/")
        return False
    
    # Update config untuk quick test
    print("\n[1/2] Creating test configuration...")
    
    test_config = {
        "project_info": {
            "name": "NCV Signal Classification - Quick Test",
            "version": "1.0.0"
        },
        "data": {
            "base_directory": available_dir,
            "classes": ["non_cts", "mild", "moderate", "severe"],
            "class_mapping": {
                "non_cts": 0,
                "mild": 1,
                "moderate": 2,
                "severe": 3
            },
            "max_files_per_class": 10  # Limit untuk quick test
        },
        "signal_processing": {
            "sampling_rate": 12804,
            "apply_bandpass_filter": True,
            "bandpass_lowcut": 10,
            "bandpass_highcut": 2500
        },
        "feature_extraction": {
            "extract_latency": True,
            "extract_amplitude": True,
            "extract_statistical": True,
            "extract_frequency": True
        },
        "models": {
            "available_models": [
                {
                    "name": "ncv_signal_dense",
                    "type": "dense_nn",
                    "description": "Dense NN for quick test"
                }
            ]
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
                "batch_size": 8,
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
            "experiments_dir": "experiments_ncv_signal_test",
            "save_models": True,
            "save_plots": True,
            "generate_report": True
        }
    }
    
    config_path = 'ncv_signal_config_test.json'
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"✓ Test config created: {config_path}")
    
    # Run pipeline
    print("\n[2/2] Running NCV signal pipeline...")
    print("This will take ~10-15 minutes...\n")
    
    try:
        from ncv_signal_main import NCVSignalPipeline
        
        pipeline = NCVSignalPipeline(config_path)
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {pipeline.experiment_dir}")
        print("\nNext steps:")
        print("1. Check FINAL_REPORT.md for results")
        print("2. View confusion matrices in plots/")
        print("3. Run full pipeline: python ncv_signal_main.py")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Check dependencies
    print("Checking dependencies...")
    try:
        import tensorflow
        import numpy
        import pandas
        import sklearn
        import scipy
        print("✓ All dependencies installed\n")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install:")
        print("  pip install tensorflow numpy pandas scikit-learn scipy")
        sys.exit(1)
    
    # Run test
    success = test_ncv_signal_pipeline()
    
    if success:
        print("\n" + "="*80)
        print("QUICK TEST SUMMARY")
        print("="*80)
        print("✓ NCV signal data loading: OK")
        print("✓ Feature extraction: OK")
        print("✓ Model training: OK")
        print("✓ Evaluation: OK")
        print("\nYou're ready to use the full pipeline!")
        print("\nFor full training:")
        print("  python ncv_signal_main.py")
        sys.exit(0)
    else:
        print("\n✗ Test failed. Please check errors above.")
        sys.exit(1)
