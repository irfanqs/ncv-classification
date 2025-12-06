"""
Quick test untuk NCV Signal V2 (Spectrogram approach)
"""
import os
import sys
import json


def test_ncv_signal_v2():
    """Test NCV signal V2 pipeline"""
    
    print("="*80)
    print("NCV SIGNAL V2 PIPELINE TEST (Spectrogram Approach)")
    print("="*80)
    print("\nThis version uses spectrogram representation (like EMG system)")
    print("Expected to have MUCH better accuracy than V1!")
    
    # Find data
    data_dirs = ['data/Full_Data', 'data/Motorik', 'data/Sensorik']
    available_dir = None
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            mild_dir = os.path.join(data_dir, 'mild')
            if os.path.exists(mild_dir):
                files = [f for f in os.listdir(mild_dir) if f.endswith('.csv')]
                if len(files) > 0:
                    available_dir = data_dir
                    print(f"\n✓ Found data in: {data_dir}")
                    print(f"  Files in mild/: {len(files)}")
                    break
    
    if available_dir is None:
        print("\n✗ No data found!")
        return False
    
    # Create test config
    print("\n[1/2] Creating test configuration...")
    
    test_config = {
        "project_info": {
            "name": "NCV Signal V2 - Quick Test",
            "version": "2.0.0"
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
            "max_files_per_class": 15  # More data for better results
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
                "epochs": 50,
                "batch_size": 16,
                "learning_rate": 0.001,
                "optimizer": "adam"
            },
            "callbacks": {
                "early_stopping": {
                    "enabled": True,
                    "patience": 15,
                    "monitor": "val_accuracy"
                },
                "reduce_lr": {
                    "enabled": True,
                    "patience": 7,
                    "factor": 0.5
                }
            }
        },
        "cross_validation": {
            "enabled": False
        },
        "output": {
            "experiments_dir": "experiments_ncv_signal_v2_test",
            "save_models": True,
            "save_plots": True,
            "generate_report": True
        }
    }
    
    config_path = 'ncv_signal_config_v2_test.json'
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"✓ Config created: {config_path}")
    
    # Run pipeline
    print("\n[2/2] Running NCV Signal V2 pipeline...")
    print("Using spectrogram approach (like EMG system)")
    print("This should give MUCH better results!")
    print("Estimated time: 15-20 minutes...\n")
    
    try:
        from ncv_signal_main_v2 import NCVSignalPipelineV2
        
        pipeline = NCVSignalPipelineV2(config_path)
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults: {pipeline.experiment_dir}")
        print("\nCheck:")
        print("  - FINAL_REPORT.md for detailed results")
        print("  - plots/ for confusion matrices")
        print("  - results/model_comparison.csv for metrics")
        print("\nExpected accuracy: 60-75% (much better than V1's 36%!)")
        
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
        print("✓ All dependencies OK\n")
    except ImportError as e:
        print(f"✗ Missing: {e}")
        sys.exit(1)
    
    # Run test
    success = test_ncv_signal_v2()
    
    if success:
        print("\n" + "="*80)
        print("V2 TEST SUMMARY")
        print("="*80)
        print("✓ Spectrogram generation: OK")
        print("✓ Model training: OK")
        print("✓ Evaluation: OK")
        print("\nV2 uses spectrogram approach (like EMG)")
        print("This should give 60-75% accuracy vs V1's 36%")
        print("\nFor full training:")
        print("  python ncv_signal_main_v2.py")
        sys.exit(0)
    else:
        print("\n✗ Test failed")
        sys.exit(1)
