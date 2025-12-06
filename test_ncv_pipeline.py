"""
Quick test script untuk NCV pipeline
Generate sample data dan run quick test
"""
import os
import sys

def test_ncv_pipeline():
    """Test NCV pipeline dengan sample data"""
    
    print("="*80)
    print("NCV PIPELINE QUICK TEST")
    print("="*80)
    
    # Step 1: Generate sample data
    print("\n[1/3] Generating sample data...")
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from utils.create_sample_ncv_data import create_sample_ncv_data
    
    try:
        create_sample_ncv_data(base_dir='data/NCV_test', samples_per_class=5)
        print("✓ Sample data generated")
    except Exception as e:
        print(f"✗ Failed to generate data: {e}")
        return False
    
    # Step 2: Update config untuk quick test
    print("\n[2/3] Creating test configuration...")
    import json
    
    test_config = {
        "project_info": {
            "name": "NCV CTS Classification - Quick Test",
            "version": "1.0.0"
        },
        "data": {
            "base_directory": "data/NCV_test",
            "classes": ["non_cts", "mild", "moderate", "severe"],
            "class_mapping": {
                "non_cts": 0,
                "mild": 1,
                "moderate": 2,
                "severe": 3
            },
            "file_extensions": [".csv"],
            "max_files_per_class": 5
        },
        "features": {
            "motor_features": [
                "distal_latency",
                "amplitude",
                "conduction_velocity",
                "f_wave_latency"
            ],
            "sensory_features": [
                "peak_latency",
                "amplitude",
                "conduction_velocity"
            ],
            "use_motor": True,
            "use_sensory": True,
            "normalization": "standard"
        },
        "models": {
            "available_models": [
                {
                    "name": "ncv_cnn",
                    "type": "cnn_1d",
                    "description": "1D CNN for NCV"
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
                "epochs": 20,
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
            "experiments_dir": "experiments_ncv_test",
            "save_models": True,
            "save_plots": True,
            "generate_report": True
        }
    }
    
    config_path = 'ncv_config_test.json'
    with open(config_path, 'w') as f:
        json.dump(test_config, f, indent=2)
    
    print(f"✓ Test config created: {config_path}")
    
    # Step 3: Run pipeline
    print("\n[3/3] Running NCV pipeline...")
    print("This will take ~5-10 minutes...\n")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from ncv_main import NCVPipeline
        
        pipeline = NCVPipeline(config_path)
        pipeline.run_complete_pipeline()
        
        print("\n" + "="*80)
        print("✓ TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: {pipeline.experiment_dir}")
        print("\nNext steps:")
        print("1. Check FINAL_REPORT.md for results")
        print("2. View confusion matrices in plots/")
        print("3. Compare with full dataset using ncv_main.py")
        
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
        import matplotlib
        import seaborn
        print("✓ All dependencies installed\n")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install:")
        print("  pip install -r ncv_requirements.txt")
        sys.exit(1)
    
    # Run test
    success = test_ncv_pipeline()
    
    if success:
        print("\n" + "="*80)
        print("QUICK TEST SUMMARY")
        print("="*80)
        print("✓ Sample data generation: OK")
        print("✓ Data loading: OK")
        print("✓ Preprocessing: OK")
        print("✓ Model training: OK")
        print("✓ Evaluation: OK")
        print("\nYou're ready to use the full pipeline!")
        print("\nFor full training:")
        print("  1. Prepare your real NCV data")
        print("  2. Update ncv_config.json")
        print("  3. Run: python ncv_main.py")
        sys.exit(0)
    else:
        print("\n✗ Test failed. Please check errors above.")
        sys.exit(1)
