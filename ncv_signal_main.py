"""
NCV Signal Classification Main Pipeline
Process NCV signal waveforms directly
"""
import os
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ncv_signal_data_loader import NCVSignalDataLoader
from ncv_signal_preprocessing import NCVSignalPreprocessor
from ncv_signal_models import NCVSignalModelBuilder, compile_model
from src.ncv_train import NCVTrainer, NCVCrossValidator
from src.ncv_evaluate import NCVEvaluator


class NCVSignalPipeline:
    """Complete NCV signal classification pipeline"""
    
    def __init__(self, config_path: str = 'ncv_signal_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.experiment_dir = None
        self.signals = None
        self.y = None
        self.signal_info = None
        self.X_features = None
        self.preprocessor = None
        self.split_data = None
        self.models = {}
        self.evaluator = None
        
    def setup_experiment(self):
        """Setup experiment directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"ncv_signal_classification_{timestamp}"
        
        exp_dir = os.path.join(self.config['output']['experiments_dir'], exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        
        self.experiment_dir = exp_dir
        
        # Save config
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"NCV SIGNAL CLASSIFICATION PIPELINE")
        print(f"{'='*80}")
        print(f"Experiment: {exp_name}")
        print(f"Directory: {exp_dir}")
        
        return exp_dir
    
    def load_signals(self):
        """Load NCV signal waveforms"""
        print(f"\n{'='*80}")
        print("STEP 1: LOADING NCV SIGNAL DATA")
        print(f"{'='*80}")
        
        data_dir = self.config['data']['base_directory']
        print(f"Data directory: {data_dir}")
        
        loader = NCVSignalDataLoader(data_dir, self.config)
        self.signals, self.y, self.signal_info = loader.load_all_signals(
            max_files_per_class=self.config['data'].get('max_files_per_class')
        )
        
        print(f"\nSignal lengths:")
        lengths = [len(s) for s in self.signals]
        print(f"  Min: {min(lengths)} samples")
        print(f"  Max: {max(lengths)} samples")
        print(f"  Mean: {np.mean(lengths):.0f} samples")
        
        return self.signals, self.y
    
    def extract_features(self):
        """Extract features from signals"""
        print(f"\n{'='*80}")
        print("STEP 2: FEATURE EXTRACTION")
        print(f"{'='*80}")
        
        self.preprocessor = NCVSignalPreprocessor(self.config)
        
        print("\nExtracting features from waveforms...")
        print("  - Latency (onset detection)")
        print("  - Amplitude (peak-to-peak)")
        print("  - Statistical features (RMS, mean, std, skewness, kurtosis)")
        print("  - Frequency features (FFT-based)")
        
        apply_filter = self.config['signal_processing'].get('apply_bandpass_filter', True)
        self.X_features = self.preprocessor.process_signals_batch(
            self.signals,
            apply_filter=apply_filter
        )
        
        print(f"\nExtracted features shape: {self.X_features.shape}")
        print(f"  Samples: {self.X_features.shape[0]}")
        print(f"  Features per sample: {self.X_features.shape[1]}")
        
        return self.X_features
    
    def preprocess_features(self):
        """Preprocess extracted features"""
        print(f"\n{'='*80}")
        print("STEP 3: PREPROCESSING FEATURES")
        print(f"{'='*80}")
        
        # Split data first
        trainer = NCVTrainer(self.config)
        self.split_data = trainer.split_data(self.X_features, self.y)
        
        # Normalize train
        print("\nNormalizing features (StandardScaler)...")
        X_train = self.preprocessor.fit_transform(self.split_data['X_train'])
        
        # Transform val and test
        X_val = self.preprocessor.transform(self.split_data['X_val'])
        X_test = self.preprocessor.transform(self.split_data['X_test'])
        
        # Update split_data
        self.split_data['X_train'] = X_train
        self.split_data['X_val'] = X_val
        self.split_data['X_test'] = X_test
        
        print(f"  Train: {X_train.shape}")
        print(f"  Val:   {X_val.shape}")
        print(f"  Test:  {X_test.shape}")
        
        return self.split_data
    
    def train_models(self):
        """Train all models"""
        print(f"\n{'='*80}")
        print("STEP 4: TRAINING MODELS")
        print(f"{'='*80}")
        
        models_config = self.config['models']['available_models']
        trainer = NCVTrainer(self.config)
        
        n_features = self.split_data['X_train'].shape[1]
        n_classes = len(self.config['data']['classes'])
        
        for model_cfg in models_config:
            model_name = model_cfg['name']
            model_type = model_cfg['type']
            
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            # Build model
            builder = NCVSignalModelBuilder(input_shape=(n_features,), num_classes=n_classes)
            model = builder.build_model(model_type)
            
            # Compile
            lr = self.config['training']['hyperparameters']['learning_rate']
            optimizer = self.config['training']['hyperparameters']['optimizer']
            model = compile_model(model, learning_rate=lr, optimizer=optimizer)
            
            # Print summary
            print(f"\nModel architecture:")
            model.summary()
            
            # Train
            history = trainer.train_model(
                model,
                self.split_data,
                model_name,
                os.path.join(self.experiment_dir, 'models')
            )
            
            self.models[model_name] = {
                'model': model,
                'history': history,
                'type': model_type
            }
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print(f"\n{'='*80}")
        print("STEP 5: EVALUATION")
        print(f"{'='*80}")
        
        class_names = self.config['data']['classes']
        self.evaluator = NCVEvaluator(class_names)
        
        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Reshape untuk evaluation (models expect 2D input)
            X_test = self.split_data['X_test']
            
            self.evaluator.evaluate_model(
                model_info['model'],
                X_test,
                self.split_data['y_test'],
                model_name
            )
            
            self.evaluator.print_classification_report(model_name)
        
        # Comparison table
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = self.evaluator.create_comparison_table()
        print(comparison_df.to_string(index=False))
        
        # Save results
        results_dir = os.path.join(self.experiment_dir, 'results')
        comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
        
        # Plot confusion matrices
        plots_dir = os.path.join(self.experiment_dir, 'plots')
        self.evaluator.plot_all_confusion_matrices(plots_dir)
        self.evaluator.plot_model_comparison(os.path.join(plots_dir, 'model_comparison.png'))
        
        return self.evaluator.results
    
    def generate_report(self):
        """Generate final report"""
        report_path = os.path.join(self.experiment_dir, 'FINAL_REPORT.md')
        
        with open(report_path, 'w') as f:
            f.write("# NCV Signal Classification - Final Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 1. Experiment Overview\n\n")
            f.write("**Objective:** Classify CTS severity from NCV signal waveforms\n\n")
            
            f.write("## 2. Dataset Information\n\n")
            f.write(f"- Total signals: {len(self.signals)}\n")
            f.write(f"- Extracted features: {self.X_features.shape[1]} per signal\n")
            f.write(f"- Classes: {self.config['data']['classes']}\n")
            f.write(f"- Class distribution:\n")
            for class_name, label in self.config['data']['class_mapping'].items():
                count = np.sum(self.y == label)
                f.write(f"  - {class_name}: {count} samples\n")
            
            f.write("\n## 3. Feature Extraction\n\n")
            f.write("Features extracted from each waveform:\n")
            f.write("- Latency (onset detection)\n")
            f.write("- Amplitude (peak-to-peak)\n")
            f.write("- RMS, mean, std, skewness, kurtosis\n")
            f.write("- Frequency domain features (FFT)\n")
            
            f.write("\n## 4. Model Architectures\n\n")
            f.write("| Model | Type | Description |\n")
            f.write("|-------|------|-------------|\n")
            for model_cfg in self.config['models']['available_models']:
                f.write(f"| {model_cfg['name']} | {model_cfg['type']} | {model_cfg['description']} |\n")
            
            f.write("\n## 5. Performance Results\n\n")
            
            if self.evaluator:
                comparison_df = self.evaluator.create_comparison_table()
                f.write("### Test Set Performance\n\n")
                f.write(comparison_df.to_markdown(index=False))
                
                best_model = comparison_df.iloc[0]
                f.write(f"\n\n**Best Model:** {best_model['Model']}\n")
                f.write(f"- Accuracy: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)\n")
                f.write(f"- F1-Score: {best_model['F1-Score']:.4f}\n")
            
            f.write("\n## 6. Files Generated\n\n")
            f.write("- Configuration: `config.json`\n")
            f.write("- Trained models: `models/`\n")
            f.write("- Evaluation results: `results/`\n")
            f.write("- Visualizations: `plots/`\n")
        
        print(f"\nFinal report saved: {report_path}")
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        try:
            self.setup_experiment()
            self.load_signals()
            self.extract_features()
            self.preprocess_features()
            self.train_models()
            self.evaluate_models()
            self.generate_report()
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Results saved to: {self.experiment_dir}")
            
        except Exception as e:
            print(f"\nPipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NCV Signal Classification Pipeline')
    parser.add_argument('--config', type=str, default='ncv_signal_config.json',
                       help='Configuration file path')
    parser.add_argument('--data-dir', type=str,
                       help='Override data directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please create ncv_signal_config.json first!")
        return
    
    # Load and update config if needed
    if args.data_dir:
        with open(args.config, 'r') as f:
            config = json.load(f)
        config['data']['base_directory'] = args.data_dir
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
    
    # Run pipeline
    pipeline = NCVSignalPipeline(args.config)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
