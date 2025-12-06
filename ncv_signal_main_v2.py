"""
NCV Signal Classification V2
Using spectrogram approach (like EMG system)
"""
import os
import json
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import EMG preprocessor (works for any time-series signal)
from data_preprocessing import EMGPreprocessor

from ncv_signal_data_loader import NCVSignalDataLoader
from src.ncv_models import NCVModelBuilder, compile_model
from src.ncv_train import NCVTrainer
from src.ncv_evaluate import NCVEvaluator


class NCVSignalPipelineV2:
    """NCV Signal pipeline using spectrogram approach"""
    
    def __init__(self, config_path: str = 'ncv_signal_config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.experiment_dir = None
        self.signals = None
        self.y = None
        self.spectrograms = None
        self.preprocessor = None
        self.split_data = None
        self.models = {}
        self.evaluator = None
        
    def setup_experiment(self):
        """Setup experiment directory"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = f"ncv_signal_v2_{timestamp}"
        
        exp_dir = os.path.join('experiments_ncv_signal_v2', exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
        
        self.experiment_dir = exp_dir
        
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"NCV SIGNAL CLASSIFICATION V2 (Spectrogram Approach)")
        print(f"{'='*80}")
        print(f"Experiment: {exp_name}")
        
        return exp_dir
    
    def load_signals(self):
        """Load NCV signal waveforms"""
        print(f"\n{'='*80}")
        print("STEP 1: LOADING NCV SIGNAL DATA")
        print(f"{'='*80}")
        
        loader = NCVSignalDataLoader(self.config['data']['base_directory'], self.config)
        self.signals, self.y, _ = loader.load_all_signals(
            max_files_per_class=self.config['data'].get('max_files_per_class')
        )
        
        print(f"\nLoaded {len(self.signals)} signals")
        return self.signals, self.y
    
    def generate_spectrograms(self):
        """Generate spectrograms from waveforms"""
        print(f"\n{'='*80}")
        print("STEP 2: GENERATING SPECTROGRAMS")
        print(f"{'='*80}")
        
        # Initialize EMG preprocessor (works for any time-series)
        sampling_rate = self.config['signal_processing']['sampling_rate']
        
        # Calculate segment length from signal
        avg_length = np.mean([len(s) for s in self.signals])
        segment_length_sec = avg_length / sampling_rate
        
        print(f"\nSignal characteristics:")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Average length: {avg_length:.0f} samples")
        print(f"  Duration: {segment_length_sec*1000:.1f} ms")
        
        self.preprocessor = EMGPreprocessor(
            sampling_rate=sampling_rate,
            segment_length=segment_length_sec,
            window_size=64,  # Smaller window for short signals
            n_frames=20
        )
        
        all_spectrograms = []
        all_labels = []
        
        print("\nProcessing signals to spectrograms...")
        for idx, (signal, label) in enumerate(zip(self.signals, self.y)):
            # Process signal to spectrogram
            spectrograms, _ = self.preprocessor.process_signal_to_spectrogram(
                signal,
                apply_bandpass=True,
                apply_notch=False,
                normalize_method='zscore',
                segment_overlap=0.5,
                augment_with_noise=False,  # No augmentation for now
                augment_time_shift=False,
                augment_amplitude=False
            )
            
            if spectrograms is not None and len(spectrograms) > 0:
                # Debug first spectrogram shape
                if idx == 0:
                    print(f"\n  First spectrogram shape: {spectrograms[0].shape}")
                    print(f"  Number of spectrograms from first signal: {len(spectrograms)}")
                
                # Process each spectrogram
                for spec in spectrograms:
                    # Ensure 2D spectrogram
                    if len(spec.shape) > 2:
                        spec = spec.squeeze()
                    
                    # Resize to 256x256 if needed
                    if spec.shape != (256, 256):
                        from scipy.ndimage import zoom
                        zoom_factors = (256 / spec.shape[0], 256 / spec.shape[1])
                        spec = zoom(spec, zoom_factors, order=1)
                    
                    all_spectrograms.append(spec)
                all_labels.extend([label] * len(spectrograms))
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(self.signals)} signals")
        
        if not all_spectrograms:
            raise ValueError("No spectrograms generated!")
        
        # Check shape before converting to array
        print(f"\n  Total spectrograms collected: {len(all_spectrograms)}")
        print(f"  Sample spectrogram shape: {all_spectrograms[0].shape}")
        
        self.spectrograms = np.array(all_spectrograms)
        self.y_spectrograms = np.array(all_labels)
        
        print(f"\n  After np.array conversion: {self.spectrograms.shape}")
        
        # Ensure correct shape for Conv2D: (n_samples, height, width, channels)
        if len(self.spectrograms.shape) == 3:
            # Shape is (n_samples, height, width) -> add channel
            self.spectrograms = np.expand_dims(self.spectrograms, axis=-1)
            print(f"  ✓ Added channel dimension: {self.spectrograms.shape}")
        elif len(self.spectrograms.shape) == 4:
            print(f"  ✓ Already has 4 dimensions: {self.spectrograms.shape}")
        else:
            raise ValueError(f"Unexpected spectrogram shape: {self.spectrograms.shape}")
        
        print(f"\nGenerated {len(self.spectrograms)} spectrograms")
        print(f"  Final shape: {self.spectrograms.shape}")
        
        return self.spectrograms, self.y_spectrograms
    
    def train_models(self):
        """Train models on spectrograms"""
        print(f"\n{'='*80}")
        print("STEP 3: TRAINING MODELS")
        print(f"{'='*80}")
        
        # Split data
        trainer = NCVTrainer(self.config)
        self.split_data = trainer.split_data(self.spectrograms, self.y_spectrograms)
        
        # Build and train models
        input_shape = self.spectrograms.shape[1:]
        n_classes = len(self.config['data']['classes'])
        
        models_to_train = [
            {'name': 'ncv_signal_cnn_v2', 'type': 'standard_cnn'},
            {'name': 'ncv_signal_lstm_v2', 'type': 'standard_lstm'},
            {'name': 'ncv_signal_hybrid_v2', 'type': 'standard_cnn_lstm'}
        ]
        
        for model_cfg in models_to_train:
            model_name = model_cfg['name']
            model_type = model_cfg['type']
            
            print(f"\n{'='*60}")
            print(f"Training: {model_name}")
            print(f"{'='*60}")
            
            # Build model
            builder = NCVModelBuilder(input_shape=input_shape, num_classes=n_classes)
            
            if model_type == 'standard_cnn':
                model = builder.build_standard_cnn()
            elif model_type == 'standard_lstm':
                model = builder.build_standard_lstm()
            elif model_type == 'standard_cnn_lstm':
                model = builder.build_standard_cnn_lstm()
            
            # Compile
            lr = self.config['training']['hyperparameters']['learning_rate']
            model = compile_model(model, learning_rate=lr)
            
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
        """Evaluate models"""
        print(f"\n{'='*80}")
        print("STEP 4: EVALUATION")
        print(f"{'='*80}")
        
        class_names = self.config['data']['classes']
        self.evaluator = NCVEvaluator(class_names)
        
        for model_name, model_info in self.models.items():
            print(f"\nEvaluating {model_name}...")
            
            self.evaluator.evaluate_model(
                model_info['model'],
                self.split_data['X_test'],
                self.split_data['y_test'],
                model_name
            )
            
            self.evaluator.print_classification_report(model_name)
        
        # Comparison
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        comparison_df = self.evaluator.create_comparison_table()
        print(comparison_df.to_string(index=False))
        
        # Save results
        results_dir = os.path.join(self.experiment_dir, 'results')
        comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
        
        # Plot
        plots_dir = os.path.join(self.experiment_dir, 'plots')
        self.evaluator.plot_all_confusion_matrices(plots_dir)
        self.evaluator.plot_model_comparison(os.path.join(plots_dir, 'model_comparison.png'))
        
        return self.evaluator.results
    
    def run_complete_pipeline(self):
        """Run complete pipeline"""
        try:
            self.setup_experiment()
            self.load_signals()
            self.generate_spectrograms()
            self.train_models()
            self.evaluate_models()
            
            print(f"\n{'='*80}")
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print(f"{'='*80}")
            print(f"Results: {self.experiment_dir}")
            
        except Exception as e:
            print(f"\nPipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='NCV Signal V2 Pipeline')
    parser.add_argument('--config', type=str, default='ncv_signal_config.json')
    parser.add_argument('--data-dir', type=str, help='Override data directory')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        return
    
    if args.data_dir:
        with open(args.config, 'r') as f:
            config = json.load(f)
        config['data']['base_directory'] = args.data_dir
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
    
    pipeline = NCVSignalPipelineV2(args.config)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
