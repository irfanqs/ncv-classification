"""
NCV Model Evaluation
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Optional
import os


class NCVEvaluator:
    """Evaluate NCV classification models"""
    
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.results = {}
        
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str) -> Dict:
        """Evaluate single model"""
        # Reshape untuk model
        X_test_reshaped = X_test.reshape(X_test.shape[0], -1, 1)
        
        # Predictions
        y_pred_proba = model.predict(X_test_reshaped, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        return results
    
    def print_classification_report(self, model_name: str):
        """Print detailed classification report"""
        if model_name not in self.results:
            print(f"No results for {model_name}")
            return
        
        result = self.results[model_name]
        
        print(f"\n{'='*60}")
        print(f"Classification Report: {model_name}")
        print(f"{'='*60}")
        print(classification_report(
            result['y_true'],
            result['y_pred'],
            target_names=self.class_names,
            digits=4
        ))
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None):
        """Plot confusion matrix"""
        if model_name not in self.results:
            print(f"No results for {model_name}")
            return
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_confusion_matrices(self, save_dir: str):
        """Plot all confusion matrices in one figure"""
        n_models = len(self.results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nAcc: {result["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'all_confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table untuk semua models"""
        if not self.results:
            return None
        
        data = []
        for model_name, result in self.results.items():
            data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy', ascending=False)
        
        return df
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot bar chart comparison"""
        df = self.create_comparison_table()
        if df is None:
            return
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df[metric], width, label=metric)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(df['Model'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
