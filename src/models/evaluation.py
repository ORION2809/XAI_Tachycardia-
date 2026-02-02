"""
Model Evaluation Module for Tachycardia Detection
Comprehensive evaluation metrics, visualizations, and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, matthews_corrcoef,
    balanced_accuracy_score, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


class ModelEvaluator:
    """
    Comprehensive model evaluation for tachycardia detection
    
    Focuses on clinically relevant metrics:
    - Sensitivity (must be high - don't miss tachycardia)
    - Specificity (reduce false alarms)
    - Positive Predictive Value
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize evaluator
        
        Args:
            class_names: Names for classes (default: ['Normal', 'Tachycardia'])
        """
        self.class_names = class_names or ['Normal', 'Tachycardia']
        self.results = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 y_proba: Optional[np.ndarray] = None,
                 model_name: str = 'model') -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for ROC/PR curves)
            model_name: Name for storing results
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Binary classification specific
        if len(np.unique(y_true)) == 2:
            # Confusion matrix values
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Additional metrics
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
            metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
            
            # Prevalence
            metrics['prevalence'] = (tp + fn) / len(y_true)
            
            # Confusion matrix
            metrics['true_positives'] = int(tp)
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            
            # ROC-AUC and PR-AUC
            if y_proba is not None:
                if len(y_proba.shape) > 1:
                    y_proba_pos = y_proba[:, 1]
                else:
                    y_proba_pos = y_proba
                
                try:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_proba_pos)
                    
                    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba_pos)
                    metrics['auc_pr'] = auc(recall_curve, precision_curve)
                except ValueError:
                    metrics['auc_roc'] = 0.0
                    metrics['auc_pr'] = 0.0
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def print_report(self, model_name: str = 'model'):
        """Print formatted evaluation report"""
        if model_name not in self.results:
            print(f"No results for model: {model_name}")
            return
        
        metrics = self.results[model_name]['metrics']
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        print("\n" + "="*60)
        print(f"EVALUATION REPORT: {model_name}")
        print("="*60)
        
        print("\n📊 CONFUSION MATRIX:")
        cm = confusion_matrix(y_true, y_pred)
        print(f"                 Predicted")
        print(f"              Neg    Pos")
        print(f"Actual Neg  [{cm[0,0]:5d}  {cm[0,1]:5d}]")
        print(f"       Pos  [{cm[1,0]:5d}  {cm[1,1]:5d}]")
        
        print("\n📈 PERFORMANCE METRICS:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        print("\n🎯 CLINICAL METRICS (Critical for Tachycardia Detection):")
        print(f"  Sensitivity (TPR): {metrics.get('sensitivity', 0):.4f}  ⚠️  Don't miss tachycardia!")
        print(f"  Specificity (TNR): {metrics.get('specificity', 0):.4f}")
        print(f"  PPV (Precision):   {metrics.get('ppv', 0):.4f}")
        print(f"  NPV:               {metrics.get('npv', 0):.4f}")
        
        print("\n📉 ERROR RATES:")
        print(f"  False Positive Rate: {metrics.get('fpr', 0):.4f}")
        print(f"  False Negative Rate: {metrics.get('fnr', 0):.4f}  ⚠️  Missed tachycardia!")
        
        print("\n📊 OVERALL SCORES:")
        print(f"  F1 Score:         {metrics['f1']:.4f}")
        print(f"  Matthews Corr:    {metrics.get('mcc', 0):.4f}")
        print(f"  Cohen's Kappa:    {metrics.get('kappa', 0):.4f}")
        
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
            print(f"  AUC-PR:           {metrics.get('auc_pr', 0):.4f}")
        
        print("\n" + "="*60)
    
    def plot_confusion_matrix(self, model_name: str = 'model',
                               save_path: Optional[str] = None):
        """Plot confusion matrix heatmap"""
        if model_name not in self.results:
            return
        
        y_true = self.results[model_name]['y_true']
        y_pred = self.results[model_name]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, model_names: Optional[List[str]] = None,
                       save_path: Optional[str] = None):
        """Plot ROC curves for one or more models"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(8, 6))
        
        for model_name in model_names:
            if model_name not in self.results:
                continue
            
            y_true = self.results[model_name]['y_true']
            y_proba = self.results[model_name]['y_proba']
            
            if y_proba is None:
                continue
            
            if len(y_proba.shape) > 1:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('ROC Curves')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, model_names: Optional[List[str]] = None,
                                     save_path: Optional[str] = None):
        """Plot Precision-Recall curves"""
        if model_names is None:
            model_names = list(self.results.keys())
        
        plt.figure(figsize=(8, 6))
        
        for model_name in model_names:
            if model_name not in self.results:
                continue
            
            y_true = self.results[model_name]['y_true']
            y_proba = self.results[model_name]['y_proba']
            
            if y_proba is None:
                continue
            
            if len(y_proba.shape) > 1:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba_pos)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall (Sensitivity)')
        plt.ylabel('Precision (PPV)')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_models(self, metric: str = 'f1') -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Args:
            metric: Metric to sort by
            
        Returns:
            DataFrame with model comparison
        """
        data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {'model': model_name}
            row.update(metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=False)
        
        return df
    
    def save_results(self, path: str):
        """Save evaluation results to JSON"""
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        
        for model_name, result in self.results.items():
            results_json[model_name] = {
                'metrics': result['metrics']
            }
        
        with open(path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"Results saved to {path}")
    
    def find_optimal_threshold(self, model_name: str = 'model',
                                target_metric: str = 'sensitivity',
                                target_value: float = 0.95) -> Dict:
        """
        Find optimal classification threshold
        
        Args:
            model_name: Model to analyze
            target_metric: 'sensitivity' or 'specificity'
            target_value: Target value for the metric
            
        Returns:
            Dictionary with optimal threshold and metrics
        """
        if model_name not in self.results:
            return {}
        
        y_true = self.results[model_name]['y_true']
        y_proba = self.results[model_name]['y_proba']
        
        if y_proba is None:
            return {}
        
        if len(y_proba.shape) > 1:
            y_proba_pos = y_proba[:, 1]
        else:
            y_proba_pos = y_proba
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba_pos)
        
        if target_metric == 'sensitivity':
            # Find threshold where sensitivity >= target
            valid_idx = np.where(tpr >= target_value)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmin(fpr[valid_idx])]
            else:
                best_idx = np.argmax(tpr)
        else:
            # Find threshold where specificity >= target
            valid_idx = np.where((1 - fpr) >= target_value)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(tpr[valid_idx])]
            else:
                best_idx = np.argmin(fpr)
        
        optimal_threshold = thresholds[best_idx]
        
        # Compute metrics at this threshold
        y_pred_optimal = (y_proba_pos >= optimal_threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()
        
        result = {
            'optimal_threshold': optimal_threshold,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1': f1_score(y_true, y_pred_optimal),
            'accuracy': accuracy_score(y_true, y_pred_optimal)
        }
        
        return result


def evaluate_all_models(models: Dict, X_test: np.ndarray, y_test: np.ndarray,
                        feature_based: bool = True) -> ModelEvaluator:
    """
    Evaluate all models in a dictionary
    
    Args:
        models: Dictionary of model name -> model object
        X_test: Test features or beats
        y_test: Test labels
        feature_based: Whether X_test contains features (True) or beats (False)
        
    Returns:
        ModelEvaluator with all results
    """
    evaluator = ModelEvaluator()
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            y_proba = None
        
        evaluator.evaluate(y_test, y_pred, y_proba, model_name=name)
        evaluator.print_report(name)
    
    return evaluator


def main():
    """Test evaluation module"""
    # Generate synthetic predictions
    np.random.seed(42)
    
    n_samples = 1000
    n_positive = 100  # Imbalanced
    
    y_true = np.zeros(n_samples, dtype=int)
    y_true[:n_positive] = 1
    np.random.shuffle(y_true)
    
    # Simulate model predictions
    y_proba = np.random.rand(n_samples)
    y_proba[y_true == 1] += 0.3  # Positives have higher probability
    y_proba = np.clip(y_proba, 0, 1)
    
    y_pred = (y_proba > 0.5).astype(int)
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba, model_name='test_model')
    
    evaluator.print_report('test_model')
    
    # Find optimal threshold
    print("\nOptimal Threshold Analysis:")
    optimal = evaluator.find_optimal_threshold('test_model', 
                                                target_metric='sensitivity',
                                                target_value=0.95)
    print(f"  Threshold: {optimal['optimal_threshold']:.3f}")
    print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
    print(f"  Specificity: {optimal['specificity']:.4f}")


if __name__ == '__main__':
    main()
