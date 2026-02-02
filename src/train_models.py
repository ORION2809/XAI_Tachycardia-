"""
Phase 2: Train All Models for Tachycardia Detection
Trains baseline ML models and deep learning models
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.baseline_models import BaselineClassifier, EnsembleClassifier, train_all_baselines
from models.deep_models import CNN1DClassifier
from models.evaluation import ModelEvaluator


def load_dataset(data_dir: str) -> dict:
    """Load the preprocessed dataset"""
    
    split_path = os.path.join(data_dir, 'processed', 'train_test_split.npz')
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Dataset not found at {split_path}. Run preprocessing pipeline first.")
    
    print("Loading dataset...")
    split = np.load(split_path, allow_pickle=True)
    
    # Load feature names
    with open(os.path.join(data_dir, 'features', 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    data = {
        'X_train': split['X_train'],
        'X_test': split['X_test'],
        'y_train': split['y_train_binary'],
        'y_test': split['y_test_binary'],
        'y_train_multi': split['y_train_multi'],
        'y_test_multi': split['y_test_multi'],
        'beats_train': split['beats_train'],
        'beats_test': split['beats_test'],
        'feature_names': feature_names
    }
    
    print(f"Train set: {data['X_train'].shape[0]} samples, {data['X_train'].shape[1]} features")
    print(f"Test set: {data['X_test'].shape[0]} samples")
    print(f"Train class distribution: {np.bincount(data['y_train'])}")
    print(f"Test class distribution: {np.bincount(data['y_test'])}")
    
    return data


def train_baseline_models(data: dict, save_dir: str) -> dict:
    """Train all baseline ML models"""
    
    print("\n" + "="*70)
    print("PHASE 2A: TRAINING BASELINE MACHINE LEARNING MODELS")
    print("="*70)
    
    results = train_all_baselines(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_test=data['X_test'],
        y_test=data['y_test'],
        feature_names=data['feature_names'],
        save_dir=save_dir
    )
    
    return results


def train_ensemble_model(data: dict, save_dir: str) -> EnsembleClassifier:
    """Train ensemble classifier"""
    
    print("\n" + "="*70)
    print("PHASE 2B: TRAINING ENSEMBLE MODEL")
    print("="*70)
    
    ensemble = EnsembleClassifier(
        models=['random_forest', 'xgboost', 'logistic_regression'],
        voting='soft'
    )
    
    ensemble.fit(data['X_train'], data['y_train'], data['feature_names'])
    
    # Evaluate
    results = ensemble.evaluate(data['X_test'], data['y_test'])
    
    print("\nEnsemble Results:")
    print("-" * 40)
    for metric, value in results['ensemble'].items():
        print(f"  {metric}: {value:.4f}")
    
    return ensemble


def train_cnn_model(data: dict, save_dir: str) -> CNN1DClassifier:
    """Train 1D-CNN on raw beats"""
    
    print("\n" + "="*70)
    print("PHASE 2C: TRAINING 1D-CNN ON RAW BEATS")
    print("="*70)
    
    # Use a subset for faster training (CNN training is slow without GPU)
    max_train = 20000
    max_test = 5000
    
    if len(data['beats_train']) > max_train:
        print(f"Subsampling training data from {len(data['beats_train'])} to {max_train}")
        idx = np.random.choice(len(data['beats_train']), max_train, replace=False)
        beats_train = data['beats_train'][idx]
        y_train = data['y_train'][idx]
    else:
        beats_train = data['beats_train']
        y_train = data['y_train']
    
    if len(data['beats_test']) > max_test:
        idx = np.random.choice(len(data['beats_test']), max_test, replace=False)
        beats_test = data['beats_test'][idx]
        y_test = data['y_test'][idx]
    else:
        beats_test = data['beats_test']
        y_test = data['y_test']
    
    print(f"Training on {len(beats_train)} beats")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    cnn = CNN1DClassifier(
        input_length=beats_train.shape[1],
        n_classes=2,
        learning_rate=0.001
    )
    
    history = cnn.fit(
        beats_train, y_train,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        verbose=True
    )
    
    # Evaluate
    print("\nCNN Test Evaluation:")
    metrics = cnn.evaluate(beats_test, y_test)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return cnn, history


def comprehensive_evaluation(models: dict, data: dict, save_dir: str):
    """Perform comprehensive evaluation of all models"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    evaluator = ModelEvaluator(class_names=['Normal', 'Tachycardia'])
    
    # Evaluate each baseline model
    for model_name, result in models.get('baseline', {}).items():
        clf = result['classifier']
        y_pred = clf.predict(data['X_test'])
        y_proba = clf.predict_proba(data['X_test'])
        
        evaluator.evaluate(data['y_test'], y_pred, y_proba, model_name=model_name)
    
    # Evaluate ensemble
    if 'ensemble' in models:
        ensemble = models['ensemble']
        y_pred = ensemble.predict(data['X_test'])
        y_proba = ensemble.predict_proba(data['X_test'])
        
        evaluator.evaluate(data['y_test'], y_pred, y_proba, model_name='ensemble')
    
    # Print all reports
    for model_name in evaluator.results.keys():
        evaluator.print_report(model_name)
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    comparison = evaluator.compare_models(metric='auc_roc')
    
    # Select columns for display
    display_cols = ['model', 'accuracy', 'sensitivity', 'specificity', 'f1', 'auc_roc']
    display_cols = [c for c in display_cols if c in comparison.columns]
    
    print(comparison[display_cols].to_string(index=False))
    
    # Save results
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    evaluator.save_results(results_path)
    
    # Find best model
    if 'auc_roc' in comparison.columns:
        best_model = comparison.iloc[0]['model']
        best_auc = comparison.iloc[0]['auc_roc']
        print(f"\n🏆 Best Model: {best_model} (AUC-ROC: {best_auc:.4f})")
    
    # Optimal threshold analysis
    print("\n" + "="*70)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("="*70)
    print("Finding threshold for 95% sensitivity (minimizing missed tachycardia):")
    
    for model_name in evaluator.results.keys():
        optimal = evaluator.find_optimal_threshold(
            model_name, 
            target_metric='sensitivity',
            target_value=0.95
        )
        
        if optimal:
            print(f"\n{model_name}:")
            print(f"  Threshold: {optimal['optimal_threshold']:.3f}")
            print(f"  Sensitivity: {optimal['sensitivity']:.4f}")
            print(f"  Specificity: {optimal['specificity']:.4f}")
            print(f"  F1 Score: {optimal['f1']:.4f}")
    
    return evaluator


def save_feature_importance(models: dict, feature_names: list, save_dir: str):
    """Save and display feature importance analysis"""
    
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    all_importance = []
    
    for model_name, result in models.get('baseline', {}).items():
        clf = result['classifier']
        importance = clf.get_feature_importance()
        
        if importance is not None:
            importance['model'] = model_name
            all_importance.append(importance)
            
            print(f"\n{model_name} - Top 15 Features:")
            print(importance.head(15).to_string(index=False))
    
    # Aggregate importance across models
    if all_importance:
        combined = pd.concat(all_importance)
        avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        print("\n" + "-"*50)
        print("AVERAGE FEATURE IMPORTANCE ACROSS MODELS")
        print("-"*50)
        print(avg_importance.head(20).to_string(index=False))
        
        # Save
        avg_importance.to_csv(
            os.path.join(save_dir, 'feature_importance.csv'),
            index=False
        )
        
        # Group by category
        categories = {
            'Heart Rate': ['rr_interval', 'rr_previous', 'heart_rate', 'rr_ratio', 'rr_diff'],
            'QRS Complex': ['qrs_duration_ms', 'qrs_amplitude', 'qrs_area', 'r_amplitude', 'q_amplitude', 's_amplitude'],
            'Statistical': ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'rms', 'energy'],
            'Frequency': ['spectral_centroid', 'spectral_spread', 'spectral_entropy', 'lf_power_ratio', 'mf_power_ratio'],
            'Wavelet': ['wavelet_detail_1_mean', 'wavelet_detail_2_mean', 'wavelet_detail_3_mean']
        }
        
        print("\n" + "-"*50)
        print("IMPORTANCE BY FEATURE CATEGORY")
        print("-"*50)
        
        for category, features in categories.items():
            cat_importance = avg_importance[avg_importance['feature'].isin(features)]['importance'].sum()
            print(f"  {category}: {cat_importance:.4f}")


def main():
    """Main training script"""
    
    print("="*70)
    print("XAI TACHYCARDIA DETECTION - PHASE 2: MODEL TRAINING")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, 'data')
    models_dir = os.path.join(project_dir, 'models')
    results_dir = os.path.join(project_dir, 'results')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    data = load_dataset(data_dir)
    
    # Store all models
    all_models = {}
    
    # Train baseline models
    baseline_results = train_baseline_models(data, models_dir)
    all_models['baseline'] = baseline_results
    
    # Train ensemble
    ensemble = train_ensemble_model(data, models_dir)
    all_models['ensemble'] = ensemble
    
    # Train CNN (optional - slower)
    try:
        cnn, cnn_history = train_cnn_model(data, models_dir)
        all_models['cnn'] = cnn
    except Exception as e:
        print(f"CNN training skipped: {e}")
    
    # Comprehensive evaluation
    evaluator = comprehensive_evaluation(all_models, data, results_dir)
    
    # Feature importance analysis
    save_feature_importance(all_models, data['feature_names'], results_dir)
    
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Models saved to: {models_dir}")
    print(f"Results saved to: {results_dir}")
    
    return all_models, evaluator


if __name__ == '__main__':
    models, evaluator = main()
