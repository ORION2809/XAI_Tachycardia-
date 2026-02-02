"""
Baseline Machine Learning Models for Tachycardia Detection
Implements interpretable models suitable for XAI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json
import os


class BaselineClassifier:
    """
    Baseline classifier wrapper for tachycardia detection
    
    Supports multiple algorithms with built-in:
    - Class imbalance handling
    - Feature scaling
    - Cross-validation
    - Model persistence
    """
    
    AVAILABLE_MODELS = {
        'random_forest': RandomForestClassifier,
        'xgboost': GradientBoostingClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'decision_tree': DecisionTreeClassifier,
        'knn': KNeighborsClassifier
    }
    
    RESAMPLING_METHODS = {
        'smote': SMOTE,
        'adasyn': ADASYN,
        'undersample': RandomUnderSampler,
        'smote_tomek': SMOTETomek
    }
    
    def __init__(self, model_type: str = 'random_forest',
                 resampling: Optional[str] = 'smote',
                 random_state: int = 42):
        """
        Initialize baseline classifier
        
        Args:
            model_type: Type of classifier to use
            resampling: Resampling method for class imbalance (None to disable)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.resampling = resampling
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.model = None
        self.resampler = None
        self.feature_names = None
        self.is_fitted = False
        
        self._init_model()
        self._init_resampler()
    
    def _init_model(self):
        """Initialize the classifier model"""
        if self.model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available: {list(self.AVAILABLE_MODELS.keys())}")
        
        model_class = self.AVAILABLE_MODELS[self.model_type]
        
        # Model-specific configurations
        if self.model_type == 'random_forest':
            self.model = model_class(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = model_class(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            )
        elif self.model_type == 'logistic_regression':
            self.model = model_class(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_type == 'svm':
            self.model = model_class(
                C=1.0,
                kernel='rbf',
                class_weight='balanced',
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == 'decision_tree':
            self.model = model_class(
                max_depth=15,
                min_samples_split=10,
                class_weight='balanced',
                random_state=self.random_state
            )
        elif self.model_type == 'knn':
            self.model = model_class(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            )
    
    def _init_resampler(self):
        """Initialize the resampling method"""
        if self.resampling is None:
            self.resampler = None
        elif self.resampling not in self.RESAMPLING_METHODS:
            raise ValueError(f"Unknown resampling method: {self.resampling}. "
                           f"Available: {list(self.RESAMPLING_METHODS.keys())}")
        else:
            resampler_class = self.RESAMPLING_METHODS[self.resampling]
            self.resampler = resampler_class(random_state=self.random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'BaselineClassifier':
        """
        Fit the classifier
        
        Args:
            X: Feature matrix [n_samples, n_features]
            y: Labels [n_samples]
            feature_names: Optional list of feature names
            
        Returns:
            self
        """
        self.feature_names = feature_names
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply resampling
        if self.resampler is not None:
            print(f"Applying {self.resampling} resampling...")
            print(f"  Before: {np.bincount(y)}")
            X_resampled, y_resampled = self.resampler.fit_resample(X_scaled, y)
            print(f"  After: {np.bincount(y_resampled)}")
        else:
            X_resampled, y_resampled = X_scaled, y
        
        # Fit model
        print(f"Training {self.model_type}...")
        self.model.fit(X_resampled, y_resampled)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            # For models without predict_proba
            predictions = self.model.predict(X_scaled)
            return np.eye(2)[predictions]
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'sensitivity': recall_score(y, y_pred, pos_label=1, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0)
        }
        
        if y_proba is not None and len(np.unique(y)) == 2:
            try:
                metrics['auc_roc'] = roc_auc_score(y, y_proba)
            except ValueError:
                metrics['auc_roc'] = 0.0
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       cv: int = 5) -> Dict[str, np.ndarray]:
        """
        Perform cross-validation
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply resampling within each fold
        if self.resampler is not None:
            X_resampled, y_resampled = self.resampler.fit_resample(X_scaled, y)
        else:
            X_resampled, y_resampled = X_scaled, y
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'accuracy': cross_val_score(self.model, X_resampled, y_resampled, 
                                       cv=skf, scoring='accuracy'),
            'f1': cross_val_score(self.model, X_resampled, y_resampled, 
                                 cv=skf, scoring='f1_weighted'),
            'recall': cross_val_score(self.model, X_resampled, y_resampled, 
                                     cv=skf, scoring='recall_weighted'),
            'roc_auc': cross_val_score(self.model, X_resampled, y_resampled, 
                                       cv=skf, scoring='roc_auc')
        }
        
        return scores
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (if available)
        
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            return None
        
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).flatten()
        
        if importance is None:
            return None
        
        feature_names = self.feature_names or [f'feature_{i}' for i in range(len(importance))]
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, path: str):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'resampling': self.resampling,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaselineClassifier':
        """Load model from disk"""
        model_data = joblib.load(path)
        
        classifier = cls(
            model_type=model_data['model_type'],
            resampling=model_data['resampling']
        )
        
        classifier.model = model_data['model']
        classifier.scaler = model_data['scaler']
        classifier.feature_names = model_data['feature_names']
        classifier.is_fitted = model_data['is_fitted']
        
        return classifier


class EnsembleClassifier:
    """
    Ensemble of multiple classifiers for robust predictions
    """
    
    def __init__(self, models: Optional[List[str]] = None,
                 voting: str = 'soft',
                 random_state: int = 42):
        """
        Initialize ensemble
        
        Args:
            models: List of model types to include
            voting: 'hard' or 'soft' voting
            random_state: Random seed
        """
        self.model_types = models or ['random_forest', 'xgboost', 'logistic_regression']
        self.voting = voting
        self.random_state = random_state
        
        self.classifiers = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None) -> 'EnsembleClassifier':
        """Fit all classifiers in the ensemble"""
        
        for model_type in self.model_types:
            print(f"\n{'='*50}")
            print(f"Training {model_type}...")
            print('='*50)
            
            clf = BaselineClassifier(
                model_type=model_type,
                resampling='smote',
                random_state=self.random_state
            )
            clf.fit(X, y, feature_names)
            self.classifiers[model_type] = clf
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction using voting"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")
        
        if self.voting == 'hard':
            predictions = np.array([clf.predict(X) for clf in self.classifiers.values()])
            # Majority voting
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
        else:  # soft voting
            probas = np.array([clf.predict_proba(X)[:, 1] for clf in self.classifiers.values()])
            avg_proba = np.mean(probas, axis=0)
            return (avg_proba > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average probability predictions"""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted.")
        
        probas = np.array([clf.predict_proba(X) for clf in self.classifiers.values()])
        return np.mean(probas, axis=0)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Evaluate ensemble and individual models"""
        results = {'individual': {}, 'ensemble': {}}
        
        # Evaluate individual models
        for name, clf in self.classifiers.items():
            results['individual'][name] = clf.evaluate(X, y)
        
        # Evaluate ensemble
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'sensitivity': recall_score(y, y_pred, pos_label=1, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0),
            'auc_roc': roc_auc_score(y, y_proba)
        }
        
        return results


def train_all_baselines(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        save_dir: Optional[str] = None) -> Dict:
    """
    Train and evaluate all baseline models
    
    Returns:
        Dictionary with all results
    """
    results = {}
    
    model_types = ['random_forest', 'xgboost', 'logistic_regression', 'decision_tree']
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()}")
        print('='*60)
        
        clf = BaselineClassifier(
            model_type=model_type,
            resampling='smote',
            random_state=42
        )
        
        clf.fit(X_train, y_train, feature_names)
        
        # Evaluate
        train_metrics = clf.evaluate(X_train, y_train)
        test_metrics = clf.evaluate(X_test, y_test)
        
        print(f"\nTrain Metrics:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        print(f"\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # Feature importance
        importance = clf.get_feature_importance()
        if importance is not None:
            print(f"\nTop 10 Features:")
            print(importance.head(10).to_string(index=False))
        
        results[model_type] = {
            'classifier': clf,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'feature_importance': importance
        }
        
        # Save model
        if save_dir:
            clf.save(os.path.join(save_dir, f'{model_type}_model.joblib'))
    
    return results


def main():
    """Test baseline models"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    
    if not os.path.exists(os.path.join(data_dir, 'train_test_split.npz')):
        print("Dataset not found. Run the preprocessing pipeline first.")
        return
    
    print("Loading dataset...")
    split = np.load(os.path.join(data_dir, 'train_test_split.npz'), allow_pickle=True)
    
    X_train = split['X_train']
    X_test = split['X_test']
    y_train = split['y_train_binary']
    y_test = split['y_test_binary']
    
    with open(os.path.join(data_dir, '..', 'features', 'feature_names.json'), 'r') as f:
        feature_names = json.load(f)
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    print(f"Test class distribution: {np.bincount(y_test)}")
    
    # Train models
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    results = train_all_baselines(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        save_dir=save_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - TEST SET PERFORMANCE")
    print("="*60)
    
    for model_type, result in results.items():
        metrics = result['test_metrics']
        print(f"\n{model_type}:")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  F1 Score:    {metrics['f1']:.4f}")
        print(f"  AUC-ROC:     {metrics.get('auc_roc', 'N/A'):.4f}")


if __name__ == '__main__':
    main()
