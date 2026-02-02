"""
SHAP-based Feature Attribution for Tachycardia Detection.

v2.4: Proper XAI using SHAP values for feature-level explanations.

Key Components:
- DeepSHAP: Fast approximation for deep learning models
- KernelSHAP: Model-agnostic explanation (slower but more accurate)
- FeatureAttributor: Unified interface for feature attributions
- ExplanationReport: Human-readable explanation generation

CRITICAL: Attention weights are NOT reliable explanations.
SHAP values satisfy theoretical axioms (local accuracy, missingness, consistency).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# SHAP VALUE RESULT STRUCTURES
# =============================================================================

@dataclass
class SHAPResult:
    """Result from SHAP computation."""
    shap_values: np.ndarray           # (n_samples, n_features) or (n_samples, seq_len)
    base_value: float                 # Expected model output
    feature_names: Optional[List[str]] = None
    target_class: int = 0
    method: str = "deep"
    
    @property
    def top_features(self) -> List[Tuple[str, float]]:
        """Get top contributing features."""
        if self.feature_names is None:
            names = [f"feature_{i}" for i in range(len(self.shap_values))]
        else:
            names = self.feature_names
        
        # Average absolute SHAP values across samples
        if self.shap_values.ndim > 1:
            importance = np.abs(self.shap_values).mean(axis=0)
        else:
            importance = np.abs(self.shap_values)
        
        indexed = list(zip(names, importance))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed
    
    def get_explanation_text(self, top_k: int = 5) -> str:
        """Generate human-readable explanation."""
        top = self.top_features[:top_k]
        
        lines = [f"Top {top_k} contributing factors:"]
        for name, importance in top:
            lines.append(f"  - {name}: {importance:.3f}")
        
        return "\n".join(lines)


@dataclass
class TemporalSHAPResult:
    """SHAP result for temporal/signal data."""
    shap_values: np.ndarray           # (seq_len,) or (batch, seq_len)
    base_value: float
    target_class: int
    sample_rate: int = 360
    
    def get_important_regions(
        self,
        window_size_sec: float = 0.2,
        top_k: int = 5,
    ) -> List[Tuple[float, float, float]]:
        """
        Get temporally important regions.
        
        Returns:
            List of (start_sec, end_sec, importance) tuples
        """
        window_samples = int(window_size_sec * self.sample_rate)
        
        if self.shap_values.ndim > 1:
            values = self.shap_values.mean(axis=0)
        else:
            values = self.shap_values
        
        n_windows = len(values) // window_samples
        window_importance = []
        
        for i in range(n_windows):
            start_sample = i * window_samples
            end_sample = (i + 1) * window_samples
            importance = np.abs(values[start_sample:end_sample]).sum()
            
            start_sec = start_sample / self.sample_rate
            end_sec = end_sample / self.sample_rate
            window_importance.append((start_sec, end_sec, importance))
        
        window_importance.sort(key=lambda x: x[2], reverse=True)
        return window_importance[:top_k]


# =============================================================================
# DEEP SHAP FOR NEURAL NETWORKS
# =============================================================================

class DeepSHAP:
    """
    DeepSHAP approximation for neural networks.
    
    Uses the DeepLIFT algorithm modified for Shapley value computation.
    Much faster than KernelSHAP for deep learning models.
    
    Reference: Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: Optional[torch.Tensor] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        self.background_data = background_data
        
        if background_data is not None:
            self.background_data = background_data.to(device)
            self._compute_expected_value()
    
    def set_background(self, data: torch.Tensor, n_samples: int = 100):
        """Set background dataset for expectation computation."""
        if len(data) > n_samples:
            indices = np.random.choice(len(data), n_samples, replace=False)
            data = data[indices]
        
        self.background_data = data.to(self.device)
        self._compute_expected_value()
    
    def _compute_expected_value(self):
        """Compute expected model output on background data."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.background_data)
            # Average over all dimensions except class
            if outputs.ndim == 3:  # (batch, seq, classes)
                self.expected_value = outputs.mean(dim=(0, 1)).cpu().numpy()
            else:
                self.expected_value = outputs.mean(dim=0).cpu().numpy()
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: int,
        n_samples: int = 50,
    ) -> TemporalSHAPResult:
        """
        Compute SHAP values for input using DeepSHAP approximation.
        
        Args:
            x: Input tensor (1, channels, seq_len)
            target_class: Class to explain
            n_samples: Number of background samples to use
            
        Returns:
            TemporalSHAPResult with SHAP values
        """
        if self.background_data is None:
            raise ValueError("Must set background data first")
        
        x = x.to(self.device)
        self.model.eval()
        
        # Sample from background
        n_bg = min(n_samples, len(self.background_data))
        indices = np.random.choice(len(self.background_data), n_bg, replace=False)
        background_sample = self.background_data[indices]
        
        # Compute SHAP via gradient-based DeepLIFT approximation
        shap_values = self._deep_lift_shap(x, background_sample, target_class)
        
        return TemporalSHAPResult(
            shap_values=shap_values,
            base_value=float(self.expected_value[target_class]),
            target_class=target_class,
        )
    
    def _deep_lift_shap(
        self,
        x: torch.Tensor,
        background: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute SHAP values using DeepLIFT-based approximation.
        
        For each background sample, compute contribution of each input feature.
        """
        n_bg = len(background)
        seq_len = x.shape[-1]
        
        shap_contributions = np.zeros(seq_len)
        
        for i in range(n_bg):
            bg = background[i:i+1]
            
            # Compute gradient along path from background to input
            contribution = self._compute_attribution_path(x, bg, target_class)
            shap_contributions += contribution
        
        # Average over background samples
        shap_contributions /= n_bg
        
        return shap_contributions
    
    def _compute_attribution_path(
        self,
        x: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        n_steps: int = 20,
    ) -> np.ndarray:
        """Compute attribution along interpolation path (similar to IG)."""
        alphas = np.linspace(0, 1, n_steps)
        
        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated = interpolated.clone().requires_grad_(True)
            
            output = self.model(interpolated)
            
            # Handle sequence output
            if output.ndim == 3:
                target_output = output[:, :, target_class].mean()
            else:
                target_output = output[:, target_class]
            
            target_output.backward()
            gradients.append(interpolated.grad.squeeze().cpu().numpy())
        
        # Integrate gradients
        avg_gradients = np.mean(gradients, axis=0)
        
        # Attribution = (x - baseline) * avg_gradients
        diff = (x - baseline).squeeze().cpu().numpy()
        attribution = diff * avg_gradients
        
        return attribution


# =============================================================================
# KERNEL SHAP (MODEL-AGNOSTIC)
# =============================================================================

class KernelSHAP:
    """
    KernelSHAP for model-agnostic explanations.
    
    Works with any model (including sklearn, XGBoost, etc.).
    Slower than DeepSHAP but more accurate and widely applicable.
    
    Uses a weighted linear regression to approximate Shapley values.
    """
    
    def __init__(
        self,
        model_fn: callable,
        background_data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            model_fn: Function that takes input and returns predictions
            background_data: Background dataset (n_samples, n_features)
            feature_names: Names for each feature
        """
        self.model_fn = model_fn
        self.background_data = background_data
        self.feature_names = feature_names
        
        # Compute expected value
        self.expected_value = self.model_fn(background_data).mean(axis=0)
    
    def explain(
        self,
        x: np.ndarray,
        target_class: int = 0,
        n_samples: int = 1000,
    ) -> SHAPResult:
        """
        Compute SHAP values using KernelSHAP.
        
        Args:
            x: Input sample (n_features,)
            target_class: Class to explain
            n_samples: Number of coalition samples
            
        Returns:
            SHAPResult with SHAP values
        """
        n_features = len(x)
        
        # Generate coalition samples
        coalitions, weights = self._generate_coalitions(n_features, n_samples)
        
        # Compute model outputs for each coalition
        outputs = []
        for coalition in coalitions:
            # Replace missing features with background
            masked_x = self._mask_input(x, coalition)
            output = self.model_fn(masked_x.reshape(1, -1))
            outputs.append(output[0, target_class])
        
        outputs = np.array(outputs)
        
        # Solve weighted linear regression
        shap_values = self._solve_weighted_regression(
            coalitions, outputs, weights, n_features
        )
        
        return SHAPResult(
            shap_values=shap_values,
            base_value=float(self.expected_value[target_class]),
            feature_names=self.feature_names,
            target_class=target_class,
            method="kernel",
        )
    
    def _generate_coalitions(
        self,
        n_features: int,
        n_samples: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random coalitions with SHAP kernel weights."""
        coalitions = []
        weights = []
        
        for _ in range(n_samples):
            # Random coalition size
            size = np.random.randint(1, n_features)
            
            # Random features in coalition
            coalition = np.zeros(n_features)
            indices = np.random.choice(n_features, size, replace=False)
            coalition[indices] = 1
            
            # SHAP kernel weight
            weight = self._shap_kernel_weight(size, n_features)
            
            coalitions.append(coalition)
            weights.append(weight)
        
        return np.array(coalitions), np.array(weights)
    
    def _shap_kernel_weight(self, size: int, n_features: int) -> float:
        """Compute SHAP kernel weight for coalition of given size."""
        if size == 0 or size == n_features:
            return 1e6  # Large weight for full/empty coalitions
        
        from math import comb
        return (n_features - 1) / (comb(n_features, size) * size * (n_features - size))
    
    def _mask_input(self, x: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """Replace missing features with background expectation."""
        masked = x.copy()
        missing_mask = coalition == 0
        
        # Use mean of background for missing features
        masked[missing_mask] = self.background_data[:, missing_mask].mean(axis=0)
        
        return masked
    
    def _solve_weighted_regression(
        self,
        coalitions: np.ndarray,
        outputs: np.ndarray,
        weights: np.ndarray,
        n_features: int,
    ) -> np.ndarray:
        """Solve weighted least squares for SHAP values."""
        # Add intercept
        X = np.column_stack([np.ones(len(coalitions)), coalitions])
        
        # Weighted least squares: (X'WX)^-1 X'Wy
        W = np.diag(weights)
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ outputs
        
        try:
            coeffs = np.linalg.solve(XtWX, XtWy)
        except np.linalg.LinAlgError:
            # Regularized solution
            coeffs = np.linalg.lstsq(XtWX + 1e-6 * np.eye(n_features + 1), XtWy, rcond=None)[0]
        
        # SHAP values are coefficients (excluding intercept)
        return coeffs[1:]


# =============================================================================
# FEATURE-LEVEL EXPLANATION FOR HRV FEATURES
# =============================================================================

class HRVFeatureExplainer:
    """
    SHAP-based explainer for HRV feature models.
    
    Provides interpretable explanations for models that use
    extracted HRV features rather than raw signals.
    """
    
    # Standard HRV feature names
    HRV_FEATURE_NAMES = [
        "mean_rr", "std_rr", "rmssd", "pnn50",
        "sdnn", "sdsd", "cv_rr",
        "lf_power", "hf_power", "lf_hf_ratio",
        "mean_hr", "std_hr", "hr_range",
        "sample_entropy", "approx_entropy",
        "dfa_alpha1", "dfa_alpha2",
    ]
    
    def __init__(
        self,
        model,
        background_features: np.ndarray,
        feature_names: List[str] = None,
    ):
        self.model = model
        self.background_features = background_features
        self.feature_names = feature_names or self.HRV_FEATURE_NAMES[:background_features.shape[1]]
        
        # Create prediction function
        if hasattr(model, 'predict_proba'):
            self.predict_fn = model.predict_proba
        else:
            self.predict_fn = lambda x: model.predict(x).reshape(-1, 1)
        
        self.explainer = KernelSHAP(
            self.predict_fn,
            background_features,
            self.feature_names,
        )
    
    def explain(
        self,
        features: np.ndarray,
        target_class: int = 1,
    ) -> SHAPResult:
        """Explain HRV feature prediction."""
        return self.explainer.explain(features, target_class)
    
    def explain_batch(
        self,
        features: np.ndarray,
        target_class: int = 1,
    ) -> List[SHAPResult]:
        """Explain multiple samples."""
        return [self.explain(f, target_class) for f in features]
    
    def get_global_importance(
        self,
        features: np.ndarray,
        target_class: int = 1,
        n_samples: int = 100,
    ) -> Dict[str, float]:
        """
        Compute global feature importance across dataset.
        
        Returns mean absolute SHAP value per feature.
        """
        if len(features) > n_samples:
            indices = np.random.choice(len(features), n_samples, replace=False)
            features = features[indices]
        
        all_shap = []
        for f in features:
            result = self.explain(f, target_class)
            all_shap.append(result.shap_values)
        
        all_shap = np.array(all_shap)
        mean_importance = np.abs(all_shap).mean(axis=0)
        
        return dict(zip(self.feature_names, mean_importance))


# =============================================================================
# EXPLANATION REPORT GENERATOR
# =============================================================================

class ExplanationReportGenerator:
    """
    Generate human-readable explanation reports.
    
    Combines SHAP values with clinical context for
    interpretable explanations.
    """
    
    def __init__(self):
        # Clinical context for features
        self.feature_context = {
            "mean_rr": "Average time between heartbeats",
            "std_rr": "Variability in heartbeat timing",
            "rmssd": "Short-term heart rate variability",
            "pnn50": "Percentage of successive RR intervals differing by >50ms",
            "lf_power": "Low frequency power (sympathetic activity)",
            "hf_power": "High frequency power (parasympathetic activity)",
            "lf_hf_ratio": "Sympathovagal balance",
            "mean_hr": "Average heart rate",
            "sample_entropy": "Signal complexity/regularity",
        }
        
        # Thresholds for feature interpretation
        self.abnormal_thresholds = {
            "mean_hr": (60, 100),  # Normal range
            "rmssd": (20, 100),
            "lf_hf_ratio": (0.5, 2.0),
        }
    
    def generate_report(
        self,
        shap_result: SHAPResult,
        feature_values: np.ndarray,
        prediction: str,
        confidence: float,
    ) -> str:
        """
        Generate comprehensive explanation report.
        
        Args:
            shap_result: SHAP values for the prediction
            feature_values: Actual feature values
            prediction: Model prediction (e.g., "VT")
            confidence: Prediction confidence
            
        Returns:
            Human-readable report string
        """
        lines = [
            "=" * 60,
            "TACHYCARDIA DETECTION EXPLANATION REPORT",
            "=" * 60,
            "",
            f"Prediction: {prediction}",
            f"Confidence: {confidence:.1%}",
            "",
            "-" * 40,
            "KEY CONTRIBUTING FACTORS",
            "-" * 40,
        ]
        
        # Get top features
        top_features = shap_result.top_features[:5]
        
        for i, (name, importance) in enumerate(top_features, 1):
            # Get feature value
            if shap_result.feature_names:
                idx = shap_result.feature_names.index(name)
                value = feature_values[idx]
            else:
                value = "N/A"
            
            # Direction
            if shap_result.shap_values.ndim > 1:
                direction = "increases" if shap_result.shap_values[:, i-1].mean() > 0 else "decreases"
            else:
                direction = "increases" if shap_result.shap_values[i-1] > 0 else "decreases"
            
            # Clinical context
            context = self.feature_context.get(name, "")
            
            lines.append(f"\n{i}. {name}")
            lines.append(f"   Value: {value:.3f}" if isinstance(value, float) else f"   Value: {value}")
            lines.append(f"   Impact: {importance:.3f} ({direction} likelihood)")
            if context:
                lines.append(f"   Context: {context}")
            
            # Check if abnormal
            if name in self.abnormal_thresholds and isinstance(value, (int, float)):
                low, high = self.abnormal_thresholds[name]
                if value < low:
                    lines.append(f"   ⚠️ Below normal range ({low}-{high})")
                elif value > high:
                    lines.append(f"   ⚠️ Above normal range ({low}-{high})")
        
        lines.extend([
            "",
            "-" * 40,
            "INTERPRETATION",
            "-" * 40,
        ])
        
        # Generate interpretation
        interpretation = self._generate_interpretation(
            shap_result, feature_values, prediction
        )
        lines.append(interpretation)
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)
    
    def _generate_interpretation(
        self,
        shap_result: SHAPResult,
        feature_values: np.ndarray,
        prediction: str,
    ) -> str:
        """Generate natural language interpretation."""
        top = shap_result.top_features[:3]
        
        if prediction in ("VT", "vt_monomorphic", "vt_polymorphic"):
            template = (
                f"The model detected ventricular tachycardia primarily based on "
                f"{top[0][0]} and {top[1][0]}. "
            )
            
            # Add specific insights
            insights = []
            for name, _ in top:
                if name == "mean_hr" or name == "mean_rr":
                    insights.append("elevated heart rate")
                elif name in ("sample_entropy", "approx_entropy"):
                    insights.append("abnormal rhythm complexity")
                elif name == "lf_hf_ratio":
                    insights.append("altered autonomic balance")
            
            if insights:
                template += f"Key observations: {', '.join(insights)}."
            
            return template
        
        return f"Prediction based on analysis of {len(shap_result.shap_values)} features."


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("SHAP Feature Attribution Demo (v2.4)")
    print("="*60)
    
    # Create synthetic model and data
    np.random.seed(42)
    
    # Simulate HRV features
    n_samples = 100
    n_features = 10
    background_data = np.random.randn(n_samples, n_features)
    
    # Simple logistic model
    weights = np.random.randn(n_features)
    
    def predict_fn(x):
        logits = x @ weights
        probs = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - probs, probs])
    
    # Create explainer
    feature_names = [f"feature_{i}" for i in range(n_features)]
    explainer = KernelSHAP(predict_fn, background_data, feature_names)
    
    # Explain a sample
    test_sample = np.random.randn(n_features)
    result = explainer.explain(test_sample, target_class=1, n_samples=100)
    
    print("\nKernelSHAP Results:")
    print(f"  Base value: {result.base_value:.4f}")
    print(f"  Top features:")
    for name, importance in result.top_features[:5]:
        print(f"    {name}: {importance:.4f}")
    
    # Test HRV Feature Explainer
    print("\n" + "-"*60)
    print("HRV Feature Explainer Demo...")
    
    # Create mock sklearn-like model
    class MockModel:
        def predict_proba(self, x):
            logits = x @ np.random.randn(x.shape[1])
            probs = 1 / (1 + np.exp(-logits))
            return np.column_stack([1 - probs, probs])
    
    hrv_explainer = HRVFeatureExplainer(
        MockModel(),
        background_data,
        feature_names,
    )
    
    hrv_result = hrv_explainer.explain(test_sample, target_class=1)
    print(f"\nHRV Explanation:")
    print(hrv_result.get_explanation_text())
    
    # Test report generator
    print("\n" + "-"*60)
    print("Explanation Report Demo...")
    
    report_gen = ExplanationReportGenerator()
    report = report_gen.generate_report(
        hrv_result,
        test_sample,
        prediction="VT",
        confidence=0.85,
    )
    print(report)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
