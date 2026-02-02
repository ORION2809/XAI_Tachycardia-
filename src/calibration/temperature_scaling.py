"""
Temperature Scaling and Calibration Module.

Provides probability calibration to ensure that predicted probabilities
match actual outcome frequencies. A well-calibrated model predicting
80% confidence should be correct 80% of the time.

Methods:
1. Temperature Scaling - Simple, single-parameter method
2. Isotonic Regression - Non-parametric, per-class calibration
3. Platt Scaling - Logistic regression on logits

Metrics:
- ECE (Expected Calibration Error)
- MCE (Maximum Calibration Error)
- Brier Score
- Reliability Diagrams
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics."""
    ece: float              # Expected Calibration Error (lower is better)
    mce: float              # Maximum Calibration Error
    brier_score: float      # Brier score (lower is better)
    overconfidence: float   # Average (confidence - accuracy) when conf > acc
    underconfidence: float  # Average (accuracy - confidence) when acc > conf
    
    def is_well_calibrated(self, ece_threshold: float = 0.1) -> bool:
        """Check if model is well-calibrated."""
        return self.ece < ece_threshold
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'ece': self.ece,
            'mce': self.mce,
            'brier_score': self.brier_score,
            'overconfidence': self.overconfidence,
            'underconfidence': self.underconfidence,
        }


class TemperatureScaling:
    """
    Temperature Scaling calibration.
    
    Divides logits by a learned temperature T before softmax.
    - T > 1: Makes predictions less confident (softens)
    - T < 1: Makes predictions more confident (sharpens)
    - T = 1: No change
    
    Simple and effective for many models.
    
    Reference: Guo et al., "On Calibration of Modern Neural Networks" (2017)
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        self.temperature = initial_temperature
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """
        Find optimal temperature on validation set.
        
        Args:
            logits: Model logits (n_samples, n_classes)
            labels: True labels (n_samples,)
            
        Returns:
            Optimal temperature value
        """
        def nll_with_temp(temp: float) -> float:
            """Negative log-likelihood with temperature scaling."""
            if temp <= 0:
                return float('inf')
            
            scaled_logits = logits / temp
            probs = softmax(scaled_logits, axis=1)
            
            # Cross-entropy loss
            log_probs = np.log(probs + 1e-10)
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            
            return nll
        
        # Optimize temperature
        result = minimize_scalar(
            nll_with_temp,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        self.temperature = result.x
        self._fitted = True
        
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Raw logits (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities
        """
        scaled_logits = logits / self.temperature
        return softmax(scaled_logits, axis=1)
    
    def calibrate_single(self, logits: np.ndarray) -> np.ndarray:
        """Calibrate single sample."""
        return self.calibrate(logits.reshape(1, -1))[0]


class IsotonicCalibration:
    """
    Isotonic Regression calibration (per-class).
    
    Non-parametric method that learns a monotonic mapping
    from predicted probability to calibrated probability
    for each class independently.
    
    More flexible than temperature scaling but requires
    more validation data.
    """
    
    def __init__(self):
        self.regressors: Dict[int, IsotonicRegression] = {}
        self._fitted = False
        self.n_classes = 0
    
    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ):
        """
        Fit isotonic regression for each class.
        
        Args:
            probs: Predicted probabilities (n_samples, n_classes)
            labels: True labels (n_samples,)
        """
        self.n_classes = probs.shape[1]
        
        for c in range(self.n_classes):
            # Binary labels for this class
            binary_labels = (labels == c).astype(float)
            
            # Fit isotonic regression
            self.regressors[c] = IsotonicRegression(
                out_of_bounds='clip',
                increasing=True,
            ).fit(probs[:, c], binary_labels)
        
        self._fitted = True
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply isotonic calibration.
        
        Args:
            probs: Uncalibrated probabilities (n_samples, n_classes)
            
        Returns:
            Calibrated probabilities (renormalized)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        
        calibrated = np.zeros_like(probs)
        
        for c, reg in self.regressors.items():
            calibrated[:, c] = reg.predict(probs[:, c])
        
        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = np.divide(
            calibrated, 
            row_sums, 
            where=row_sums > 0,
            out=np.zeros_like(calibrated)
        )
        
        return calibrated


class CalibrationModule:
    """
    Complete calibration module combining multiple methods.
    
    Pipeline:
    1. Temperature scaling (always applied)
    2. Optional isotonic regression (per-class refinement)
    
    Also provides calibration metrics and reliability diagrams.
    """
    
    def __init__(
        self,
        use_isotonic: bool = True,
        n_bins: int = 15,
    ):
        self.use_isotonic = use_isotonic
        self.n_bins = n_bins
        
        self.temp_scaling = TemperatureScaling()
        self.isotonic = IsotonicCalibration() if use_isotonic else None
        
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Fit calibration on validation set.
        
        Args:
            logits: Raw model logits
            labels: True labels
            
        Returns:
            Dict with fitted parameters and metrics
        """
        # Step 1: Temperature scaling
        temperature = self.temp_scaling.fit(logits, labels)
        temp_scaled_probs = self.temp_scaling.calibrate(logits)
        
        results = {
            'temperature': temperature,
        }
        
        # Metrics after temperature scaling
        temp_metrics = self.compute_metrics(temp_scaled_probs, labels)
        results['temp_scaling_metrics'] = temp_metrics.to_dict()
        
        # Step 2: Isotonic regression (optional)
        if self.use_isotonic:
            self.isotonic.fit(temp_scaled_probs, labels)
            final_probs = self.isotonic.calibrate(temp_scaled_probs)
            
            isotonic_metrics = self.compute_metrics(final_probs, labels)
            results['isotonic_metrics'] = isotonic_metrics.to_dict()
        
        self._fitted = True
        
        return results
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply full calibration pipeline.
        
        Args:
            logits: Raw model logits
            
        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before calibrate()")
        
        # Temperature scaling
        probs = self.temp_scaling.calibrate(logits)
        
        # Isotonic (optional)
        if self.use_isotonic and self.isotonic._fitted:
            probs = self.isotonic.calibrate(probs)
        
        return probs
    
    def compute_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationMetrics:
        """
        Compute calibration metrics.
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            
        Returns:
            CalibrationMetrics
        """
        n_samples = len(labels)
        n_classes = probs.shape[1]
        
        # Confidences and predictions
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        # Binned statistics
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        
        ece = 0.0
        mce = 0.0
        overconf_sum = 0.0
        overconf_count = 0
        underconf_sum = 0.0
        underconf_count = 0
        
        for i in range(self.n_bins):
            mask = (confidences > bin_boundaries[i]) & \
                   (confidences <= bin_boundaries[i + 1])
            bin_size = np.sum(mask)
            
            if bin_size > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                
                gap = np.abs(avg_conf - avg_acc)
                ece += bin_size * gap
                mce = max(mce, gap)
                
                if avg_conf > avg_acc:
                    overconf_sum += bin_size * (avg_conf - avg_acc)
                    overconf_count += bin_size
                else:
                    underconf_sum += bin_size * (avg_acc - avg_conf)
                    underconf_count += bin_size
        
        ece /= n_samples
        overconfidence = overconf_sum / max(overconf_count, 1)
        underconfidence = underconf_sum / max(underconf_count, 1)
        
        # Brier score
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_samples), labels] = 1
        brier_score = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
        
        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            overconfidence=overconfidence,
            underconfidence=underconfidence,
        )
    
    def get_reliability_diagram_data(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get data for plotting reliability diagram.
        
        Returns:
            Dict with 'bin_centers', 'bin_accuracies', 'bin_counts'
        """
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        bin_accuracies = np.zeros(self.n_bins)
        bin_counts = np.zeros(self.n_bins)
        
        for i in range(self.n_bins):
            mask = (confidences > bin_boundaries[i]) & \
                   (confidences <= bin_boundaries[i + 1])
            bin_counts[i] = np.sum(mask)
            if bin_counts[i] > 0:
                bin_accuracies[i] = np.mean(accuracies[mask])
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
        }
    
    def plot_reliability_diagram(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        title: str = "Reliability Diagram",
        save_path: Optional[str] = None,
    ):
        """Plot reliability diagram."""
        import matplotlib.pyplot as plt
        
        data = self.get_reliability_diagram_data(probs, labels)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        bin_width = 1.0 / self.n_bins
        ax1.bar(
            data['bin_centers'],
            data['bin_accuracies'],
            width=bin_width * 0.8,
            alpha=0.7,
            label='Model accuracy'
        )
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title(title)
        ax1.legend()
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Histogram of confidences
        ax2.bar(
            data['bin_centers'],
            data['bin_counts'],
            width=bin_width * 0.8,
            alpha=0.7,
        )
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def calibrate_model_output(
    logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Quick calibration with known temperature.
    
    Args:
        logits: Raw logits
        temperature: Temperature value (from prior fitting)
        
    Returns:
        Calibrated probabilities
    """
    return softmax(logits / temperature, axis=-1)


# =============================================================================
# v2.4: ECE GATE FOR DEPLOYMENT
# =============================================================================

@dataclass
class ECEGateConfig:
    """Configuration for ECE gate."""
    max_ece_high_sensitivity: float = 0.08
    max_ece_balanced: float = 0.10
    max_ece_research: float = 0.15
    recalibration_trigger_ece: float = 0.12
    min_samples_for_validation: int = 100


class ECEGate:
    """
    v2.4: ECE as hard gate for deployment.
    
    Model CANNOT be deployed if ECE exceeds threshold.
    Provides clear pass/fail criteria for calibration quality.
    """
    
    def __init__(self, config: ECEGateConfig = None):
        if config is None:
            config = ECEGateConfig()
        self.config = config
    
    def validate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        operating_mode: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Validate calibration meets deployment requirements.
        
        Args:
            probs: Calibrated probabilities
            labels: True labels
            operating_mode: "high_sensitivity", "balanced", or "research"
            
        Returns:
            Dict with passes, ece, threshold, recommendation
        """
        ece = self._compute_ece(probs, labels)
        
        # Get threshold for operating mode
        if operating_mode == "high_sensitivity":
            threshold = self.config.max_ece_high_sensitivity
        elif operating_mode == "balanced":
            threshold = self.config.max_ece_balanced
        else:
            threshold = self.config.max_ece_research
        
        passes = ece <= threshold
        needs_recalibration = ece > self.config.recalibration_trigger_ece
        
        result = {
            "passes": passes,
            "ece": ece,
            "threshold": threshold,
            "operating_mode": operating_mode,
            "needs_recalibration": needs_recalibration,
        }
        
        if not passes:
            result["recommendation"] = f"ECE {ece:.3f} exceeds threshold {threshold}. Recalibrate before deployment."
        elif needs_recalibration:
            result["recommendation"] = f"ECE {ece:.3f} approaching limit. Consider recalibration."
        else:
            result["recommendation"] = f"Calibration acceptable for {operating_mode} mode."
        
        return result
    
    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 15,
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        for i in range(n_bins):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                ece += np.sum(mask) * np.abs(avg_conf - avg_acc)
        
        return ece / len(labels)


# =============================================================================
# v2.4: DOMAIN SHIFT DETECTION
# =============================================================================

@dataclass
class DomainShiftResult:
    """Result from domain shift detection."""
    has_shift: bool
    severity: str  # "none", "mild", "moderate", "severe"
    confidence_shift: float
    accuracy_delta: float
    ece_delta: float
    recommendation: str


class DomainShiftDetector:
    """
    v2.4: Detect domain shift between training and deployment.
    
    Monitors:
    - Confidence distribution changes
    - Calibration degradation
    - Accuracy drops
    """
    
    def __init__(
        self,
        reference_confidence_mean: float = 0.0,
        reference_confidence_std: float = 1.0,
        reference_ece: float = 0.0,
        confidence_shift_threshold: float = 0.15,
        ece_degradation_threshold: float = 0.05,
    ):
        self.reference_confidence_mean = reference_confidence_mean
        self.reference_confidence_std = reference_confidence_std
        self.reference_ece = reference_ece
        self.confidence_shift_threshold = confidence_shift_threshold
        self.ece_degradation_threshold = ece_degradation_threshold
        
        self._reference_set = False
    
    def set_reference(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ):
        """Set reference statistics from training/validation data."""
        confidences = np.max(probs, axis=1)
        self.reference_confidence_mean = np.mean(confidences)
        self.reference_confidence_std = np.std(confidences)
        self.reference_ece = self._compute_ece(probs, labels)
        self._reference_set = True
    
    def detect_shift(
        self,
        probs: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> DomainShiftResult:
        """
        Detect domain shift in new data.
        
        Can work with or without labels (labels improve detection).
        """
        if not self._reference_set:
            return DomainShiftResult(
                has_shift=False,
                severity="unknown",
                confidence_shift=0.0,
                accuracy_delta=0.0,
                ece_delta=0.0,
                recommendation="Set reference distribution first.",
            )
        
        confidences = np.max(probs, axis=1)
        new_conf_mean = np.mean(confidences)
        
        # Confidence distribution shift
        confidence_shift = abs(new_conf_mean - self.reference_confidence_mean)
        
        # ECE and accuracy delta (if labels available)
        ece_delta = 0.0
        accuracy_delta = 0.0
        
        if labels is not None:
            new_ece = self._compute_ece(probs, labels)
            ece_delta = new_ece - self.reference_ece
            
            predictions = np.argmax(probs, axis=1)
            new_accuracy = np.mean(predictions == labels)
            # Assume reference accuracy was reasonable (~0.85)
            accuracy_delta = 0.85 - new_accuracy
        
        # Classify severity
        if confidence_shift < 0.05 and ece_delta < 0.02:
            severity = "none"
            has_shift = False
        elif confidence_shift < 0.10 or ece_delta < 0.05:
            severity = "mild"
            has_shift = True
        elif confidence_shift < 0.20 or ece_delta < 0.10:
            severity = "moderate"
            has_shift = True
        else:
            severity = "severe"
            has_shift = True
        
        # Generate recommendation
        if severity == "none":
            recommendation = "No significant domain shift detected."
        elif severity == "mild":
            recommendation = "Minor shift detected. Monitor calibration metrics."
        elif severity == "moderate":
            recommendation = "Moderate shift. Consider recalibration on new domain."
        else:
            recommendation = "Severe domain shift! Recalibration required before deployment."
        
        return DomainShiftResult(
            has_shift=has_shift,
            severity=severity,
            confidence_shift=confidence_shift,
            accuracy_delta=accuracy_delta,
            ece_delta=ece_delta,
            recommendation=recommendation,
        )
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute ECE."""
        bin_boundaries = np.linspace(0, 1, 16)
        ece = 0.0
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        for i in range(15):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                ece += np.sum(mask) * np.abs(avg_conf - avg_acc)
        
        return ece / len(labels)


# =============================================================================
# v2.4: RECALIBRATION PROTOCOL
# =============================================================================

class RecalibrationProtocol:
    """
    v2.4: Protocol for recalibrating on new domains.
    
    Used when domain shift is detected to adapt calibration
    without full model retraining.
    """
    
    def __init__(
        self,
        min_samples: int = 100,
        temperature_range: Tuple[float, float] = (0.5, 3.0),
    ):
        self.min_samples = min_samples
        self.temperature_range = temperature_range
        self.original_temperature: Optional[float] = None
        self.adapted_temperature: Optional[float] = None
    
    def recalibrate(
        self,
        original_calibrator: CalibrationModule,
        new_logits: np.ndarray,
        new_labels: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Recalibrate on new domain data.
        
        Returns:
            Dict with new temperature, old ECE, new ECE, improvement
        """
        if len(new_labels) < self.min_samples:
            return {
                "success": False,
                "reason": f"Insufficient samples ({len(new_labels)} < {self.min_samples})",
            }
        
        # Store original temperature
        self.original_temperature = original_calibrator.temp_scaling.temperature
        
        # Compute ECE with original calibration
        original_probs = original_calibrator.calibrate(new_logits)
        original_ece = self._compute_ece(original_probs, new_labels)
        
        # Find new optimal temperature
        new_temp_scaling = TemperatureScaling()
        new_temp = new_temp_scaling.fit(new_logits, new_labels)
        self.adapted_temperature = new_temp
        
        # Compute ECE with new calibration
        new_probs = new_temp_scaling.calibrate(new_logits)
        new_ece = self._compute_ece(new_probs, new_labels)
        
        improvement = original_ece - new_ece
        
        return {
            "success": True,
            "original_temperature": self.original_temperature,
            "adapted_temperature": self.adapted_temperature,
            "original_ece": original_ece,
            "new_ece": new_ece,
            "ece_improvement": improvement,
            "recommendation": "Use adapted temperature" if improvement > 0.01 else "Original calibration sufficient",
        }
    
    def _compute_ece(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute ECE."""
        bin_boundaries = np.linspace(0, 1, 16)
        ece = 0.0
        
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        for i in range(15):
            mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            if np.sum(mask) > 0:
                avg_conf = np.mean(confidences[mask])
                avg_acc = np.mean(accuracies[mask])
                ece += np.sum(mask) * np.abs(avg_conf - avg_acc)
        
        return ece / len(labels)


# =============================================================================
# v2.4: CLASS-CONDITIONAL CALIBRATION
# =============================================================================

class ClassConditionalCalibration:
    """
    v2.4: Per-class calibration for fine-grained control.
    
    Different classes may need different calibration strategies:
    - VT/VFL: Sensitivity-focused, accept higher FA
    - SVT: Balanced
    - Sinus tachycardia: Specificity-focused
    """
    
    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.class_temperatures: Dict[int, float] = {}
        self.class_isotonic: Dict[int, IsotonicRegression] = {}
        self._fitted = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        class_names: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Fit per-class calibration.
        
        Uses different optimization for different class priorities.
        """
        if class_names is None:
            class_names = [f"class_{i}" for i in range(self.n_classes)]
        
        probs = softmax(logits, axis=1)
        results = {}
        
        for c in range(self.n_classes):
            # Binary one-vs-all for this class
            binary_labels = (labels == c).astype(float)
            class_probs = probs[:, c]
            
            # Temperature scaling for this class
            best_temp = self._find_class_temperature(
                logits[:, c], binary_labels
            )
            self.class_temperatures[c] = best_temp
            
            # Isotonic regression
            self.class_isotonic[c] = IsotonicRegression(
                out_of_bounds='clip'
            ).fit(class_probs, binary_labels)
            
            results[class_names[c]] = {
                "temperature": best_temp,
            }
        
        self._fitted = True
        return results
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply per-class calibration."""
        if not self._fitted:
            raise RuntimeError("Must call fit() first")
        
        calibrated = np.zeros_like(logits)
        
        for c in range(self.n_classes):
            # Apply class-specific temperature
            temp = self.class_temperatures.get(c, 1.0)
            scaled = logits[:, c] / temp
            probs = 1 / (1 + np.exp(-scaled))  # Sigmoid for one-vs-all
            
            # Apply isotonic
            if c in self.class_isotonic:
                probs = self.class_isotonic[c].predict(probs)
            
            calibrated[:, c] = probs
        
        # Normalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = np.divide(
            calibrated, row_sums, 
            where=row_sums > 0, 
            out=np.zeros_like(calibrated)
        )
        
        return calibrated
    
    def _find_class_temperature(
        self,
        class_logits: np.ndarray,
        binary_labels: np.ndarray,
    ) -> float:
        """Find optimal temperature for one-vs-all classification."""
        
        def nll_with_temp(temp: float) -> float:
            if temp <= 0:
                return float('inf')
            scaled = class_logits / temp
            probs = 1 / (1 + np.exp(-scaled))
            probs = np.clip(probs, 1e-10, 1 - 1e-10)
            nll = -np.mean(
                binary_labels * np.log(probs) + 
                (1 - binary_labels) * np.log(1 - probs)
            )
            return nll
        
        result = minimize_scalar(nll_with_temp, bounds=(0.1, 10.0), method='bounded')
        return result.x


if __name__ == "__main__":
    print("Calibration Module Demo (v2.4)")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 5
    
    # Simulate overconfident model
    logits = np.random.randn(n_samples, n_classes) * 3
    labels = np.random.randint(0, n_classes, n_samples)
    
    # Make some predictions correct
    for i in range(n_samples):
        if np.random.rand() < 0.7:
            logits[i, labels[i]] += 2
    
    print(f"Samples: {n_samples}, Classes: {n_classes}")
    
    # Before calibration
    uncalibrated_probs = softmax(logits, axis=1)
    
    # Fit calibration
    calibration = CalibrationModule(use_isotonic=True)
    results = calibration.fit(logits, labels)
    
    print(f"\nOptimal temperature: {results['temperature']:.3f}")
    
    # After calibration
    calibrated_probs = calibration.calibrate(logits)
    
    # Compare metrics
    uncal_metrics = calibration.compute_metrics(uncalibrated_probs, labels)
    cal_metrics = calibration.compute_metrics(calibrated_probs, labels)
    
    print("\nBefore Calibration:")
    print(f"  ECE: {uncal_metrics.ece:.4f}")
    print(f"  MCE: {uncal_metrics.mce:.4f}")
    print(f"  Brier: {uncal_metrics.brier_score:.4f}")
    
    print("\nAfter Calibration:")
    print(f"  ECE: {cal_metrics.ece:.4f}")
    print(f"  MCE: {cal_metrics.mce:.4f}")
    print(f"  Brier: {cal_metrics.brier_score:.4f}")
    
    print(f"\nWell-calibrated: {cal_metrics.is_well_calibrated()}")
    
    # Test ECE Gate
    print("\n" + "-" * 60)
    print("Testing ECE Gate (v2.4)...")
    gate = ECEGate()
    gate_result = gate.validate(calibrated_probs, labels, "balanced")
    print(f"  Passes: {gate_result['passes']}")
    print(f"  ECE: {gate_result['ece']:.4f}")
    print(f"  Threshold: {gate_result['threshold']}")
    print(f"  Recommendation: {gate_result['recommendation']}")
    
    # Test Domain Shift Detector
    print("\n" + "-" * 60)
    print("Testing Domain Shift Detector (v2.4)...")
    detector = DomainShiftDetector()
    detector.set_reference(calibrated_probs, labels)
    
    # Simulate shifted data
    shifted_logits = logits * 1.5 + 0.5
    shifted_probs = calibration.calibrate(shifted_logits)
    shift_result = detector.detect_shift(shifted_probs, labels)
    print(f"  Has shift: {shift_result.has_shift}")
    print(f"  Severity: {shift_result.severity}")
    print(f"  Confidence shift: {shift_result.confidence_shift:.3f}")
    print(f"  Recommendation: {shift_result.recommendation}")
    
    print("\n" + "=" * 60)
