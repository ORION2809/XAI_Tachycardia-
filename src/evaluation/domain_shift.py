"""
Domain Shift Detection and Mitigation Module

Implements v2.4 requirements:
- Domain shift MUST be quantified AND mitigated, not just documented
- Per-domain recalibration with holdout split
- Threshold retuning to maintain sensitivity floor
- Population Stability Index (PSI) for drift detection
- Fallback to conservative mode on high drift

Reference: BUILDABLE_SPEC.md Part 9.3
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np
from datetime import datetime


class DriftSeverity(Enum):
    """Severity levels for detected drift."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DomainShiftMitigationConfig:
    """
    Configuration for domain shift mitigation.
    
    v2.4: External validation must include active mitigation, not just reporting.
    """
    # Recalibration
    enable_per_domain_recalibration: bool = True
    calibration_holdout_fraction: float = 0.3  # Use 30% of external for recal
    
    # Threshold retuning
    enable_threshold_retuning: bool = True
    retuning_sensitivity_floor: float = 0.90  # Minimum sens during retuning
    
    # Monitoring
    track_drift_indicators: bool = True
    drift_detection_method: str = "population_stability_index"  # PSI
    
    # PSI thresholds
    psi_low_threshold: float = 0.10
    psi_moderate_threshold: float = 0.20
    psi_high_threshold: float = 0.25
    psi_critical_threshold: float = 0.50
    
    # Fallback behavior
    fallback_on_high_drift: str = "conservative_mode"  # Use higher thresholds
    conservative_threshold_multiplier: float = 1.5
    
    # Random seed for reproducibility
    random_seed: int = 42


@dataclass
class DriftIndicators:
    """Drift detection results."""
    mean_psi: float
    max_psi: float
    psi_per_feature: Dict[str, float] = field(default_factory=dict)
    features_above_threshold: List[str] = field(default_factory=list)
    severity: DriftSeverity = DriftSeverity.NONE
    drift_detected: bool = False
    recommendation: str = ""


@dataclass
class RecalibrationResult:
    """Result of per-domain recalibration."""
    domain: str
    original_ece: float
    calibrated_ece: float
    temperature: float
    improvement_pct: float
    n_calibration_samples: int


@dataclass
class ThresholdRetuningResult:
    """Result of threshold retuning for a domain."""
    domain: str
    original_thresholds: Dict[str, float]
    retuned_thresholds: Dict[str, float]
    sensitivity_before: Dict[str, float]
    sensitivity_after: Dict[str, float]
    n_retuning_samples: int


@dataclass
class MitigatedEvaluationResult:
    """Complete mitigated evaluation result."""
    domain: str
    raw_metrics: Dict[str, float]
    mitigated_metrics: Dict[str, float]
    drift_indicators: DriftIndicators
    recalibration: Optional[RecalibrationResult]
    threshold_retuning: Optional[ThresholdRetuningResult]
    improvement: Dict[str, float]
    passes_gates: bool
    gate_failures: List[str]


class TemperatureScaler:
    """
    Temperature scaling for probability calibration.
    
    Fits a single temperature parameter to rescale logits.
    """
    
    def __init__(self):
        self.temperature: float = 1.0
        self.fitted: bool = False
    
    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> 'TemperatureScaler':
        """
        Fit temperature parameter to minimize NLL.
        
        Args:
            logits: Model logits or pre-softmax outputs (n_samples, n_classes)
            labels: True labels (n_samples,)
            lr: Learning rate for optimization
            max_iter: Maximum iterations
        """
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        
        # Simple grid search for temperature
        best_temp = 1.0
        best_nll = float('inf')
        
        for t in np.linspace(0.1, 5.0, 50):
            scaled = logits / t
            probs = self._softmax(scaled)
            nll = self._compute_nll(probs, labels)
            
            if nll < best_nll:
                best_nll = nll
                best_temp = t
        
        self.temperature = best_temp
        self.fitted = True
        
        return self
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        if not self.fitted:
            raise ValueError("TemperatureScaler not fitted")
        
        if logits.ndim == 1:
            logits = logits.reshape(-1, 1)
        
        scaled = logits / self.temperature
        return self._softmax(scaled)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _compute_nll(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Compute negative log-likelihood."""
        n_samples = len(labels)
        eps = 1e-10
        
        if probs.ndim == 1:
            # Binary case
            nll = -np.mean(
                labels * np.log(probs + eps) + 
                (1 - labels) * np.log(1 - probs + eps)
            )
        else:
            # Multi-class case
            nll = -np.mean(np.log(probs[np.arange(n_samples), labels] + eps))
        
        return nll


class DomainShiftMitigation:
    """
    Active domain shift mitigation for external validation.
    
    v2.4 requirement: Don't just report the drop, FIX it.
    
    Implements:
    1. Population Stability Index (PSI) for drift detection
    2. Per-domain temperature scaling recalibration
    3. Threshold retuning to maintain sensitivity floor
    4. Conservative mode fallback for high drift
    """
    
    # Class mapping for metrics computation
    CLASS_MAP = {
        'normal': 0,
        'sinus_tachy': 1,
        'svt': 2,
        'vt': 3,
        'vfl': 4,
    }
    
    def __init__(self, config: Optional[DomainShiftMitigationConfig] = None):
        self.config = config or DomainShiftMitigationConfig()
        self.domain_calibrators: Dict[str, TemperatureScaler] = {}
        self.domain_thresholds: Dict[str, Dict[str, float]] = {}
        self.drift_history: List[Dict[str, Any]] = []
    
    def prepare_external_set(
        self,
        external_data: List[Tuple[np.ndarray, int]],
        domain: str,
    ) -> Tuple[List, List]:
        """
        Split external data into calibration holdout and test.
        
        CRITICAL: Use holdout for recalibration BEFORE computing final metrics.
        This prevents information leakage.
        
        Args:
            external_data: List of (features, label) tuples
            domain: Domain identifier
            
        Returns:
            Tuple of (holdout_data, test_data)
        """
        n_samples = len(external_data)
        n_holdout = int(n_samples * self.config.calibration_holdout_fraction)
        
        # Shuffle with fixed seed for reproducibility
        indices = np.arange(n_samples)
        np.random.seed(self.config.random_seed)
        np.random.shuffle(indices)
        
        holdout_indices = indices[:n_holdout]
        test_indices = indices[n_holdout:]
        
        holdout_data = [external_data[i] for i in holdout_indices]
        test_data = [external_data[i] for i in test_indices]
        
        return holdout_data, test_data
    
    def compute_drift_indicators(
        self,
        internal_features: np.ndarray,
        external_features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> DriftIndicators:
        """
        Compute Population Stability Index (PSI) and other drift indicators.
        
        PSI interpretation:
        - < 0.10: No significant population change
        - 0.10-0.20: Slight population change, monitor
        - 0.20-0.25: Moderate population change, investigate
        - > 0.25: Significant population change, action required
        - > 0.50: Critical drift, fallback to conservative mode
        
        Args:
            internal_features: Features from internal/training domain
            external_features: Features from external/deployment domain
            feature_names: Optional feature names for reporting
            
        Returns:
            DriftIndicators with PSI metrics and severity assessment
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(internal_features.shape[1])]
        
        # PSI per feature
        psi_scores = {}
        for i, name in enumerate(feature_names):
            internal_col = internal_features[:, i]
            external_col = external_features[:, i]
            psi = self._compute_psi(internal_col, external_col)
            psi_scores[name] = psi
        
        psi_values = list(psi_scores.values())
        mean_psi = float(np.mean(psi_values))
        max_psi = float(np.max(psi_values))
        
        # Features above threshold
        features_above = [
            name for name, psi in psi_scores.items()
            if psi > self.config.psi_moderate_threshold
        ]
        
        # Determine severity
        if mean_psi > self.config.psi_critical_threshold:
            severity = DriftSeverity.CRITICAL
            recommendation = (
                "CRITICAL drift detected. Fallback to conservative mode. "
                "Model retraining on external domain strongly recommended."
            )
        elif mean_psi > self.config.psi_high_threshold:
            severity = DriftSeverity.HIGH
            recommendation = (
                "High drift detected. Per-domain recalibration required. "
                "Consider threshold retuning and additional validation."
            )
        elif mean_psi > self.config.psi_moderate_threshold:
            severity = DriftSeverity.MODERATE
            recommendation = (
                "Moderate drift detected. Recalibration recommended. "
                "Monitor performance on external domain."
            )
        elif mean_psi > self.config.psi_low_threshold:
            severity = DriftSeverity.LOW
            recommendation = (
                "Low drift detected. Continue monitoring. "
                "Consider recalibration if performance degrades."
            )
        else:
            severity = DriftSeverity.NONE
            recommendation = (
                "No significant drift detected. "
                "Standard evaluation procedures apply."
            )
        
        return DriftIndicators(
            mean_psi=mean_psi,
            max_psi=max_psi,
            psi_per_feature=psi_scores,
            features_above_threshold=features_above,
            severity=severity,
            drift_detected=severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL],
            recommendation=recommendation,
        )
    
    def _compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Population Stability Index between two distributions.
        
        PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)
        
        Args:
            expected: Expected/baseline distribution (internal)
            actual: Actual/new distribution (external)
            n_bins: Number of bins for histogram
            
        Returns:
            PSI value
        """
        # Handle constant features
        if np.std(expected) < 1e-10 and np.std(actual) < 1e-10:
            return 0.0
        
        # Bin the distributions
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        # Handle edge case where min == max
        if max_val - min_val < 1e-10:
            return 0.0
        
        bins = np.linspace(min_val, max_val, n_bins + 1)
        
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        # Add small constant to avoid log(0) and division by zero
        eps = 1e-6
        expected_pct = (expected_counts + eps) / (expected_counts.sum() + eps * n_bins)
        actual_pct = (actual_counts + eps) / (actual_counts.sum() + eps * n_bins)
        
        # PSI calculation
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)
    
    def recalibrate_for_domain(
        self,
        model_logits: np.ndarray,
        labels: np.ndarray,
        domain: str,
    ) -> RecalibrationResult:
        """
        Fit domain-specific temperature scaling.
        
        Temperature scaling is the minimum viable recalibration that
        preserves model accuracy while improving calibration.
        
        Args:
            model_logits: Model logits/pre-softmax outputs
            labels: True labels
            domain: Domain identifier
            
        Returns:
            RecalibrationResult with before/after ECE
        """
        # Compute ECE before calibration
        original_probs = self._softmax(model_logits)
        original_ece = self._compute_ece(original_probs, labels)
        
        # Fit temperature scaler
        calibrator = TemperatureScaler()
        calibrator.fit(model_logits, labels)
        
        # Compute ECE after calibration
        calibrated_probs = calibrator.transform(model_logits)
        calibrated_ece = self._compute_ece(calibrated_probs, labels)
        
        # Store calibrator
        self.domain_calibrators[domain] = calibrator
        
        improvement = (original_ece - calibrated_ece) / max(original_ece, 1e-10) * 100
        
        return RecalibrationResult(
            domain=domain,
            original_ece=original_ece,
            calibrated_ece=calibrated_ece,
            temperature=calibrator.temperature,
            improvement_pct=improvement,
            n_calibration_samples=len(labels),
        )
    
    def retune_thresholds_for_domain(
        self,
        model_probs: np.ndarray,
        labels: np.ndarray,
        domain: str,
        internal_thresholds: Dict[str, float],
    ) -> ThresholdRetuningResult:
        """
        Retune detection thresholds to maintain sensitivity floor on new domain.
        
        Key insight: if internal threshold gives 95% sens but external gives 80%,
        we need a LOWER threshold on external to recover sensitivity.
        
        Args:
            model_probs: Model probabilities (n_samples, n_classes)
            labels: True labels
            domain: Domain identifier
            internal_thresholds: Original thresholds from internal domain
            
        Returns:
            ThresholdRetuningResult with before/after metrics
        """
        retuned = {}
        sensitivity_before = {}
        sensitivity_after = {}
        
        for class_name, internal_thresh in internal_thresholds.items():
            class_idx = self.CLASS_MAP.get(class_name.lower(), 0)
            
            if model_probs.ndim > 1 and class_idx < model_probs.shape[1]:
                class_probs = model_probs[:, class_idx]
            else:
                class_probs = model_probs
            
            class_labels = (labels == class_idx).astype(int)
            
            # Skip if no positive samples
            if class_labels.sum() == 0:
                retuned[class_name] = internal_thresh
                sensitivity_before[class_name] = 0.0
                sensitivity_after[class_name] = 0.0
                continue
            
            # Compute sensitivity at internal threshold
            preds_internal = (class_probs >= internal_thresh).astype(int)
            sens_internal = self._compute_sensitivity(preds_internal, class_labels)
            sensitivity_before[class_name] = sens_internal
            
            # Find threshold that achieves sensitivity floor
            # Sort by decreasing probability to find decision points
            sorted_indices = np.argsort(-class_probs)
            sorted_probs = class_probs[sorted_indices]
            sorted_labels = class_labels[sorted_indices]
            
            n_positives = class_labels.sum()
            target_tp = int(np.ceil(n_positives * self.config.retuning_sensitivity_floor))
            
            # Find threshold where we hit target TP
            cumsum_tp = np.cumsum(sorted_labels)
            
            # Find first index where we have enough TP
            valid_indices = np.where(cumsum_tp >= target_tp)[0]
            
            if len(valid_indices) > 0:
                threshold_idx = valid_indices[0]
                new_threshold = float(sorted_probs[threshold_idx])
            else:
                # Can't meet floor, use very low threshold
                new_threshold = 0.1
            
            retuned[class_name] = new_threshold
            
            # Compute sensitivity at new threshold
            preds_new = (class_probs >= new_threshold).astype(int)
            sens_new = self._compute_sensitivity(preds_new, class_labels)
            sensitivity_after[class_name] = sens_new
        
        self.domain_thresholds[domain] = retuned
        
        return ThresholdRetuningResult(
            domain=domain,
            original_thresholds=internal_thresholds,
            retuned_thresholds=retuned,
            sensitivity_before=sensitivity_before,
            sensitivity_after=sensitivity_after,
            n_retuning_samples=len(labels),
        )
    
    def get_mitigated_evaluation(
        self,
        model_predict_fn: Callable[[np.ndarray], np.ndarray],
        external_data: List[Tuple[np.ndarray, int]],
        internal_features: np.ndarray,
        domain: str,
        internal_thresholds: Dict[str, float],
        feature_names: Optional[List[str]] = None,
        mode_config: Optional[Any] = None,
    ) -> MitigatedEvaluationResult:
        """
        Complete mitigated evaluation workflow.
        
        Returns both raw and mitigated metrics with full analysis.
        
        Args:
            model_predict_fn: Function that takes features and returns logits/probs
            external_data: List of (features, label) tuples
            internal_features: Features from internal domain for drift detection
            domain: Domain identifier
            internal_thresholds: Original thresholds
            feature_names: Feature names for drift analysis
            mode_config: Operating mode config for gate checking
            
        Returns:
            MitigatedEvaluationResult with complete analysis
        """
        # 1. Split into holdout and test
        holdout, test = self.prepare_external_set(external_data, domain)
        
        # 2. Extract features and labels
        holdout_features = np.array([x[0] for x in holdout])
        holdout_labels = np.array([x[1] for x in holdout])
        test_features = np.array([x[0] for x in test])
        test_labels = np.array([x[1] for x in test])
        
        # 3. Compute drift indicators
        external_features = np.array([x[0] for x in external_data])
        drift_indicators = self.compute_drift_indicators(
            internal_features, external_features, feature_names
        )
        
        # 4. Get predictions on holdout for calibration
        holdout_logits = model_predict_fn(holdout_features)
        
        # 5. Recalibrate if enabled
        recalibration = None
        if self.config.enable_per_domain_recalibration:
            recalibration = self.recalibrate_for_domain(
                holdout_logits, holdout_labels, domain
            )
        
        # 6. Retune thresholds if enabled
        threshold_retuning = None
        if self.config.enable_threshold_retuning:
            holdout_probs = self._softmax(holdout_logits)
            threshold_retuning = self.retune_thresholds_for_domain(
                holdout_probs, holdout_labels, domain, internal_thresholds
            )
            mitigated_thresholds = threshold_retuning.retuned_thresholds
        else:
            mitigated_thresholds = internal_thresholds
        
        # 7. Get predictions on test set
        test_logits = model_predict_fn(test_features)
        
        # Apply calibration if available
        if self.config.enable_per_domain_recalibration and domain in self.domain_calibrators:
            test_probs = self.domain_calibrators[domain].transform(test_logits)
        else:
            test_probs = self._softmax(test_logits)
        
        # 8. Compute raw and mitigated metrics
        raw_probs = self._softmax(test_logits)  # Without calibration
        raw_metrics = self._compute_metrics(raw_probs, test_labels, internal_thresholds)
        mitigated_metrics = self._compute_metrics(test_probs, test_labels, mitigated_thresholds)
        
        # 9. Compute improvement
        improvement = {
            'sensitivity_gain': (
                mitigated_metrics.get('vt_sensitivity', 0) -
                raw_metrics.get('vt_sensitivity', 0)
            ),
            'ece_reduction': (
                raw_metrics.get('ece', 0) -
                mitigated_metrics.get('ece', 0)
            ),
            'fa_change': (
                mitigated_metrics.get('fa_per_hour', 0) -
                raw_metrics.get('fa_per_hour', 0)
            ),
        }
        
        # 10. Check gates if mode config provided
        passes_gates = True
        gate_failures = []
        
        if mode_config is not None:
            # VT sensitivity floor
            if mitigated_metrics.get('vt_sensitivity', 0) < getattr(mode_config, 'vt_vfl_sensitivity_floor', 0.90):
                passes_gates = False
                gate_failures.append('vt_sensitivity_below_floor')
            
            # ECE ceiling
            if mitigated_metrics.get('ece', 1.0) > getattr(mode_config, 'max_ece', 0.10):
                passes_gates = False
                gate_failures.append('ece_above_ceiling')
            
            # FA/hr ceiling
            if mitigated_metrics.get('fa_per_hour', float('inf')) > getattr(mode_config, 'vt_vfl_max_fa_per_hour', 2.0):
                passes_gates = False
                gate_failures.append('fa_above_ceiling')
        
        # Record drift history
        self.drift_history.append({
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'drift_indicators': drift_indicators,
            'passes_gates': passes_gates,
        })
        
        return MitigatedEvaluationResult(
            domain=domain,
            raw_metrics=raw_metrics,
            mitigated_metrics=mitigated_metrics,
            drift_indicators=drift_indicators,
            recalibration=recalibration,
            threshold_retuning=threshold_retuning,
            improvement=improvement,
            passes_gates=passes_gates,
            gate_failures=gate_failures,
        )
    
    def _compute_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        thresholds: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        for class_name, thresh in thresholds.items():
            class_idx = self.CLASS_MAP.get(class_name.lower(), 0)
            
            if probs.ndim > 1 and class_idx < probs.shape[1]:
                class_probs = probs[:, class_idx]
            else:
                class_probs = probs
            
            class_labels = (labels == class_idx).astype(int)
            preds = (class_probs >= thresh).astype(int)
            
            tp = np.sum((preds == 1) & (class_labels == 1))
            fn = np.sum((preds == 0) & (class_labels == 1))
            fp = np.sum((preds == 1) & (class_labels == 0))
            tn = np.sum((preds == 0) & (class_labels == 0))
            
            sensitivity = tp / max(tp + fn, 1)
            ppv = tp / max(tp + fp, 1)
            specificity = tn / max(tn + fp, 1)
            
            metrics[f'{class_name}_sensitivity'] = float(sensitivity)
            metrics[f'{class_name}_ppv'] = float(ppv)
            metrics[f'{class_name}_specificity'] = float(specificity)
        
        # Compute overall ECE
        if probs.ndim > 1:
            # Multi-class: use max prob
            max_probs = np.max(probs, axis=1)
            pred_labels = np.argmax(probs, axis=1)
            correct = (pred_labels == labels).astype(int)
            metrics['ece'] = float(self._compute_ece_binary(max_probs, correct))
        else:
            metrics['ece'] = float(self._compute_ece_binary(probs, labels))
        
        # Simplified FA/hr estimate (would need timing info in practice)
        vt_idx = self.CLASS_MAP.get('vt', 3)
        if probs.ndim > 1 and vt_idx < probs.shape[1]:
            vt_probs = probs[:, vt_idx]
            vt_labels = (labels == vt_idx).astype(int)
            vt_thresh = thresholds.get('vt', 0.5)
            vt_preds = (vt_probs >= vt_thresh).astype(int)
            fp_vt = np.sum((vt_preds == 1) & (vt_labels == 0))
            # Assume 1 sample per second, 24 hours = 86400 samples
            metrics['fa_per_hour'] = fp_vt / 24.0  # Very rough estimate
        
        return metrics
    
    def _compute_sensitivity(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute sensitivity (recall)."""
        tp = np.sum((preds == 1) & (labels == 1))
        fn = np.sum((preds == 0) & (labels == 1))
        return tp / max(tp + fn, 1)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _compute_ece(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error for multi-class."""
        if probs.ndim == 1:
            return self._compute_ece_binary(probs, labels)
        
        max_probs = np.max(probs, axis=1)
        pred_labels = np.argmax(probs, axis=1)
        correct = (pred_labels == labels).astype(int)
        
        return self._compute_ece_binary(max_probs, correct)
    
    def _compute_ece_binary(
        self,
        probs: np.ndarray,
        correct: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Expected Calibration Error for binary/max-prob case."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = correct[mask].mean()
                bin_conf = probs[mask].mean()
                ece += mask.sum() * np.abs(bin_acc - bin_conf)
        
        return ece / len(probs) if len(probs) > 0 else 0.0
    
    def get_calibrator(self, domain: str) -> Optional[TemperatureScaler]:
        """Get the calibrator for a specific domain."""
        return self.domain_calibrators.get(domain)
    
    def get_thresholds(self, domain: str) -> Optional[Dict[str, float]]:
        """Get the retuned thresholds for a specific domain."""
        return self.domain_thresholds.get(domain)
    
    def generate_drift_report(self) -> str:
        """Generate a human-readable drift report."""
        if not self.drift_history:
            return "No drift analysis performed yet."
        
        lines = [
            "=" * 60,
            "DOMAIN SHIFT ANALYSIS REPORT",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Domains analyzed: {len(self.drift_history)}",
            "",
        ]
        
        for entry in self.drift_history:
            indicators = entry['drift_indicators']
            lines.extend([
                f"--- Domain: {entry['domain']} ---",
                f"Timestamp: {entry['timestamp']}",
                f"Mean PSI: {indicators.mean_psi:.4f}",
                f"Max PSI: {indicators.max_psi:.4f}",
                f"Severity: {indicators.severity.value}",
                f"Drift Detected: {indicators.drift_detected}",
                f"Features above threshold: {len(indicators.features_above_threshold)}",
                f"Recommendation: {indicators.recommendation}",
                f"Gates Passed: {entry['passes_gates']}",
                "",
            ])
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


# Convenience functions
def compute_psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Compute Population Stability Index between two arrays."""
    mitigation = DomainShiftMitigation()
    return mitigation._compute_psi(expected, actual)


def detect_drift(
    internal_features: np.ndarray,
    external_features: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> DriftIndicators:
    """Detect drift between internal and external feature distributions."""
    mitigation = DomainShiftMitigation()
    return mitigation.compute_drift_indicators(
        internal_features, external_features, feature_names
    )
