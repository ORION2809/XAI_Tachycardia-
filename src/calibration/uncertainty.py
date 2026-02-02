"""
Uncertainty Estimation and Policy Module.

Provides uncertainty quantification for model predictions using:
1. MC Dropout - Multiple forward passes with dropout enabled
2. Ensemble variance - Disagreement between ensemble members
3. Entropy-based - Prediction entropy

Also defines policies for how to handle high-uncertainty predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Callable
from dataclasses import dataclass


@dataclass
class UncertaintyResult:
    """Result from uncertainty estimation."""
    mean_probs: np.ndarray          # Mean probabilities
    uncertainty: np.ndarray         # Per-timestep uncertainty (0-1)
    aleatoric: Optional[np.ndarray] = None  # Data uncertainty
    epistemic: Optional[np.ndarray] = None  # Model uncertainty
    method: str = "mc_dropout"
    n_samples: int = 10
    
    @property
    def mean_uncertainty(self) -> float:
        """Average uncertainty across timesteps."""
        return float(np.mean(self.uncertainty))
    
    @property
    def max_uncertainty(self) -> float:
        """Maximum uncertainty."""
        return float(np.max(self.uncertainty))
    
    def is_high_uncertainty(self, threshold: float = 0.4) -> bool:
        """Check if average uncertainty exceeds threshold."""
        return self.mean_uncertainty > threshold
    
    def get_uncertain_regions(
        self, 
        threshold: float = 0.5
    ) -> List[Tuple[int, int]]:
        """Get regions where uncertainty exceeds threshold."""
        high_unc = self.uncertainty > threshold
        regions = []
        in_region = False
        start = 0
        
        for i, is_uncertain in enumerate(high_unc):
            if is_uncertain and not in_region:
                in_region = True
                start = i
            elif not is_uncertain and in_region:
                in_region = False
                regions.append((start, i))
        
        if in_region:
            regions.append((start, len(high_unc)))
        
        return regions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mean_uncertainty': self.mean_uncertainty,
            'max_uncertainty': self.max_uncertainty,
            'method': self.method,
            'n_samples': self.n_samples,
            'is_high_uncertainty': self.is_high_uncertainty(),
        }


class MCDropoutEstimator:
    """
    Monte Carlo Dropout uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty. High variance across samples indicates
    high epistemic (model) uncertainty.
    
    Reference: Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.n_samples = n_samples
        self.device = device
    
    def estimate(
        self,
        x: torch.Tensor,
        n_samples: Optional[int] = None,
    ) -> UncertaintyResult:
        """
        Estimate uncertainty using MC Dropout.
        
        Args:
            x: Input tensor (batch, channels, seq_len)
            n_samples: Override number of samples
            
        Returns:
            UncertaintyResult with mean probs and uncertainty
        """
        n_samples = n_samples or self.n_samples
        x = x.to(self.device)
        
        # Enable dropout for inference
        self.model.train()
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.cpu().numpy())
        
        # Return to eval mode
        self.model.eval()
        
        # Stack: (n_samples, batch, seq, classes)
        samples = np.stack(samples, axis=0)
        
        # Mean probabilities
        mean_probs = samples.mean(axis=0)
        
        # Epistemic uncertainty: variance across samples
        var_probs = samples.var(axis=0)
        epistemic = var_probs.sum(axis=-1)  # Sum across classes
        
        # Normalize epistemic by maximum possible variance
        max_var = 0.25 * mean_probs.shape[-1]  # Max variance for uniform
        epistemic = epistemic / max_var
        epistemic = np.clip(epistemic, 0, 1)
        
        # Total uncertainty: entropy of mean prediction
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=-1)
        max_entropy = np.log(mean_probs.shape[-1])
        total_uncertainty = entropy / max_entropy
        
        # Squeeze batch dimension if single sample
        if mean_probs.shape[0] == 1:
            mean_probs = mean_probs.squeeze(0)
            total_uncertainty = total_uncertainty.squeeze(0)
            epistemic = epistemic.squeeze(0)
        
        return UncertaintyResult(
            mean_probs=mean_probs,
            uncertainty=total_uncertainty,
            epistemic=epistemic,
            method="mc_dropout",
            n_samples=n_samples,
        )


class EntropyEstimator:
    """
    Entropy-based uncertainty estimation.
    
    Uses prediction entropy as uncertainty measure.
    Fast but doesn't distinguish aleatoric from epistemic uncertainty.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
    
    def estimate(self, x: torch.Tensor) -> UncertaintyResult:
        """
        Estimate uncertainty using prediction entropy.
        
        Args:
            x: Input tensor
            
        Returns:
            UncertaintyResult
        """
        x = x.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=-1)
        
        probs_np = probs.cpu().numpy()
        
        # Entropy
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10), axis=-1)
        max_entropy = np.log(probs_np.shape[-1])
        uncertainty = entropy / max_entropy
        
        # Squeeze if single sample
        if probs_np.shape[0] == 1:
            probs_np = probs_np.squeeze(0)
            uncertainty = uncertainty.squeeze(0)
        
        return UncertaintyResult(
            mean_probs=probs_np,
            uncertainty=uncertainty,
            method="entropy",
            n_samples=1,
        )


class EnsembleEstimator:
    """
    Ensemble-based uncertainty estimation.
    
    Uses disagreement between ensemble members as uncertainty.
    Requires multiple trained models.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.models = models
        self.device = device
        
        for model in self.models:
            model.to(device)
            model.eval()
    
    def estimate(self, x: torch.Tensor) -> UncertaintyResult:
        """
        Estimate uncertainty using ensemble disagreement.
        
        Args:
            x: Input tensor
            
        Returns:
            UncertaintyResult
        """
        x = x.to(self.device)
        
        samples = []
        for model in self.models:
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs.cpu().numpy())
        
        samples = np.stack(samples, axis=0)
        
        # Mean and variance
        mean_probs = samples.mean(axis=0)
        var_probs = samples.var(axis=0)
        
        # Epistemic: variance across ensemble
        epistemic = var_probs.sum(axis=-1)
        max_var = 0.25 * mean_probs.shape[-1]
        epistemic = np.clip(epistemic / max_var, 0, 1)
        
        # Total: entropy of mean
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=-1)
        max_entropy = np.log(mean_probs.shape[-1])
        uncertainty = entropy / max_entropy
        
        if mean_probs.shape[0] == 1:
            mean_probs = mean_probs.squeeze(0)
            uncertainty = uncertainty.squeeze(0)
            epistemic = epistemic.squeeze(0)
        
        return UncertaintyResult(
            mean_probs=mean_probs,
            uncertainty=uncertainty,
            epistemic=epistemic,
            method="ensemble",
            n_samples=len(self.models),
        )


class UncertaintyEstimator:
    """
    Unified uncertainty estimator supporting multiple methods.
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: str = 'mc_dropout',
        n_samples: int = 10,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.method = method
        self.device = device
        
        if method == 'mc_dropout':
            self.estimator = MCDropoutEstimator(model, n_samples, device)
        elif method == 'entropy':
            self.estimator = EntropyEstimator(model, device)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate(self, x: torch.Tensor, **kwargs) -> UncertaintyResult:
        """Estimate uncertainty."""
        return self.estimator.estimate(x, **kwargs)


class UncertaintyPolicy:
    """
    Policy for handling predictions based on uncertainty.
    
    Defines how the system should respond to different uncertainty levels:
    - Low: Normal operation
    - Medium: Add review flag
    - High: Suppress hard alarm, issue soft alert
    - Very High: Suppress all alarms
    """
    
    # Thresholds
    LOW_THRESHOLD = 0.2
    MEDIUM_THRESHOLD = 0.4
    HIGH_THRESHOLD = 0.6
    
    def __init__(
        self,
        medium_threshold: float = 0.4,
        high_threshold: float = 0.6,
    ):
        self.MEDIUM_THRESHOLD = medium_threshold
        self.HIGH_THRESHOLD = high_threshold
    
    def get_uncertainty_level(self, uncertainty: float) -> str:
        """
        Categorize uncertainty level.
        
        Returns: 'low', 'medium', 'high', or 'very_high'
        """
        if uncertainty < self.LOW_THRESHOLD:
            return 'low'
        elif uncertainty < self.MEDIUM_THRESHOLD:
            return 'medium'
        elif uncertainty < self.HIGH_THRESHOLD:
            return 'high'
        else:
            return 'very_high'
    
    def apply_policy(
        self,
        prediction: Dict[str, Any],
        uncertainty_result: UncertaintyResult,
    ) -> Dict[str, Any]:
        """
        Apply uncertainty policy to prediction.
        
        Modifies prediction based on uncertainty level.
        
        Args:
            prediction: Model prediction dict
            uncertainty_result: Uncertainty estimation result
            
        Returns:
            Modified prediction dict
        """
        result = prediction.copy()
        mean_unc = uncertainty_result.mean_uncertainty
        
        # Add uncertainty info
        result['uncertainty'] = mean_unc
        result['uncertainty_level'] = self.get_uncertainty_level(mean_unc)
        result['uncertainty_method'] = uncertainty_result.method
        
        level = result['uncertainty_level']
        
        if level == 'very_high':
            # Suppress all alarms
            result['alarm_type'] = 'suppressed'
            result['requires_review'] = True
            result['suppression_reason'] = 'very_high_uncertainty'
            
        elif level == 'high':
            # Downgrade to soft alert
            if result.get('alarm_type') == 'alarm':
                result['alarm_type'] = 'soft_alert'
            result['requires_review'] = True
            result['uncertainty_warning'] = True
            
        elif level == 'medium':
            # Add review flag
            result['requires_review'] = True
            result['uncertainty_warning'] = True
            
        else:  # low
            result['requires_review'] = result.get('requires_review', False)
            result['uncertainty_warning'] = False
        
        # Adjust confidence
        if 'confidence' in result:
            # Reduce confidence based on uncertainty
            adjustment = 1.0 - (mean_unc * 0.5)  # Max 50% reduction
            result['adjusted_confidence'] = result['confidence'] * adjustment
        
        return result
    
    def should_suppress_alarm(
        self,
        uncertainty_result: UncertaintyResult,
    ) -> Tuple[bool, str]:
        """
        Quick check if alarm should be suppressed.
        
        Returns:
            (should_suppress, reason)
        """
        level = self.get_uncertainty_level(uncertainty_result.mean_uncertainty)
        
        if level == 'very_high':
            return True, "very_high_uncertainty"
        
        return False, ""
    
    def get_confidence_adjustment(
        self,
        uncertainty_result: UncertaintyResult,
    ) -> float:
        """
        Get multiplier for confidence adjustment.
        
        Returns value in [0.5, 1.0].
        """
        mean_unc = uncertainty_result.mean_uncertainty
        return max(0.5, 1.0 - mean_unc * 0.5)
    
    def format_uncertainty_message(
        self,
        uncertainty_result: UncertaintyResult,
    ) -> str:
        """
        Get human-readable uncertainty message.
        """
        level = self.get_uncertainty_level(uncertainty_result.mean_uncertainty)
        mean_unc = uncertainty_result.mean_uncertainty
        
        messages = {
            'low': f"Model is confident (uncertainty: {mean_unc:.1%})",
            'medium': f"Moderate uncertainty ({mean_unc:.1%}). Review recommended.",
            'high': f"High uncertainty ({mean_unc:.1%}). Treat with caution.",
            'very_high': f"Very high uncertainty ({mean_unc:.1%}). Manual review required.",
        }
        
        return messages[level]


def compute_predictive_uncertainty(
    probs: np.ndarray,
    method: str = 'entropy'
) -> np.ndarray:
    """
    Compute uncertainty from probabilities.
    
    Args:
        probs: Probability array (..., n_classes)
        method: 'entropy' or 'margin'
        
    Returns:
        Uncertainty array (same shape without last dim)
    """
    if method == 'entropy':
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        max_entropy = np.log(probs.shape[-1])
        return entropy / max_entropy
    
    elif method == 'margin':
        # Uncertainty = 1 - (top prob - second prob)
        sorted_probs = np.sort(probs, axis=-1)
        margin = sorted_probs[..., -1] - sorted_probs[..., -2]
        return 1.0 - margin
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("Uncertainty Estimation Demo")
    print("=" * 60)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 5),
            )
        
        def forward(self, x):
            # Flatten and process
            batch = x.shape[0]
            x = x.view(batch, -1)[:, :100]  # Take first 100
            return self.fc(x).unsqueeze(1)  # Add seq dim
    
    model = SimpleModel()
    
    # Test input
    x = torch.randn(1, 1, 360)
    
    # MC Dropout
    print("\nMC Dropout Estimation:")
    mc_estimator = MCDropoutEstimator(model, n_samples=20, device='cpu')
    mc_result = mc_estimator.estimate(x)
    print(f"  Mean uncertainty: {mc_result.mean_uncertainty:.3f}")
    print(f"  Max uncertainty: {mc_result.max_uncertainty:.3f}")
    print(f"  Is high uncertainty: {mc_result.is_high_uncertainty()}")
    
    # Entropy
    print("\nEntropy Estimation:")
    entropy_estimator = EntropyEstimator(model, device='cpu')
    entropy_result = entropy_estimator.estimate(x)
    print(f"  Mean uncertainty: {entropy_result.mean_uncertainty:.3f}")
    
    # Test policy
    print("\nTesting Uncertainty Policy:")
    policy = UncertaintyPolicy()
    
    prediction = {
        'episode_type': 'VT',
        'confidence': 0.85,
        'alarm_type': 'alarm',
    }
    
    result = policy.apply_policy(prediction, mc_result)
    print(f"  Uncertainty level: {result['uncertainty_level']}")
    print(f"  Alarm type: {result['alarm_type']}")
    print(f"  Requires review: {result['requires_review']}")
    print(f"  Message: {policy.format_uncertainty_message(mc_result)}")
    
    print("\n" + "=" * 60)
