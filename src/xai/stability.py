"""
XAI Stability Checker.

Validates that XAI explanations are:
1. Stable under small input perturbations
2. Aligned with detected episode boundaries
3. Consistent across similar inputs

Unstable explanations are unreliable and should not be presented to users.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .saliency import XAIModule, AttributionResult


@dataclass
class StabilityResult:
    """Result from stability check."""
    is_stable: bool
    mean_similarity: float
    min_similarity: float
    max_similarity: float
    std_similarity: float
    n_trials: int
    noise_std: float
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_stable': self.is_stable,
            'mean_similarity': self.mean_similarity,
            'min_similarity': self.min_similarity,
            'max_similarity': self.max_similarity,
            'std_similarity': self.std_similarity,
            'n_trials': self.n_trials,
            'noise_std': self.noise_std,
            'recommendations': self.recommendations,
        }


@dataclass
class AlignmentResult:
    """Result from alignment check."""
    is_aligned: bool
    iou: float                      # Intersection over Union
    precision: float                # Fraction of high-attr in episode
    recall: float                   # Fraction of episode with high-attr
    episode_coverage: float         # How much of episode has attribution
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_aligned': self.is_aligned,
            'iou': self.iou,
            'precision': self.precision,
            'recall': self.recall,
            'episode_coverage': self.episode_coverage,
            'recommendations': self.recommendations,
        }


class XAIStabilityChecker:
    """
    Verify XAI explanations are stable and meaningful.
    
    An explanation is considered good if:
    1. It's stable under small noise (similarity > 0.75)
    2. High-attribution regions align with detected episodes (precision > 0.5)
    3. Explanations are consistent for similar inputs
    
    These checks are essential for trustworthy XAI.
    """
    
    # Thresholds
    STABILITY_THRESHOLD = 0.75
    ALIGNMENT_PRECISION_THRESHOLD = 0.4
    ALIGNMENT_IOU_THRESHOLD = 0.3
    
    def __init__(self, xai_module: XAIModule):
        """
        Initialize stability checker.
        
        Args:
            xai_module: XAI module to check
        """
        self.xai = xai_module
    
    def check_noise_stability(
        self,
        x: torch.Tensor,
        target_class: int,
        noise_std: float = 0.01,
        n_trials: int = 5,
        method: str = 'integrated_gradients',
    ) -> StabilityResult:
        """
        Check if explanations are stable under small noise perturbations.
        
        A stable explanation should not change significantly when
        small noise is added to the input.
        
        Args:
            x: Input signal
            target_class: Class to explain
            noise_std: Standard deviation of Gaussian noise
            n_trials: Number of noisy samples to test
            method: Attribution method to use
            
        Returns:
            StabilityResult with stability metrics
        """
        # Get base attribution
        base_result = self.xai.explain(x, target_class, method=method)
        base_attr = base_result.attributions
        
        similarities = []
        recommendations = []
        
        for i in range(n_trials):
            # Add small noise
            noise = torch.randn_like(x) * noise_std
            x_noisy = x + noise
            
            # Get attribution for noisy input
            noisy_result = self.xai.explain(x_noisy, target_class, method=method)
            noisy_attr = noisy_result.attributions
            
            # Compute cosine similarity
            sim = self._cosine_similarity(base_attr, noisy_attr)
            similarities.append(sim)
        
        # Aggregate results
        mean_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        std_sim = np.std(similarities)
        
        is_stable = mean_sim >= self.STABILITY_THRESHOLD
        
        if not is_stable:
            recommendations.append(
                f"Explanation unstable (mean sim={mean_sim:.3f} < {self.STABILITY_THRESHOLD})"
            )
            recommendations.append(
                "Consider using more IG steps or averaging over baselines"
            )
        
        if std_sim > 0.1:
            recommendations.append(
                f"High variance in stability (std={std_sim:.3f})"
            )
        
        return StabilityResult(
            is_stable=is_stable,
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            max_similarity=max_sim,
            std_similarity=std_sim,
            n_trials=n_trials,
            noise_std=noise_std,
            recommendations=recommendations,
        )
    
    def check_episode_alignment(
        self,
        attributions: np.ndarray,
        episode_start: int,
        episode_end: int,
        threshold_percentile: float = 90,
    ) -> AlignmentResult:
        """
        Check if high-attribution regions align with detected episode.
        
        Good explanations should highlight regions that correspond
        to the detected arrhythmia episode.
        
        Args:
            attributions: Attribution array
            episode_start: Start sample of episode
            episode_end: End sample of episode
            threshold_percentile: Percentile for "high" attribution
            
        Returns:
            AlignmentResult with alignment metrics
        """
        recommendations = []
        
        # Find high-attribution regions
        threshold = np.percentile(np.abs(attributions), threshold_percentile)
        high_attr_mask = np.abs(attributions) > threshold
        
        # Episode mask
        episode_mask = np.zeros_like(attributions, dtype=bool)
        episode_mask[episode_start:episode_end] = True
        
        # Compute metrics
        intersection = np.sum(high_attr_mask & episode_mask)
        high_attr_count = np.sum(high_attr_mask)
        episode_count = np.sum(episode_mask)
        union = np.sum(high_attr_mask | episode_mask)
        
        # IoU
        iou = intersection / (union + 1e-8)
        
        # Precision: What fraction of high-attr is in episode?
        precision = intersection / (high_attr_count + 1e-8)
        
        # Recall: What fraction of episode has high-attr?
        recall = intersection / (episode_count + 1e-8)
        
        # Episode coverage: average attribution in episode region
        if episode_count > 0:
            episode_coverage = np.mean(np.abs(attributions[episode_start:episode_end]))
            outside_coverage = np.mean(np.abs(np.concatenate([
                attributions[:episode_start],
                attributions[episode_end:]
            ])))
            coverage_ratio = episode_coverage / (outside_coverage + 1e-8)
        else:
            coverage_ratio = 0.0
        
        is_aligned = (
            precision >= self.ALIGNMENT_PRECISION_THRESHOLD or
            iou >= self.ALIGNMENT_IOU_THRESHOLD
        )
        
        if not is_aligned:
            recommendations.append(
                f"Poor alignment (precision={precision:.3f}, IoU={iou:.3f})"
            )
            recommendations.append(
                "Explanation may not reflect the detected episode"
            )
        
        if precision < 0.3:
            recommendations.append(
                "Many high-attribution regions are outside the episode"
            )
        
        if recall < 0.3:
            recommendations.append(
                "Much of the episode has low attribution"
            )
        
        return AlignmentResult(
            is_aligned=is_aligned,
            iou=iou,
            precision=precision,
            recall=recall,
            episode_coverage=coverage_ratio,
            recommendations=recommendations,
        )
    
    def check_consistency(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        target_class: int,
        expected_similarity: float = 0.6,
        method: str = 'integrated_gradients',
    ) -> Tuple[float, bool]:
        """
        Check if similar inputs get similar explanations.
        
        Args:
            x1, x2: Two similar input signals
            target_class: Class to explain
            expected_similarity: Minimum expected similarity
            method: Attribution method
            
        Returns:
            Tuple of (similarity, is_consistent)
        """
        result1 = self.xai.explain(x1, target_class, method=method)
        result2 = self.xai.explain(x2, target_class, method=method)
        
        # Handle different lengths by padding/truncating
        attr1 = result1.attributions
        attr2 = result2.attributions
        
        min_len = min(len(attr1), len(attr2))
        attr1 = attr1[:min_len]
        attr2 = attr2[:min_len]
        
        similarity = self._cosine_similarity(attr1, attr2)
        is_consistent = similarity >= expected_similarity
        
        return similarity, is_consistent
    
    def run_full_sanity_check(
        self,
        x: torch.Tensor,
        target_class: int,
        episode_start: Optional[int] = None,
        episode_end: Optional[int] = None,
        method: str = 'integrated_gradients',
    ) -> Dict[str, Any]:
        """
        Run comprehensive sanity checks on explanation.
        
        Args:
            x: Input signal
            target_class: Class to explain
            episode_start: Optional episode start for alignment check
            episode_end: Optional episode end for alignment check
            method: Attribution method
            
        Returns:
            Dict with all check results
        """
        results = {}
        
        # Get attribution
        attribution = self.xai.explain(x, target_class, method=method)
        results['attribution'] = attribution
        
        # Check noise stability
        stability = self.check_noise_stability(x, target_class, method=method)
        results['stability'] = stability.to_dict()
        
        # Check alignment if episode provided
        if episode_start is not None and episode_end is not None:
            alignment = self.check_episode_alignment(
                attribution.attributions,
                episode_start,
                episode_end
            )
            results['alignment'] = alignment.to_dict()
        
        # Overall quality score
        quality_score = stability.mean_similarity
        if 'alignment' in results:
            quality_score = (quality_score + results['alignment']['precision']) / 2
        
        results['quality_score'] = quality_score
        results['is_trustworthy'] = (
            stability.is_stable and 
            (episode_start is None or results.get('alignment', {}).get('is_aligned', True))
        )
        
        # Aggregate recommendations
        all_recs = list(stability.recommendations)
        if 'alignment' in results:
            all_recs.extend(results['alignment']['recommendations'])
        results['recommendations'] = all_recs
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)


class XAIQualityGate:
    """
    Gate for deciding whether to show XAI explanation to user.
    
    Only shows explanations that pass quality checks.
    """
    
    def __init__(
        self,
        stability_threshold: float = 0.75,
        alignment_threshold: float = 0.4,
    ):
        self.stability_threshold = stability_threshold
        self.alignment_threshold = alignment_threshold
    
    def should_show_explanation(
        self,
        sanity_check_result: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Decide whether to show explanation to user.
        
        Returns:
            Tuple of (should_show, reason)
        """
        # Check trustworthiness
        if not sanity_check_result.get('is_trustworthy', False):
            return False, "Explanation did not pass quality checks"
        
        # Check stability
        stability = sanity_check_result.get('stability', {})
        if stability.get('mean_similarity', 0) < self.stability_threshold:
            return False, f"Unstable explanation (similarity={stability.get('mean_similarity', 0):.2f})"
        
        # Check alignment if available
        alignment = sanity_check_result.get('alignment', {})
        if alignment and alignment.get('precision', 1.0) < self.alignment_threshold:
            return False, f"Poor alignment (precision={alignment.get('precision', 0):.2f})"
        
        return True, "Explanation passed quality checks"
    
    def get_explanation_confidence(
        self,
        sanity_check_result: Dict[str, Any]
    ) -> float:
        """
        Get confidence score for explanation (0-1).
        
        Higher = more trustworthy explanation.
        """
        quality_score = sanity_check_result.get('quality_score', 0.5)
        
        stability = sanity_check_result.get('stability', {})
        stability_score = stability.get('mean_similarity', 0.5)
        
        # Weight stability more heavily
        confidence = 0.6 * stability_score + 0.4 * quality_score
        
        return confidence


if __name__ == "__main__":
    print("XAI Stability Checker Demo")
    print("=" * 60)
    
    import torch.nn.functional as F
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 16, 7, padding=3)
            self.pool = nn.MaxPool1d(8)
            self.fc = nn.Linear(16, 5)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.permute(0, 2, 1)
            return self.fc(x)
    
    model = SimpleModel()
    xai = XAIModule(model, device='cpu')
    checker = XAIStabilityChecker(xai)
    
    # Test input
    x = torch.randn(1, 1, 360)
    
    print("\nRunning full sanity check...")
    results = checker.run_full_sanity_check(
        x, 
        target_class=3,
        episode_start=100,
        episode_end=200,
    )
    
    print(f"\nStability: {results['stability']}")
    print(f"Alignment: {results.get('alignment', 'N/A')}")
    print(f"Quality Score: {results['quality_score']:.3f}")
    print(f"Trustworthy: {results['is_trustworthy']}")
    
    if results['recommendations']:
        print("\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")
    
    # Test quality gate
    gate = XAIQualityGate()
    should_show, reason = gate.should_show_explanation(results)
    confidence = gate.get_explanation_confidence(results)
    
    print(f"\nQuality Gate:")
    print(f"  Should show: {should_show}")
    print(f"  Reason: {reason}")
    print(f"  Confidence: {confidence:.3f}")
    
    print("\n" + "=" * 60)
