"""
Unit tests for evaluation metrics module.

Tests for:
- EvaluationProtocol episode matching
- Per-class FA calculation
- OnsetCriticalEvaluator
- ConfidenceAwareEvaluator
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.metrics import (
    EpisodeLabel,
    EpisodeType,
    LabelConfidenceTier,
    EvaluationMetrics,
    EvaluationProtocol,
    PerClassFACalculator,
    OnsetCriticalEvaluator,
    ConfidenceAwareEvaluator,
    PerPatientEvaluator,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_ground_truth():
    """Sample ground truth episodes for testing."""
    return [
        EpisodeLabel(
            EpisodeType.VT_MONOMORPHIC, 
            start_sample=1000, 
            end_sample=2000,
            start_time_sec=2.78,
            end_time_sec=5.56,
            label_tier=LabelConfidenceTier.EXPERT_RHYTHM,
            patient_id="P001"
        ),
        EpisodeLabel(
            EpisodeType.VT_POLYMORPHIC, 
            start_sample=5000, 
            end_sample=6500,
            start_time_sec=13.89,
            end_time_sec=18.06,
            label_tier=LabelConfidenceTier.EXPERT_RHYTHM,
            patient_id="P001"
        ),
        EpisodeLabel(
            EpisodeType.SVT, 
            start_sample=10000, 
            end_sample=12000,
            start_time_sec=27.78,
            end_time_sec=33.33,
            label_tier=LabelConfidenceTier.DERIVED_RHYTHM,
            patient_id="P002"
        ),
    ]


@pytest.fixture
def sample_predictions():
    """Sample predictions with mixed quality."""
    return [
        # Good match for VT_MONOMORPHIC
        EpisodeLabel(
            EpisodeType.VT_MONOMORPHIC,
            start_sample=1050,
            end_sample=2100,
            patient_id="P001"
        ),
        # Good match for VT_POLYMORPHIC  
        EpisodeLabel(
            EpisodeType.VT_POLYMORPHIC,
            start_sample=5200,
            end_sample=6600,
            patient_id="P001"
        ),
        # Good match for SVT
        EpisodeLabel(
            EpisodeType.SVT,
            start_sample=10100,
            end_sample=11900,
            patient_id="P002"
        ),
        # False positive
        EpisodeLabel(
            EpisodeType.VT_MONOMORPHIC,
            start_sample=30000,
            end_sample=31000,
            patient_id="P003"
        ),
    ]


# =============================================================================
# EVALUATION PROTOCOL TESTS
# =============================================================================

class TestEvaluationProtocol:
    """Tests for EvaluationProtocol class."""
    
    def test_iou_calculation_perfect_overlap(self):
        """IoU should be 1.0 for identical episodes."""
        protocol = EvaluationProtocol()
        
        ep1 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)
        ep2 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)
        
        iou = protocol._compute_iou(ep1, ep2)
        assert iou == 1.0
    
    def test_iou_calculation_no_overlap(self):
        """IoU should be 0.0 for non-overlapping episodes."""
        protocol = EvaluationProtocol()
        
        ep1 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)
        ep2 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 3000, 4000)
        
        iou = protocol._compute_iou(ep1, ep2)
        assert iou == 0.0
    
    def test_iou_calculation_partial_overlap(self):
        """IoU should be correct for partial overlap."""
        protocol = EvaluationProtocol()
        
        ep1 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)  # 1000 samples
        ep2 = EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1500, 2500)  # 1000 samples
        # Intersection: 1500-2000 = 500 samples
        # Union: 1000 + 1000 - 500 = 1500 samples
        # IoU = 500 / 1500 = 0.333...
        
        iou = protocol._compute_iou(ep1, ep2)
        assert abs(iou - 1/3) < 0.01
    
    def test_type_compatibility_vt_types(self):
        """VT types should be compatible with each other."""
        protocol = EvaluationProtocol()
        
        assert protocol._types_compatible(EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC)
        assert protocol._types_compatible(EpisodeType.VT_MONOMORPHIC, EpisodeType.VFL)
        assert protocol._types_compatible(EpisodeType.VFL, EpisodeType.VFIB)
    
    def test_type_compatibility_svt_types(self):
        """SVT types should be compatible with each other."""
        protocol = EvaluationProtocol()
        
        assert protocol._types_compatible(EpisodeType.SVT, EpisodeType.AFIB_RVR)
        assert protocol._types_compatible(EpisodeType.SVT, EpisodeType.AFLUTTER)
    
    def test_type_incompatibility(self):
        """VT and SVT types should not be compatible."""
        protocol = EvaluationProtocol()
        
        assert not protocol._types_compatible(EpisodeType.VT_MONOMORPHIC, EpisodeType.SVT)
        assert not protocol._types_compatible(EpisodeType.VFL, EpisodeType.AFIB_RVR)
    
    def test_episode_matching(self, sample_ground_truth, sample_predictions):
        """Episode matching should work correctly."""
        protocol = EvaluationProtocol()
        
        matches = protocol._match_episodes(sample_predictions, sample_ground_truth)
        
        # Should match 3 episodes (all ground truth should be matched)
        assert len(matches) == 3
    
    def test_evaluate_returns_correct_metrics(self, sample_ground_truth, sample_predictions):
        """Evaluation should return correct metrics."""
        protocol = EvaluationProtocol()
        
        metrics = protocol.evaluate(sample_predictions, sample_ground_truth, total_duration_hours=2.0)
        
        # 3 matches, 0 FN (assuming all GT matched), 1 FP
        assert metrics.episode_sensitivity == 1.0  # All GT matched
        assert metrics.episode_ppv == 0.75  # 3 out of 4 predictions correct
        assert metrics.false_alarms_per_hour == 0.5  # 1 FP / 2 hours
    
    def test_vt_sensitivity_calculation(self, sample_ground_truth, sample_predictions):
        """VT sensitivity should be computed correctly."""
        protocol = EvaluationProtocol()
        
        metrics = protocol.evaluate(sample_predictions, sample_ground_truth, total_duration_hours=2.0)
        
        # 2 VT episodes in ground truth, both matched
        assert metrics.vt_sensitivity == 1.0


# =============================================================================
# PER-CLASS FA TESTS
# =============================================================================

class TestPerClassFACalculator:
    """Tests for PerClassFACalculator."""
    
    def test_per_class_fa_no_false_positives(self, sample_ground_truth):
        """No false positives should give 0 FA/hr for all classes."""
        # Predictions that exactly match ground truth (no FP)
        predictions = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
            EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6500, patient_id="P001"),
            EpisodeLabel(EpisodeType.SVT, 10000, 12000, patient_id="P002"),
        ]
        
        calc = PerClassFACalculator()
        fa_rates = calc.compute(predictions, sample_ground_truth, total_duration_hours=2.0)
        
        assert fa_rates['vt_vfl'] == 0.0
        assert fa_rates['svt'] == 0.0
        assert fa_rates['sinus_tachy'] == 0.0
    
    def test_per_class_fa_with_vt_fp(self, sample_ground_truth):
        """VT false positive should only increment VT FA."""
        predictions = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
            EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6500, patient_id="P001"),
            EpisodeLabel(EpisodeType.SVT, 10000, 12000, patient_id="P002"),
            # False positive VT
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 50000, 51000, patient_id="P999"),
        ]
        
        calc = PerClassFACalculator()
        fa_rates = calc.compute(predictions, sample_ground_truth, total_duration_hours=2.0)
        
        assert fa_rates['vt_vfl'] == 0.5  # 1 FP / 2 hours
        assert fa_rates['svt'] == 0.0
        assert fa_rates['sinus_tachy'] == 0.0


# =============================================================================
# ONSET-CRITICAL EVALUATOR TESTS
# =============================================================================

class TestOnsetCriticalEvaluator:
    """Tests for OnsetCriticalEvaluator."""
    
    def test_onset_accuracy_perfect_timing(self):
        """Perfect onset timing should give 100% accuracy."""
        gt = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)]
        pred = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)]
        
        evaluator = OnsetCriticalEvaluator(fs=360)
        result = evaluator.evaluate_onset_accuracy(pred, gt)
        
        assert result['mean_onset_error_ms'] == 0.0
        assert result['onset_accuracy'] == 1.0
    
    def test_onset_accuracy_late_detection(self):
        """Late detection should be captured in error."""
        gt = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)]
        # Detection starts 180 samples late (500ms at 360Hz)
        pred = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1180, 2100)]
        
        evaluator = OnsetCriticalEvaluator(fs=360, max_onset_error_ms=500)
        result = evaluator.evaluate_onset_accuracy(pred, gt)
        
        # Error should be 500ms
        assert abs(result['mean_onset_error_ms'] - 500.0) < 1.0
        assert result['onset_accuracy'] == 1.0  # Within tolerance
    
    def test_onset_accuracy_early_detection(self):
        """Early detection should give negative error."""
        gt = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000)]
        # Detection starts 180 samples early (500ms at 360Hz)
        pred = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 820, 1900)]
        
        evaluator = OnsetCriticalEvaluator(fs=360, max_onset_error_ms=500)
        result = evaluator.evaluate_onset_accuracy(pred, gt)
        
        # Error should be -500ms (early)
        assert abs(result['mean_onset_error_ms'] + 500.0) < 1.0
        assert result['late_detection_fraction'] == 0.0  # All early
    
    def test_detection_latencies(self):
        """Detection latency calculation should work."""
        events = [
            {'gt_onset_sec': 10.0, 'first_detection_sec': 11.0, 'warning_sec': 12.0, 'alarm_sec': 13.0},
            {'gt_onset_sec': 20.0, 'first_detection_sec': 21.5, 'warning_sec': 22.5, 'alarm_sec': 24.0},
        ]
        
        evaluator = OnsetCriticalEvaluator()
        result = evaluator.evaluate_detection_latencies(events)
        
        assert result['first_detection']['mean_sec'] == 1.25  # (1.0 + 1.5) / 2
        assert result['warning']['mean_sec'] == 2.25  # (2.0 + 2.5) / 2
        assert result['alarm']['mean_sec'] == 3.5  # (3.0 + 4.0) / 2


# =============================================================================
# CONFIDENCE-AWARE EVALUATOR TESTS
# =============================================================================

class TestConfidenceAwareEvaluator:
    """Tests for ConfidenceAwareEvaluator."""
    
    def test_stratification_by_tier(self):
        """Metrics should be stratified by label tier."""
        # Ground truth with different tiers
        gt = [
            EpisodeLabel(
                EpisodeType.VT_MONOMORPHIC, 1000, 2000,
                label_tier=LabelConfidenceTier.EXPERT_RHYTHM
            ),
            EpisodeLabel(
                EpisodeType.VT_POLYMORPHIC, 5000, 6000,
                label_tier=LabelConfidenceTier.DERIVED_RHYTHM
            ),
        ]
        
        # Only match the expert tier episode
        pred = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000),
        ]
        
        evaluator = ConfidenceAwareEvaluator()
        result = evaluator.evaluate_with_confidence(pred, gt, total_duration_hours=1.0)
        
        assert result['expert_tier']['vt_sensitivity'] == 1.0
        assert result['derived_tier']['vt_sensitivity'] == 0.0
    
    def test_weighted_sensitivity(self):
        """Weighted sensitivity should account for tier weights."""
        gt = [
            EpisodeLabel(
                EpisodeType.VT_MONOMORPHIC, 1000, 2000,
                label_tier=LabelConfidenceTier.EXPERT_RHYTHM  # Weight 1.0
            ),
            EpisodeLabel(
                EpisodeType.VT_POLYMORPHIC, 5000, 6000,
                label_tier=LabelConfidenceTier.HEURISTIC  # Weight 0.5
            ),
        ]
        
        # Match only expert tier
        pred = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000),
        ]
        
        evaluator = ConfidenceAwareEvaluator()
        result = evaluator.evaluate_with_confidence(pred, gt, total_duration_hours=1.0)
        
        # Expert: 1.0 sens * 1 ep * 1.0 weight = 1.0
        # Heuristic: 0.0 sens * 1 ep * 0.5 weight = 0.0
        # Weighted = 1.0 / (1.0 + 0.5) = 0.667
        assert abs(result['weighted_vt_sensitivity'] - 2/3) < 0.01


# =============================================================================
# PER-PATIENT EVALUATOR TESTS
# =============================================================================

class TestPerPatientEvaluator:
    """Tests for PerPatientEvaluator."""
    
    def test_per_patient_metrics(self, sample_ground_truth, sample_predictions):
        """Per-patient metrics should be computed correctly."""
        hours = {"P001": 2.0, "P002": 1.0, "P003": 1.0}
        
        evaluator = PerPatientEvaluator()
        result = evaluator.compute_per_patient_metrics(
            sample_predictions, sample_ground_truth, hours
        )
        
        # P001 should have VT sensitivity = 1.0 (2 VT matched)
        assert result["P001"]['vt_sensitivity'] == 1.0
        assert result["P001"]['n_vt_episodes'] == 2
    
    def test_worst_patients_identification(self):
        """Should identify patients with worst performance."""
        per_patient = {
            "P001": {'vt_sensitivity': 1.0, 'n_vt_episodes': 2, 'fa_per_hour': 0.5},
            "P002": {'vt_sensitivity': 0.5, 'n_vt_episodes': 2, 'fa_per_hour': 1.0},
            "P003": {'vt_sensitivity': 0.0, 'n_vt_episodes': 1, 'fa_per_hour': 3.0},
        }
        
        evaluator = PerPatientEvaluator()
        result = evaluator.get_worst_patients(per_patient, n_worst=2)
        
        # Worst by sensitivity should be P003 (0.0) then P002 (0.5)
        assert result['worst_by_sensitivity'][0]['patient_id'] == "P003"
        assert result['worst_by_sensitivity'][1]['patient_id'] == "P002"
        
        # Worst by FA should be P003 (3.0) then P002 (1.0)
        assert result['worst_by_fa'][0]['patient_id'] == "P003"


# =============================================================================
# EVALUATION METRICS DATACLASS TESTS
# =============================================================================

class TestEvaluationMetrics:
    """Tests for EvaluationMetrics dataclass."""
    
    def test_sensitivity_floor_check(self):
        """Sensitivity floor check should work."""
        metrics = EvaluationMetrics(vt_sensitivity=0.92)
        assert metrics.passes_sensitivity_floor(0.90) is True
        assert metrics.passes_sensitivity_floor(0.95) is False
    
    def test_fa_ceiling_check(self):
        """FA ceiling check should work."""
        metrics = EvaluationMetrics(false_alarms_per_hour=1.5)
        assert metrics.passes_fa_ceiling(2.0) is True
        assert metrics.passes_fa_ceiling(1.0) is False
    
    def test_to_dict_conversion(self):
        """Metrics should convert to dict correctly."""
        metrics = EvaluationMetrics(
            episode_sensitivity=0.95,
            vt_sensitivity=0.92,
            false_alarms_per_hour=1.5
        )
        
        d = metrics.to_dict()
        
        assert d['episode_sensitivity'] == 0.95
        assert d['vt_sensitivity'] == 0.92
        assert d['false_alarms_per_hour'] == 1.5


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
