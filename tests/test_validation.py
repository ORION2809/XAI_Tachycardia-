"""
Unit tests for validation module.

Tests for:
- SubCohortValidator
- PatientSplitValidator
- CrossValidator
- AcceptanceTests
- DomainShiftValidator
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
)

from evaluation.validation import (
    SubCohort,
    PatientMetadata,
    SubCohortValidator,
    PatientSplitValidator,
    CrossValidator,
    CrossValidationConfig,
    AcceptanceTests,
    AcceptanceConfig,
    DomainShiftValidator,
    DomainShiftConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def patient_metadata():
    """Sample patient metadata for testing."""
    return {
        "P001": PatientMetadata("P001", age=65, is_paced=False, mean_sqi=0.85),
        "P002": PatientMetadata("P002", age=78, is_paced=True, mean_sqi=0.55),  # Paced, low SQI
        "P003": PatientMetadata("P003", age=45, has_bbb=True, mean_sqi=0.90),  # BBB
        "P004": PatientMetadata("P004", age=55, baseline_hr=110, mean_sqi=0.75),  # High HR
        "P005": PatientMetadata("P005", age=80, mean_sqi=0.40),  # Elderly, low SQI
    }


@pytest.fixture
def sample_episodes():
    """Sample episodes for testing."""
    ground_truth = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6500, patient_id="P002"),
        EpisodeLabel(EpisodeType.SVT, 10000, 12000, patient_id="P003"),
    ]
    
    predictions = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5100, 6600, patient_id="P002"),
        EpisodeLabel(EpisodeType.SVT, 10000, 12000, patient_id="P003"),
    ]
    
    return ground_truth, predictions


# =============================================================================
# PATIENT METADATA TESTS
# =============================================================================

class TestPatientMetadata:
    """Tests for PatientMetadata dataclass."""
    
    def test_subcohort_detection_paced(self, patient_metadata):
        """Paced patients should be identified."""
        cohorts = patient_metadata["P002"].get_subcohorts()
        assert SubCohort.PACED in cohorts
    
    def test_subcohort_detection_low_sqi(self, patient_metadata):
        """Low SQI patients should be identified."""
        cohorts = patient_metadata["P005"].get_subcohorts()
        assert SubCohort.LOW_SQI in cohorts
    
    def test_subcohort_detection_bbb(self, patient_metadata):
        """BBB patients should be identified."""
        cohorts = patient_metadata["P003"].get_subcohorts()
        assert SubCohort.BBB in cohorts
    
    def test_subcohort_detection_high_hr(self, patient_metadata):
        """High HR patients should be identified."""
        cohorts = patient_metadata["P004"].get_subcohorts()
        assert SubCohort.HIGH_HR in cohorts
    
    def test_subcohort_detection_elderly(self, patient_metadata):
        """Elderly patients should be identified."""
        cohorts = patient_metadata["P005"].get_subcohorts()
        assert SubCohort.ELDERLY in cohorts
    
    def test_multiple_subcohorts(self, patient_metadata):
        """Patients can belong to multiple sub-cohorts."""
        # P002 is paced and has low SQI
        cohorts = patient_metadata["P002"].get_subcohorts()
        assert SubCohort.PACED in cohorts
        assert SubCohort.LOW_SQI in cohorts


# =============================================================================
# SUBCOHORT VALIDATOR TESTS
# =============================================================================

class TestSubCohortValidator:
    """Tests for SubCohortValidator."""
    
    def test_validation_returns_results_for_all_cohorts(
        self, sample_episodes, patient_metadata
    ):
        """Validation should return results for each sub-cohort."""
        gt, pred = sample_episodes
        hours = {"P001": 1.0, "P002": 1.0, "P003": 1.0, "P004": 1.0, "P005": 1.0}
        
        validator = SubCohortValidator()
        results = validator.validate_subcohorts(pred, gt, patient_metadata, hours)
        
        # Should have results for each SubCohort enum value
        for cohort in SubCohort:
            assert cohort.value in results['subcohorts']
    
    def test_validation_relaxed_thresholds(self):
        """Sub-cohorts should use relaxed thresholds."""
        validator = SubCohortValidator(sensitivity_floor=0.90, fa_ceiling=2.0)
        
        # Relaxed floor should be 0.765 (85% of 0.90)
        assert abs(validator.relaxed_sensitivity - 0.765) < 0.01
        
        # Relaxed ceiling should be 4.0 (2x of 2.0)
        assert validator.relaxed_fa == 4.0
    
    def test_empty_subcohort_skipped(self):
        """Sub-cohorts with no patients should be skipped."""
        gt = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001")]
        pred = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001")]
        
        # Only one patient, not in any special sub-cohort
        metadata = {"P001": PatientMetadata("P001", age=50, mean_sqi=0.90)}
        hours = {"P001": 1.0}
        
        validator = SubCohortValidator()
        results = validator.validate_subcohorts(pred, gt, metadata, hours)
        
        # Most sub-cohorts should be skipped
        paced_result = results['subcohorts'][SubCohort.PACED.value]
        assert paced_result.get('skipped') is True


# =============================================================================
# PATIENT SPLIT VALIDATOR TESTS
# =============================================================================

class TestPatientSplitValidator:
    """Tests for PatientSplitValidator."""
    
    def test_valid_split_no_overlap(self):
        """Valid split should have no patient overlap."""
        train = {"P001", "P002", "P003"}
        test = {"P004", "P005"}
        
        result = PatientSplitValidator.validate_split(train, test)
        
        assert result['valid'] is True
        assert result['n_overlapping'] == 0
    
    def test_invalid_split_with_overlap(self):
        """Invalid split should detect overlap."""
        train = {"P001", "P002", "P003"}
        test = {"P003", "P004", "P005"}  # P003 in both!
        
        result = PatientSplitValidator.validate_split(train, test)
        
        assert result['valid'] is False
        assert result['n_overlapping'] == 1
        assert "P003" in result['overlapping_patients']
    
    def test_episode_assignment_validation(self, sample_episodes):
        """Episode assignment validation should work."""
        gt, _ = sample_episodes
        
        train = {"P001"}
        test = {"P002", "P003"}
        
        result = PatientSplitValidator.validate_episode_assignments(gt, train, test)
        
        assert result['valid'] is True
        assert result['n_train_episodes'] == 1
        assert result['n_test_episodes'] == 2


# =============================================================================
# CROSS-VALIDATOR TESTS
# =============================================================================

class TestCrossValidator:
    """Tests for CrossValidator."""
    
    def test_fold_creation(self):
        """Folds should be created correctly."""
        config = CrossValidationConfig(n_folds=3)
        cv = CrossValidator(config)
        
        patient_ids = ["P001", "P002", "P003", "P004", "P005", "P006"]
        patient_has_vt = {p: i % 2 == 0 for i, p in enumerate(patient_ids)}
        
        folds = cv.create_folds(patient_ids, patient_has_vt)
        
        assert len(folds) == 3
        
        # Each fold should have train and test sets
        for train, test in folds:
            assert len(train) > 0
            assert len(test) > 0
            
            # No overlap between train and test
            assert len(set(train) & set(test)) == 0
    
    def test_all_patients_used_once_in_test(self):
        """Each patient should appear in test exactly once."""
        config = CrossValidationConfig(n_folds=3)
        cv = CrossValidator(config)
        
        patient_ids = ["P001", "P002", "P003", "P004", "P005", "P006"]
        patient_has_vt = {p: True for p in patient_ids}
        
        folds = cv.create_folds(patient_ids, patient_has_vt)
        
        # Collect all test patients
        all_test = []
        for _, test in folds:
            all_test.extend(test)
        
        # Each patient should appear exactly once
        assert sorted(all_test) == sorted(patient_ids)


# =============================================================================
# ACCEPTANCE TESTS TESTS
# =============================================================================

class TestAcceptanceTests:
    """Tests for AcceptanceTests class."""
    
    def test_sensitivity_floor_pass(self):
        """Should pass when VT sensitivity meets floor."""
        config = AcceptanceConfig(vt_vfl_sensitivity_floor=0.90)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(vt_sensitivity=0.92, svt_sensitivity=0.75)
        
        result = tests.test_sensitivity_floors(metrics)
        
        assert result['passed'] is True
        assert result['tests']['vt_sensitivity']['passed'] is True
    
    def test_sensitivity_floor_fail(self):
        """Should fail when VT sensitivity below floor."""
        config = AcceptanceConfig(vt_vfl_sensitivity_floor=0.90)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(vt_sensitivity=0.85, svt_sensitivity=0.75)
        
        result = tests.test_sensitivity_floors(metrics)
        
        assert result['passed'] is False
        assert result['tests']['vt_sensitivity']['passed'] is False
    
    def test_fa_ceiling_pass(self):
        """Should pass when FA below ceiling."""
        config = AcceptanceConfig(total_max_fa_per_hour=2.0)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(
            false_alarms_per_hour=1.5,
            vt_vfl_fa_per_hour=0.5,
            svt_fa_per_hour=0.3,
            sinus_tachy_fa_per_hour=0.2
        )
        
        result = tests.test_fa_ceilings(metrics)
        
        assert result['passed'] is True
    
    def test_fa_ceiling_fail(self):
        """Should fail when FA above ceiling."""
        config = AcceptanceConfig(total_max_fa_per_hour=2.0)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(false_alarms_per_hour=3.0)
        
        result = tests.test_fa_ceilings(metrics)
        
        assert result['passed'] is False
    
    def test_calibration_quality_pass(self):
        """Should pass when ECE below threshold."""
        config = AcceptanceConfig(max_ece=0.10)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(ece=0.08)
        
        result = tests.test_calibration_quality(metrics)
        
        assert result['passed'] is True
    
    def test_calibration_quality_fail(self):
        """Should fail when ECE above threshold."""
        config = AcceptanceConfig(max_ece=0.10)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(ece=0.15)
        
        result = tests.test_calibration_quality(metrics)
        
        assert result['passed'] is False
    
    def test_detection_latency_pass(self):
        """Should pass when latency below threshold."""
        config = AcceptanceConfig(max_p95_latency_sec=5.0)
        tests = AcceptanceTests(config)
        
        metrics = EvaluationMetrics(p95_detection_latency_sec=4.0)
        
        result = tests.test_detection_latency(metrics)
        
        assert result['passed'] is True
    
    def test_end_to_end_latency_gate(self):
        """End-to-end latency test should work as gate."""
        config = AcceptanceConfig(max_vt_onset_to_alarm_sec=10.0)
        tests = AcceptanceTests(config)
        
        # All within limit
        measurements_pass = [
            {'vt_onset_sec': 10.0, 'alarm_time_sec': 18.0},  # 8s
            {'vt_onset_sec': 20.0, 'alarm_time_sec': 29.0},  # 9s
        ]
        
        result = tests.test_end_to_end_latency(measurements_pass)
        assert result['passed'] is True
        
        # One violation
        measurements_fail = [
            {'vt_onset_sec': 10.0, 'alarm_time_sec': 18.0},  # 8s
            {'vt_onset_sec': 20.0, 'alarm_time_sec': 35.0},  # 15s - violation!
        ]
        
        result = tests.test_end_to_end_latency(measurements_fail)
        assert result['passed'] is False
        assert len(result['violations']) == 1
    
    def test_run_all_tests_integration(self):
        """Integration test for all acceptance tests."""
        config = AcceptanceConfig()
        tests = AcceptanceTests(config)
        
        # Passing metrics
        metrics = EvaluationMetrics(
            vt_sensitivity=0.92,
            svt_sensitivity=0.75,
            false_alarms_per_hour=1.5,
            vt_vfl_fa_per_hour=0.5,
            svt_fa_per_hour=0.3,
            sinus_tachy_fa_per_hour=0.2,
            ece=0.08,
            p95_detection_latency_sec=4.0
        )
        
        result = tests.run_all_tests(metrics)
        
        assert result['passed'] is True
        assert 'sensitivity_floors' in result['tests']
        assert 'fa_ceilings' in result['tests']
        assert 'calibration' in result['tests']
        assert 'detection_latency' in result['tests']


# =============================================================================
# DOMAIN SHIFT VALIDATOR TESTS
# =============================================================================

class TestDomainShiftValidator:
    """Tests for DomainShiftValidator."""
    
    def test_split_for_recalibration(self):
        """Should split data correctly for recalibration."""
        config = DomainShiftConfig(calibration_holdout_fraction=0.3)
        validator = DomainShiftValidator(config)
        
        episodes = [EpisodeLabel(EpisodeType.VT_MONOMORPHIC, i*1000, i*1000+500) 
                   for i in range(10)]
        
        holdout, test = validator.split_for_recalibration(episodes)
        
        assert len(holdout) == 3  # 30% of 10
        assert len(test) == 7     # 70% of 10
    
    def test_compute_external_drop(self):
        """Should compute external validation drop correctly."""
        config = DomainShiftConfig()
        validator = DomainShiftValidator(config)
        
        internal = EvaluationMetrics(
            vt_sensitivity=0.95,
            svt_sensitivity=0.80,
            false_alarms_per_hour=1.0,
            ece=0.05
        )
        
        external = EvaluationMetrics(
            vt_sensitivity=0.88,
            svt_sensitivity=0.70,
            false_alarms_per_hour=1.5,
            ece=0.10
        )
        
        drops = validator.compute_external_drop(internal, external)
        
        assert abs(drops['vt_sensitivity_drop'] - 0.07) < 0.01
        assert abs(drops['svt_sensitivity_drop'] - 0.10) < 0.01
        assert abs(drops['fa_increase'] - 0.5) < 0.01
        assert abs(drops['ece_increase'] - 0.05) < 0.01
    
    def test_acceptable_drop_check(self):
        """Should check if drop is acceptable."""
        config = DomainShiftConfig()
        validator = DomainShiftValidator(config)
        
        # Acceptable drop
        drops_ok = {
            'vt_sensitivity_drop': 0.03,
            'svt_sensitivity_drop': 0.05,
            'fa_increase': 0.3,
            'ece_increase': 0.02
        }
        
        result = validator.check_acceptable_drop(drops_ok, max_sens_drop=0.05)
        assert result['passed'] is True
        
        # Unacceptable drop
        drops_bad = {
            'vt_sensitivity_drop': 0.10,  # Too much
            'svt_sensitivity_drop': 0.05,
            'fa_increase': 0.3,
            'ece_increase': 0.02
        }
        
        result = validator.check_acceptable_drop(drops_bad, max_sens_drop=0.05)
        assert result['passed'] is False


# =============================================================================
# KNOWN VT DETECTION TEST
# =============================================================================

class TestKnownVTDetection:
    """Tests for known VT detection acceptance test."""
    
    def test_all_known_detected(self):
        """Should pass when all known VT episodes detected."""
        config = AcceptanceConfig()
        tests = AcceptanceTests(config)
        
        known_vt = [
            {'record_id': "R001", 'onset_sample': 1000, 'fs': 360, 'patient_id': "P001"},
            {'record_id': "R002", 'onset_sample': 5000, 'fs': 360, 'patient_id': "P002"},
        ]
        
        detected = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
            EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6000, patient_id="P002"),
        ]
        
        result = tests.test_known_vt_detected(known_vt, detected)
        
        assert result['passed'] is True
        assert result['sensitivity'] == 1.0
    
    def test_missed_known_vt(self):
        """Should fail when known VT episodes missed."""
        config = AcceptanceConfig()
        tests = AcceptanceTests(config)
        
        known_vt = [
            {'record_id': "R001", 'onset_sample': 1000, 'fs': 360, 'patient_id': "P001"},
            {'record_id': "R002", 'onset_sample': 5000, 'fs': 360, 'patient_id': "P002"},
        ]
        
        # Only detect one
        detected = [
            EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
        ]
        
        result = tests.test_known_vt_detected(known_vt, detected)
        
        assert result['passed'] is False
        assert result['sensitivity'] == 0.5
        assert len(result['missed']) == 1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
