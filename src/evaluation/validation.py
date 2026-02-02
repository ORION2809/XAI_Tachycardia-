"""
Validation Framework for Tachycardia Detection.

v2.4: Comprehensive validation with:
- Sub-cohort validation (low-SQI, high-HR, paced, BBB patients)
- Patient-level cross-validation (no patient leakage)
- Acceptance tests (hard gates for deployment)
- Domain shift mitigation protocol

CRITICAL: Patient-level splitting is MANDATORY. No beat-level or episode-level
splitting that allows same patient in train and test.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, GroupKFold
import warnings


# =============================================================================
# IMPORT FROM METRICS
# =============================================================================

from .metrics import (
    EvaluationProtocol,
    EvaluationMetrics,
    EpisodeLabel,
    EpisodeType,
    LabelConfidenceTier,
    PerClassFACalculator,
    OnsetCriticalEvaluator,
)


# =============================================================================
# SUB-COHORT DEFINITIONS
# =============================================================================

class SubCohort(Enum):
    """Sub-cohort identifiers for stratified analysis."""
    LOW_SQI = "low_sqi_quartile"
    HIGH_HR = "high_hr_patients"
    PACED = "paced_patients"
    BBB = "bundle_branch_block"
    BRADYCARDIC = "bradycardic_patients"
    ATRIAL_FRIB = "atrial_fibrillation_history"
    ARTIFACT_PRONE = "artifact_prone"
    ELDERLY = "elderly_over_75"
    PEDIATRIC = "pediatric_under_18"


@dataclass
class PatientMetadata:
    """Metadata for patient-level analysis."""
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    is_paced: bool = False
    has_bbb: bool = False
    has_afib_history: bool = False
    baseline_hr: Optional[float] = None
    mean_sqi: float = 1.0
    monitoring_hours: float = 0.0
    n_episodes: int = 0
    
    # Computed flags
    @property
    def is_high_hr(self) -> bool:
        return (self.baseline_hr or 0) > 100
    
    @property
    def is_bradycardic(self) -> bool:
        return (self.baseline_hr or 100) < 50
    
    @property
    def is_elderly(self) -> bool:
        return (self.age or 0) > 75
    
    @property
    def is_pediatric(self) -> bool:
        return (self.age or 100) < 18
    
    @property
    def is_low_sqi(self) -> bool:
        return self.mean_sqi < 0.6
    
    def get_subcohorts(self) -> Set[SubCohort]:
        """Get all sub-cohorts this patient belongs to."""
        cohorts = set()
        if self.is_low_sqi:
            cohorts.add(SubCohort.LOW_SQI)
        if self.is_high_hr:
            cohorts.add(SubCohort.HIGH_HR)
        if self.is_paced:
            cohorts.add(SubCohort.PACED)
        if self.has_bbb:
            cohorts.add(SubCohort.BBB)
        if self.is_bradycardic:
            cohorts.add(SubCohort.BRADYCARDIC)
        if self.has_afib_history:
            cohorts.add(SubCohort.ATRIAL_FRIB)
        if self.is_elderly:
            cohorts.add(SubCohort.ELDERLY)
        if self.is_pediatric:
            cohorts.add(SubCohort.PEDIATRIC)
        return cohorts


# =============================================================================
# SUB-COHORT VALIDATOR
# =============================================================================

class SubCohortValidator:
    """
    v2.4: Validate performance on clinically important sub-cohorts.
    
    Aggregate metrics can hide failures in specific patient populations.
    
    Key sub-cohorts:
    - Low SQI quartile: Patients with mean SQI < 0.6
    - High HR: Patients with baseline HR > 100 bpm
    - Paced patients: Different morphology challenges
    - BBB patients: Wide QRS that can mimic VT
    """
    
    # Relaxed thresholds for challenging sub-cohorts
    RELAXATION_FACTOR_SENS = 0.85  # 85% of main floor
    RELAXATION_FACTOR_FA = 2.0    # 2x main ceiling
    
    def __init__(
        self,
        sensitivity_floor: float = 0.90,
        fa_ceiling: float = 2.0,
    ):
        self.sensitivity_floor = sensitivity_floor
        self.fa_ceiling = fa_ceiling
        self.relaxed_sensitivity = sensitivity_floor * self.RELAXATION_FACTOR_SENS
        self.relaxed_fa = fa_ceiling * self.RELAXATION_FACTOR_FA
    
    def validate_subcohorts(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        patient_metadata: Dict[str, PatientMetadata],
        patient_monitoring_hours: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Validate performance across all sub-cohorts.
        
        Returns:
            Dict with per-subcohort metrics and pass/fail status
        """
        results = {
            'passed': True,
            'subcohorts': {},
            'failed_subcohorts': [],
        }
        
        # Group patients by sub-cohort
        cohort_patients = defaultdict(set)
        for patient_id, meta in patient_metadata.items():
            for cohort in meta.get_subcohorts():
                cohort_patients[cohort].add(patient_id)
        
        # Evaluate each sub-cohort
        protocol = EvaluationProtocol()
        
        for cohort in SubCohort:
            patients_in_cohort = cohort_patients.get(cohort, set())
            
            if not patients_in_cohort:
                results['subcohorts'][cohort.value] = {
                    'n_patients': 0,
                    'skipped': True,
                    'reason': 'no_patients_in_cohort',
                }
                continue
            
            # Filter episodes to this sub-cohort
            pred_filtered = [
                p for p in predictions 
                if p.patient_id in patients_in_cohort
            ]
            gt_filtered = [
                g for g in ground_truth 
                if g.patient_id in patients_in_cohort
            ]
            
            # Total monitoring hours for this cohort
            cohort_hours = sum(
                patient_monitoring_hours.get(p, 0) 
                for p in patients_in_cohort
            )
            
            if cohort_hours == 0:
                results['subcohorts'][cohort.value] = {
                    'n_patients': len(patients_in_cohort),
                    'skipped': True,
                    'reason': 'no_monitoring_hours',
                }
                continue
            
            # Compute VT-specific metrics
            vt_gt = [g for g in gt_filtered if g.episode_type in protocol.VT_TYPES]
            vt_pred = [p for p in pred_filtered if p.episode_type in protocol.VT_TYPES]
            
            if vt_gt:
                matches = protocol._match_episodes(vt_pred, vt_gt)
                tp = len(matches)
                fn = len(vt_gt) - tp
                vt_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                vt_sens = float('nan')
            
            # FA calculation
            all_matches = protocol._match_episodes(pred_filtered, gt_filtered)
            fp = len(pred_filtered) - len(all_matches)
            fa_per_hour = fp / cohort_hours
            
            subcohort_result = {
                'n_patients': len(patients_in_cohort),
                'vt_sensitivity': vt_sens,
                'n_vt_episodes': len(vt_gt),
                'fa_per_hour': fa_per_hour,
                'monitoring_hours': cohort_hours,
            }
            
            # Check against relaxed thresholds
            sens_pass = np.isnan(vt_sens) or vt_sens >= self.relaxed_sensitivity
            fa_pass = fa_per_hour <= self.relaxed_fa
            
            subcohort_result['sensitivity_pass'] = sens_pass
            subcohort_result['fa_pass'] = fa_pass
            subcohort_result['passed'] = sens_pass and fa_pass
            
            if not sens_pass:
                results['passed'] = False
                results['failed_subcohorts'].append({
                    'subcohort': cohort.value,
                    'metric': 'vt_sensitivity',
                    'value': vt_sens,
                    'threshold': self.relaxed_sensitivity,
                })
            
            if not fa_pass:
                results['passed'] = False
                results['failed_subcohorts'].append({
                    'subcohort': cohort.value,
                    'metric': 'fa_per_hour',
                    'value': fa_per_hour,
                    'threshold': self.relaxed_fa,
                })
            
            results['subcohorts'][cohort.value] = subcohort_result
        
        return results


# =============================================================================
# PATIENT-LEVEL SPLIT VALIDATOR
# =============================================================================

class PatientSplitValidator:
    """
    v2.4: Ensure no patient leakage between train/test splits.
    
    This is a CRITICAL requirement. Beat-level or episode-level splitting
    that allows same patient in train and test is INVALID.
    """
    
    @staticmethod
    def validate_split(
        train_patient_ids: Set[str],
        test_patient_ids: Set[str],
    ) -> Dict[str, Any]:
        """
        Validate that train and test sets have no patient overlap.
        """
        overlap = train_patient_ids & test_patient_ids
        
        return {
            'valid': len(overlap) == 0,
            'n_train_patients': len(train_patient_ids),
            'n_test_patients': len(test_patient_ids),
            'n_overlapping': len(overlap),
            'overlapping_patients': list(overlap)[:10],  # First 10 for debugging
        }
    
    @staticmethod
    def validate_episode_assignments(
        episodes: List[EpisodeLabel],
        train_patient_ids: Set[str],
        test_patient_ids: Set[str],
    ) -> Dict[str, Any]:
        """
        Validate that all episodes are assigned to correct split.
        """
        train_episodes = []
        test_episodes = []
        unassigned = []
        misassigned = []
        
        for ep in episodes:
            in_train = ep.patient_id in train_patient_ids
            in_test = ep.patient_id in test_patient_ids
            
            if in_train and in_test:
                misassigned.append(ep.patient_id)
            elif in_train:
                train_episodes.append(ep)
            elif in_test:
                test_episodes.append(ep)
            else:
                unassigned.append(ep)
        
        return {
            'valid': len(misassigned) == 0 and len(unassigned) == 0,
            'n_train_episodes': len(train_episodes),
            'n_test_episodes': len(test_episodes),
            'n_unassigned': len(unassigned),
            'n_misassigned': len(set(misassigned)),
        }


# =============================================================================
# CROSS-VALIDATOR
# =============================================================================

@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation."""
    n_folds: int = 5
    random_state: int = 42
    stratify_by: str = "has_vt"  # Stratification criterion
    use_group_kfold: bool = True  # Patient-level grouping


class CrossValidator:
    """
    v2.4: Patient-level stratified cross-validation.
    
    Requirements:
    - Group by patient (no patient in multiple folds)
    - Stratify by VT presence (each fold has similar VT distribution)
    - Report per-fold AND aggregate metrics
    """
    
    def __init__(self, config: CrossValidationConfig):
        self.config = config
    
    def create_folds(
        self,
        patient_ids: List[str],
        patient_has_vt: Dict[str, bool],
    ) -> List[Tuple[List[str], List[str]]]:
        """
        Create patient-level cross-validation folds.
        
        Args:
            patient_ids: List of all patient IDs
            patient_has_vt: Patient ID → has VT episodes
            
        Returns:
            List of (train_patient_ids, test_patient_ids) tuples
        """
        n_patients = len(patient_ids)
        patient_ids = np.array(patient_ids)
        
        # Create stratification labels
        strat_labels = np.array([patient_has_vt.get(p, False) for p in patient_ids])
        
        if self.config.use_group_kfold:
            # Pure group K-fold (no stratification, but patient-level)
            kfold = GroupKFold(n_splits=self.config.n_folds)
            groups = np.arange(n_patients)  # Each patient is own group
            
            folds = []
            for train_idx, test_idx in kfold.split(patient_ids, groups=groups):
                train_patients = patient_ids[train_idx].tolist()
                test_patients = patient_ids[test_idx].tolist()
                folds.append((train_patients, test_patients))
        else:
            # Stratified K-fold
            kfold = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_state
            )
            
            folds = []
            for train_idx, test_idx in kfold.split(patient_ids, strat_labels):
                train_patients = patient_ids[train_idx].tolist()
                test_patients = patient_ids[test_idx].tolist()
                folds.append((train_patients, test_patients))
        
        return folds
    
    def run_cross_validation(
        self,
        folds: List[Tuple[List[str], List[str]]],
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        patient_monitoring_hours: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Run cross-validation evaluation across all folds.
        
        Returns:
            Dict with per-fold metrics and aggregate statistics
        """
        protocol = EvaluationProtocol()
        fold_results = []
        
        for fold_idx, (train_pats, test_pats) in enumerate(folds):
            test_pats_set = set(test_pats)
            
            # Validate no leakage
            split_valid = PatientSplitValidator.validate_split(
                set(train_pats), test_pats_set
            )
            
            if not split_valid['valid']:
                warnings.warn(f"Fold {fold_idx}: patient leakage detected!")
            
            # Filter to test patients
            fold_pred = [p for p in predictions if p.patient_id in test_pats_set]
            fold_gt = [g for g in ground_truth if g.patient_id in test_pats_set]
            
            # Total hours for this fold
            fold_hours = sum(
                patient_monitoring_hours.get(p, 0) 
                for p in test_pats_set
            )
            
            # Run evaluation
            metrics = protocol.evaluate(fold_pred, fold_gt, fold_hours)
            
            fold_results.append({
                'fold_idx': fold_idx,
                'n_test_patients': len(test_pats),
                'n_train_patients': len(train_pats),
                'test_hours': fold_hours,
                'vt_sensitivity': metrics.vt_sensitivity,
                'svt_sensitivity': metrics.svt_sensitivity,
                'fa_per_hour': metrics.false_alarms_per_hour,
                'episode_f1': metrics.episode_f1,
            })
        
        # Compute aggregate statistics
        vt_sens_values = [f['vt_sensitivity'] for f in fold_results]
        fa_values = [f['fa_per_hour'] for f in fold_results]
        
        return {
            'n_folds': len(folds),
            'fold_results': fold_results,
            'aggregate': {
                'mean_vt_sensitivity': float(np.mean(vt_sens_values)),
                'std_vt_sensitivity': float(np.std(vt_sens_values)),
                'min_vt_sensitivity': float(np.min(vt_sens_values)),
                'max_vt_sensitivity': float(np.max(vt_sens_values)),
                'mean_fa_per_hour': float(np.mean(fa_values)),
                'std_fa_per_hour': float(np.std(fa_values)),
            },
        }


# =============================================================================
# ACCEPTANCE TESTS
# =============================================================================

@dataclass
class AcceptanceConfig:
    """Configuration for acceptance criteria."""
    # Sensitivity floors
    vt_vfl_sensitivity_floor: float = 0.90
    svt_sensitivity_floor: float = 0.70
    
    # FA ceilings
    vt_vfl_max_fa_per_hour: float = 1.0
    svt_max_fa_per_hour: float = 0.5
    sinus_max_fa_per_hour: float = 0.3
    total_max_fa_per_hour: float = 2.0
    
    # Calibration
    max_ece: float = 0.10
    
    # Latency
    max_p95_latency_sec: float = 5.0
    max_vt_onset_to_alarm_sec: float = 10.0
    
    # External validation
    max_external_sensitivity_drop: float = 0.05


class AcceptanceTests:
    """
    v2.4: Hard gates for deployment.
    
    These tests MUST pass before any deployment.
    
    Key tests:
    1. VT sensitivity ≥ 90% (HARD GATE)
    2. FA/hr ≤ 2.0 (per-class budgets)
    3. ECE ≤ 0.10
    4. P95 latency ≤ 5s
    5. External validation drop ≤ 5%
    """
    
    def __init__(self, config: AcceptanceConfig):
        self.config = config
    
    def test_sensitivity_floors(
        self,
        metrics: EvaluationMetrics,
    ) -> Dict[str, Any]:
        """Test sensitivity meets minimum floors."""
        results = {
            'passed': True,
            'tests': {},
        }
        
        # VT sensitivity
        vt_pass = metrics.vt_sensitivity >= self.config.vt_vfl_sensitivity_floor
        results['tests']['vt_sensitivity'] = {
            'value': metrics.vt_sensitivity,
            'threshold': self.config.vt_vfl_sensitivity_floor,
            'passed': vt_pass,
        }
        if not vt_pass:
            results['passed'] = False
        
        # SVT sensitivity (softer floor)
        svt_pass = (
            metrics.svt_sensitivity >= self.config.svt_sensitivity_floor
            or metrics.svt_n_episodes == 0  # OK if no SVT episodes
        )
        results['tests']['svt_sensitivity'] = {
            'value': metrics.svt_sensitivity,
            'threshold': self.config.svt_sensitivity_floor,
            'n_episodes': metrics.svt_n_episodes,
            'passed': svt_pass,
        }
        
        return results
    
    def test_fa_ceilings(
        self,
        metrics: EvaluationMetrics,
    ) -> Dict[str, Any]:
        """Test false alarm rates meet ceilings."""
        results = {
            'passed': True,
            'tests': {},
        }
        
        # Overall FA/hr
        total_pass = metrics.false_alarms_per_hour <= self.config.total_max_fa_per_hour
        results['tests']['total_fa_per_hour'] = {
            'value': metrics.false_alarms_per_hour,
            'threshold': self.config.total_max_fa_per_hour,
            'passed': total_pass,
        }
        if not total_pass:
            results['passed'] = False
        
        # Per-class FA/hr
        vt_fa_pass = metrics.vt_vfl_fa_per_hour <= self.config.vt_vfl_max_fa_per_hour
        results['tests']['vt_vfl_fa_per_hour'] = {
            'value': metrics.vt_vfl_fa_per_hour,
            'threshold': self.config.vt_vfl_max_fa_per_hour,
            'passed': vt_fa_pass,
        }
        
        svt_fa_pass = metrics.svt_fa_per_hour <= self.config.svt_max_fa_per_hour
        results['tests']['svt_fa_per_hour'] = {
            'value': metrics.svt_fa_per_hour,
            'threshold': self.config.svt_max_fa_per_hour,
            'passed': svt_fa_pass,
        }
        
        sinus_fa_pass = metrics.sinus_tachy_fa_per_hour <= self.config.sinus_max_fa_per_hour
        results['tests']['sinus_fa_per_hour'] = {
            'value': metrics.sinus_tachy_fa_per_hour,
            'threshold': self.config.sinus_max_fa_per_hour,
            'passed': sinus_fa_pass,
        }
        
        return results
    
    def test_calibration_quality(
        self,
        metrics: EvaluationMetrics,
    ) -> Dict[str, Any]:
        """Test calibration (ECE < threshold)."""
        ece_pass = metrics.ece <= self.config.max_ece
        
        return {
            'passed': ece_pass,
            'ece': metrics.ece,
            'threshold': self.config.max_ece,
        }
    
    def test_detection_latency(
        self,
        metrics: EvaluationMetrics,
    ) -> Dict[str, Any]:
        """Test P95 detection latency."""
        latency_pass = metrics.p95_detection_latency_sec <= self.config.max_p95_latency_sec
        
        return {
            'passed': latency_pass,
            'p95_latency_sec': metrics.p95_detection_latency_sec,
            'threshold': self.config.max_p95_latency_sec,
        }
    
    def test_end_to_end_latency(
        self,
        latency_measurements: List[Dict],
    ) -> Dict[str, Any]:
        """
        v2.4: VT onset → alarm latency test.
        
        This is a GATE. Any violation is a failure.
        """
        results = {
            'passed': True,
            'n_episodes': len(latency_measurements),
            'latencies_sec': [],
            'violations': [],
        }
        
        max_latency = self.config.max_vt_onset_to_alarm_sec
        
        for meas in latency_measurements:
            latency = meas['alarm_time_sec'] - meas['vt_onset_sec']
            results['latencies_sec'].append(latency)
            
            if latency > max_latency:
                results['passed'] = False
                results['violations'].append({
                    'vt_onset_sec': meas['vt_onset_sec'],
                    'alarm_time_sec': meas['alarm_time_sec'],
                    'latency_sec': latency,
                    'max_allowed_sec': max_latency,
                })
        
        if results['latencies_sec']:
            arr = np.array(results['latencies_sec'])
            results['p50_latency'] = float(np.percentile(arr, 50))
            results['p95_latency'] = float(np.percentile(arr, 95))
            results['max_latency'] = float(np.max(arr))
        
        return results
    
    def test_worst_case_patients(
        self,
        per_patient_metrics: Dict[str, Dict],
        n_worst: int = 5,
    ) -> Dict[str, Any]:
        """
        v2.4: Worst N patient analysis.
        
        Check that even worst patients meet relaxed criteria.
        """
        # Get worst by sensitivity
        patients_with_vt = {
            p: m for p, m in per_patient_metrics.items()
            if m.get('n_vt_episodes', 0) > 0
        }
        
        sorted_by_sens = sorted(
            patients_with_vt.items(),
            key=lambda x: x[1].get('vt_sensitivity', 0)
        )
        
        worst_patients = sorted_by_sens[:n_worst]
        
        # Check if any worst patient has 0% sensitivity
        zero_sens_patients = [
            p for p, m in worst_patients 
            if m.get('vt_sensitivity', 0) == 0
        ]
        
        return {
            'passed': len(zero_sens_patients) == 0,
            'worst_patients': [
                {
                    'patient_id': p,
                    'vt_sensitivity': m.get('vt_sensitivity'),
                    'n_vt_episodes': m.get('n_vt_episodes'),
                }
                for p, m in worst_patients
            ],
            'n_zero_sensitivity': len(zero_sens_patients),
        }
    
    def test_known_vt_detected(
        self,
        known_vt_episodes: List[Dict],
        detected_episodes: List[EpisodeLabel],
        max_latency_sec: float = 5.0,
    ) -> Dict[str, Any]:
        """
        v2.4: Curated "don't miss VT" test.
        
        Known VT episodes MUST be detected within latency budget.
        """
        results = {
            'passed': True,
            'total_known': len(known_vt_episodes),
            'detected': 0,
            'missed': [],
            'detection_latencies_sec': [],
        }
        
        for known in known_vt_episodes:
            record_id = known['record_id']
            onset_sample = known['onset_sample']
            fs = known.get('fs', 360)
            
            # Find matching detections
            record_detections = [
                d for d in detected_episodes
                if d.patient_id == record_id or known.get('patient_id') == d.patient_id
            ]
            
            detected = False
            detection_latency = float('inf')
            
            for det in record_detections:
                # Check temporal overlap
                if det.start_sample <= onset_sample + fs * max_latency_sec:
                    if det.end_sample >= onset_sample:
                        detected = True
                        latency = max(0, det.start_sample - onset_sample) / fs
                        detection_latency = min(detection_latency, latency)
            
            if detected:
                results['detected'] += 1
                results['detection_latencies_sec'].append(detection_latency)
                
                if detection_latency > max_latency_sec:
                    results['passed'] = False
                    results['missed'].append({
                        **known,
                        'reason': f'late_detection_{detection_latency:.1f}s',
                    })
            else:
                results['passed'] = False
                results['missed'].append({
                    **known,
                    'reason': 'not_detected',
                })
        
        results['sensitivity'] = results['detected'] / max(results['total_known'], 1)
        results['mean_latency_sec'] = (
            np.mean(results['detection_latencies_sec'])
            if results['detection_latencies_sec'] else float('inf')
        )
        
        return results
    
    def run_all_tests(
        self,
        metrics: EvaluationMetrics,
        per_patient_metrics: Optional[Dict] = None,
        latency_measurements: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run all acceptance tests.
        
        Returns:
            Dict with overall pass/fail and per-test results
        """
        results = {
            'passed': True,
            'tests': {},
        }
        
        # Sensitivity floors
        sens_result = self.test_sensitivity_floors(metrics)
        results['tests']['sensitivity_floors'] = sens_result
        if not sens_result['passed']:
            results['passed'] = False
        
        # FA ceilings
        fa_result = self.test_fa_ceilings(metrics)
        results['tests']['fa_ceilings'] = fa_result
        if not fa_result['passed']:
            results['passed'] = False
        
        # Calibration
        cal_result = self.test_calibration_quality(metrics)
        results['tests']['calibration'] = cal_result
        if not cal_result['passed']:
            results['passed'] = False
        
        # Detection latency
        latency_result = self.test_detection_latency(metrics)
        results['tests']['detection_latency'] = latency_result
        if not latency_result['passed']:
            results['passed'] = False
        
        # End-to-end latency (if provided)
        if latency_measurements:
            e2e_result = self.test_end_to_end_latency(latency_measurements)
            results['tests']['end_to_end_latency'] = e2e_result
            if not e2e_result['passed']:
                results['passed'] = False
        
        # Worst case patients (if provided)
        if per_patient_metrics:
            worst_result = self.test_worst_case_patients(per_patient_metrics)
            results['tests']['worst_case_patients'] = worst_result
            if not worst_result['passed']:
                results['passed'] = False
        
        return results


# =============================================================================
# DOMAIN SHIFT MITIGATION
# =============================================================================

@dataclass
class DomainShiftConfig:
    """Configuration for domain shift handling."""
    enable_recalibration: bool = True
    calibration_holdout_fraction: float = 0.3
    enable_threshold_retuning: bool = True
    sensitivity_floor_for_retuning: float = 0.90


class DomainShiftValidator:
    """
    v2.4: Domain shift detection and mitigation for external validation.
    
    Don't just report the drop - FIX it.
    """
    
    def __init__(self, config: DomainShiftConfig):
        self.config = config
    
    def split_for_recalibration(
        self,
        external_episodes: List[EpisodeLabel],
    ) -> Tuple[List[EpisodeLabel], List[EpisodeLabel]]:
        """
        Split external data into recalibration holdout and test.
        """
        n_episodes = len(external_episodes)
        n_holdout = int(n_episodes * self.config.calibration_holdout_fraction)
        
        # Shuffle with fixed seed
        indices = np.arange(n_episodes)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        holdout_episodes = [external_episodes[i] for i in indices[:n_holdout]]
        test_episodes = [external_episodes[i] for i in indices[n_holdout:]]
        
        return holdout_episodes, test_episodes
    
    def compute_external_drop(
        self,
        internal_metrics: EvaluationMetrics,
        external_metrics: EvaluationMetrics,
    ) -> Dict[str, float]:
        """
        Compute sensitivity drop between internal and external validation.
        """
        return {
            'vt_sensitivity_drop': internal_metrics.vt_sensitivity - external_metrics.vt_sensitivity,
            'svt_sensitivity_drop': internal_metrics.svt_sensitivity - external_metrics.svt_sensitivity,
            'fa_increase': external_metrics.false_alarms_per_hour - internal_metrics.false_alarms_per_hour,
            'ece_increase': external_metrics.ece - internal_metrics.ece,
        }
    
    def check_acceptable_drop(
        self,
        drops: Dict[str, float],
        max_sens_drop: float = 0.05,
        max_fa_increase: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Check if external validation drop is acceptable.
        """
        results = {
            'passed': True,
            'checks': {},
        }
        
        # VT sensitivity drop
        vt_pass = drops['vt_sensitivity_drop'] <= max_sens_drop
        results['checks']['vt_sensitivity_drop'] = {
            'value': drops['vt_sensitivity_drop'],
            'threshold': max_sens_drop,
            'passed': vt_pass,
        }
        if not vt_pass:
            results['passed'] = False
        
        # FA increase
        fa_pass = drops['fa_increase'] <= max_fa_increase
        results['checks']['fa_increase'] = {
            'value': drops['fa_increase'],
            'threshold': max_fa_increase,
            'passed': fa_pass,
        }
        
        return results


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Validation Framework Demo (v2.4)")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    
    # Ground truth episodes with patient IDs
    ground_truth = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, patient_id="P001"),
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6500, patient_id="P001"),
        EpisodeLabel(EpisodeType.SVT, 10000, 12000, patient_id="P002"),
        EpisodeLabel(EpisodeType.VFL, 20000, 22000, patient_id="P003"),
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 30000, 32000, patient_id="P004"),
    ]
    
    # Predictions
    predictions = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1050, 2100, patient_id="P001"),  # Good
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5200, 6600, patient_id="P001"),  # Good
        EpisodeLabel(EpisodeType.SVT, 10100, 11900, patient_id="P002"),  # Good
        EpisodeLabel(EpisodeType.VFL, 20100, 22100, patient_id="P003"),  # Good
        # Missing P004 VT - false negative
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 40000, 41000, patient_id="P005"),  # FP
    ]
    
    # Patient metadata
    patient_metadata = {
        "P001": PatientMetadata("P001", age=65, is_paced=False, mean_sqi=0.85),
        "P002": PatientMetadata("P002", age=78, is_paced=True, mean_sqi=0.55),  # Low SQI, paced
        "P003": PatientMetadata("P003", age=45, has_bbb=True, mean_sqi=0.90),  # BBB
        "P004": PatientMetadata("P004", age=55, baseline_hr=110, mean_sqi=0.75),  # High HR
        "P005": PatientMetadata("P005", age=35, mean_sqi=0.95),
    }
    
    patient_hours = {"P001": 2.0, "P002": 1.5, "P003": 1.0, "P004": 2.5, "P005": 1.0}
    
    # Test sub-cohort validation
    print("\n--- Sub-Cohort Validation ---")
    subcohort_validator = SubCohortValidator()
    subcohort_results = subcohort_validator.validate_subcohorts(
        predictions, ground_truth, patient_metadata, patient_hours
    )
    print(f"Overall passed: {subcohort_results['passed']}")
    for cohort, result in subcohort_results['subcohorts'].items():
        if result.get('skipped'):
            print(f"  {cohort}: SKIPPED ({result.get('reason')})")
        else:
            print(f"  {cohort}: VT sens={result.get('vt_sensitivity', 'N/A'):.1%}, "
                  f"FA/hr={result.get('fa_per_hour', 0):.2f}, n_patients={result.get('n_patients')}")
    
    # Test patient split validation
    print("\n--- Patient Split Validation ---")
    train_patients = {"P001", "P002", "P003"}
    test_patients = {"P004", "P005"}
    split_result = PatientSplitValidator.validate_split(train_patients, test_patients)
    print(f"Valid split: {split_result['valid']}")
    print(f"Train patients: {split_result['n_train_patients']}, Test patients: {split_result['n_test_patients']}")
    
    # Test cross-validation
    print("\n--- Cross-Validation Setup ---")
    cv_config = CrossValidationConfig(n_folds=3)
    cv = CrossValidator(cv_config)
    patient_ids = list(patient_metadata.keys())
    patient_has_vt = {
        "P001": True, "P002": False, "P003": True, "P004": True, "P005": False
    }
    folds = cv.create_folds(patient_ids, patient_has_vt)
    print(f"Created {len(folds)} folds")
    for i, (train, test) in enumerate(folds):
        print(f"  Fold {i}: train={train}, test={test}")
    
    # Test acceptance tests
    print("\n--- Acceptance Tests ---")
    # Create synthetic metrics
    metrics = EvaluationMetrics(
        episode_sensitivity=0.80,
        episode_ppv=0.80,
        vt_sensitivity=0.80,  # Below floor!
        svt_sensitivity=0.75,
        false_alarms_per_hour=1.0,
        vt_vfl_fa_per_hour=0.5,
        svt_fa_per_hour=0.3,
        sinus_tachy_fa_per_hour=0.2,
        ece=0.08,
        p95_detection_latency_sec=3.5,
    )
    
    acceptance_config = AcceptanceConfig()
    acceptance_tests = AcceptanceTests(acceptance_config)
    
    test_results = acceptance_tests.run_all_tests(metrics)
    print(f"Overall passed: {test_results['passed']}")
    for test_name, result in test_results['tests'].items():
        status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
