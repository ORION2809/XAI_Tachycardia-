"""
Evaluation Metrics for Tachycardia Detection.

v2.4: Comprehensive metrics with:
- Sensitivity-first evaluation (VT sens ≥90% as hard gate)
- Per-class FA/hr tracking (VT/VFL, SVT, sinus_tachy separately)
- Onset-critical metrics (onset error, time-to-detection, time-to-alarm)
- Confidence-tier stratified metrics
- Per-patient metrics for worst-case analysis

CRITICAL: Episode-level metrics are PRIMARY. Beat-level metrics are SECONDARY.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


# =============================================================================
# EPISODE TYPE AND LABEL STRUCTURES
# =============================================================================

class EpisodeType(Enum):
    """Episode classification types."""
    NORMAL = "normal"
    SINUS_TACHY = "sinus_tachycardia"
    SVT = "svt"
    VT_MONOMORPHIC = "vt_monomorphic"
    VT_POLYMORPHIC = "vt_polymorphic"
    VFL = "vfl"
    VFIB = "vfib"
    AFIB_RVR = "afib_rvr"
    AFLUTTER = "aflutter"


class LabelConfidenceTier(Enum):
    """Truth hierarchy for labels - higher = more trustworthy."""
    EXPERT_RHYTHM = "expert_rhythm"
    DERIVED_RHYTHM = "derived_rhythm"
    HEURISTIC = "heuristic"
    UNCERTAIN = "uncertain"


@dataclass
class EpisodeLabel:
    """Episode label for evaluation."""
    episode_type: EpisodeType
    start_sample: int
    end_sample: int
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    confidence: float = 1.0
    label_tier: LabelConfidenceTier = LabelConfidenceTier.EXPERT_RHYTHM
    patient_id: str = ""
    dataset: str = ""
    
    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample
    
    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


# =============================================================================
# MAIN EVALUATION METRICS DATACLASS
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Complete metrics suite for tachycardia detection."""
    
    # Episode-level metrics (PRIMARY)
    episode_sensitivity: float = 0.0
    episode_ppv: float = 0.0
    episode_f1: float = 0.0
    false_alarms_per_hour: float = 0.0
    
    # Per-class episode metrics
    vt_sensitivity: float = 0.0
    vt_ppv: float = 0.0
    vt_n_episodes: int = 0
    svt_sensitivity: float = 0.0
    svt_ppv: float = 0.0
    svt_n_episodes: int = 0
    
    # v2.4: Per-class FA/hr
    vt_vfl_fa_per_hour: float = 0.0
    svt_fa_per_hour: float = 0.0
    sinus_tachy_fa_per_hour: float = 0.0
    
    # Timing metrics (onset-critical)
    mean_detection_latency_sec: float = 0.0
    median_detection_latency_sec: float = 0.0
    p95_detection_latency_sec: float = 0.0
    mean_onset_error_ms: float = 0.0
    p95_onset_error_ms: float = 0.0
    onset_accuracy: float = 0.0
    
    # v2.4: Separate latencies
    time_to_first_detection_sec: float = 0.0
    time_to_warning_sec: float = 0.0
    time_to_alarm_sec: float = 0.0
    
    # Beat-level (secondary, for comparison)
    beat_sensitivity: float = 0.0
    beat_specificity: float = 0.0
    beat_f1: float = 0.0
    
    # Calibration
    ece: float = 0.0
    brier_score: float = 0.0
    
    # XAI quality
    xai_stability_score: float = 0.0
    xai_alignment_score: float = 0.0
    
    # v2.4: Confidence-stratified metrics
    expert_tier_vt_sensitivity: float = 0.0
    derived_tier_vt_sensitivity: float = 0.0
    heuristic_tier_vt_sensitivity: float = 0.0
    confidence_weighted_sensitivity: float = 0.0
    
    # Monitoring context
    total_duration_hours: float = 0.0
    n_patients: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_sensitivity': self.episode_sensitivity,
            'episode_ppv': self.episode_ppv,
            'episode_f1': self.episode_f1,
            'false_alarms_per_hour': self.false_alarms_per_hour,
            'vt_sensitivity': self.vt_sensitivity,
            'vt_ppv': self.vt_ppv,
            'svt_sensitivity': self.svt_sensitivity,
            'svt_ppv': self.svt_ppv,
            'vt_vfl_fa_per_hour': self.vt_vfl_fa_per_hour,
            'svt_fa_per_hour': self.svt_fa_per_hour,
            'mean_detection_latency_sec': self.mean_detection_latency_sec,
            'p95_detection_latency_sec': self.p95_detection_latency_sec,
            'ece': self.ece,
            'expert_tier_vt_sensitivity': self.expert_tier_vt_sensitivity,
            'derived_tier_vt_sensitivity': self.derived_tier_vt_sensitivity,
        }
    
    def passes_sensitivity_floor(self, min_vt_sens: float = 0.90) -> bool:
        """Check if VT sensitivity meets minimum threshold."""
        return self.vt_sensitivity >= min_vt_sens
    
    def passes_fa_ceiling(self, max_fa_hr: float = 2.0) -> bool:
        """Check if FA/hr meets maximum threshold."""
        return self.false_alarms_per_hour <= max_fa_hr


# =============================================================================
# EVALUATION PROTOCOL
# =============================================================================

class EvaluationProtocol:
    """
    Exact evaluation rules for tachycardia detection.
    
    Key principles:
    - Episode-level matching with IoU threshold
    - Sensitivity-first model selection
    - FA/hr normalization per monitoring hour
    """
    
    # Episode matching rules
    EPISODE_OVERLAP_THRESHOLD: float = 0.5
    DETECTION_LATENCY_TOLERANCE_SEC: float = 5.0
    
    # Episode type groups
    VT_TYPES = {EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                EpisodeType.VFL, EpisodeType.VFIB}
    SVT_TYPES = {EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER}
    
    def __init__(
        self,
        overlap_threshold: float = 0.5,
        fs: int = 360,
    ):
        self.overlap_threshold = overlap_threshold
        self.fs = fs
    
    def evaluate(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        total_duration_hours: float,
    ) -> EvaluationMetrics:
        """
        Run complete evaluation.
        
        Args:
            predictions: Predicted episodes
            ground_truth: Ground truth episodes
            total_duration_hours: Total monitoring duration
            
        Returns:
            EvaluationMetrics with all computed metrics
        """
        # Match predictions to ground truth
        matches = self._match_episodes(predictions, ground_truth)
        
        # Compute overall episode metrics
        tp = len(matches)
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * sensitivity * ppv / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0
        fa_per_hour = fp / total_duration_hours if total_duration_hours > 0 else 0
        
        # Detection latencies
        latencies = [m["latency"] for m in matches if m["latency"] is not None and m["latency"] >= 0]
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        median_latency = float(np.median(latencies)) if latencies else 0.0
        p95_latency = float(np.percentile(latencies, 95)) if len(latencies) >= 2 else mean_latency
        
        # Per-class metrics
        vt_metrics = self._compute_class_metrics(predictions, ground_truth, self.VT_TYPES)
        svt_metrics = self._compute_class_metrics(predictions, ground_truth, self.SVT_TYPES)
        
        # Per-class FA/hr
        fa_calc = PerClassFACalculator()
        per_class_fa = fa_calc.compute(predictions, ground_truth, total_duration_hours)
        
        return EvaluationMetrics(
            episode_sensitivity=sensitivity,
            episode_ppv=ppv,
            episode_f1=f1,
            false_alarms_per_hour=fa_per_hour,
            vt_sensitivity=vt_metrics["sensitivity"],
            vt_ppv=vt_metrics["ppv"],
            vt_n_episodes=vt_metrics["n_ground_truth"],
            svt_sensitivity=svt_metrics["sensitivity"],
            svt_ppv=svt_metrics["ppv"],
            svt_n_episodes=svt_metrics["n_ground_truth"],
            vt_vfl_fa_per_hour=per_class_fa["vt_vfl"],
            svt_fa_per_hour=per_class_fa["svt"],
            sinus_tachy_fa_per_hour=per_class_fa["sinus_tachy"],
            mean_detection_latency_sec=mean_latency,
            median_detection_latency_sec=median_latency,
            p95_detection_latency_sec=p95_latency,
            total_duration_hours=total_duration_hours,
        )
    
    def _match_episodes(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
    ) -> List[Dict]:
        """Match predicted episodes to ground truth."""
        matches = []
        matched_gt = set()
        
        for pred in predictions:
            best_iou = 0
            best_gt_idx = None
            best_latency = None
            
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                # Check type compatibility
                if not self._types_compatible(pred.episode_type, gt.episode_type):
                    continue
                
                # Compute IoU
                iou = self._compute_iou(pred, gt)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
                    best_latency = pred.start_time_sec - gt.start_time_sec
            
            if best_iou >= self.overlap_threshold:
                matches.append({
                    "prediction": pred,
                    "ground_truth": ground_truth[best_gt_idx],
                    "iou": best_iou,
                    "latency": best_latency,
                })
                matched_gt.add(best_gt_idx)
        
        return matches
    
    def _compute_iou(self, ep1: EpisodeLabel, ep2: EpisodeLabel) -> float:
        """Compute temporal IoU between episodes."""
        start = max(ep1.start_sample, ep2.start_sample)
        end = min(ep1.end_sample, ep2.end_sample)
        
        intersection = max(0, end - start)
        union = (ep1.duration_samples + ep2.duration_samples - intersection)
        
        return intersection / union if union > 0 else 0
    
    def _types_compatible(self, t1: EpisodeType, t2: EpisodeType) -> bool:
        """Check if episode types are compatible for matching."""
        if t1 in self.VT_TYPES and t2 in self.VT_TYPES:
            return True
        if t1 in self.SVT_TYPES and t2 in self.SVT_TYPES:
            return True
        return t1 == t2
    
    def _compute_class_metrics(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        target_types: Set[EpisodeType],
    ) -> Dict[str, float]:
        """Compute metrics for specific episode types."""
        pred_filtered = [p for p in predictions if p.episode_type in target_types]
        gt_filtered = [g for g in ground_truth if g.episode_type in target_types]
        
        if not gt_filtered:
            return {"sensitivity": 0.0, "ppv": 0.0, "n_ground_truth": 0, "n_predictions": 0}
        
        matches = self._match_episodes(pred_filtered, gt_filtered)
        
        tp = len(matches)
        fp = len(pred_filtered) - tp
        fn = len(gt_filtered) - tp
        
        return {
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "ppv": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "n_ground_truth": len(gt_filtered),
            "n_predictions": len(pred_filtered),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }


# =============================================================================
# PER-CLASS FA CALCULATOR
# =============================================================================

class PerClassFACalculator:
    """
    v2.4: Per-class false alarm rate calculation.
    
    Prevents one class from consuming the entire FA budget.
    """
    
    VT_TYPES = {EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                EpisodeType.VFL, EpisodeType.VFIB}
    SVT_TYPES = {EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER}
    SINUS_TYPES = {EpisodeType.SINUS_TACHY}
    
    def compute(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        total_duration_hours: float,
    ) -> Dict[str, float]:
        """
        Compute per-class FA rates.
        
        Returns:
            Dict with vt_vfl, svt, sinus_tachy FA/hr
        """
        # Get matched predictions
        protocol = EvaluationProtocol()
        matches = protocol._match_episodes(predictions, ground_truth)
        matched_pred_ids = {id(m["prediction"]) for m in matches}
        
        # False positives by class
        vt_fp = 0
        svt_fp = 0
        sinus_fp = 0
        other_fp = 0
        
        for pred in predictions:
            if id(pred) not in matched_pred_ids:
                # This is a false positive
                if pred.episode_type in self.VT_TYPES:
                    vt_fp += 1
                elif pred.episode_type in self.SVT_TYPES:
                    svt_fp += 1
                elif pred.episode_type in self.SINUS_TYPES:
                    sinus_fp += 1
                else:
                    other_fp += 1
        
        hours = max(total_duration_hours, 0.001)  # Avoid div by zero
        
        return {
            "vt_vfl": vt_fp / hours,
            "svt": svt_fp / hours,
            "sinus_tachy": sinus_fp / hours,
            "other": other_fp / hours,
            "total": (vt_fp + svt_fp + sinus_fp + other_fp) / hours,
        }


# =============================================================================
# ONSET-CRITICAL EVALUATOR
# =============================================================================

class OnsetCriticalEvaluator:
    """
    v2.4: Onset-critical episode matching metrics.
    
    IoU is fine for overlap, but clinicians care about:
    1. Onset error distribution (how accurate is onset timing?)
    2. Time-to-first-detection (first alert after true onset)
    3. Time-to-alarm (after confirmation gates)
    """
    
    def __init__(
        self,
        fs: int = 360,
        max_onset_error_ms: float = 500,
    ):
        self.fs = fs
        self.max_onset_error_ms = max_onset_error_ms
    
    def evaluate_onset_accuracy(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
    ) -> Dict[str, Any]:
        """
        Evaluate onset timing accuracy.
        
        Returns:
            Dict with onset error statistics
        """
        onset_errors = []
        
        # Match episodes
        matched_gt = set()
        
        for pred in predictions:
            best_match_idx = None
            best_overlap = 0
            
            for i, gt in enumerate(ground_truth):
                if i in matched_gt:
                    continue
                
                overlap = self._compute_overlap(pred, gt)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match_idx = i
            
            if best_match_idx is not None and best_overlap > 0:
                gt = ground_truth[best_match_idx]
                matched_gt.add(best_match_idx)
                
                # Onset error: pred_start - gt_start (in ms)
                error_samples = pred.start_sample - gt.start_sample
                error_ms = error_samples / self.fs * 1000
                onset_errors.append(error_ms)
        
        if not onset_errors:
            return {
                'onset_errors_ms': [],
                'mean_onset_error_ms': float('nan'),
                'median_onset_error_ms': float('nan'),
                'p95_onset_error_ms': float('nan'),
                'onset_accuracy': 0.0,
                'n_matched': 0,
            }
        
        errors_arr = np.array(onset_errors)
        within_tolerance = np.abs(errors_arr) <= self.max_onset_error_ms
        
        return {
            'onset_errors_ms': onset_errors,
            'mean_onset_error_ms': float(np.mean(errors_arr)),
            'median_onset_error_ms': float(np.median(errors_arr)),
            'std_onset_error_ms': float(np.std(errors_arr)),
            'p95_onset_error_ms': float(np.percentile(np.abs(errors_arr), 95)),
            'onset_accuracy': float(np.mean(within_tolerance)),
            'n_matched': len(onset_errors),
            'mean_signed_error_ms': float(np.mean(errors_arr)),
            'late_detection_fraction': float(np.mean(errors_arr > 0)),
        }
    
    def evaluate_detection_latencies(
        self,
        detection_events: List[Dict],
    ) -> Dict[str, Any]:
        """
        Evaluate detection and alarm latencies separately.
        
        Args:
            detection_events: List of {
                'gt_onset_sec': float,
                'first_detection_sec': float,
                'warning_sec': float,
                'alarm_sec': float,
            }
        """
        first_detection_latencies = []
        warning_latencies = []
        alarm_latencies = []
        
        for event in detection_events:
            gt_onset = event['gt_onset_sec']
            
            if event.get('first_detection_sec') is not None:
                latency = event['first_detection_sec'] - gt_onset
                if latency >= 0:
                    first_detection_latencies.append(latency)
            
            if event.get('warning_sec') is not None:
                latency = event['warning_sec'] - gt_onset
                if latency >= 0:
                    warning_latencies.append(latency)
            
            if event.get('alarm_sec') is not None:
                latency = event['alarm_sec'] - gt_onset
                if latency >= 0:
                    alarm_latencies.append(latency)
        
        def compute_stats(latencies):
            if not latencies:
                return {'mean_sec': float('nan'), 'median_sec': float('nan'), 
                        'p95_sec': float('nan'), 'n_events': 0}
            return {
                'mean_sec': float(np.mean(latencies)),
                'median_sec': float(np.median(latencies)),
                'p95_sec': float(np.percentile(latencies, 95)) if len(latencies) >= 2 else float(np.mean(latencies)),
                'n_events': len(latencies),
            }
        
        return {
            'first_detection': compute_stats(first_detection_latencies),
            'warning': compute_stats(warning_latencies),
            'alarm': compute_stats(alarm_latencies),
            'confirmation_overhead_sec': (
                float(np.mean(alarm_latencies)) - float(np.mean(first_detection_latencies))
                if alarm_latencies and first_detection_latencies else float('nan')
            ),
        }
    
    def _compute_overlap(self, ep1: EpisodeLabel, ep2: EpisodeLabel) -> int:
        """Compute overlap in samples."""
        start = max(ep1.start_sample, ep2.start_sample)
        end = min(ep1.end_sample, ep2.end_sample)
        return max(0, end - start)
    
    def generate_latency_report(
        self,
        onset_metrics: Dict,
        latency_metrics: Dict,
    ) -> str:
        """Generate human-readable latency report."""
        report = []
        report.append("# Onset-Critical Evaluation Report (v2.4)")
        
        report.append("\n## Onset Accuracy")
        report.append(f"- Mean onset error: {onset_metrics['mean_onset_error_ms']:.0f} ms")
        report.append(f"- Median onset error: {onset_metrics['median_onset_error_ms']:.0f} ms")
        report.append(f"- P95 onset error: {onset_metrics['p95_onset_error_ms']:.0f} ms")
        report.append(f"- Within {self.max_onset_error_ms}ms tolerance: {onset_metrics['onset_accuracy']:.1%}")
        report.append(f"- Late detection fraction: {onset_metrics.get('late_detection_fraction', 0):.1%}")
        
        report.append("\n## Detection Latencies")
        fd = latency_metrics['first_detection']
        report.append(f"- First Detection:")
        report.append(f"  - Mean: {fd['mean_sec']:.2f}s, P95: {fd['p95_sec']:.2f}s")
        
        warn = latency_metrics['warning']
        report.append(f"- Warning Issued:")
        report.append(f"  - Mean: {warn['mean_sec']:.2f}s, P95: {warn['p95_sec']:.2f}s")
        
        alarm = latency_metrics['alarm']
        report.append(f"- Alarm Fired:")
        report.append(f"  - Mean: {alarm['mean_sec']:.2f}s, P95: {alarm['p95_sec']:.2f}s")
        
        overhead = latency_metrics['confirmation_overhead_sec']
        report.append(f"\n- Confirmation overhead: {overhead:.2f}s")
        
        return "\n".join(report)


# =============================================================================
# CONFIDENCE-AWARE EVALUATOR
# =============================================================================

class ConfidenceAwareEvaluator:
    """
    v2.4: Evaluator that respects truth hierarchy.
    
    Different label sources have different reliability.
    Metrics MUST be stratified by confidence tier.
    """
    
    # Tier weights for weighted average
    TIER_WEIGHTS = {
        LabelConfidenceTier.EXPERT_RHYTHM: 1.0,
        LabelConfidenceTier.DERIVED_RHYTHM: 0.8,
        LabelConfidenceTier.HEURISTIC: 0.5,
        LabelConfidenceTier.UNCERTAIN: 0.0,
    }
    
    def evaluate_with_confidence(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        total_duration_hours: float,
    ) -> Dict[str, Any]:
        """
        Evaluate with explicit label confidence stratification.
        
        Returns:
            Dict with per-tier metrics and weighted aggregate
        """
        # Stratify ground truth by tier
        gt_by_tier = defaultdict(list)
        for gt in ground_truth:
            gt_by_tier[gt.label_tier].append(gt)
        
        # Compute metrics per tier
        protocol = EvaluationProtocol()
        tier_metrics = {}
        
        for tier in LabelConfidenceTier:
            tier_gt = gt_by_tier.get(tier, [])
            if tier_gt:
                # Filter predictions to VT types for VT metrics
                vt_pred = [p for p in predictions if p.episode_type in protocol.VT_TYPES]
                vt_gt = [g for g in tier_gt if g.episode_type in protocol.VT_TYPES]
                
                if vt_gt:
                    matches = protocol._match_episodes(vt_pred, vt_gt)
                    tp = len(matches)
                    fn = len(vt_gt) - tp
                    
                    tier_metrics[tier] = {
                        "vt_sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
                        "n_vt_episodes": len(vt_gt),
                        "n_matched": tp,
                    }
                else:
                    tier_metrics[tier] = {"vt_sensitivity": float('nan'), "n_vt_episodes": 0}
            else:
                tier_metrics[tier] = {"vt_sensitivity": float('nan'), "n_vt_episodes": 0}
        
        # Compute weighted aggregate
        weighted_sens_num = 0
        weighted_sens_den = 0
        
        for tier, metrics in tier_metrics.items():
            n_ep = metrics.get("n_vt_episodes", 0)
            sens = metrics.get("vt_sensitivity", 0)
            
            if n_ep > 0 and not np.isnan(sens):
                weight = self.TIER_WEIGHTS.get(tier, 0)
                weighted_sens_num += sens * n_ep * weight
                weighted_sens_den += n_ep * weight
        
        weighted_sensitivity = weighted_sens_num / weighted_sens_den if weighted_sens_den > 0 else 0
        
        return {
            'expert_tier': tier_metrics.get(LabelConfidenceTier.EXPERT_RHYTHM, {}),
            'derived_tier': tier_metrics.get(LabelConfidenceTier.DERIVED_RHYTHM, {}),
            'heuristic_tier': tier_metrics.get(LabelConfidenceTier.HEURISTIC, {}),
            'uncertain_tier': tier_metrics.get(LabelConfidenceTier.UNCERTAIN, {}),
            'weighted_vt_sensitivity': weighted_sensitivity,
        }


# =============================================================================
# PER-PATIENT METRICS
# =============================================================================

class PerPatientEvaluator:
    """
    v2.4: Per-patient metrics for worst-case analysis.
    
    Aggregate metrics can hide complete failures on individual patients.
    """
    
    def compute_per_patient_metrics(
        self,
        predictions: List[EpisodeLabel],
        ground_truth: List[EpisodeLabel],
        patient_monitoring_hours: Dict[str, float],
    ) -> Dict[str, Dict]:
        """
        Compute metrics for each patient.
        
        Args:
            predictions: All predictions
            ground_truth: All ground truth
            patient_monitoring_hours: Patient ID → monitoring hours
            
        Returns:
            Patient ID → metrics dict
        """
        # Group by patient
        pred_by_patient = defaultdict(list)
        gt_by_patient = defaultdict(list)
        
        for pred in predictions:
            pred_by_patient[pred.patient_id].append(pred)
        for gt in ground_truth:
            gt_by_patient[gt.patient_id].append(gt)
        
        all_patients = set(pred_by_patient.keys()) | set(gt_by_patient.keys())
        
        protocol = EvaluationProtocol()
        per_patient = {}
        
        for patient_id in all_patients:
            patient_pred = pred_by_patient.get(patient_id, [])
            patient_gt = gt_by_patient.get(patient_id, [])
            hours = patient_monitoring_hours.get(patient_id, 1.0)
            
            # VT-specific metrics
            vt_gt = [g for g in patient_gt if g.episode_type in protocol.VT_TYPES]
            vt_pred = [p for p in patient_pred if p.episode_type in protocol.VT_TYPES]
            
            if vt_gt:
                matches = protocol._match_episodes(vt_pred, vt_gt)
                tp = len(matches)
                fn = len(vt_gt) - tp
                vt_sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            else:
                vt_sens = float('nan')
            
            # FA calculation
            all_matches = protocol._match_episodes(patient_pred, patient_gt)
            fp = len(patient_pred) - len(all_matches)
            fa_per_hour = fp / hours if hours > 0 else 0
            
            per_patient[patient_id] = {
                'vt_sensitivity': vt_sens,
                'n_vt_episodes': len(vt_gt),
                'fa_per_hour': fa_per_hour,
                'n_predictions': len(patient_pred),
                'n_ground_truth': len(patient_gt),
                'monitoring_hours': hours,
            }
        
        return per_patient
    
    def get_worst_patients(
        self,
        per_patient_metrics: Dict[str, Dict],
        n_worst: int = 5,
    ) -> Dict[str, List]:
        """
        Get worst N patients by sensitivity and FA.
        
        Returns:
            Dict with worst_by_sensitivity and worst_by_fa lists
        """
        # Filter to patients with VT events
        patients_with_vt = {
            p: m for p, m in per_patient_metrics.items()
            if m.get('n_vt_episodes', 0) > 0
        }
        
        # Worst by sensitivity
        sorted_by_sens = sorted(
            patients_with_vt.items(),
            key=lambda x: x[1].get('vt_sensitivity', 0)
        )
        
        worst_by_sens = [
            {
                'patient_id': p,
                'vt_sensitivity': m.get('vt_sensitivity'),
                'n_vt_episodes': m.get('n_vt_episodes'),
            }
            for p, m in sorted_by_sens[:n_worst]
        ]
        
        # Worst by FA
        sorted_by_fa = sorted(
            per_patient_metrics.items(),
            key=lambda x: x[1].get('fa_per_hour', 0),
            reverse=True
        )
        
        worst_by_fa = [
            {
                'patient_id': p,
                'fa_per_hour': m.get('fa_per_hour'),
                'monitoring_hours': m.get('monitoring_hours'),
            }
            for p, m in sorted_by_fa[:n_worst]
        ]
        
        return {
            'worst_by_sensitivity': worst_by_sens,
            'worst_by_fa': worst_by_fa,
        }


# =============================================================================
# RESULTS REPORTER
# =============================================================================

class ResultsReporter:
    """
    Generate standardized results reports.
    
    Follows the ReportingContract from spec.
    """
    
    def generate_report(
        self,
        metrics: EvaluationMetrics,
        confidence_metrics: Dict[str, Any],
        run_config: Dict[str, Any],
    ) -> str:
        """Generate markdown-formatted results report."""
        report = []
        report.append("# Evaluation Results Report")
        report.append(f"\n## Run Configuration")
        report.append(f"- Model: {run_config.get('model_name', 'Unknown')}")
        report.append(f"- Dataset: {run_config.get('dataset', 'Unknown')}")
        report.append(f"- Date: {run_config.get('date', 'Unknown')}")
        report.append(f"- Total Duration: {metrics.total_duration_hours:.1f} hours")
        
        # Headline metrics table
        report.append("\n## Headline Metrics (Expert + Derived Tiers)")
        report.append("\n| Metric | Value | Target | Status |")
        report.append("|--------|-------|--------|--------|")
        
        vt_sens = metrics.vt_sensitivity
        vt_target = 0.90
        vt_status = "✅ PASS" if vt_sens >= vt_target else "❌ FAIL"
        report.append(f"| VT Sensitivity | {vt_sens:.1%} | ≥{vt_target:.0%} | {vt_status} |")
        
        fa = metrics.false_alarms_per_hour
        fa_target = 2.0
        fa_status = "✅ PASS" if fa <= fa_target else "❌ FAIL"
        report.append(f"| FA/hour | {fa:.2f} | ≤{fa_target} | {fa_status} |")
        
        latency = metrics.p95_detection_latency_sec
        latency_target = 5.0
        latency_status = "✅ PASS" if latency <= latency_target else "❌ FAIL"
        report.append(f"| P95 Latency | {latency:.2f}s | ≤{latency_target}s | {latency_status} |")
        
        ece = metrics.ece
        ece_target = 0.10
        ece_status = "✅ PASS" if ece <= ece_target else "❌ FAIL"
        report.append(f"| ECE | {ece:.3f} | ≤{ece_target} | {ece_status} |")
        
        # Per-class FA breakdown
        report.append("\n## Per-Class False Alarm Rates")
        report.append("\n| Class | FA/hr | Target | Status |")
        report.append("|-------|-------|--------|--------|")
        report.append(f"| VT/VFL | {metrics.vt_vfl_fa_per_hour:.2f} | ≤1.0 | {'✅' if metrics.vt_vfl_fa_per_hour <= 1.0 else '❌'} |")
        report.append(f"| SVT | {metrics.svt_fa_per_hour:.2f} | ≤0.5 | {'✅' if metrics.svt_fa_per_hour <= 0.5 else '❌'} |")
        report.append(f"| Sinus Tachy | {metrics.sinus_tachy_fa_per_hour:.2f} | ≤0.3 | {'✅' if metrics.sinus_tachy_fa_per_hour <= 0.3 else '❌'} |")
        
        # Stratified metrics
        report.append("\n## Stratified Metrics by Label Tier")
        report.append("\n| Tier | VT Sensitivity | N Episodes | Notes |")
        report.append("|------|----------------|------------|-------|")
        
        expert = confidence_metrics.get('expert_tier', {})
        report.append(f"| Expert | {expert.get('vt_sensitivity', 0):.1%} | {expert.get('n_vt_episodes', 0)} | PRIMARY |")
        
        derived = confidence_metrics.get('derived_tier', {})
        report.append(f"| Derived | {derived.get('vt_sensitivity', 0):.1%} | {derived.get('n_vt_episodes', 0)} | PRIMARY |")
        
        heuristic = confidence_metrics.get('heuristic_tier', {})
        report.append(f"| Heuristic | {heuristic.get('vt_sensitivity', 0):.1%} | {heuristic.get('n_vt_episodes', 0)} | SECONDARY |")
        
        return "\n".join(report)


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Evaluation Metrics Demo (v2.4)")
    print("="*60)
    
    # Create synthetic episodes
    np.random.seed(42)
    
    # Ground truth episodes
    ground_truth = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1000, 2000, 2.78, 5.56, label_tier=LabelConfidenceTier.EXPERT_RHYTHM),
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5000, 6500, 13.89, 18.06, label_tier=LabelConfidenceTier.EXPERT_RHYTHM),
        EpisodeLabel(EpisodeType.SVT, 10000, 12000, 27.78, 33.33, label_tier=LabelConfidenceTier.DERIVED_RHYTHM),
        EpisodeLabel(EpisodeType.VFL, 20000, 22000, 55.56, 61.11, label_tier=LabelConfidenceTier.EXPERT_RHYTHM),
    ]
    
    # Predictions (some correct, some false positives)
    predictions = [
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 1050, 2100, 2.92, 5.83),  # Good match
        EpisodeLabel(EpisodeType.VT_POLYMORPHIC, 5200, 6600, 14.44, 18.33),  # Good match
        EpisodeLabel(EpisodeType.SVT, 10100, 11900, 28.06, 33.06),  # Good match
        # Missing VFL - false negative
        EpisodeLabel(EpisodeType.VT_MONOMORPHIC, 30000, 31000, 83.33, 86.11),  # False positive
    ]
    
    total_hours = 2.0
    
    # Run evaluation
    protocol = EvaluationProtocol()
    metrics = protocol.evaluate(predictions, ground_truth, total_hours)
    
    print("\n--- Overall Metrics ---")
    print(f"Episode Sensitivity: {metrics.episode_sensitivity:.2%}")
    print(f"Episode PPV: {metrics.episode_ppv:.2%}")
    print(f"FA/hour: {metrics.false_alarms_per_hour:.2f}")
    print(f"VT Sensitivity: {metrics.vt_sensitivity:.2%}")
    print(f"SVT Sensitivity: {metrics.svt_sensitivity:.2%}")
    
    # Test per-class FA
    print("\n--- Per-Class FA/hr ---")
    print(f"VT/VFL: {metrics.vt_vfl_fa_per_hour:.2f}")
    print(f"SVT: {metrics.svt_fa_per_hour:.2f}")
    print(f"Sinus Tachy: {metrics.sinus_tachy_fa_per_hour:.2f}")
    
    # Test onset evaluator
    print("\n--- Onset Accuracy ---")
    onset_eval = OnsetCriticalEvaluator()
    onset_metrics = onset_eval.evaluate_onset_accuracy(predictions, ground_truth)
    print(f"Mean onset error: {onset_metrics['mean_onset_error_ms']:.0f} ms")
    print(f"Onset accuracy: {onset_metrics['onset_accuracy']:.1%}")
    
    # Test confidence-aware evaluation
    print("\n--- Confidence-Stratified Metrics ---")
    conf_eval = ConfidenceAwareEvaluator()
    conf_metrics = conf_eval.evaluate_with_confidence(predictions, ground_truth, total_hours)
    print(f"Expert tier VT sens: {conf_metrics['expert_tier'].get('vt_sensitivity', 0):.2%}")
    print(f"Derived tier VT sens: {conf_metrics['derived_tier'].get('vt_sensitivity', 0):.2%}")
    print(f"Weighted VT sens: {conf_metrics['weighted_vt_sensitivity']:.2%}")
    
    # Test report generation
    print("\n" + "-"*60)
    reporter = ResultsReporter()
    report = reporter.generate_report(
        metrics, 
        conf_metrics,
        {'model_name': 'CausalGRU', 'dataset': 'MIT-BIH', 'date': '2026-01-24'}
    )
    print(report)
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
