"""
Deployment Readiness Checklist Module

Implements v2.4 requirements:
- Formal deployment readiness assessment
- All mode-specific gates checked
- Sub-cohort analysis included
- Domain shift quantified and mitigated
- Human-readable reports for regulatory/clinical review

Reference: BUILDABLE_SPEC.md Part 9.4
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import numpy as np


class ReadinessStatus(Enum):
    """Overall readiness status."""
    READY = "READY"
    READY_WITH_CONDITIONS = "READY_WITH_CONDITIONS"
    NOT_READY = "NOT_READY"
    BLOCKED = "BLOCKED"


class GateStatus(Enum):
    """Individual gate status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    NOT_EVALUATED = "NOT_EVALUATED"


@dataclass
class GateResult:
    """Result of a single deployment gate check."""
    name: str
    status: GateStatus
    value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    is_blocking: bool = True


@dataclass
class SubcohortAnalysis:
    """Analysis results for a specific sub-cohort."""
    name: str
    n_patients: int
    n_episodes: int
    vt_sensitivity: float
    fa_per_hour: float
    ece: float
    passes_relaxed_thresholds: bool
    notes: str = ""


@dataclass 
class DeploymentReadinessReport:
    """
    v2.4: Formal deployment readiness assessment.
    
    This is what a regulatory or clinical review would expect.
    Contains all metrics, sub-cohort analysis, and gate status.
    """
    # Identity
    model_version: str
    operating_mode: str
    evaluation_date: str
    evaluator: str = "Automated System"
    
    # Overall status
    overall_status: ReadinessStatus = ReadinessStatus.NOT_READY
    
    # Core metrics - Internal
    internal_vt_sensitivity: float = 0.0
    internal_vfl_sensitivity: float = 0.0
    internal_svt_sensitivity: float = 0.0
    internal_vt_fa_per_hour: float = float('inf')
    internal_vfl_fa_per_hour: float = float('inf')
    
    # Core metrics - External
    external_vt_sensitivity: float = 0.0
    external_vfl_sensitivity: float = 0.0
    external_vt_fa_per_hour: float = float('inf')
    
    # Calibration
    internal_ece: float = 1.0
    external_ece: float = 1.0
    
    # Latency
    p50_latency_sec: float = float('inf')
    p95_latency_sec: float = float('inf')
    p99_latency_sec: float = float('inf')
    max_latency_sec: float = float('inf')
    
    # Sub-cohort analysis
    worst_patient_sensitivity: float = 0.0
    low_sqi_quartile_sensitivity: float = 0.0
    paced_patient_sensitivity: float = 0.0
    high_hr_patient_sensitivity: float = 0.0
    bbb_patient_sensitivity: float = 0.0
    
    # Domain shift
    external_sensitivity_drop: float = 1.0
    mitigation_applied: bool = False
    post_mitigation_drop: Optional[float] = None
    drift_severity: str = "unknown"
    
    # Gate results
    gate_results: List[GateResult] = field(default_factory=list)
    failed_gates: List[str] = field(default_factory=list)
    warning_gates: List[str] = field(default_factory=list)
    
    # Sub-cohort details
    subcohort_analyses: List[SubcohortAnalysis] = field(default_factory=list)
    
    # Additional notes
    notes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def generate_summary(self) -> str:
        """Generate human-readable summary for review."""
        status_emoji = {
            ReadinessStatus.READY: "✅",
            ReadinessStatus.READY_WITH_CONDITIONS: "⚠️",
            ReadinessStatus.NOT_READY: "❌",
            ReadinessStatus.BLOCKED: "🚫",
        }
        
        emoji = status_emoji.get(self.overall_status, "❓")
        
        summary = f"""
========================================
DEPLOYMENT READINESS REPORT
========================================
Model Version: {self.model_version}
Operating Mode: {self.operating_mode}
Date: {self.evaluation_date}
Evaluator: {self.evaluator}

Overall Status: {emoji} {self.overall_status.value}

--- CORE PERFORMANCE (INTERNAL) ---
VT Sensitivity: {self.internal_vt_sensitivity:.1%}
VFL Sensitivity: {self.internal_vfl_sensitivity:.1%}
SVT Sensitivity: {self.internal_svt_sensitivity:.1%}
VT FA/hr: {self.internal_vt_fa_per_hour:.2f}
VFL FA/hr: {self.internal_vfl_fa_per_hour:.2f}

--- CORE PERFORMANCE (EXTERNAL) ---
VT Sensitivity: {self.external_vt_sensitivity:.1%}
VFL Sensitivity: {self.external_vfl_sensitivity:.1%}
VT FA/hr: {self.external_vt_fa_per_hour:.2f}

--- CALIBRATION ---
ECE (internal): {self.internal_ece:.4f}
ECE (external): {self.external_ece:.4f}

--- LATENCY ---
P50: {self.p50_latency_sec:.2f}s
P95: {self.p95_latency_sec:.2f}s
P99: {self.p99_latency_sec:.2f}s
Max: {self.max_latency_sec:.2f}s

--- SUB-COHORT ANALYSIS ---
Worst patient VT sens: {self.worst_patient_sensitivity:.1%}
Low-SQI quartile sens: {self.low_sqi_quartile_sensitivity:.1%}
Paced patients sens: {self.paced_patient_sensitivity:.1%}
High HR patients sens: {self.high_hr_patient_sensitivity:.1%}
BBB patients sens: {self.bbb_patient_sensitivity:.1%}

--- DOMAIN SHIFT ---
External sensitivity drop: {self.external_sensitivity_drop:.1%}
Mitigation applied: {self.mitigation_applied}
Post-mitigation drop: {f'{self.post_mitigation_drop:.1%}' if self.post_mitigation_drop is not None else 'N/A'}
Drift severity: {self.drift_severity}

--- GATE STATUS ---
Total gates: {len(self.gate_results)}
Passed: {sum(1 for g in self.gate_results if g.status == GateStatus.PASSED)}
Failed: {sum(1 for g in self.gate_results if g.status == GateStatus.FAILED)}
Warnings: {sum(1 for g in self.gate_results if g.status == GateStatus.WARNING)}
"""
        
        if self.failed_gates:
            summary += f"\nFailed gates: {', '.join(self.failed_gates)}"
        
        if self.warning_gates:
            summary += f"\nWarning gates: {', '.join(self.warning_gates)}"
        
        if self.recommendations:
            summary += "\n\n--- RECOMMENDATIONS ---\n"
            for i, rec in enumerate(self.recommendations, 1):
                summary += f"{i}. {rec}\n"
        
        if self.notes:
            summary += "\n--- NOTES ---\n"
            for note in self.notes:
                summary += f"• {note}\n"
        
        summary += "\n========================================"
        
        return summary
    
    def export_for_review(self) -> Dict:
        """Export as dictionary for JSON/reporting."""
        data = asdict(self)
        # Convert enums to strings
        data['overall_status'] = self.overall_status.value
        data['gate_results'] = [
            {
                'name': g.name,
                'status': g.status.value,
                'value': g.value,
                'threshold': g.threshold,
                'message': g.message,
                'is_blocking': g.is_blocking,
            }
            for g in self.gate_results
        ]
        return data
    
    def to_json(self) -> str:
        """Export as JSON string."""
        import json
        return json.dumps(self.export_for_review(), indent=2, default=str)


class DeploymentReadinessChecker:
    """
    Deployment readiness checker that evaluates all gates.
    
    v2.4: Comprehensive pre-deployment validation.
    """
    
    def __init__(
        self,
        model_version: str,
        operating_mode: str,
        mode_config: Optional[Any] = None,
    ):
        """
        Initialize the checker.
        
        Args:
            model_version: Model version string
            operating_mode: Operating mode name
            mode_config: Operating mode configuration with thresholds
        """
        self.model_version = model_version
        self.operating_mode = operating_mode
        self.mode_config = mode_config
        
        # Default thresholds if no config provided
        self.thresholds = {
            'vt_sensitivity_floor': 0.95,
            'vfl_sensitivity_floor': 0.95,
            'svt_sensitivity_floor': 0.80,
            'vt_max_fa_per_hour': 1.0,
            'vfl_max_fa_per_hour': 1.0,
            'max_ece': 0.10,
            'max_latency_sec': 5.0,
            'max_external_drop': 0.20,
            'subcohort_relaxation': 0.85,  # 85% of main threshold
        }
        
        # Override with mode config if provided
        if mode_config is not None:
            self._load_mode_config(mode_config)
    
    def _load_mode_config(self, config: Any) -> None:
        """Load thresholds from mode config."""
        if hasattr(config, 'vt_vfl_sensitivity_floor'):
            self.thresholds['vt_sensitivity_floor'] = config.vt_vfl_sensitivity_floor
            self.thresholds['vfl_sensitivity_floor'] = config.vt_vfl_sensitivity_floor
        if hasattr(config, 'svt_sensitivity_floor'):
            self.thresholds['svt_sensitivity_floor'] = config.svt_sensitivity_floor
        if hasattr(config, 'vt_vfl_max_fa_per_hour'):
            self.thresholds['vt_max_fa_per_hour'] = config.vt_vfl_max_fa_per_hour
            self.thresholds['vfl_max_fa_per_hour'] = config.vt_vfl_max_fa_per_hour
        if hasattr(config, 'max_ece'):
            self.thresholds['max_ece'] = config.max_ece
        if hasattr(config, 'max_vt_onset_to_alarm_sec'):
            self.thresholds['max_latency_sec'] = config.max_vt_onset_to_alarm_sec
        if hasattr(config, 'max_external_sensitivity_drop'):
            self.thresholds['max_external_drop'] = config.max_external_sensitivity_drop
    
    def run_full_check(
        self,
        internal_metrics: Dict[str, float],
        external_metrics: Dict[str, float],
        latency_metrics: Dict[str, float],
        subcohort_metrics: Dict[str, Dict[str, float]],
        per_patient_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        drift_analysis: Optional[Dict[str, Any]] = None,
        mitigation_applied: bool = False,
        post_mitigation_metrics: Optional[Dict[str, float]] = None,
    ) -> DeploymentReadinessReport:
        """
        Run full deployment readiness check.
        
        Args:
            internal_metrics: Metrics from internal validation set
            external_metrics: Metrics from external validation set
            latency_metrics: Detection latency measurements
            subcohort_metrics: Per-subcohort performance
            per_patient_metrics: Optional per-patient breakdown
            drift_analysis: Domain shift analysis results
            mitigation_applied: Whether mitigation was applied
            post_mitigation_metrics: Metrics after mitigation
            
        Returns:
            DeploymentReadinessReport with all gates evaluated
        """
        report = DeploymentReadinessReport(
            model_version=self.model_version,
            operating_mode=self.operating_mode,
            evaluation_date=datetime.now().isoformat(),
        )
        
        # Populate core metrics
        report.internal_vt_sensitivity = internal_metrics.get('vt_sensitivity', 0)
        report.internal_vfl_sensitivity = internal_metrics.get('vfl_sensitivity', 0)
        report.internal_svt_sensitivity = internal_metrics.get('svt_sensitivity', 0)
        report.internal_vt_fa_per_hour = internal_metrics.get('vt_fa_per_hour', float('inf'))
        report.internal_vfl_fa_per_hour = internal_metrics.get('vfl_fa_per_hour', float('inf'))
        report.internal_ece = internal_metrics.get('ece', 1.0)
        
        report.external_vt_sensitivity = external_metrics.get('vt_sensitivity', 0)
        report.external_vfl_sensitivity = external_metrics.get('vfl_sensitivity', 0)
        report.external_vt_fa_per_hour = external_metrics.get('vt_fa_per_hour', float('inf'))
        report.external_ece = external_metrics.get('ece', 1.0)
        
        # Latency
        report.p50_latency_sec = latency_metrics.get('p50', float('inf'))
        report.p95_latency_sec = latency_metrics.get('p95', float('inf'))
        report.p99_latency_sec = latency_metrics.get('p99', float('inf'))
        report.max_latency_sec = latency_metrics.get('max', float('inf'))
        
        # Domain shift
        report.external_sensitivity_drop = (
            report.internal_vt_sensitivity - report.external_vt_sensitivity
        )
        report.mitigation_applied = mitigation_applied
        if post_mitigation_metrics:
            report.post_mitigation_drop = (
                report.internal_vt_sensitivity - 
                post_mitigation_metrics.get('vt_sensitivity', 0)
            )
        if drift_analysis:
            report.drift_severity = drift_analysis.get('severity', 'unknown')
        
        # Run all gate checks
        gates = []
        
        # Gate 1: VT Sensitivity (Internal)
        gates.append(self._check_gate(
            name="vt_sensitivity_internal",
            value=report.internal_vt_sensitivity,
            threshold=self.thresholds['vt_sensitivity_floor'],
            comparison=">=",
            is_blocking=True,
        ))
        
        # Gate 2: VFL Sensitivity (Internal)
        gates.append(self._check_gate(
            name="vfl_sensitivity_internal",
            value=report.internal_vfl_sensitivity,
            threshold=self.thresholds['vfl_sensitivity_floor'],
            comparison=">=",
            is_blocking=True,
        ))
        
        # Gate 3: SVT Sensitivity (Internal)
        gates.append(self._check_gate(
            name="svt_sensitivity_internal",
            value=report.internal_svt_sensitivity,
            threshold=self.thresholds['svt_sensitivity_floor'],
            comparison=">=",
            is_blocking=False,  # Warning only
        ))
        
        # Gate 4: VT FA/hr (Internal)
        gates.append(self._check_gate(
            name="vt_fa_per_hour_internal",
            value=report.internal_vt_fa_per_hour,
            threshold=self.thresholds['vt_max_fa_per_hour'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 5: VFL FA/hr (Internal)
        gates.append(self._check_gate(
            name="vfl_fa_per_hour_internal",
            value=report.internal_vfl_fa_per_hour,
            threshold=self.thresholds['vfl_max_fa_per_hour'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 6: ECE (Internal)
        gates.append(self._check_gate(
            name="ece_internal",
            value=report.internal_ece,
            threshold=self.thresholds['max_ece'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 7: ECE (External)
        gates.append(self._check_gate(
            name="ece_external",
            value=report.external_ece,
            threshold=self.thresholds['max_ece'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 8: P95 Latency
        gates.append(self._check_gate(
            name="p95_latency",
            value=report.p95_latency_sec,
            threshold=self.thresholds['max_latency_sec'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 9: External sensitivity drop
        effective_drop = (
            report.post_mitigation_drop 
            if report.post_mitigation_drop is not None 
            else report.external_sensitivity_drop
        )
        gates.append(self._check_gate(
            name="external_sensitivity_drop",
            value=effective_drop,
            threshold=self.thresholds['max_external_drop'],
            comparison="<=",
            is_blocking=True,
        ))
        
        # Gate 10: VT Sensitivity (External)
        # Relaxed threshold for external
        external_thresh = self.thresholds['vt_sensitivity_floor'] * 0.90
        gates.append(self._check_gate(
            name="vt_sensitivity_external",
            value=report.external_vt_sensitivity,
            threshold=external_thresh,
            comparison=">=",
            is_blocking=True,
        ))
        
        # Process subcohort metrics
        for subcohort_name, metrics in subcohort_metrics.items():
            analysis = self._analyze_subcohort(subcohort_name, metrics)
            report.subcohort_analyses.append(analysis)
            
            # Update summary metrics
            if subcohort_name == 'low_sqi_quartile':
                report.low_sqi_quartile_sensitivity = metrics.get('vt_sensitivity', 0)
            elif subcohort_name == 'paced_patients':
                report.paced_patient_sensitivity = metrics.get('vt_sensitivity', 0)
            elif subcohort_name == 'high_hr_patients':
                report.high_hr_patient_sensitivity = metrics.get('vt_sensitivity', 0)
            elif subcohort_name == 'bbb_patients':
                report.bbb_patient_sensitivity = metrics.get('vt_sensitivity', 0)
        
        # Find worst patient
        if per_patient_metrics:
            worst_sens = float('inf')
            for patient_id, metrics in per_patient_metrics.items():
                if metrics.get('n_vt_episodes', 0) > 0:
                    sens = metrics.get('vt_sensitivity', 0)
                    if sens < worst_sens:
                        worst_sens = sens
            report.worst_patient_sensitivity = worst_sens if worst_sens != float('inf') else 0.0
        
        # Collect gate results
        report.gate_results = gates
        report.failed_gates = [g.name for g in gates if g.status == GateStatus.FAILED]
        report.warning_gates = [g.name for g in gates if g.status == GateStatus.WARNING]
        
        # Determine overall status
        blocking_failures = [g for g in gates if g.status == GateStatus.FAILED and g.is_blocking]
        non_blocking_failures = [g for g in gates if g.status == GateStatus.FAILED and not g.is_blocking]
        warnings = [g for g in gates if g.status == GateStatus.WARNING]
        
        if len(blocking_failures) > 0:
            report.overall_status = ReadinessStatus.NOT_READY
            report.recommendations.append(
                f"Address {len(blocking_failures)} blocking gate failure(s): "
                f"{', '.join(g.name for g in blocking_failures)}"
            )
        elif len(non_blocking_failures) > 0 or len(warnings) > 0:
            report.overall_status = ReadinessStatus.READY_WITH_CONDITIONS
            report.recommendations.append(
                f"Review {len(non_blocking_failures)} non-blocking failure(s) and "
                f"{len(warnings)} warning(s) before deployment."
            )
        else:
            report.overall_status = ReadinessStatus.READY
            report.notes.append("All gates passed. Model is ready for deployment.")
        
        # Add recommendations based on metrics
        if report.drift_severity in ['high', 'critical']:
            report.recommendations.append(
                "High domain drift detected. Consider retraining on external domain data."
            )
        
        if report.worst_patient_sensitivity < 0.5:
            report.recommendations.append(
                f"Worst patient sensitivity is {report.worst_patient_sensitivity:.1%}. "
                "Investigate failure modes for edge cases."
            )
        
        return report
    
    def _check_gate(
        self,
        name: str,
        value: float,
        threshold: float,
        comparison: str,
        is_blocking: bool,
    ) -> GateResult:
        """Check a single gate."""
        if comparison == ">=":
            passed = value >= threshold
            message = f"{value:.4f} >= {threshold:.4f}" if passed else f"{value:.4f} < {threshold:.4f}"
        elif comparison == "<=":
            passed = value <= threshold
            message = f"{value:.4f} <= {threshold:.4f}" if passed else f"{value:.4f} > {threshold:.4f}"
        elif comparison == "==":
            passed = abs(value - threshold) < 1e-6
            message = f"{value:.4f} == {threshold:.4f}" if passed else f"{value:.4f} != {threshold:.4f}"
        else:
            passed = False
            message = f"Unknown comparison: {comparison}"
        
        if passed:
            status = GateStatus.PASSED
        elif is_blocking:
            status = GateStatus.FAILED
        else:
            status = GateStatus.WARNING
        
        return GateResult(
            name=name,
            status=status,
            value=value,
            threshold=threshold,
            message=message,
            is_blocking=is_blocking,
        )
    
    def _analyze_subcohort(
        self,
        name: str,
        metrics: Dict[str, float],
    ) -> SubcohortAnalysis:
        """Analyze a sub-cohort."""
        vt_sens = metrics.get('vt_sensitivity', 0)
        fa_hr = metrics.get('fa_per_hour', float('inf'))
        ece = metrics.get('ece', 1.0)
        
        # Relaxed thresholds for subcohorts
        relaxed_sens = self.thresholds['vt_sensitivity_floor'] * self.thresholds['subcohort_relaxation']
        relaxed_fa = self.thresholds['vt_max_fa_per_hour'] * 2.0
        relaxed_ece = self.thresholds['max_ece'] * 1.5
        
        passes = (
            vt_sens >= relaxed_sens and
            fa_hr <= relaxed_fa and
            ece <= relaxed_ece
        )
        
        notes = []
        if vt_sens < relaxed_sens:
            notes.append(f"Sensitivity below relaxed threshold ({relaxed_sens:.1%})")
        if fa_hr > relaxed_fa:
            notes.append(f"FA/hr above relaxed threshold ({relaxed_fa:.1f})")
        if ece > relaxed_ece:
            notes.append(f"ECE above relaxed threshold ({relaxed_ece:.3f})")
        
        return SubcohortAnalysis(
            name=name,
            n_patients=metrics.get('n_patients', 0),
            n_episodes=metrics.get('n_episodes', 0),
            vt_sensitivity=vt_sens,
            fa_per_hour=fa_hr,
            ece=ece,
            passes_relaxed_thresholds=passes,
            notes="; ".join(notes) if notes else "Passes all relaxed thresholds",
        )
    
    def quick_check(
        self,
        vt_sensitivity: float,
        fa_per_hour: float,
        ece: float,
        p95_latency: float,
    ) -> Tuple[bool, List[str]]:
        """
        Quick check of core gates.
        
        Returns:
            Tuple of (passes_all, list_of_failures)
        """
        failures = []
        
        if vt_sensitivity < self.thresholds['vt_sensitivity_floor']:
            failures.append(
                f"VT sensitivity {vt_sensitivity:.1%} < "
                f"{self.thresholds['vt_sensitivity_floor']:.1%}"
            )
        
        if fa_per_hour > self.thresholds['vt_max_fa_per_hour']:
            failures.append(
                f"FA/hr {fa_per_hour:.2f} > "
                f"{self.thresholds['vt_max_fa_per_hour']:.2f}"
            )
        
        if ece > self.thresholds['max_ece']:
            failures.append(
                f"ECE {ece:.3f} > {self.thresholds['max_ece']:.3f}"
            )
        
        if p95_latency > self.thresholds['max_latency_sec']:
            failures.append(
                f"P95 latency {p95_latency:.2f}s > "
                f"{self.thresholds['max_latency_sec']:.2f}s"
            )
        
        return len(failures) == 0, failures


class PreDeploymentGates:
    """
    Pre-deployment gate checks from BUILDABLE_SPEC v2.4.
    
    These are hard gates that MUST pass before deployment.
    """
    
    @staticmethod
    def check_patient_split_integrity(
        train_ids: set,
        test_ids: set,
    ) -> GateResult:
        """No patient appears in both train and test."""
        overlap = train_ids & test_ids
        passed = len(overlap) == 0
        
        return GateResult(
            name="patient_split_integrity",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=len(overlap),
            threshold=0,
            message=f"Overlap: {len(overlap)} patients" if not passed else "No overlap",
            is_blocking=True,
        )
    
    @staticmethod
    def check_known_vt_detected(
        known_vt_episodes: List[Dict],
        detected_episodes: List[Dict],
        max_latency_sec: float = 5.0,
    ) -> GateResult:
        """
        All known VT episodes must be detected within latency budget.
        
        This is the "don't miss VT" test.
        """
        if not known_vt_episodes:
            return GateResult(
                name="known_vt_detected",
                status=GateStatus.NOT_EVALUATED,
                message="No known VT episodes provided",
                is_blocking=True,
            )
        
        detected_count = 0
        missed = []
        
        for known in known_vt_episodes:
            record_id = known.get('record_id')
            onset_sample = known.get('onset_sample', 0)
            fs = known.get('fs', 360)
            
            # Find matching detection
            found = False
            for det in detected_episodes:
                if det.get('record_id') == record_id:
                    det_start = det.get('start_sample', 0)
                    latency_sec = (det_start - onset_sample) / fs
                    if 0 <= latency_sec <= max_latency_sec:
                        found = True
                        break
            
            if found:
                detected_count += 1
            else:
                missed.append(record_id)
        
        sensitivity = detected_count / len(known_vt_episodes)
        passed = len(missed) == 0
        
        return GateResult(
            name="known_vt_detected",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=sensitivity,
            threshold=1.0,
            message=f"Detected {detected_count}/{len(known_vt_episodes)}" + 
                    (f", missed: {missed}" if missed else ""),
            is_blocking=True,
        )
    
    @staticmethod
    def check_vf_not_suppressed_by_sqi(
        vf_segments: List[Dict],
        suppressed_segments: List[Dict],
    ) -> GateResult:
        """
        VF/VFL must not be suppressed by SQI even when QRS detectability is low.
        """
        if not vf_segments:
            return GateResult(
                name="vf_not_suppressed_by_sqi",
                status=GateStatus.NOT_EVALUATED,
                message="No VF segments provided",
                is_blocking=True,
            )
        
        incorrectly_suppressed = []
        
        for vf in vf_segments:
            segment_id = vf.get('segment_id')
            for supp in suppressed_segments:
                if supp.get('segment_id') == segment_id:
                    incorrectly_suppressed.append(segment_id)
                    break
        
        passed = len(incorrectly_suppressed) == 0
        
        return GateResult(
            name="vf_not_suppressed_by_sqi",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=len(incorrectly_suppressed),
            threshold=0,
            message=f"Incorrectly suppressed: {len(incorrectly_suppressed)}" if not passed else "None suppressed",
            is_blocking=True,
        )
    
    @staticmethod
    def check_alarm_burst_rate(
        alarm_history: List[Dict],
        max_alarms_per_window: int = 3,
        window_sec: float = 60.0,
    ) -> GateResult:
        """
        No more than X alarms in Y seconds (burst limiting).
        
        A system that fires 10 alarms in 1 minute is unusable.
        """
        if not alarm_history:
            return GateResult(
                name="alarm_burst_rate",
                status=GateStatus.PASSED,
                value=0,
                threshold=max_alarms_per_window,
                message="No alarms to check",
                is_blocking=True,
            )
        
        sorted_alarms = sorted(alarm_history, key=lambda x: x.get('timestamp_sec', 0))
        max_burst = 0
        violations = 0
        
        for i, alarm in enumerate(sorted_alarms):
            t_start = alarm.get('timestamp_sec', 0)
            t_end = t_start + window_sec
            
            count = sum(
                1 for a in sorted_alarms
                if t_start <= a.get('timestamp_sec', 0) < t_end
            )
            
            max_burst = max(max_burst, count)
            if count > max_alarms_per_window:
                violations += 1
        
        passed = max_burst <= max_alarms_per_window
        
        return GateResult(
            name="alarm_burst_rate",
            status=GateStatus.PASSED if passed else GateStatus.FAILED,
            value=max_burst,
            threshold=max_alarms_per_window,
            message=f"Max burst: {max_burst} alarms in {window_sec}s window" +
                    (f" ({violations} violations)" if not passed else ""),
            is_blocking=True,
        )


# Convenience function
def check_deployment_readiness(
    model_version: str,
    operating_mode: str,
    internal_metrics: Dict[str, float],
    external_metrics: Dict[str, float],
    latency_metrics: Dict[str, float],
    subcohort_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    mode_config: Optional[Any] = None,
) -> DeploymentReadinessReport:
    """
    Quick deployment readiness check.
    
    Returns a DeploymentReadinessReport with all gates evaluated.
    """
    checker = DeploymentReadinessChecker(
        model_version=model_version,
        operating_mode=operating_mode,
        mode_config=mode_config,
    )
    
    return checker.run_full_check(
        internal_metrics=internal_metrics,
        external_metrics=external_metrics,
        latency_metrics=latency_metrics,
        subcohort_metrics=subcohort_metrics or {},
    )
