"""
Evaluation Module for Tachycardia Detection.

v2.4: Comprehensive evaluation with sensitivity-first metrics,
per-class FA/hr, onset-critical metrics, and sub-cohort validation.
"""

from .metrics import (
    EpisodeLabel,
    EpisodeType,
    LabelConfidenceTier,
    EvaluationMetrics,
    EvaluationProtocol,
    OnsetCriticalEvaluator,
    PerClassFACalculator,
    ConfidenceAwareEvaluator,
    PerPatientEvaluator,
    ResultsReporter,
)

from .validation import (
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

from .domain_shift import (
    DriftSeverity,
    DomainShiftMitigationConfig,
    DriftIndicators,
    RecalibrationResult,
    ThresholdRetuningResult,
    MitigatedEvaluationResult,
    TemperatureScaler,
    DomainShiftMitigation,
    compute_psi,
    detect_drift,
)

from .deployment_readiness import (
    ReadinessStatus,
    GateStatus,
    GateResult,
    SubcohortAnalysis,
    DeploymentReadinessReport,
    DeploymentReadinessChecker,
    PreDeploymentGates,
    check_deployment_readiness,
)

__all__ = [
    # Episode structures
    'EpisodeLabel',
    'EpisodeType',
    'LabelConfidenceTier',
    # Metrics
    'EvaluationMetrics',
    'EvaluationProtocol',
    'OnsetCriticalEvaluator',
    'PerClassFACalculator',
    'ConfidenceAwareEvaluator',
    'PerPatientEvaluator',
    'ResultsReporter',
    # Validation
    'SubCohort',
    'PatientMetadata',
    'SubCohortValidator',
    'PatientSplitValidator',
    'CrossValidator',
    'CrossValidationConfig',
    'AcceptanceTests',
    'AcceptanceConfig',
    'DomainShiftValidator',
    'DomainShiftConfig',
    # Domain Shift (v2.4)
    'DriftSeverity',
    'DomainShiftMitigationConfig',
    'DriftIndicators',
    'RecalibrationResult',
    'ThresholdRetuningResult',
    'MitigatedEvaluationResult',
    'TemperatureScaler',
    'DomainShiftMitigation',
    'compute_psi',
    'detect_drift',
    # Deployment Readiness (v2.4)
    'ReadinessStatus',
    'GateStatus',
    'GateResult',
    'SubcohortAnalysis',
    'DeploymentReadinessReport',
    'DeploymentReadinessChecker',
    'PreDeploymentGates',
    'check_deployment_readiness',
]
