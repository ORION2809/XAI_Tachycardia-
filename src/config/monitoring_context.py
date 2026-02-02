"""
Monitoring Context for XAI Tachycardia Detection.

v2.4: Explicit FA/hr semantics and monitoring context definitions.

CRITICAL: FA/hr is MEANINGLESS without context. Define it explicitly:
- Continuous ambulatory vs ICU telemetry
- Per-patient vs global
- Artifact handling
- Expected noise levels

Without this context, comparing FA/hr across studies is comparing apples to oranges.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# =============================================================================
# CONTEXT TYPES
# =============================================================================

class ContextType(Enum):
    """Types of monitoring contexts."""
    ICU_TELEMETRY = "icu_telemetry"           # High-acuity, clean signals
    STEP_DOWN_UNIT = "step_down_unit"         # Moderate acuity
    AMBULATORY_HOLTER = "ambulatory_holter"   # 24-48h ambulatory monitoring
    WEARABLE = "wearable"                      # Consumer/clinical wearables
    STRESS_TEST = "stress_test"               # Exercise/stress testing
    EVENT_MONITOR = "event_monitor"           # Patient-triggered recording


class FACalculationScope(Enum):
    """How FA/hr should be calculated."""
    PER_PATIENT_HOUR = "per_patient_hour"     # FA rate per patient-hour (standard)
    GLOBAL_HOUR = "global_hour"               # FA rate across all recordings
    PER_RECORDING = "per_recording"           # FA count per recording segment


class NoiseLevel(Enum):
    """Expected noise level in the monitoring context."""
    LOW = "low"         # Clean ICU signals, minimal artifact
    MEDIUM = "medium"   # Holter, some motion artifact expected
    HIGH = "high"       # Wearables, significant artifact expected


# =============================================================================
# MONITORING CONTEXT
# =============================================================================

@dataclass
class MonitoringContext:
    """
    Define the monitoring context for FA/hr interpretation.
    
    v2.4: FA/hr means nothing unless you define:
    - Continuous ambulatory vs ICU telemetry
    - Per-patient vs global
    - Artifact handling
    
    Usage:
        context = ICU_TELEMETRY_CONTEXT
        effective_hours = context.get_effective_hours(
            total_hours=24.0,
            artifact_hours=2.0
        )
        fa_rate = false_alarms / effective_hours
    """
    
    # Context identification
    context_type: ContextType
    description: str = ""
    
    # Temporal scope for FA calculation
    fa_calculation_scope: FACalculationScope = FACalculationScope.PER_PATIENT_HOUR
    
    # Patient population characteristics
    expected_noise_level: NoiseLevel = NoiseLevel.MEDIUM
    includes_paced_patients: bool = True
    includes_bundle_branch_block: bool = True
    
    # Artifact handling (CRITICAL for FA/hr interpretation)
    artifact_time_excluded_from_denominator: bool = True  # If True, FA/hr only counts "monitorable" time
    min_monitorable_fraction: float = 0.70  # Minimum fraction of time that must be monitorable
    
    # Recording characteristics
    typical_recording_duration_hours: float = 24.0
    continuous_monitoring: bool = True  # vs intermittent/event-triggered
    
    # Expected baseline rates (for sanity checking)
    expected_artifact_fraction: float = 0.10  # 10% artifact expected
    expected_baseline_alarm_rate_per_hour: float = 0.5  # Before optimization
    
    def get_effective_hours(
        self,
        total_hours: float,
        artifact_hours: float,
    ) -> float:
        """
        Get effective monitoring hours for FA/hr calculation.
        
        If artifact_time_excluded_from_denominator is True, we only count
        "monitorable" time where the signal was usable.
        
        Args:
            total_hours: Total recording duration
            artifact_hours: Hours classified as artifact/unusable
            
        Returns:
            Effective hours to use as FA/hr denominator
        """
        if self.artifact_time_excluded_from_denominator:
            effective = total_hours - artifact_hours
            return max(effective, 0.0)
        return total_hours
    
    def check_monitorable_fraction(
        self,
        total_hours: float,
        artifact_hours: float,
    ) -> tuple:
        """
        Check if recording meets minimum monitorable fraction.
        
        Returns:
            (is_acceptable, fraction, message)
        """
        if total_hours <= 0:
            return False, 0.0, "No recording time"
        
        monitorable_hours = total_hours - artifact_hours
        fraction = monitorable_hours / total_hours
        
        if fraction >= self.min_monitorable_fraction:
            return True, fraction, "Acceptable monitorable fraction"
        else:
            return False, fraction, f"Monitorable fraction {fraction:.1%} below minimum {self.min_monitorable_fraction:.1%}"
    
    def calculate_fa_rate(
        self,
        false_alarm_count: int,
        total_hours: float,
        artifact_hours: float = 0.0,
    ) -> Dict[str, float]:
        """
        Calculate FA rate with full context.
        
        Returns:
            Dict with FA rate and metadata
        """
        effective_hours = self.get_effective_hours(total_hours, artifact_hours)
        
        if effective_hours <= 0:
            fa_per_hour = float('inf')
        else:
            fa_per_hour = false_alarm_count / effective_hours
        
        return {
            "fa_count": false_alarm_count,
            "total_hours": total_hours,
            "artifact_hours": artifact_hours,
            "effective_hours": effective_hours,
            "fa_per_hour": fa_per_hour,
            "context_type": self.context_type.value,
            "scope": self.fa_calculation_scope.value,
        }
    
    def validate(self) -> bool:
        """Validate context configuration."""
        assert 0.0 <= self.min_monitorable_fraction <= 1.0
        assert 0.0 <= self.expected_artifact_fraction <= 1.0
        assert self.typical_recording_duration_hours > 0
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "context_type": self.context_type.value,
            "description": self.description,
            "fa_calculation_scope": self.fa_calculation_scope.value,
            "expected_noise_level": self.expected_noise_level.value,
            "includes_paced_patients": self.includes_paced_patients,
            "includes_bundle_branch_block": self.includes_bundle_branch_block,
            "artifact_time_excluded_from_denominator": self.artifact_time_excluded_from_denominator,
            "min_monitorable_fraction": self.min_monitorable_fraction,
            "typical_recording_duration_hours": self.typical_recording_duration_hours,
        }


# =============================================================================
# Pre-defined Monitoring Contexts
# =============================================================================

ICU_TELEMETRY_CONTEXT = MonitoringContext(
    context_type=ContextType.ICU_TELEMETRY,
    description="ICU continuous telemetry monitoring with high-quality signals",
    fa_calculation_scope=FACalculationScope.PER_PATIENT_HOUR,
    expected_noise_level=NoiseLevel.LOW,
    includes_paced_patients=True,
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=True,
    min_monitorable_fraction=0.90,  # ICU should have very clean signals
    typical_recording_duration_hours=24.0,
    continuous_monitoring=True,
    expected_artifact_fraction=0.05,  # Very low artifact expected
    expected_baseline_alarm_rate_per_hour=0.3,
)


STEP_DOWN_CONTEXT = MonitoringContext(
    context_type=ContextType.STEP_DOWN_UNIT,
    description="Step-down unit telemetry with moderate signal quality",
    fa_calculation_scope=FACalculationScope.PER_PATIENT_HOUR,
    expected_noise_level=NoiseLevel.LOW,
    includes_paced_patients=True,
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=True,
    min_monitorable_fraction=0.85,
    typical_recording_duration_hours=24.0,
    continuous_monitoring=True,
    expected_artifact_fraction=0.10,
    expected_baseline_alarm_rate_per_hour=0.4,
)


AMBULATORY_HOLTER_CONTEXT = MonitoringContext(
    context_type=ContextType.AMBULATORY_HOLTER,
    description="Ambulatory Holter monitoring with expected motion artifact",
    fa_calculation_scope=FACalculationScope.PER_PATIENT_HOUR,
    expected_noise_level=NoiseLevel.MEDIUM,
    includes_paced_patients=True,
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=True,
    min_monitorable_fraction=0.70,  # More artifact expected
    typical_recording_duration_hours=24.0,
    continuous_monitoring=True,
    expected_artifact_fraction=0.20,
    expected_baseline_alarm_rate_per_hour=0.5,
)


WEARABLE_CONTEXT = MonitoringContext(
    context_type=ContextType.WEARABLE,
    description="Consumer/clinical wearable with significant expected artifact",
    fa_calculation_scope=FACalculationScope.PER_PATIENT_HOUR,
    expected_noise_level=NoiseLevel.HIGH,
    includes_paced_patients=False,  # Wearables typically exclude paced patients
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=True,
    min_monitorable_fraction=0.50,  # High artifact tolerance
    typical_recording_duration_hours=168.0,  # Week-long monitoring
    continuous_monitoring=True,
    expected_artifact_fraction=0.30,
    expected_baseline_alarm_rate_per_hour=0.8,
)


STRESS_TEST_CONTEXT = MonitoringContext(
    context_type=ContextType.STRESS_TEST,
    description="Exercise stress testing with expected motion and baseline wander",
    fa_calculation_scope=FACalculationScope.PER_RECORDING,  # Usually per-test
    expected_noise_level=NoiseLevel.MEDIUM,
    includes_paced_patients=False,  # Usually excluded
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=True,
    min_monitorable_fraction=0.80,
    typical_recording_duration_hours=0.5,  # 30 minutes
    continuous_monitoring=True,
    expected_artifact_fraction=0.15,
    expected_baseline_alarm_rate_per_hour=0.5,
)


EVENT_MONITOR_CONTEXT = MonitoringContext(
    context_type=ContextType.EVENT_MONITOR,
    description="Patient-triggered event monitor recordings",
    fa_calculation_scope=FACalculationScope.PER_RECORDING,
    expected_noise_level=NoiseLevel.MEDIUM,
    includes_paced_patients=True,
    includes_bundle_branch_block=True,
    artifact_time_excluded_from_denominator=False,  # Short recordings, count all
    min_monitorable_fraction=0.60,
    typical_recording_duration_hours=0.05,  # ~3 minutes per event
    continuous_monitoring=False,
    expected_artifact_fraction=0.10,
    expected_baseline_alarm_rate_per_hour=0.3,
)


# Context lookup table
MONITORING_CONTEXTS: Dict[ContextType, MonitoringContext] = {
    ContextType.ICU_TELEMETRY: ICU_TELEMETRY_CONTEXT,
    ContextType.STEP_DOWN_UNIT: STEP_DOWN_CONTEXT,
    ContextType.AMBULATORY_HOLTER: AMBULATORY_HOLTER_CONTEXT,
    ContextType.WEARABLE: WEARABLE_CONTEXT,
    ContextType.STRESS_TEST: STRESS_TEST_CONTEXT,
    ContextType.EVENT_MONITOR: EVENT_MONITOR_CONTEXT,
}


def get_context(context_type: ContextType) -> MonitoringContext:
    """Get predefined monitoring context."""
    return MONITORING_CONTEXTS[context_type]


def get_default_context() -> MonitoringContext:
    """Get default monitoring context (ICU telemetry)."""
    return ICU_TELEMETRY_CONTEXT


# =============================================================================
# FA/HR REPORTING UTILITIES
# =============================================================================

@dataclass
class FAReportCard:
    """
    Complete FA rate report with context.
    
    This ensures that FA rates are always reported with full context,
    preventing misleading comparisons.
    """
    # Context
    context: MonitoringContext
    
    # Counts
    vt_false_alarms: int = 0
    svt_false_alarms: int = 0
    sinus_tachy_false_alarms: int = 0
    total_false_alarms: int = 0
    
    # Duration
    total_hours: float = 0.0
    artifact_hours: float = 0.0
    patient_count: int = 0
    
    def calculate_rates(self) -> Dict[str, float]:
        """Calculate all FA rates."""
        effective_hours = self.context.get_effective_hours(
            self.total_hours, self.artifact_hours
        )
        
        if effective_hours <= 0:
            return {
                "vt_fa_per_hour": float('inf'),
                "svt_fa_per_hour": float('inf'),
                "sinus_tachy_fa_per_hour": float('inf'),
                "total_fa_per_hour": float('inf'),
            }
        
        return {
            "vt_fa_per_hour": self.vt_false_alarms / effective_hours,
            "svt_fa_per_hour": self.svt_false_alarms / effective_hours,
            "sinus_tachy_fa_per_hour": self.sinus_tachy_false_alarms / effective_hours,
            "total_fa_per_hour": self.total_false_alarms / effective_hours,
            "effective_hours": effective_hours,
            "monitorable_fraction": (self.total_hours - self.artifact_hours) / self.total_hours if self.total_hours > 0 else 0,
        }
    
    def generate_report(self) -> str:
        """Generate human-readable FA report."""
        rates = self.calculate_rates()
        
        lines = [
            "=" * 60,
            "FALSE ALARM RATE REPORT",
            "=" * 60,
            f"\nContext: {self.context.context_type.value}",
            f"Noise Level: {self.context.expected_noise_level.value}",
            f"FA Scope: {self.context.fa_calculation_scope.value}",
            f"\nRecording Summary:",
            f"  Total Hours: {self.total_hours:.1f}",
            f"  Artifact Hours: {self.artifact_hours:.1f}",
            f"  Effective Hours: {rates['effective_hours']:.1f}",
            f"  Monitorable Fraction: {rates['monitorable_fraction']:.1%}",
            f"  Patient Count: {self.patient_count}",
            f"\nFA Rates (per hour):",
            f"  VT/VFL: {rates['vt_fa_per_hour']:.2f}",
            f"  SVT: {rates['svt_fa_per_hour']:.2f}",
            f"  Sinus Tachy: {rates['sinus_tachy_fa_per_hour']:.2f}",
            f"  Total: {rates['total_fa_per_hour']:.2f}",
            "\n" + "=" * 60,
        ]
        
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MONITORING CONTEXT DEMO")
    print("=" * 60)
    
    # Compare FA rates across contexts
    print("\nFA/hr interpretation varies by context:")
    print("-" * 60)
    
    for ctx_type, ctx in MONITORING_CONTEXTS.items():
        print(f"\n{ctx_type.value}:")
        print(f"  Noise Level: {ctx.expected_noise_level.value}")
        print(f"  Min Monitorable: {ctx.min_monitorable_fraction:.0%}")
        print(f"  Artifact Excluded: {ctx.artifact_time_excluded_from_denominator}")
    
    # Example FA calculation
    print("\n" + "=" * 60)
    print("Example FA Rate Calculation")
    print("=" * 60)
    
    context = ICU_TELEMETRY_CONTEXT
    result = context.calculate_fa_rate(
        false_alarm_count=5,
        total_hours=24.0,
        artifact_hours=2.0
    )
    
    print(f"\nContext: {result['context_type']}")
    print(f"False Alarms: {result['fa_count']}")
    print(f"Total Hours: {result['total_hours']}")
    print(f"Artifact Hours: {result['artifact_hours']}")
    print(f"Effective Hours: {result['effective_hours']}")
    print(f"FA/hr: {result['fa_per_hour']:.2f}")
    
    print("\n" + "=" * 60)
