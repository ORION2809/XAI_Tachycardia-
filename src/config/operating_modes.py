"""
Operating Modes for XAI Tachycardia Detection.

v2.4: Defines deployment-grade operating configurations with explicit gates.

This is the deployment-grade specification. Run experiments against these modes,
not ad-hoc configurations. Each mode has specific sensitivity floors, FA limits,
latency bounds, and calibration requirements.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class OperatingMode(Enum):
    """
    Operating modes for the detection system.
    
    Each mode represents a different trade-off between sensitivity and specificity,
    targeted at different clinical use cases.
    """
    HIGH_SENSITIVITY = "high_sensitivity"  # ICU, never-miss studies
    BALANCED = "balanced"                   # Production default, general telemetry
    RESEARCH = "research"                   # Algorithm development, ablation studies


@dataclass
class OperatingModeConfig:
    """
    Complete configuration for an operating mode.
    
    This defines ALL operational parameters for a deployment mode.
    No parameter should be left to ad-hoc configuration.
    """
    mode: OperatingMode
    
    # === Sensitivity floors by clinical tier ===
    vt_vfl_sensitivity_floor: float         # Tier 0: MUST NOT MISS
    svt_sensitivity_floor: float            # Tier 1: Clinically relevant
    sinus_tachy_sensitivity_floor: float    # Tier 2: Advisory
    
    # === False alarm limits by tier ===
    vt_vfl_max_fa_per_hour: float
    svt_max_fa_per_hour: float
    sinus_tachy_max_fa_per_hour: float
    
    # === Confirmation timing ===
    vt_confirmation_sec: float      # How long VT must persist before alarm
    svt_confirmation_sec: float     # How long SVT must persist before alarm
    
    # === SQI behavior ===
    sqi_suppress_threshold: float   # Below this: suppress (unless VF-like)
    sqi_defer_threshold: float      # Below this but above suppress: defer
    
    # === Burst limiting ===
    max_alarms_per_burst_window: int    # Max alarms in burst window
    burst_window_sec: float             # Burst window duration
    
    # === Latency bounds (HARD GATES) ===
    max_vt_onset_to_alarm_sec: float    # P95 must be below this
    
    # === Calibration requirements ===
    max_ece: float                      # Expected Calibration Error ceiling
    
    # === External validation requirements ===
    max_external_sensitivity_drop: float  # Max allowed drop on external set
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        assert 0.0 < self.vt_vfl_sensitivity_floor <= 1.0
        assert 0.0 < self.svt_sensitivity_floor <= 1.0
        assert 0.0 < self.sinus_tachy_sensitivity_floor <= 1.0
        
        assert self.vt_vfl_max_fa_per_hour > 0
        assert self.svt_max_fa_per_hour > 0
        
        assert self.vt_confirmation_sec > 0
        assert self.svt_confirmation_sec > 0
        
        assert 0.0 <= self.sqi_suppress_threshold <= 1.0
        assert self.sqi_suppress_threshold < self.sqi_defer_threshold <= 1.0
        
        assert self.max_alarms_per_burst_window > 0
        assert self.burst_window_sec > 0
        
        assert self.max_vt_onset_to_alarm_sec > 0
        assert 0.0 < self.max_ece <= 1.0
        
        return True
    
    def get_sensitivity_floor(self, episode_type: str) -> float:
        """Get sensitivity floor for a given episode type."""
        if episode_type in ('vt_mono', 'vt_poly', 'vfl', 'vfib', 'VT_MONOMORPHIC', 'VT_POLYMORPHIC', 'VFL', 'VFIB'):
            return self.vt_vfl_sensitivity_floor
        elif episode_type in ('svt', 'afib_rvr', 'aflutter', 'SVT', 'AFIB_RVR', 'AFLUTTER'):
            return self.svt_sensitivity_floor
        elif episode_type in ('sinus_tachy', 'SINUS_TACHYCARDIA'):
            return self.sinus_tachy_sensitivity_floor
        else:
            return 0.0  # Unknown type
    
    def get_fa_limit(self, episode_type: str) -> float:
        """Get FA/hr limit for a given episode type."""
        if episode_type in ('vt_mono', 'vt_poly', 'vfl', 'vfib', 'VT_MONOMORPHIC', 'VT_POLYMORPHIC', 'VFL', 'VFIB'):
            return self.vt_vfl_max_fa_per_hour
        elif episode_type in ('svt', 'afib_rvr', 'aflutter', 'SVT', 'AFIB_RVR', 'AFLUTTER'):
            return self.svt_max_fa_per_hour
        elif episode_type in ('sinus_tachy', 'SINUS_TACHYCARDIA'):
            return self.sinus_tachy_max_fa_per_hour
        else:
            return 1.0  # Default limit


# =============================================================================
# Pre-defined Operating Modes
# =============================================================================

OPERATING_MODES: Dict[OperatingMode, OperatingModeConfig] = {
    
    OperatingMode.HIGH_SENSITIVITY: OperatingModeConfig(
        mode=OperatingMode.HIGH_SENSITIVITY,
        
        # Very high sensitivity floors - "never miss" VT
        vt_vfl_sensitivity_floor=0.98,      # 98% - absolute minimum
        svt_sensitivity_floor=0.92,
        sinus_tachy_sensitivity_floor=0.85,
        
        # Allow more FA in exchange for sensitivity
        vt_vfl_max_fa_per_hour=3.0,
        svt_max_fa_per_hour=2.0,
        sinus_tachy_max_fa_per_hour=1.0,
        
        # Faster confirmation for quicker response
        vt_confirmation_sec=1.0,
        svt_confirmation_sec=1.5,
        
        # Almost never suppress based on SQI
        sqi_suppress_threshold=0.2,
        sqi_defer_threshold=0.5,
        
        # More alarms allowed in burst
        max_alarms_per_burst_window=5,
        burst_window_sec=300,  # 5 minutes
        
        # Tight latency bound
        max_vt_onset_to_alarm_sec=4.0,
        
        # Strict calibration
        max_ece=0.08,
        
        # Strict external validation
        max_external_sensitivity_drop=0.10,
    ),
    
    OperatingMode.BALANCED: OperatingModeConfig(
        mode=OperatingMode.BALANCED,
        
        # Production-grade sensitivity floors
        vt_vfl_sensitivity_floor=0.95,      # 95% - production standard
        svt_sensitivity_floor=0.88,
        sinus_tachy_sensitivity_floor=0.80,
        
        # Balanced FA limits
        vt_vfl_max_fa_per_hour=1.5,
        svt_max_fa_per_hour=1.0,
        sinus_tachy_max_fa_per_hour=0.5,
        
        # Standard confirmation timing
        vt_confirmation_sec=1.5,
        svt_confirmation_sec=2.0,
        
        # Standard SQI thresholds
        sqi_suppress_threshold=0.3,
        sqi_defer_threshold=0.6,
        
        # Moderate burst limiting
        max_alarms_per_burst_window=3,
        burst_window_sec=300,
        
        # Standard latency bound
        max_vt_onset_to_alarm_sec=5.0,
        
        # Production calibration standard
        max_ece=0.10,
        
        # Allow slightly more external drop
        max_external_sensitivity_drop=0.15,
    ),
    
    OperatingMode.RESEARCH: OperatingModeConfig(
        mode=OperatingMode.RESEARCH,
        
        # Relaxed sensitivity floors for research
        vt_vfl_sensitivity_floor=0.90,      # 90% - minimum acceptable
        svt_sensitivity_floor=0.80,
        sinus_tachy_sensitivity_floor=0.70,
        
        # Moderate FA limits
        vt_vfl_max_fa_per_hour=2.0,
        svt_max_fa_per_hour=1.0,
        sinus_tachy_max_fa_per_hour=0.5,
        
        # Standard confirmation timing
        vt_confirmation_sec=1.5,
        svt_confirmation_sec=2.0,
        
        # More aggressive SQI filtering
        sqi_suppress_threshold=0.5,
        sqi_defer_threshold=0.7,
        
        # Standard burst limiting
        max_alarms_per_burst_window=3,
        burst_window_sec=300,
        
        # Relaxed latency bound
        max_vt_onset_to_alarm_sec=6.0,
        
        # Relaxed calibration standard
        max_ece=0.15,
        
        # Allow more external drop
        max_external_sensitivity_drop=0.20,
    ),
}


def get_mode_config(mode: OperatingMode) -> OperatingModeConfig:
    """Get configuration for a specific operating mode."""
    return OPERATING_MODES[mode]


def get_default_mode() -> OperatingMode:
    """Get the default operating mode (BALANCED)."""
    return OperatingMode.BALANCED


def print_mode_comparison():
    """Print a comparison table of all operating modes."""
    print("\n" + "="*100)
    print("OPERATING MODE COMPARISON")
    print("="*100)
    
    headers = ["Parameter", "HIGH_SENSITIVITY", "BALANCED", "RESEARCH"]
    
    rows = [
        ("VT Sensitivity Floor", 
         f"≥{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].vt_vfl_sensitivity_floor:.0%}",
         f"≥{OPERATING_MODES[OperatingMode.BALANCED].vt_vfl_sensitivity_floor:.0%}",
         f"≥{OPERATING_MODES[OperatingMode.RESEARCH].vt_vfl_sensitivity_floor:.0%}"),
        
        ("SVT Sensitivity Floor",
         f"≥{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].svt_sensitivity_floor:.0%}",
         f"≥{OPERATING_MODES[OperatingMode.BALANCED].svt_sensitivity_floor:.0%}",
         f"≥{OPERATING_MODES[OperatingMode.RESEARCH].svt_sensitivity_floor:.0%}"),
        
        ("VT FA/hr Allowed",
         f"≤{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].vt_vfl_max_fa_per_hour:.1f}",
         f"≤{OPERATING_MODES[OperatingMode.BALANCED].vt_vfl_max_fa_per_hour:.1f}",
         f"≤{OPERATING_MODES[OperatingMode.RESEARCH].vt_vfl_max_fa_per_hour:.1f}"),
        
        ("VT Confirmation (sec)",
         f"{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].vt_confirmation_sec:.1f}",
         f"{OPERATING_MODES[OperatingMode.BALANCED].vt_confirmation_sec:.1f}",
         f"{OPERATING_MODES[OperatingMode.RESEARCH].vt_confirmation_sec:.1f}"),
        
        ("Max Latency (sec)",
         f"≤{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].max_vt_onset_to_alarm_sec:.1f}",
         f"≤{OPERATING_MODES[OperatingMode.BALANCED].max_vt_onset_to_alarm_sec:.1f}",
         f"≤{OPERATING_MODES[OperatingMode.RESEARCH].max_vt_onset_to_alarm_sec:.1f}"),
        
        ("Max ECE",
         f"≤{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].max_ece:.2f}",
         f"≤{OPERATING_MODES[OperatingMode.BALANCED].max_ece:.2f}",
         f"≤{OPERATING_MODES[OperatingMode.RESEARCH].max_ece:.2f}"),
        
        ("Burst Limit (alarms/5min)",
         f"{OPERATING_MODES[OperatingMode.HIGH_SENSITIVITY].max_alarms_per_burst_window}",
         f"{OPERATING_MODES[OperatingMode.BALANCED].max_alarms_per_burst_window}",
         f"{OPERATING_MODES[OperatingMode.RESEARCH].max_alarms_per_burst_window}"),
    ]
    
    # Print table
    col_width = 25
    print(f"\n{'Parameter':<30} {'HIGH_SENSITIVITY':>20} {'BALANCED':>20} {'RESEARCH':>20}")
    print("-"*100)
    
    for row in rows:
        print(f"{row[0]:<30} {row[1]:>20} {row[2]:>20} {row[3]:>20}")
    
    print("="*100 + "\n")


if __name__ == "__main__":
    print_mode_comparison()
