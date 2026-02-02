"""
Clinical Priority Tiers for XAI Tachycardia Detection.

v2.4: Defines the clinical importance hierarchy for different arrhythmia types.

CRITICAL: Not all tachycardias are equal. The system MUST treat them differently
based on clinical priority and life-threatening potential.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

from ..data.contracts import EpisodeType


class ClinicalPriorityTier(Enum):
    """
    Clinical priority tiers for arrhythmia types.
    
    This is the PRODUCT CONTRACT: how we treat each arrhythmia class.
    
    Tier 0: MUST NOT MISS - Life-threatening arrhythmias
    Tier 1: Clinically relevant - Needs clinical attention
    Tier 2: Advisory - Contextual, lower priority
    """
    TIER_0_MUST_NOT_MISS = 0        # VT, VFL, VF - life-threatening
    TIER_1_CLINICALLY_RELEVANT = 1  # SVT, AFIB_RVR, AFlutter - needs attention
    TIER_2_ADVISORY = 2             # Sinus tachycardia - contextual


@dataclass
class TierOperatingParameters:
    """
    Operating parameters for each clinical tier.
    
    These parameters define how the system should behave for each tier
    across different operating modes.
    """
    tier: ClinicalPriorityTier
    
    # Sensitivity requirements (mode-dependent)
    sensitivity_floor_high_sens: float
    sensitivity_floor_balanced: float
    sensitivity_floor_research: float
    
    # FA tolerance
    max_fa_per_hour: float
    
    # Alarm behavior
    can_be_suppressed_for_budget: bool  # Can this tier be suppressed when budget exhausted?
    alarm_sound_priority: str           # "critical", "warning", "info"
    
    # Confirmation requirements
    min_confirmation_sec: float
    requires_hr_validation: bool
    requires_morphology: bool
    
    def get_sensitivity_floor(self, mode: str) -> float:
        """Get sensitivity floor for the given mode."""
        if mode == "high_sensitivity":
            return self.sensitivity_floor_high_sens
        elif mode == "balanced":
            return self.sensitivity_floor_balanced
        elif mode == "research":
            return self.sensitivity_floor_research
        else:
            return self.sensitivity_floor_balanced  # Default


# =============================================================================
# Tier Parameter Definitions
# =============================================================================

TIER_PARAMETERS: Dict[ClinicalPriorityTier, TierOperatingParameters] = {
    
    ClinicalPriorityTier.TIER_0_MUST_NOT_MISS: TierOperatingParameters(
        tier=ClinicalPriorityTier.TIER_0_MUST_NOT_MISS,
        
        # Highest sensitivity requirements
        sensitivity_floor_high_sens=0.98,   # 98% - "never miss"
        sensitivity_floor_balanced=0.95,    # 95% - production default
        sensitivity_floor_research=0.90,    # 90% - minimum acceptable
        
        # Allow more FA for VT to maintain sensitivity
        max_fa_per_hour=2.0,
        
        # NEVER suppress VT for budget reasons
        can_be_suppressed_for_budget=False,
        alarm_sound_priority="critical",
        
        # VT requires full validation
        min_confirmation_sec=1.5,
        requires_hr_validation=True,
        requires_morphology=True,
    ),
    
    ClinicalPriorityTier.TIER_1_CLINICALLY_RELEVANT: TierOperatingParameters(
        tier=ClinicalPriorityTier.TIER_1_CLINICALLY_RELEVANT,
        
        # Moderate sensitivity requirements
        sensitivity_floor_high_sens=0.92,
        sensitivity_floor_balanced=0.88,
        sensitivity_floor_research=0.80,
        
        # Moderate FA tolerance
        max_fa_per_hour=1.0,
        
        # Can be suppressed if budget exhausted (rare)
        can_be_suppressed_for_budget=True,
        alarm_sound_priority="warning",
        
        # SVT requires HR but not necessarily morphology
        min_confirmation_sec=2.0,
        requires_hr_validation=True,
        requires_morphology=False,
    ),
    
    ClinicalPriorityTier.TIER_2_ADVISORY: TierOperatingParameters(
        tier=ClinicalPriorityTier.TIER_2_ADVISORY,
        
        # Lower sensitivity requirements
        sensitivity_floor_high_sens=0.85,
        sensitivity_floor_balanced=0.80,
        sensitivity_floor_research=0.70,
        
        # Low FA tolerance (these are less critical)
        max_fa_per_hour=0.5,
        
        # Can be suppressed to reduce alarm fatigue
        can_be_suppressed_for_budget=True,
        alarm_sound_priority="info",
        
        # Longer confirmation, less stringent requirements
        min_confirmation_sec=3.0,
        requires_hr_validation=True,
        requires_morphology=False,
    ),
}


# =============================================================================
# Episode Type to Tier Mapping
# =============================================================================

ARRHYTHMIA_PRIORITY_MAP: Dict[EpisodeType, ClinicalPriorityTier] = {
    # Tier 0: MUST NOT MISS - life-threatening
    EpisodeType.VT_MONOMORPHIC: ClinicalPriorityTier.TIER_0_MUST_NOT_MISS,
    EpisodeType.VT_POLYMORPHIC: ClinicalPriorityTier.TIER_0_MUST_NOT_MISS,
    EpisodeType.VFL: ClinicalPriorityTier.TIER_0_MUST_NOT_MISS,
    EpisodeType.VFIB: ClinicalPriorityTier.TIER_0_MUST_NOT_MISS,
    
    # Tier 1: Clinically relevant but not immediately catastrophic
    EpisodeType.SVT: ClinicalPriorityTier.TIER_1_CLINICALLY_RELEVANT,
    EpisodeType.AFIB_RVR: ClinicalPriorityTier.TIER_1_CLINICALLY_RELEVANT,
    EpisodeType.AFLUTTER: ClinicalPriorityTier.TIER_1_CLINICALLY_RELEVANT,
    
    # Tier 2: Advisory / contextual
    EpisodeType.SINUS_TACHYCARDIA: ClinicalPriorityTier.TIER_2_ADVISORY,
    
    # Not applicable
    EpisodeType.NORMAL: None,
    EpisodeType.UNKNOWN: None,
    EpisodeType.ARTIFACT: None,
}


def get_tier_for_episode_type(episode_type: EpisodeType) -> ClinicalPriorityTier:
    """
    Get the clinical priority tier for an episode type.
    
    Args:
        episode_type: The type of episode
        
    Returns:
        The clinical priority tier, or None if not applicable
    """
    return ARRHYTHMIA_PRIORITY_MAP.get(episode_type)


def get_tier_parameters(tier: ClinicalPriorityTier) -> TierOperatingParameters:
    """
    Get the operating parameters for a clinical tier.
    
    Args:
        tier: The clinical priority tier
        
    Returns:
        The operating parameters for this tier
    """
    return TIER_PARAMETERS[tier]


def is_must_not_miss(episode_type: EpisodeType) -> bool:
    """
    Check if an episode type is in the "must not miss" tier.
    
    Args:
        episode_type: The type of episode
        
    Returns:
        True if this is a must-not-miss arrhythmia
    """
    tier = get_tier_for_episode_type(episode_type)
    return tier == ClinicalPriorityTier.TIER_0_MUST_NOT_MISS


def can_suppress_for_budget(episode_type: EpisodeType) -> bool:
    """
    Check if an episode type can be suppressed for alarm budget reasons.
    
    Args:
        episode_type: The type of episode
        
    Returns:
        True if this can be suppressed when alarm budget is exhausted
    """
    tier = get_tier_for_episode_type(episode_type)
    if tier is None:
        return True
    return TIER_PARAMETERS[tier].can_be_suppressed_for_budget


def print_tier_summary():
    """Print a summary of clinical priority tiers."""
    print("\n" + "="*80)
    print("CLINICAL PRIORITY TIER SUMMARY")
    print("="*80)
    
    for tier, params in TIER_PARAMETERS.items():
        print(f"\n{tier.name}")
        print("-" * len(tier.name))
        
        # Episode types in this tier
        types_in_tier = [
            et.value for et, t in ARRHYTHMIA_PRIORITY_MAP.items()
            if t == tier
        ]
        print(f"  Episode types: {', '.join(types_in_tier)}")
        
        print(f"  Sensitivity floors: "
              f"HIGH={params.sensitivity_floor_high_sens:.0%}, "
              f"BALANCED={params.sensitivity_floor_balanced:.0%}, "
              f"RESEARCH={params.sensitivity_floor_research:.0%}")
        print(f"  Max FA/hr: {params.max_fa_per_hour:.1f}")
        print(f"  Can suppress for budget: {'No' if not params.can_be_suppressed_for_budget else 'Yes'}")
        print(f"  Alarm priority: {params.alarm_sound_priority}")
        print(f"  Min confirmation: {params.min_confirmation_sec:.1f}s")
        print(f"  Requires: HR={params.requires_hr_validation}, "
              f"Morphology={params.requires_morphology}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    print_tier_summary()
