"""
Configuration module for XAI Tachycardia Detection.

v2.4: Contains operating modes, clinical tiers, monitoring context, and deployment configurations.
"""

from .operating_modes import (
    OperatingMode,
    OperatingModeConfig,
    OPERATING_MODES,
    get_mode_config,
    get_default_mode,
)
from .clinical_tiers import (
    ClinicalPriorityTier,
    TierOperatingParameters,
    TIER_PARAMETERS,
    ARRHYTHMIA_PRIORITY_MAP,
    get_tier_for_episode_type,
    get_tier_parameters,
    is_must_not_miss,
)
from .monitoring_context import (
    ContextType,
    FACalculationScope,
    NoiseLevel,
    MonitoringContext,
    FAReportCard,
    ICU_TELEMETRY_CONTEXT,
    STEP_DOWN_CONTEXT,
    AMBULATORY_HOLTER_CONTEXT,
    WEARABLE_CONTEXT,
    STRESS_TEST_CONTEXT,
    EVENT_MONITOR_CONTEXT,
    MONITORING_CONTEXTS,
    get_context,
    get_default_context,
)

__all__ = [
    # Operating modes
    'OperatingMode',
    'OperatingModeConfig',
    'OPERATING_MODES',
    'get_mode_config',
    'get_default_mode',
    
    # Clinical tiers
    'ClinicalPriorityTier',
    'TierOperatingParameters',
    'TIER_PARAMETERS',
    'ARRHYTHMIA_PRIORITY_MAP',
    'get_tier_for_episode_type',
    'get_tier_parameters',
    'is_must_not_miss',
    
    # Monitoring context
    'ContextType',
    'FACalculationScope',
    'NoiseLevel',
    'MonitoringContext',
    'FAReportCard',
    'ICU_TELEMETRY_CONTEXT',
    'STEP_DOWN_CONTEXT',
    'AMBULATORY_HOLTER_CONTEXT',
    'WEARABLE_CONTEXT',
    'STRESS_TEST_CONTEXT',
    'EVENT_MONITOR_CONTEXT',
    'MONITORING_CONTEXTS',
    'get_context',
    'get_default_context',
]
