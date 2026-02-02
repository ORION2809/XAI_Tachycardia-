# Detection module
# v2.4: Complete detection pipeline with two-lane architecture

# Two-lane pipeline (Lane 1: Detection, Lane 2: Confirmation)
from .two_lane_pipeline import (
    TwoLanePipeline,
    DetectionLane,
    ConfirmationLane,
    DetectionConfig,
    EpisodeType,
    DetectedEpisode,
    TemporalConfig,
    AlignmentConfig,
)

# Episode detector
from .episode_detector import (
    EpisodeDetector,
    EpisodeDetectorConfig,
    SQIResult,
)

# Alarm system (two-tier with budget tracking)
from .alarm_system import (
    TwoTierAlarmSystem,
    AlarmConfig,
    AlarmTier,
    AlarmOutput,
    AlarmBudgetTracker,
    AlarmStateTracker,
    create_alarm_system,
)

# Decision machine (unified decision policy)
from .decision_machine import (
    UnifiedDecisionPolicy,
    DecisionPolicyConfig,
    DecisionInput,
    DecisionOutput,
    DecisionAction,
    AlarmBudget,
)

__all__ = [
    # Two-lane pipeline
    'TwoLanePipeline',
    'DetectionLane',
    'ConfirmationLane',
    'DetectionConfig',
    'EpisodeType',
    'DetectedEpisode',
    'TemporalConfig',
    'AlignmentConfig',
    
    # Episode detector
    'EpisodeDetector',
    'EpisodeDetectorConfig',
    'SQIResult',
    
    # Alarm system
    'TwoTierAlarmSystem',
    'AlarmConfig',
    'AlarmTier',
    'AlarmOutput',
    'AlarmBudgetTracker',
    'AlarmStateTracker',
    'create_alarm_system',
    
    # Decision machine
    'UnifiedDecisionPolicy',
    'DecisionPolicyConfig',
    'DecisionInput',
    'DecisionOutput',
    'DecisionAction',
    'AlarmBudget',
]
