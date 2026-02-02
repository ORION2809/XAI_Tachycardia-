"""
Decision Machine for Tachycardia Detection.

v2.4: Unified decision policy with single authority for all alarm decisions.

Key Components:
- UnifiedDecisionPolicy: SINGLE decision engine for all ALARM/WARNING/SUPPRESS decisions
- AlarmStateTracker: THIN state supplier (NOT a decision engine)
- BurstSuppressor: Rate limiting for alarm fatigue prevention
- DecisionInput/Output: Explicit contract for all decision inputs

CRITICAL: UnifiedDecisionPolicy is the ONLY decision authority.
All other components supply context, not decisions.
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import time


# =============================================================================
# ENUMS AND CONTRACTS
# =============================================================================

class DecisionAction(Enum):
    """Possible decision outputs."""
    SUPPRESS = "suppress"       # Do not alert (bad quality or low confidence)
    WARNING = "warning"         # Soft alert, continue monitoring
    ALARM = "alarm"             # Hard alarm, immediate attention
    DEFER = "defer"             # High uncertainty, request clinician review


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


@dataclass
class EpisodeLabel:
    """Episode label data structure."""
    episode_type: EpisodeType
    start_time: float
    end_time: float
    confidence: float = 1.0
    source: str = "model"
    
    @property
    def duration_sec(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DecisionInput:
    """
    All inputs to the decision policy in one place.
    
    This is the CONTRACT: any decision MUST have all these inputs.
    """
    # Episode information
    episode: EpisodeLabel
    episode_type: EpisodeType
    episode_duration_sec: float
    
    # Model outputs (calibrated)
    calibrated_probability: float   # After temperature/isotonic calibration
    raw_probability: float          # Before calibration
    
    # Uncertainty
    uncertainty: float              # 0-1, from MC dropout or ensemble
    uncertainty_tier: str           # "low", "medium", "high", "very_high"
    
    # Signal quality
    sqi_score: float                # 0-1
    sqi_is_usable: bool
    sqi_qrs_detectability: float    # QRS detectability component (0-1)
    
    # HR sanity (coupled to SQI for ALARM gate)
    hr_computed: bool               # Was HR successfully computed?
    hr_value_bpm: Optional[float]   # Computed HR, or None if failed
    hr_in_valid_range: bool         # Is HR within clinical bounds?
    
    # Fields with defaults (must come after required fields)
    sqi_recommendations: List[str] = field(default_factory=list)
    
    # Morphology (soft score from v2.2)
    morphology_score: float = 0.5   # 0 (narrow/SVT) to 1 (wide/VT)
    morphology_confidence: float = 0.0  # Confidence in morphology assessment
    
    # Episode persistence (temporal context)
    consecutive_detections: int = 0     # How many consecutive windows
    persistence_sec: float = 0.0        # How long has episode persisted
    previous_tier: Optional[str] = None # Previous decision tier
    
    # Context
    current_time: float = 0.0
    time_since_last_alarm: float = float('inf')
    alarms_in_last_hour: int = 0


@dataclass
class DecisionOutput:
    """Decision policy output with full explanation."""
    action: DecisionAction
    confidence: float               # Decision confidence
    explanation: str                # Human-readable reason
    
    # Audit trail
    contributing_factors: Dict[str, Any] = field(default_factory=dict)
    overriding_factors: List[str] = field(default_factory=list)
    
    # For downstream
    requires_clinician_review: bool = False
    suppress_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "contributing_factors": self.contributing_factors,
            "overriding_factors": self.overriding_factors,
            "requires_clinician_review": self.requires_clinician_review,
            "suppress_reason": self.suppress_reason,
        }


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DecisionPolicyConfig:
    """Unified decision policy configuration."""
    
    # Probability thresholds (on CALIBRATED probabilities)
    warning_prob_threshold: float = 0.4
    alarm_prob_threshold: float = 0.65
    
    # Uncertainty thresholds
    max_uncertainty_for_alarm: float = 0.35
    defer_uncertainty_threshold: float = 0.55
    
    # SQI thresholds
    min_sqi_for_warning: float = 0.5
    min_sqi_for_alarm: float = 0.6
    
    # Persistence requirements (in seconds - deterministic)
    min_persistence_for_warning_sec: float = 0.3
    min_persistence_for_alarm_sec: float = 0.75
    
    # Rate limiting
    max_alarms_per_hour: int = 2
    cooldown_after_alarm_sec: float = 30.0
    
    # Episode type weights (VT more urgent than SVT)
    vt_urgency_multiplier: float = 1.5
    svt_urgency_multiplier: float = 1.0
    
    # HR+SQI coupling thresholds
    min_qrs_detectability_for_alarm: float = 0.6
    min_qrs_detectability_for_hr_trust: float = 0.7


@dataclass
class AlarmConfig:
    """Alarm rate limiting configuration."""
    max_alarm_rate_per_hour: int = 2
    cooldown_after_alarm_sec: float = 30.0
    burst_window_sec: float = 300.0     # 5 minutes
    max_alarms_per_burst: int = 3


# =============================================================================
# ALARM BUDGET TRACKING
# =============================================================================

@dataclass 
class AlarmBudget:
    """
    v2.2: Per-class alarm budget tracking.
    
    Partition available alarms across episode types to prevent
    one class from consuming all alerts.
    """
    vt_vfl_alarms: List[float] = field(default_factory=list)
    svt_alarms: List[float] = field(default_factory=list)
    sinus_tachy_alarms: List[float] = field(default_factory=list)
    
    # Hourly limits
    vt_vfl_limit: float = 1.0       # FA/hr for VT/VFL
    svt_limit: float = 0.5          # FA/hr for SVT
    sinus_tachy_limit: float = 0.3  # FA/hr for sinus tachycardia
    
    def can_alarm(self, episode_type: EpisodeType, current_time: float) -> bool:
        """Check if alarm budget allows another alarm for this type."""
        hour_ago = current_time - 3600
        
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            recent = [t for t in self.vt_vfl_alarms if t > hour_ago]
            return len(recent) < self.vt_vfl_limit
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR):
            recent = [t for t in self.svt_alarms if t > hour_ago]
            return len(recent) < self.svt_limit
        elif episode_type == EpisodeType.SINUS_TACHY:
            recent = [t for t in self.sinus_tachy_alarms if t > hour_ago]
            return len(recent) < self.sinus_tachy_limit
        
        return True  # Allow by default
    
    def record_alarm(self, episode_type: EpisodeType, current_time: float):
        """Record that an alarm was issued."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC,
                           EpisodeType.VFL, EpisodeType.VFIB):
            self.vt_vfl_alarms.append(current_time)
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR):
            self.svt_alarms.append(current_time)
        elif episode_type == EpisodeType.SINUS_TACHY:
            self.sinus_tachy_alarms.append(current_time)
    
    def prune_old(self, current_time: float, max_age_sec: float = 7200):
        """Remove alarms older than max_age_sec."""
        cutoff = current_time - max_age_sec
        self.vt_vfl_alarms = [t for t in self.vt_vfl_alarms if t > cutoff]
        self.svt_alarms = [t for t in self.svt_alarms if t > cutoff]
        self.sinus_tachy_alarms = [t for t in self.sinus_tachy_alarms if t > cutoff]


# =============================================================================
# ALARM STATE TRACKER (THIN STATE SUPPLIER)
# =============================================================================

class AlarmStateTracker:
    """
    THIN state tracker for alarm persistence.
    
    NOTE: This is NOT a decision engine. It only tracks:
    - Consecutive detection count
    - Alarm history (for rate limiting)
    - Warning/alarm state transitions
    
    All actual decisions go through UnifiedDecisionPolicy.
    This class supplies state context to DecisionInput.
    """
    
    def __init__(self, config: AlarmConfig = None):
        if config is None:
            config = AlarmConfig()
        self.config = config
        self.alarm_history: List[float] = []
        self.consecutive_count: int = 0
        self.first_detection_time: Optional[float] = None
        self.current_state: str = "idle"  # idle, warning, alarm
        
    def update_on_detection(
        self,
        is_detection: bool,
        current_time: float,
    ) -> Dict[str, Any]:
        """
        Update state tracking on each detection window.
        
        Returns context for DecisionInput, NOT a decision.
        """
        if is_detection:
            if self.consecutive_count == 0:
                self.first_detection_time = current_time
            self.consecutive_count += 1
        else:
            self.consecutive_count = 0
            self.first_detection_time = None
            self.current_state = "idle"
        
        # Compute persistence duration
        persistence_sec = 0.0
        if self.first_detection_time is not None:
            persistence_sec = current_time - self.first_detection_time
        
        # Count recent alarms for rate limiting
        recent_alarms = [t for t in self.alarm_history 
                        if current_time - t < 3600]
        
        return {
            "consecutive_count": self.consecutive_count,
            "persistence_sec": persistence_sec,
            "first_detection_time": self.first_detection_time,
            "recent_alarm_count": len(recent_alarms),
            "rate_limit_available": len(recent_alarms) < self.config.max_alarm_rate_per_hour,
            "current_state": self.current_state,
        }
    
    def record_alarm_fired(self, current_time: float):
        """Record that an alarm was fired (called by UnifiedDecisionPolicy)."""
        self.alarm_history.append(current_time)
        self.current_state = "alarm"
        # Prune old alarms (keep last 24 hours)
        self.alarm_history = [t for t in self.alarm_history 
                             if current_time - t < 86400]
    
    def record_warning_issued(self):
        """Record transition to warning state."""
        self.current_state = "warning"
    
    def reset(self):
        """Reset all state (e.g., new patient)."""
        self.consecutive_count = 0
        self.first_detection_time = None
        self.current_state = "idle"
    
    def check_cooldown(self, current_time: float) -> bool:
        """Check if cooldown is active."""
        if not self.alarm_history:
            return False
        time_since_last = current_time - self.alarm_history[-1]
        return time_since_last < self.config.cooldown_after_alarm_sec
    
    def get_state_context(self, current_time: float) -> Dict[str, Any]:
        """Get full state context for DecisionInput."""
        recent_alarms = [t for t in self.alarm_history 
                        if current_time - t < 3600]
        
        persistence_sec = 0.0
        if self.first_detection_time is not None:
            persistence_sec = current_time - self.first_detection_time
        
        time_since_last_alarm = float('inf')
        if self.alarm_history:
            time_since_last_alarm = current_time - self.alarm_history[-1]
            
        return {
            "consecutive_detections": self.consecutive_count,
            "persistence_sec": persistence_sec,
            "recent_alarm_count": len(recent_alarms),
            "rate_limit_active": len(recent_alarms) >= self.config.max_alarm_rate_per_hour,
            "cooldown_active": self.check_cooldown(current_time),
            "current_state": self.current_state,
            "time_since_last_alarm": time_since_last_alarm,
        }


# =============================================================================
# BURST SUPPRESSOR
# =============================================================================

class BurstSuppressor:
    """
    v2.2: Prevent alarm fatigue from rapid-fire alerts.
    
    Limits number of alarms in a short window to prevent
    overwhelming clinicians during noisy periods.
    """
    
    def __init__(
        self,
        burst_window_sec: float = 300.0,
        max_alarms_per_burst: int = 3,
    ):
        self.burst_window_sec = burst_window_sec
        self.max_alarms_per_burst = max_alarms_per_burst
        self.recent_alarms: deque = deque()
    
    def can_alarm(self, current_time: float) -> Tuple[bool, str]:
        """
        Check if burst limit allows another alarm.
        
        Returns:
            (can_alarm, reason)
        """
        self._prune_old(current_time)
        
        if len(self.recent_alarms) >= self.max_alarms_per_burst:
            return False, f"Burst limit reached ({self.max_alarms_per_burst} in {self.burst_window_sec}s)"
        
        return True, ""
    
    def record_alarm(self, current_time: float):
        """Record an alarm."""
        self.recent_alarms.append(current_time)
    
    def _prune_old(self, current_time: float):
        """Remove alarms outside burst window."""
        cutoff = current_time - self.burst_window_sec
        while self.recent_alarms and self.recent_alarms[0] < cutoff:
            self.recent_alarms.popleft()
    
    def get_remaining_budget(self, current_time: float) -> int:
        """Get number of alarms remaining in burst window."""
        self._prune_old(current_time)
        return max(0, self.max_alarms_per_burst - len(self.recent_alarms))
    
    def reset(self):
        """Clear alarm history."""
        self.recent_alarms.clear()


# =============================================================================
# UNIFIED DECISION POLICY (SINGLE AUTHORITY)
# =============================================================================

class UnifiedDecisionPolicy:
    """
    Single decision policy that integrates all inputs.
    
    This replaces scattered threshold checks with ONE coherent policy.
    Every decision has a clear audit trail.
    
    CRITICAL: This is the ONLY decision authority in the system.
    All other components supply context, not decisions.
    """
    
    def __init__(
        self,
        config: DecisionPolicyConfig = None,
        alarm_budget: AlarmBudget = None,
        burst_suppressor: BurstSuppressor = None,
    ):
        if config is None:
            config = DecisionPolicyConfig()
        self.config = config
        self.alarm_budget = alarm_budget or AlarmBudget()
        self.burst_suppressor = burst_suppressor or BurstSuppressor()
    
    def decide(self, input: DecisionInput) -> DecisionOutput:
        """
        Make decision based on all available inputs.
        
        Decision flow:
        1. Check hard gates (SQI, rate limiting, burst suppression)
        2. Compute weighted decision score
        3. Apply HR+SQI coupling for ALARM gate
        4. Apply threshold logic
        5. Generate explanation
        """
        factors: Dict[str, Any] = {}
        overriding: List[str] = []
        
        # ===== STAGE 1: Hard Gates =====
        
        # Gate 1: SQI unusable
        if not input.sqi_is_usable:
            return DecisionOutput(
                action=DecisionAction.SUPPRESS,
                confidence=1.0,
                explanation="Signal quality too poor for reliable detection",
                contributing_factors={"sqi_unusable": True},
                overriding_factors=["Signal quality gate"],
                requires_clinician_review=False,
                suppress_reason="sqi_unusable",
            )
        
        # Gate 2: Rate limiting (hourly)
        if input.alarms_in_last_hour >= self.config.max_alarms_per_hour:
            return DecisionOutput(
                action=DecisionAction.SUPPRESS,
                confidence=0.8,
                explanation=f"Alarm rate limit reached ({self.config.max_alarms_per_hour}/hour)",
                contributing_factors={"rate_limit": True},
                overriding_factors=["Rate limit gate"],
                requires_clinician_review=True,
                suppress_reason="rate_limit",
            )
        
        # Gate 3: Cooldown
        if input.time_since_last_alarm < self.config.cooldown_after_alarm_sec:
            return DecisionOutput(
                action=DecisionAction.SUPPRESS,
                confidence=0.7,
                explanation=f"Cooldown active ({input.time_since_last_alarm:.1f}s < {self.config.cooldown_after_alarm_sec}s)",
                contributing_factors={"cooldown": True},
                overriding_factors=["Cooldown gate"],
                requires_clinician_review=False,
                suppress_reason="cooldown",
            )
        
        # Gate 4: Burst suppression
        can_burst, burst_reason = self.burst_suppressor.can_alarm(input.current_time)
        if not can_burst:
            return DecisionOutput(
                action=DecisionAction.SUPPRESS,
                confidence=0.7,
                explanation=burst_reason,
                contributing_factors={"burst_suppression": True},
                overriding_factors=["Burst suppression gate"],
                requires_clinician_review=True,
                suppress_reason="burst_limit",
            )
        
        # Gate 5: Per-class alarm budget
        if not self.alarm_budget.can_alarm(input.episode_type, input.current_time):
            return DecisionOutput(
                action=DecisionAction.WARNING,
                confidence=0.6,
                explanation=f"Alarm budget exhausted for {input.episode_type.value}",
                contributing_factors={"alarm_budget_exhausted": True},
                overriding_factors=["Per-class alarm budget"],
                requires_clinician_review=True,
                suppress_reason=None,
            )
        
        # ===== STAGE 2: Compute Decision Score =====
        
        # Get urgency multiplier based on episode type
        is_ventricular = input.episode_type in [
            EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
            EpisodeType.VFL, EpisodeType.VFIB
        ]
        urgency = self.config.vt_urgency_multiplier if is_ventricular else self.config.svt_urgency_multiplier
        
        # Weighted decision score
        prob_component = input.calibrated_probability * urgency
        factors["calibrated_prob"] = input.calibrated_probability
        factors["urgency_multiplier"] = urgency
        
        # Uncertainty penalty
        uncertainty_penalty = 1.0 - input.uncertainty
        prob_component *= uncertainty_penalty
        factors["uncertainty"] = input.uncertainty
        factors["uncertainty_penalty"] = uncertainty_penalty
        
        # SQI scaling (reduce confidence if SQI borderline)
        sqi_scale = min(input.sqi_score / 0.8, 1.0)
        prob_component *= sqi_scale
        factors["sqi_score"] = input.sqi_score
        factors["sqi_scale"] = sqi_scale
        
        # Morphology score (soft factor for VT)
        if is_ventricular:
            morphology_factor = 0.7 + 0.3 * input.morphology_score
            morphology_factor *= input.morphology_confidence
            morphology_factor = max(morphology_factor, 0.5)
            prob_component *= morphology_factor
            factors["morphology_score"] = input.morphology_score
            factors["morphology_confidence"] = input.morphology_confidence
            factors["morphology_factor"] = morphology_factor
        
        # Persistence bonus
        if input.persistence_sec > self.config.min_persistence_for_alarm_sec:
            persistence_bonus = 1.1
        else:
            persistence_bonus = 1.0
        prob_component *= persistence_bonus
        factors["persistence_sec"] = input.persistence_sec
        factors["persistence_bonus"] = persistence_bonus
        
        final_score = prob_component
        factors["final_score"] = final_score
        
        # ===== STAGE 3: High Uncertainty DEFER Check =====
        
        if input.uncertainty > self.config.defer_uncertainty_threshold:
            return DecisionOutput(
                action=DecisionAction.DEFER,
                confidence=0.5,
                explanation=f"High uncertainty ({input.uncertainty:.2f}) requires clinician review",
                contributing_factors=factors,
                overriding_factors=["High uncertainty override"],
                requires_clinician_review=True,
                suppress_reason=None,
            )
        
        # ===== STAGE 4: HR+SQI Coupling for ALARM Gate =====
        
        hr_sqi_block_alarm = False
        hr_sqi_reason = None
        
        if not input.hr_computed:
            if input.sqi_qrs_detectability < self.config.min_qrs_detectability_for_alarm:
                hr_sqi_block_alarm = True
                hr_sqi_reason = "HR cannot be computed and QRS detectability is low"
            else:
                factors["hr_warning"] = "HR computation failed with good QRS detectability"
        elif not input.hr_in_valid_range:
            if input.sqi_qrs_detectability < self.config.min_qrs_detectability_for_hr_trust:
                hr_sqi_block_alarm = True
                hr_sqi_reason = f"HR ({input.hr_value_bpm:.0f} BPM) outside valid range with marginal QRS detectability"
        
        factors["hr_computed"] = input.hr_computed
        factors["hr_value_bpm"] = input.hr_value_bpm
        factors["hr_in_valid_range"] = input.hr_in_valid_range
        factors["sqi_qrs_detectability"] = input.sqi_qrs_detectability
        factors["hr_sqi_block_alarm"] = hr_sqi_block_alarm
        
        # ===== STAGE 5: Decision Logic =====
        
        # Check for ALARM
        alarm_conditions = (
            final_score >= self.config.alarm_prob_threshold and
            input.uncertainty <= self.config.max_uncertainty_for_alarm and
            input.sqi_score >= self.config.min_sqi_for_alarm and
            input.persistence_sec >= self.config.min_persistence_for_alarm_sec and
            not hr_sqi_block_alarm
        )
        
        if alarm_conditions:
            # Record alarm for tracking
            self.burst_suppressor.record_alarm(input.current_time)
            self.alarm_budget.record_alarm(input.episode_type, input.current_time)
            
            return DecisionOutput(
                action=DecisionAction.ALARM,
                confidence=min(final_score, 1.0),
                explanation=self._generate_alarm_explanation(input, factors),
                contributing_factors=factors,
                overriding_factors=[],
                requires_clinician_review=False,
                suppress_reason=None,
            )
        
        # If ALARM was blocked by HR+SQI, explain why
        if hr_sqi_block_alarm:
            overriding.append(f"ALARM blocked: {hr_sqi_reason}")
        
        # Check for WARNING
        warning_conditions = (
            final_score >= self.config.warning_prob_threshold and
            input.sqi_score >= self.config.min_sqi_for_warning and
            input.persistence_sec >= self.config.min_persistence_for_warning_sec
        )
        
        if warning_conditions:
            # Determine what's preventing ALARM
            if final_score < self.config.alarm_prob_threshold:
                overriding.append(f"Probability below alarm threshold ({final_score:.2f} < {self.config.alarm_prob_threshold})")
            if input.uncertainty > self.config.max_uncertainty_for_alarm:
                overriding.append(f"Uncertainty too high for alarm ({input.uncertainty:.2f})")
            if input.persistence_sec < self.config.min_persistence_for_alarm_sec:
                overriding.append(f"Insufficient persistence for alarm ({input.persistence_sec:.2f}s)")
            
            return DecisionOutput(
                action=DecisionAction.WARNING,
                confidence=min(final_score, 1.0),
                explanation=f"Possible {input.episode_type.value} detected, monitoring",
                contributing_factors=factors,
                overriding_factors=overriding,
                requires_clinician_review=hr_sqi_block_alarm,
                suppress_reason=None,
            )
        
        # Default: SUPPRESS
        return DecisionOutput(
            action=DecisionAction.SUPPRESS,
            confidence=1.0 - final_score,
            explanation="No concerning arrhythmia detected",
            contributing_factors=factors,
            overriding_factors=[],
            requires_clinician_review=False,
            suppress_reason="below_threshold",
        )
    
    def _generate_alarm_explanation(self, input: DecisionInput, factors: Dict) -> str:
        """Generate human-readable alarm explanation."""
        type_name = input.episode_type.value.replace("_", " ").upper()
        duration = input.episode_duration_sec
        prob = input.calibrated_probability
        
        explanation = f"{type_name} detected with {prob:.0%} confidence, duration {duration:.1f}s"
        
        if input.hr_computed and input.hr_value_bpm:
            explanation += f", HR {input.hr_value_bpm:.0f} BPM"
        
        if input.sqi_score < 0.8:
            explanation += f" (signal quality: {input.sqi_score:.0%})"
        
        return explanation
    
    def reset(self):
        """Reset all state (for new patient/session)."""
        self.burst_suppressor.reset()
        # Note: alarm_budget is typically retained per-patient


# =============================================================================
# DECISION PIPELINE INTEGRATION
# =============================================================================

class DetectionPipeline:
    """
    v2.4: Full detection pipeline integrating all components.
    
    Flow:
    1. Model predicts episode probabilities
    2. Calibration adjusts probabilities
    3. State tracker provides context
    4. UnifiedDecisionPolicy makes decision
    """
    
    def __init__(
        self,
        policy: UnifiedDecisionPolicy = None,
        state_tracker: AlarmStateTracker = None,
    ):
        self.policy = policy or UnifiedDecisionPolicy()
        self.state_tracker = state_tracker or AlarmStateTracker()
    
    def process(
        self,
        episode: EpisodeLabel,
        calibrated_prob: float,
        raw_prob: float,
        uncertainty: float,
        sqi_score: float,
        sqi_is_usable: bool,
        sqi_qrs_detectability: float,
        hr_computed: bool,
        hr_value_bpm: Optional[float],
        current_time: float,
        morphology_score: float = 0.5,
        morphology_confidence: float = 0.0,
    ) -> DecisionOutput:
        """
        Process a detected episode through the full pipeline.
        
        Returns the final decision.
        """
        # Get state context
        is_detection = calibrated_prob > 0.3  # Soft threshold for "detection"
        self.state_tracker.update_on_detection(is_detection, current_time)
        state_context = self.state_tracker.get_state_context(current_time)
        
        # Determine HR validity
        hr_in_valid_range = False
        if hr_computed and hr_value_bpm is not None:
            hr_in_valid_range = 30 <= hr_value_bpm <= 300
        
        # Build decision input
        decision_input = DecisionInput(
            episode=episode,
            episode_type=episode.episode_type,
            episode_duration_sec=episode.duration_sec,
            calibrated_probability=calibrated_prob,
            raw_probability=raw_prob,
            uncertainty=uncertainty,
            uncertainty_tier=self._classify_uncertainty(uncertainty),
            sqi_score=sqi_score,
            sqi_is_usable=sqi_is_usable,
            sqi_qrs_detectability=sqi_qrs_detectability,
            hr_computed=hr_computed,
            hr_value_bpm=hr_value_bpm,
            hr_in_valid_range=hr_in_valid_range,
            morphology_score=morphology_score,
            morphology_confidence=morphology_confidence,
            consecutive_detections=state_context["consecutive_detections"],
            persistence_sec=state_context["persistence_sec"],
            previous_tier=state_context["current_state"] if state_context["current_state"] != "idle" else None,
            current_time=current_time,
            time_since_last_alarm=state_context["time_since_last_alarm"],
            alarms_in_last_hour=state_context["recent_alarm_count"],
        )
        
        # Get decision
        decision = self.policy.decide(decision_input)
        
        # Update state tracker based on decision
        if decision.action == DecisionAction.ALARM:
            self.state_tracker.record_alarm_fired(current_time)
        elif decision.action == DecisionAction.WARNING:
            self.state_tracker.record_warning_issued()
        
        return decision
    
    def _classify_uncertainty(self, uncertainty: float) -> str:
        """Classify uncertainty into tier."""
        if uncertainty < 0.2:
            return "low"
        elif uncertainty < 0.4:
            return "medium"
        elif uncertainty < 0.6:
            return "high"
        else:
            return "very_high"
    
    def reset(self):
        """Reset for new patient/session."""
        self.state_tracker.reset()
        self.policy.reset()


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Decision Machine Demo (v2.4)")
    print("="*60)
    
    # Create pipeline
    pipeline = DetectionPipeline()
    
    # Create test episode
    episode = EpisodeLabel(
        episode_type=EpisodeType.VT_MONOMORPHIC,
        start_time=0.0,
        end_time=2.5,
        confidence=0.85,
    )
    
    # Process with high confidence, good SQI
    print("\n--- Test 1: High confidence VT ---")
    decision = pipeline.process(
        episode=episode,
        calibrated_prob=0.85,
        raw_prob=0.90,
        uncertainty=0.15,
        sqi_score=0.85,
        sqi_is_usable=True,
        sqi_qrs_detectability=0.9,
        hr_computed=True,
        hr_value_bpm=180,
        current_time=100.0,
        morphology_score=0.8,
        morphology_confidence=0.9,
    )
    print(f"Action: {decision.action.value}")
    print(f"Explanation: {decision.explanation}")
    print(f"Confidence: {decision.confidence:.2f}")
    
    # Process with low SQI
    print("\n--- Test 2: Low SQI ---")
    decision = pipeline.process(
        episode=episode,
        calibrated_prob=0.85,
        raw_prob=0.90,
        uncertainty=0.15,
        sqi_score=0.25,
        sqi_is_usable=False,
        sqi_qrs_detectability=0.3,
        hr_computed=False,
        hr_value_bpm=None,
        current_time=200.0,
    )
    print(f"Action: {decision.action.value}")
    print(f"Explanation: {decision.explanation}")
    
    # Process with high uncertainty
    print("\n--- Test 3: High uncertainty ---")
    decision = pipeline.process(
        episode=episode,
        calibrated_prob=0.75,
        raw_prob=0.80,
        uncertainty=0.65,
        sqi_score=0.8,
        sqi_is_usable=True,
        sqi_qrs_detectability=0.85,
        hr_computed=True,
        hr_value_bpm=175,
        current_time=300.0,
    )
    print(f"Action: {decision.action.value}")
    print(f"Explanation: {decision.explanation}")
    
    # Test burst suppression
    print("\n--- Test 4: Burst suppression ---")
    pipeline.reset()
    for i in range(5):
        decision = pipeline.process(
            episode=episode,
            calibrated_prob=0.90,
            raw_prob=0.95,
            uncertainty=0.10,
            sqi_score=0.9,
            sqi_is_usable=True,
            sqi_qrs_detectability=0.95,
            hr_computed=True,
            hr_value_bpm=190,
            current_time=1000.0 + i * 30,  # 30s apart
        )
        print(f"  Alarm {i+1}: {decision.action.value}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
