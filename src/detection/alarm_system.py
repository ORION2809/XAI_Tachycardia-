"""
Two-Tier Alarm System for Tachycardia Detection.

v2.4: Complete alarm management with per-class budget tracking.

From BUILDABLE_SPEC.md Part 8:
- Two tiers: WARNING (soft alert) → ALARM (hard alarm)
- Per-class FA/hr targets to prevent budget exhaustion
- Rate limiting and cooldown
- Priority-based suppression (VT/VFL > SVT > Sinus)

ARCHITECTURE:
- AlarmConfig: All alarm thresholds and budgets
- AlarmBudgetTracker: Per-class alarm budget tracking
- AlarmStateTracker: Thin state supplier for persistence
- TwoTierAlarmSystem: Orchestrates alarm logic with UnifiedDecisionPolicy
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import time


# =============================================================================
# IMPORTS
# =============================================================================

from .two_lane_pipeline import EpisodeType, DetectedEpisode


# =============================================================================
# ALARM TIER ENUM
# =============================================================================

class AlarmTier(Enum):
    """Alarm tier levels."""
    NONE = "none"
    WARNING = "warning"
    ALARM = "alarm"
    SUPPRESSED = "suppressed"


# =============================================================================
# ALARM CONFIGURATION
# =============================================================================

@dataclass
class AlarmConfig:
    """
    Alarm system configuration.
    
    v2.3 FIX: Per-class FA/hr targets and alarm budget partitioning.
    A system can meet overall FA/hr while producing unacceptably many
    SVT warnings that burn clinician trust and get disabled.
    """
    
    # =========================================================================
    # TIER 1: WARNING THRESHOLDS
    # =========================================================================
    warning_vt_prob: float = 0.5
    warning_svt_prob: float = 0.5
    warning_sinus_tachy_prob: float = 0.6
    warning_consecutive: int = 2  # Number of consecutive detections
    
    # =========================================================================
    # TIER 2: ALARM THRESHOLDS
    # =========================================================================
    alarm_vt_prob: float = 0.7
    alarm_svt_prob: float = 0.65
    alarm_sinus_tachy_prob: float = 0.75
    alarm_consecutive: int = 3  # Number of consecutive detections
    alarm_hr_check: bool = True
    alarm_morphology_check: bool = True
    
    # =========================================================================
    # GLOBAL RATE LIMITING
    # =========================================================================
    max_alarm_rate_per_hour: float = 2.0
    cooldown_after_alarm_sec: float = 30.0
    burst_window_sec: float = 300.0  # 5 minutes
    max_alarms_per_burst: int = 3
    
    # =========================================================================
    # PER-CLASS FA/HR TARGETS (v2.3)
    # =========================================================================
    # CRITICAL: Overall FA/hr isn't enough - need per-class nuisance control
    vt_vfl_max_fa_per_hour: float = 1.0    # Lethal arrhythmia false alarms
    svt_max_fa_per_hour: float = 0.5       # Non-lethal fast rhythms
    sinus_tachy_max_fa_per_hour: float = 0.5  # Usually not clinically urgent
    
    # =========================================================================
    # PER-CLASS PPV FLOORS
    # =========================================================================
    # If PPV drops below these, rate-limit that class
    vt_min_ppv: float = 0.50  # At least 50% of VT alarms should be true
    svt_min_ppv: float = 0.30  # Can tolerate more SVT false alarms
    
    # =========================================================================
    # ALARM BUDGET PRIORITY
    # =========================================================================
    # CRITICAL: VT/VFL gets priority - if budget is limited, suppress SVT first
    priority_order: List[str] = field(
        default_factory=lambda: ['VFL', 'VT_POLYMORPHIC', 'VT_MONOMORPHIC', 
                                  'SVT', 'AFIB_RVR', 'SINUS_TACHY']
    )
    
    # =========================================================================
    # PERSISTENCE REQUIREMENTS
    # =========================================================================
    min_persistence_for_warning_sec: float = 0.3
    min_persistence_for_alarm_sec: float = 0.75
    
    # =========================================================================
    # UNCERTAINTY THRESHOLDS
    # =========================================================================
    max_uncertainty_for_alarm: float = 0.35
    defer_uncertainty_threshold: float = 0.55


# =============================================================================
# ALARM OUTPUT
# =============================================================================

@dataclass
class AlarmOutput:
    """Output from alarm system."""
    tier: AlarmTier
    episode_type: EpisodeType
    confidence: float
    
    # Timing
    timestamp: float
    persistence_sec: float
    consecutive_count: int
    
    # Explanation
    explanation: str
    contributing_factors: Dict[str, float] = field(default_factory=dict)
    
    # Review requirements
    requires_clinician_review: bool = False
    suppress_reason: Optional[str] = None
    
    # Budget info
    budget_remaining: Dict[str, float] = field(default_factory=dict)
    
    def is_alarm(self) -> bool:
        """Check if this is a hard alarm."""
        return self.tier == AlarmTier.ALARM
    
    def is_warning(self) -> bool:
        """Check if this is a warning."""
        return self.tier == AlarmTier.WARNING
    
    def is_suppressed(self) -> bool:
        """Check if this was suppressed."""
        return self.tier == AlarmTier.SUPPRESSED


# =============================================================================
# ALARM BUDGET TRACKER
# =============================================================================

@dataclass
class AlarmBudgetTracker:
    """
    v2.3: Track per-class alarm counts for budget partitioning.
    
    Prevents one class from consuming entire alarm budget.
    """
    
    # Rolling hour alarm counts per class
    vt_vfl_alarms: List[float] = field(default_factory=list)
    svt_alarms: List[float] = field(default_factory=list)
    sinus_tachy_alarms: List[float] = field(default_factory=list)
    
    # PPV tracking for adaptive rate limiting
    vt_vfl_true_positives: int = 0
    vt_vfl_false_positives: int = 0
    svt_true_positives: int = 0
    svt_false_positives: int = 0
    
    def record_alarm(self, episode_type: EpisodeType, timestamp: float):
        """Record an alarm for budget tracking."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            self.vt_vfl_alarms.append(timestamp)
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            self.svt_alarms.append(timestamp)
        elif episode_type == EpisodeType.SINUS_TACHY:
            self.sinus_tachy_alarms.append(timestamp)
    
    def get_hourly_counts(self, current_time: float) -> Dict[str, int]:
        """Get alarm counts for last hour."""
        hour_ago = current_time - 3600
        return {
            'vt_vfl': len([t for t in self.vt_vfl_alarms if t > hour_ago]),
            'svt': len([t for t in self.svt_alarms if t > hour_ago]),
            'sinus_tachy': len([t for t in self.sinus_tachy_alarms if t > hour_ago]),
        }
    
    def get_burst_counts(self, current_time: float, burst_window_sec: float = 300.0) -> Dict[str, int]:
        """Get alarm counts for burst window (last N minutes)."""
        burst_start = current_time - burst_window_sec
        return {
            'vt_vfl': len([t for t in self.vt_vfl_alarms if t > burst_start]),
            'svt': len([t for t in self.svt_alarms if t > burst_start]),
            'sinus_tachy': len([t for t in self.sinus_tachy_alarms if t > burst_start]),
        }
    
    def check_budget_available(
        self,
        episode_type: EpisodeType,
        config: AlarmConfig,
        current_time: float,
    ) -> Tuple[bool, str]:
        """
        Check if alarm budget is available for this episode type.
        
        Returns:
            (available, reason)
        """
        counts = self.get_hourly_counts(current_time)
        total = sum(counts.values())
        
        # VT/VFL priority: always allow if under limit
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            if counts['vt_vfl'] >= config.vt_vfl_max_fa_per_hour:
                return False, f"VT/VFL budget exhausted ({counts['vt_vfl']}/{config.vt_vfl_max_fa_per_hour}/hr)"
            return True, "vt_budget_available"
        
        # SVT: check class-specific AND global budget
        if episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            if counts['svt'] >= config.svt_max_fa_per_hour:
                return False, f"SVT budget exhausted ({counts['svt']}/{config.svt_max_fa_per_hour}/hr)"
            if total >= config.max_alarm_rate_per_hour:
                return False, f"Global budget exhausted ({total}/{config.max_alarm_rate_per_hour}/hr)"
            return True, "svt_budget_available"
        
        # Sinus tachy: lowest priority
        if episode_type == EpisodeType.SINUS_TACHY:
            if counts['sinus_tachy'] >= config.sinus_tachy_max_fa_per_hour:
                return False, f"Sinus tachy budget exhausted"
            # Stricter: only allow if well under global budget
            if total >= config.max_alarm_rate_per_hour * 0.8:
                return False, "Global budget nearly exhausted - suppressing low-priority"
            return True, "sinus_tachy_budget_available"
        
        return True, "unknown_class_allowed"
    
    def get_remaining_budget(
        self,
        config: AlarmConfig,
        current_time: float,
    ) -> Dict[str, float]:
        """Get remaining alarm budget per class."""
        counts = self.get_hourly_counts(current_time)
        return {
            'vt_vfl': max(0, config.vt_vfl_max_fa_per_hour - counts['vt_vfl']),
            'svt': max(0, config.svt_max_fa_per_hour - counts['svt']),
            'sinus_tachy': max(0, config.sinus_tachy_max_fa_per_hour - counts['sinus_tachy']),
            'global': max(0, config.max_alarm_rate_per_hour - sum(counts.values())),
        }
    
    def record_feedback(
        self, 
        episode_type: EpisodeType, 
        was_true_positive: bool
    ):
        """Record clinician feedback for PPV tracking."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            if was_true_positive:
                self.vt_vfl_true_positives += 1
            else:
                self.vt_vfl_false_positives += 1
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            if was_true_positive:
                self.svt_true_positives += 1
            else:
                self.svt_false_positives += 1
    
    def get_ppv(self, episode_type: EpisodeType) -> Optional[float]:
        """Get current PPV for episode type."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            total = self.vt_vfl_true_positives + self.vt_vfl_false_positives
            if total > 0:
                return self.vt_vfl_true_positives / total
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            total = self.svt_true_positives + self.svt_false_positives
            if total > 0:
                return self.svt_true_positives / total
        return None
    
    def prune_old(self, current_time: float, max_age_sec: float = 7200):
        """Remove alarms older than max_age_sec."""
        cutoff = current_time - max_age_sec
        self.vt_vfl_alarms = [t for t in self.vt_vfl_alarms if t > cutoff]
        self.svt_alarms = [t for t in self.svt_alarms if t > cutoff]
        self.sinus_tachy_alarms = [t for t in self.sinus_tachy_alarms if t > cutoff]
    
    def reset(self):
        """Reset all tracking (new patient)."""
        self.vt_vfl_alarms = []
        self.svt_alarms = []
        self.sinus_tachy_alarms = []


# =============================================================================
# ALARM STATE TRACKER
# =============================================================================

class AlarmStateTracker:
    """
    THIN state tracker for alarm persistence.
    
    NOTE: This is NOT a decision engine. It only tracks:
    - Consecutive detection count
    - Alarm history (for rate limiting)
    - Warning/alarm state transitions
    
    All actual decisions go through TwoTierAlarmSystem.
    This class supplies state context.
    """
    
    def __init__(self, config: AlarmConfig = None):
        if config is None:
            config = AlarmConfig()
        self.config = config
        
        # Alarm history (timestamps)
        self.alarm_history: List[float] = []
        self.warning_history: List[float] = []
        
        # Detection persistence
        self.consecutive_count: int = 0
        self.first_detection_time: Optional[float] = None
        
        # Current state
        self.current_tier: AlarmTier = AlarmTier.NONE
        self.current_episode_type: Optional[EpisodeType] = None
    
    def update_on_detection(
        self,
        is_detection: bool,
        episode_type: Optional[EpisodeType],
        current_time: float,
    ) -> Dict[str, Any]:
        """
        Update state tracking on each detection window.
        
        Returns context for alarm decisions, NOT a decision.
        """
        if is_detection:
            # Check if same episode type or new
            if episode_type != self.current_episode_type:
                # New episode type, reset consecutive count
                self.consecutive_count = 1
                self.first_detection_time = current_time
                self.current_episode_type = episode_type
            else:
                # Same type, increment
                if self.consecutive_count == 0:
                    self.first_detection_time = current_time
                self.consecutive_count += 1
        else:
            # No detection, reset
            self.consecutive_count = 0
            self.first_detection_time = None
            self.current_tier = AlarmTier.NONE
            self.current_episode_type = None
        
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
            "current_tier": self.current_tier,
            "current_episode_type": self.current_episode_type,
        }
    
    def record_alarm_fired(self, current_time: float):
        """Record that an alarm was fired."""
        self.alarm_history.append(current_time)
        self.current_tier = AlarmTier.ALARM
        # Prune old alarms (keep last 24 hours)
        self.alarm_history = [t for t in self.alarm_history 
                             if current_time - t < 86400]
    
    def record_warning_issued(self, current_time: float):
        """Record transition to warning state."""
        self.warning_history.append(current_time)
        self.current_tier = AlarmTier.WARNING
        # Prune old warnings
        self.warning_history = [t for t in self.warning_history 
                               if current_time - t < 86400]
    
    def check_cooldown(self, current_time: float) -> Tuple[bool, float]:
        """
        Check if cooldown is active.
        
        Returns:
            (is_active, time_remaining)
        """
        if not self.alarm_history:
            return False, 0.0
        time_since_last = current_time - self.alarm_history[-1]
        if time_since_last < self.config.cooldown_after_alarm_sec:
            return True, self.config.cooldown_after_alarm_sec - time_since_last
        return False, 0.0
    
    def get_state_context(self, current_time: float) -> Dict[str, Any]:
        """Get full state context for alarm decisions."""
        recent_alarms = [t for t in self.alarm_history 
                        if current_time - t < 3600]
        
        persistence_sec = 0.0
        if self.first_detection_time is not None:
            persistence_sec = current_time - self.first_detection_time
        
        cooldown_active, cooldown_remaining = self.check_cooldown(current_time)
        
        return {
            "consecutive_detections": self.consecutive_count,
            "persistence_sec": persistence_sec,
            "recent_alarm_count": len(recent_alarms),
            "rate_limit_active": len(recent_alarms) >= self.config.max_alarm_rate_per_hour,
            "cooldown_active": cooldown_active,
            "cooldown_remaining_sec": cooldown_remaining,
            "current_tier": self.current_tier,
            "current_episode_type": self.current_episode_type,
        }
    
    def reset(self):
        """Reset all state (new patient)."""
        self.consecutive_count = 0
        self.first_detection_time = None
        self.current_tier = AlarmTier.NONE
        self.current_episode_type = None
        # Note: alarm_history retained for rate limiting
    
    def reset_all(self):
        """Full reset including history."""
        self.reset()
        self.alarm_history = []
        self.warning_history = []


# =============================================================================
# TWO-TIER ALARM SYSTEM
# =============================================================================

class TwoTierAlarmSystem:
    """
    Two-tier alarm system with per-class budget tracking.
    
    Tier 1 (WARNING): Low threshold, short duration
    Tier 2 (ALARM): High threshold, longer duration, HR check
    
    Features:
    - Per-class FA/hr budget tracking
    - Priority-based suppression (VT > SVT > Sinus)
    - Cooldown after alarms
    - Rate limiting
    - Uncertainty-based deferral
    
    This is the main entry point for alarm decisions.
    """
    
    def __init__(self, config: AlarmConfig = None):
        if config is None:
            config = AlarmConfig()
        self.config = config
        
        # State trackers
        self.state_tracker = AlarmStateTracker(config)
        self.budget_tracker = AlarmBudgetTracker()
        
        # Per-class state (for independent tracking)
        self.per_class_state: Dict[EpisodeType, AlarmStateTracker] = {}
    
    def _get_class_state(self, episode_type: EpisodeType) -> AlarmStateTracker:
        """Get or create state tracker for episode type."""
        if episode_type not in self.per_class_state:
            self.per_class_state[episode_type] = AlarmStateTracker(self.config)
        return self.per_class_state[episode_type]
    
    def _get_warning_threshold(self, episode_type: EpisodeType) -> float:
        """Get warning probability threshold for episode type."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            return self.config.warning_vt_prob
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            return self.config.warning_svt_prob
        elif episode_type == EpisodeType.SINUS_TACHY:
            return self.config.warning_sinus_tachy_prob
        return 0.5  # Default
    
    def _get_alarm_threshold(self, episode_type: EpisodeType) -> float:
        """Get alarm probability threshold for episode type."""
        if episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                           EpisodeType.VFL, EpisodeType.VFIB):
            return self.config.alarm_vt_prob
        elif episode_type in (EpisodeType.SVT, EpisodeType.AFIB_RVR, EpisodeType.AFLUTTER):
            return self.config.alarm_svt_prob
        elif episode_type == EpisodeType.SINUS_TACHY:
            return self.config.alarm_sinus_tachy_prob
        return 0.7  # Default
    
    def evaluate(
        self,
        episode: DetectedEpisode,
        sqi_score: float,
        sqi_is_usable: bool,
        uncertainty: float = 0.0,
        hr_bpm: Optional[float] = None,
        morphology_score: float = 0.5,
        current_time: Optional[float] = None,
    ) -> AlarmOutput:
        """
        Evaluate an episode for alarm/warning.
        
        Args:
            episode: Detected episode from EpisodeDetector
            sqi_score: Signal quality score (0-1)
            sqi_is_usable: Whether signal quality is usable
            uncertainty: Model uncertainty (0-1)
            hr_bpm: Computed heart rate (optional)
            morphology_score: QRS morphology score (0=narrow, 1=wide)
            current_time: Current timestamp (defaults to time.time())
        
        Returns:
            AlarmOutput with tier, explanation, and budget info
        """
        if current_time is None:
            current_time = time.time()
        
        factors: Dict[str, float] = {}
        
        # =====================================================================
        # STAGE 1: HARD GATES
        # =====================================================================
        
        # Gate 1: SQI unusable (bypass for VF at high confidence)
        vf_bypass = (episode.episode_type == EpisodeType.VFIB and 
                    episode.confidence > 0.8)
        
        if not sqi_is_usable and not vf_bypass:
            return AlarmOutput(
                tier=AlarmTier.SUPPRESSED,
                episode_type=episode.episode_type,
                confidence=0.0,
                timestamp=current_time,
                persistence_sec=0.0,
                consecutive_count=0,
                explanation="Signal quality too poor for reliable detection",
                contributing_factors={"sqi_unusable": 1.0},
                suppress_reason="sqi_unusable",
            )
        
        # Gate 2: Cooldown check
        cooldown_active, cooldown_remaining = self.state_tracker.check_cooldown(current_time)
        if cooldown_active:
            # Allow VT/VFL to bypass cooldown if very high confidence
            vt_bypass = (episode.episode_type in (EpisodeType.VT_MONOMORPHIC, 
                                                   EpisodeType.VT_POLYMORPHIC,
                                                   EpisodeType.VFL, EpisodeType.VFIB) 
                        and episode.confidence > 0.9)
            
            if not vt_bypass:
                return AlarmOutput(
                    tier=AlarmTier.SUPPRESSED,
                    episode_type=episode.episode_type,
                    confidence=episode.confidence,
                    timestamp=current_time,
                    persistence_sec=0.0,
                    consecutive_count=0,
                    explanation=f"Cooldown active ({cooldown_remaining:.1f}s remaining)",
                    contributing_factors={"cooldown": 1.0},
                    suppress_reason="cooldown",
                )
        
        # Gate 3: Budget check
        budget_available, budget_reason = self.budget_tracker.check_budget_available(
            episode.episode_type, self.config, current_time
        )
        if not budget_available:
            return AlarmOutput(
                tier=AlarmTier.SUPPRESSED,
                episode_type=episode.episode_type,
                confidence=episode.confidence,
                timestamp=current_time,
                persistence_sec=0.0,
                consecutive_count=0,
                explanation=budget_reason,
                contributing_factors={"budget_exhausted": 1.0},
                suppress_reason="budget_exhausted",
                requires_clinician_review=True,  # Clinician should know
            )
        
        # =====================================================================
        # STAGE 2: UPDATE STATE TRACKING
        # =====================================================================
        
        # Update global state
        is_detection = episode.confidence > 0.3  # Any meaningful detection
        state = self.state_tracker.update_on_detection(
            is_detection, episode.episode_type, current_time
        )
        
        # Get per-class state
        class_state_tracker = self._get_class_state(episode.episode_type)
        class_state = class_state_tracker.update_on_detection(
            is_detection, episode.episode_type, current_time
        )
        
        factors["consecutive_count"] = class_state["consecutive_count"]
        factors["persistence_sec"] = class_state["persistence_sec"]
        
        # =====================================================================
        # STAGE 3: COMPUTE ALARM TIER
        # =====================================================================
        
        warning_threshold = self._get_warning_threshold(episode.episode_type)
        alarm_threshold = self._get_alarm_threshold(episode.episode_type)
        
        factors["episode_confidence"] = episode.confidence
        factors["warning_threshold"] = warning_threshold
        factors["alarm_threshold"] = alarm_threshold
        factors["sqi_score"] = sqi_score
        factors["uncertainty"] = uncertainty
        
        # Check for alarm tier (highest priority)
        if self._check_alarm_conditions(
            episode=episode,
            alarm_threshold=alarm_threshold,
            consecutive_count=class_state["consecutive_count"],
            persistence_sec=class_state["persistence_sec"],
            uncertainty=uncertainty,
            hr_bpm=hr_bpm,
            morphology_score=morphology_score,
            sqi_score=sqi_score,
        ):
            # Record alarm
            self.state_tracker.record_alarm_fired(current_time)
            class_state_tracker.record_alarm_fired(current_time)
            self.budget_tracker.record_alarm(episode.episode_type, current_time)
            
            return AlarmOutput(
                tier=AlarmTier.ALARM,
                episode_type=episode.episode_type,
                confidence=episode.confidence,
                timestamp=current_time,
                persistence_sec=class_state["persistence_sec"],
                consecutive_count=class_state["consecutive_count"],
                explanation=self._build_alarm_explanation(episode, class_state, factors),
                contributing_factors=factors,
                budget_remaining=self.budget_tracker.get_remaining_budget(
                    self.config, current_time
                ),
            )
        
        # Check for warning tier
        if self._check_warning_conditions(
            episode=episode,
            warning_threshold=warning_threshold,
            consecutive_count=class_state["consecutive_count"],
            persistence_sec=class_state["persistence_sec"],
            uncertainty=uncertainty,
        ):
            # Record warning
            self.state_tracker.record_warning_issued(current_time)
            class_state_tracker.record_warning_issued(current_time)
            
            return AlarmOutput(
                tier=AlarmTier.WARNING,
                episode_type=episode.episode_type,
                confidence=episode.confidence,
                timestamp=current_time,
                persistence_sec=class_state["persistence_sec"],
                consecutive_count=class_state["consecutive_count"],
                explanation=self._build_warning_explanation(episode, class_state, factors),
                contributing_factors=factors,
                budget_remaining=self.budget_tracker.get_remaining_budget(
                    self.config, current_time
                ),
            )
        
        # No alarm or warning
        return AlarmOutput(
            tier=AlarmTier.NONE,
            episode_type=episode.episode_type,
            confidence=episode.confidence,
            timestamp=current_time,
            persistence_sec=class_state["persistence_sec"],
            consecutive_count=class_state["consecutive_count"],
            explanation="Detection below threshold or insufficient persistence",
            contributing_factors=factors,
            budget_remaining=self.budget_tracker.get_remaining_budget(
                self.config, current_time
            ),
        )
    
    def _check_warning_conditions(
        self,
        episode: DetectedEpisode,
        warning_threshold: float,
        consecutive_count: int,
        persistence_sec: float,
        uncertainty: float,
    ) -> bool:
        """Check if warning conditions are met."""
        # Probability threshold
        if episode.confidence < warning_threshold:
            return False
        
        # Consecutive count
        if consecutive_count < self.config.warning_consecutive:
            return False
        
        # Persistence
        if persistence_sec < self.config.min_persistence_for_warning_sec:
            return False
        
        # High uncertainty blocks warning
        if uncertainty > self.config.defer_uncertainty_threshold:
            return False
        
        return True
    
    def _check_alarm_conditions(
        self,
        episode: DetectedEpisode,
        alarm_threshold: float,
        consecutive_count: int,
        persistence_sec: float,
        uncertainty: float,
        hr_bpm: Optional[float],
        morphology_score: float,
        sqi_score: float,
    ) -> bool:
        """Check if alarm conditions are met."""
        # Probability threshold
        if episode.confidence < alarm_threshold:
            return False
        
        # Consecutive count
        if consecutive_count < self.config.alarm_consecutive:
            return False
        
        # Persistence
        if persistence_sec < self.config.min_persistence_for_alarm_sec:
            return False
        
        # Uncertainty threshold
        if uncertainty > self.config.max_uncertainty_for_alarm:
            return False
        
        # HR check for VT (optional)
        if self.config.alarm_hr_check:
            if episode.episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC):
                if hr_bpm is not None:
                    # VT should have HR > 100 bpm
                    if hr_bpm < 100:
                        return False
        
        # Morphology check for VT (optional, soft)
        if self.config.alarm_morphology_check:
            if episode.episode_type in (EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC):
                # Wide QRS expected for VT
                if morphology_score < 0.3:
                    # Low morphology score (narrow QRS) - reduce confidence
                    # This is a soft check, not a hard gate
                    pass
        
        # SQI should be good enough
        if sqi_score < 0.5:
            return False
        
        return True
    
    def _build_alarm_explanation(
        self,
        episode: DetectedEpisode,
        state: Dict[str, Any],
        factors: Dict[str, float],
    ) -> str:
        """Build human-readable alarm explanation."""
        ep_name = episode.episode_type.name if hasattr(episode.episode_type, 'name') else str(episode.episode_type)
        return (
            f"ALARM: {ep_name} detected with {episode.confidence:.0%} confidence. "
            f"Persisted for {state['persistence_sec']:.1f}s over {state['consecutive_count']} beats. "
            f"SQI={factors.get('sqi_score', 0):.2f}, uncertainty={factors.get('uncertainty', 0):.2f}."
        )
    
    def _build_warning_explanation(
        self,
        episode: DetectedEpisode,
        state: Dict[str, Any],
        factors: Dict[str, float],
    ) -> str:
        """Build human-readable warning explanation."""
        ep_name = episode.episode_type.name if hasattr(episode.episode_type, 'name') else str(episode.episode_type)
        return (
            f"WARNING: {ep_name} detected with {episode.confidence:.0%} confidence. "
            f"Monitoring - {state['consecutive_count']} consecutive detections. "
            f"Continue observation for possible escalation to ALARM."
        )
    
    def get_budget_status(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        """Get current alarm budget status."""
        if current_time is None:
            current_time = time.time()
        
        remaining = self.budget_tracker.get_remaining_budget(self.config, current_time)
        hourly = self.budget_tracker.get_hourly_counts(current_time)
        
        return {
            "remaining": remaining,
            "used_this_hour": hourly,
            "limits": {
                "vt_vfl": self.config.vt_vfl_max_fa_per_hour,
                "svt": self.config.svt_max_fa_per_hour,
                "sinus_tachy": self.config.sinus_tachy_max_fa_per_hour,
                "global": self.config.max_alarm_rate_per_hour,
            },
        }
    
    def record_feedback(
        self,
        episode_type: EpisodeType,
        was_true_positive: bool,
    ):
        """Record clinician feedback for PPV tracking."""
        self.budget_tracker.record_feedback(episode_type, was_true_positive)
    
    def reset(self):
        """Reset for new patient (keep budget tracking)."""
        self.state_tracker.reset()
        self.per_class_state.clear()
    
    def reset_all(self):
        """Full reset including budget."""
        self.state_tracker.reset_all()
        self.budget_tracker.reset()
        self.per_class_state.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_alarm_system(
    max_alarm_rate_per_hour: float = 2.0,
    cooldown_sec: float = 30.0,
    vt_vfl_limit: float = 1.0,
    svt_limit: float = 0.5,
) -> TwoTierAlarmSystem:
    """
    Create alarm system with common configuration.
    
    Args:
        max_alarm_rate_per_hour: Global alarm rate limit
        cooldown_sec: Cooldown period after alarm
        vt_vfl_limit: VT/VFL hourly limit
        svt_limit: SVT hourly limit
    
    Returns:
        Configured TwoTierAlarmSystem
    """
    config = AlarmConfig(
        max_alarm_rate_per_hour=max_alarm_rate_per_hour,
        cooldown_after_alarm_sec=cooldown_sec,
        vt_vfl_max_fa_per_hour=vt_vfl_limit,
        svt_max_fa_per_hour=svt_limit,
    )
    return TwoTierAlarmSystem(config)


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("TWO-TIER ALARM SYSTEM DEMO")
    print("=" * 60)
    
    # Create alarm system
    alarm_system = create_alarm_system()
    
    # Create test episodes
    current_time = time.time()
    
    # Episode 1: VT with high confidence
    episode1 = DetectedEpisode(
        start_sample=0,
        end_sample=1000,
        start_time_sec=0.0,
        end_time_sec=2.5,
        episode_type=EpisodeType.VT_MONOMORPHIC,
        severity="high",
        confidence=0.85,
    )
    
    print("\n--- Episode 1: VT at 85% confidence ---")
    
    # Simulate multiple consecutive detections
    for i in range(5):
        result = alarm_system.evaluate(
            episode=episode1,
            sqi_score=0.8,
            sqi_is_usable=True,
            uncertainty=0.1,
            hr_bpm=180.0,
            morphology_score=0.7,
            current_time=current_time + i * 0.5,
        )
        print(f"  Detection {i+1}: {result.tier.value} - {result.explanation[:50]}...")
    
    # Episode 2: SVT with medium confidence
    episode2 = DetectedEpisode(
        start_sample=1000,
        end_sample=2000,
        start_time_sec=2.5,
        end_time_sec=5.0,
        episode_type=EpisodeType.SVT,
        severity="medium",
        confidence=0.65,
    )
    
    print("\n--- Episode 2: SVT at 65% confidence (during cooldown) ---")
    result = alarm_system.evaluate(
        episode=episode2,
        sqi_score=0.7,
        sqi_is_usable=True,
        uncertainty=0.2,
        current_time=current_time + 3.0,
    )
    print(f"  Result: {result.tier.value}")
    print(f"  Explanation: {result.explanation}")
    
    # Budget status
    print("\n--- Budget Status ---")
    status = alarm_system.get_budget_status(current_time + 5.0)
    print(f"  Used this hour: {status['used_this_hour']}")
    print(f"  Remaining: {status['remaining']}")
    
    # Episode 3: Low SQI
    episode3 = DetectedEpisode(
        start_sample=2000,
        end_sample=3000,
        start_time_sec=5.0,
        end_time_sec=7.5,
        episode_type=EpisodeType.VT_MONOMORPHIC,
        severity="high",
        confidence=0.9,
    )
    
    print("\n--- Episode 3: VT at 90% confidence but low SQI ---")
    result = alarm_system.evaluate(
        episode=episode3,
        sqi_score=0.2,
        sqi_is_usable=False,
        uncertainty=0.1,
        current_time=current_time + 60.0,  # After cooldown
    )
    print(f"  Result: {result.tier.value}")
    print(f"  Suppress reason: {result.suppress_reason}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
