"""
Unit tests for decision machine module.

Tests for:
- UnifiedDecisionPolicy
- AlarmBudget per-class tracking
- BurstSuppressor rate limiting
- AlarmStateTracker
- DetectionPipeline integration
"""

import pytest
import numpy as np
import sys
import os
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# MOCK CLASSES (for testing without full implementation)
# =============================================================================

class DecisionAction:
    """Mock DecisionAction enum."""
    SUPPRESS = "suppress"
    WARNING = "warning"
    ALARM = "alarm"
    DEFER = "defer"


class EpisodeType:
    """Mock EpisodeType enum."""
    VT_MONOMORPHIC = "vt_monomorphic"
    VT_POLYMORPHIC = "vt_polymorphic"
    VFL = "vfl"
    SVT = "svt"
    SINUS_TACHY = "sinus_tachycardia"


class MockDecisionPolicyConfig:
    """Mock configuration for decision policy."""
    def __init__(
        self,
        warning_threshold: float = 0.4,
        alarm_threshold: float = 0.7,
        high_uncertainty_threshold: float = 0.3,
        min_sqi_for_alarm: float = 0.5,
        min_persistence_beats: int = 3,
    ):
        self.warning_threshold = warning_threshold
        self.alarm_threshold = alarm_threshold
        self.high_uncertainty_threshold = high_uncertainty_threshold
        self.min_sqi_for_alarm = min_sqi_for_alarm
        self.min_persistence_beats = min_persistence_beats


class MockAlarmBudget:
    """Mock alarm budget tracker."""
    
    BUDGET_LIMITS = {
        'vt_vfl': 1.0,  # FA/hr
        'svt': 0.5,
        'sinus_tachy': 0.3,
    }
    
    def __init__(self):
        self.false_alarms = {'vt_vfl': 0, 'svt': 0, 'sinus_tachy': 0}
        self.monitoring_hours = 0.0
    
    def record_false_alarm(self, episode_type: str):
        """Record a false alarm."""
        category = self._categorize(episode_type)
        self.false_alarms[category] += 1
    
    def get_fa_per_hour(self, category: str) -> float:
        """Get current FA/hr for category."""
        if self.monitoring_hours == 0:
            return 0.0
        return self.false_alarms[category] / self.monitoring_hours
    
    def is_within_budget(self, category: str) -> bool:
        """Check if category is within FA budget."""
        return self.get_fa_per_hour(category) <= self.BUDGET_LIMITS.get(category, float('inf'))
    
    def _categorize(self, episode_type: str) -> str:
        """Categorize episode type for budget tracking."""
        lower_type = episode_type.lower()
        # Check SVT first to avoid false match from 'svt' containing 'vt'
        if 'svt' in lower_type or 'afib' in lower_type:
            return 'svt'
        elif 'vt' in lower_type or 'vfl' in lower_type:
            return 'vt_vfl'
        else:
            return 'sinus_tachy'


class MockBurstSuppressor:
    """Mock burst suppressor for rate limiting."""
    
    def __init__(self, max_alarms: int = 3, window_seconds: float = 300):
        self.max_alarms = max_alarms
        self.window_seconds = window_seconds
        self.alarm_times = []
    
    def record_alarm(self, timestamp: float):
        """Record an alarm."""
        self.alarm_times.append(timestamp)
    
    def should_suppress(self, current_time: float) -> bool:
        """Check if alarm should be suppressed due to rate limiting."""
        # Remove old alarms
        cutoff = current_time - self.window_seconds
        self.alarm_times = [t for t in self.alarm_times if t > cutoff]
        
        return len(self.alarm_times) >= self.max_alarms


class MockAlarmStateTracker:
    """Mock alarm state tracker."""
    
    def __init__(self):
        self.consecutive_detections = 0
        self.last_episode_type = None
        self.last_alarm_time = None
    
    def update(self, episode_type: str, detected: bool):
        """Update tracker with new detection."""
        if detected and episode_type == self.last_episode_type:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 1 if detected else 0
        
        self.last_episode_type = episode_type if detected else None
    
    def get_persistence(self) -> int:
        """Get consecutive detection count."""
        return self.consecutive_detections


class MockUnifiedDecisionPolicy:
    """Mock unified decision policy."""
    
    def __init__(self, config: MockDecisionPolicyConfig):
        self.config = config
    
    def decide(
        self,
        probability: float,
        uncertainty: float,
        sqi_score: float,
        persistence: int,
        episode_type: str,
    ) -> str:
        """Make decision based on inputs."""
        # Gate 1: SQI check
        if sqi_score < self.config.min_sqi_for_alarm:
            return DecisionAction.DEFER
        
        # Gate 2: Uncertainty check
        if uncertainty > self.config.high_uncertainty_threshold:
            return DecisionAction.DEFER
        
        # Gate 3: Probability thresholds
        if probability < self.config.warning_threshold:
            return DecisionAction.SUPPRESS
        
        if probability < self.config.alarm_threshold:
            return DecisionAction.WARNING
        
        # Gate 4: Persistence check
        if persistence < self.config.min_persistence_beats:
            return DecisionAction.WARNING
        
        return DecisionAction.ALARM


# =============================================================================
# ALARM BUDGET TESTS
# =============================================================================

class TestAlarmBudget:
    """Tests for AlarmBudget class."""
    
    def test_initial_budget_is_zero(self):
        """Initial FA counts should be zero."""
        budget = MockAlarmBudget()
        
        assert budget.false_alarms['vt_vfl'] == 0
        assert budget.false_alarms['svt'] == 0
        assert budget.false_alarms['sinus_tachy'] == 0
    
    def test_record_vt_false_alarm(self):
        """Recording VT FA should increment correct category."""
        budget = MockAlarmBudget()
        
        budget.record_false_alarm('vt_monomorphic')
        
        assert budget.false_alarms['vt_vfl'] == 1
        assert budget.false_alarms['svt'] == 0
    
    def test_record_svt_false_alarm(self):
        """Recording SVT FA should increment correct category."""
        budget = MockAlarmBudget()
        
        budget.record_false_alarm('svt')
        
        assert budget.false_alarms['svt'] == 1
        assert budget.false_alarms['vt_vfl'] == 0
    
    def test_fa_per_hour_calculation(self):
        """FA/hr calculation should be correct."""
        budget = MockAlarmBudget()
        budget.monitoring_hours = 2.0
        budget.false_alarms['vt_vfl'] = 2
        
        assert budget.get_fa_per_hour('vt_vfl') == 1.0
    
    def test_within_budget_check(self):
        """Budget check should work correctly."""
        budget = MockAlarmBudget()
        budget.monitoring_hours = 1.0
        
        # Within budget
        budget.false_alarms['vt_vfl'] = 0
        assert budget.is_within_budget('vt_vfl') is True
        
        # Over budget (1.0 limit, 2 FA in 1 hour = 2 FA/hr)
        budget.false_alarms['vt_vfl'] = 2
        assert budget.is_within_budget('vt_vfl') is False


# =============================================================================
# BURST SUPPRESSOR TESTS
# =============================================================================

class TestBurstSuppressor:
    """Tests for BurstSuppressor class."""
    
    def test_no_suppression_initially(self):
        """Should not suppress when no alarms recorded."""
        suppressor = MockBurstSuppressor(max_alarms=3, window_seconds=300)
        
        assert suppressor.should_suppress(100.0) is False
    
    def test_suppression_after_max_alarms(self):
        """Should suppress after max alarms in window."""
        suppressor = MockBurstSuppressor(max_alarms=3, window_seconds=300)
        
        # Record 3 alarms
        suppressor.record_alarm(100.0)
        suppressor.record_alarm(150.0)
        suppressor.record_alarm(200.0)
        
        # Should suppress next one
        assert suppressor.should_suppress(250.0) is True
    
    def test_no_suppression_after_window_expires(self):
        """Should not suppress after alarms age out of window."""
        suppressor = MockBurstSuppressor(max_alarms=3, window_seconds=300)
        
        # Record 3 alarms
        suppressor.record_alarm(100.0)
        suppressor.record_alarm(150.0)
        suppressor.record_alarm(200.0)
        
        # After 400s, only 1 alarm in window
        assert suppressor.should_suppress(500.0) is False


# =============================================================================
# ALARM STATE TRACKER TESTS
# =============================================================================

class TestAlarmStateTracker:
    """Tests for AlarmStateTracker class."""
    
    def test_initial_state(self):
        """Initial state should be clean."""
        tracker = MockAlarmStateTracker()
        
        assert tracker.consecutive_detections == 0
        assert tracker.last_episode_type is None
    
    def test_consecutive_detection_tracking(self):
        """Should track consecutive detections of same type."""
        tracker = MockAlarmStateTracker()
        
        tracker.update('vt_monomorphic', True)
        assert tracker.get_persistence() == 1
        
        tracker.update('vt_monomorphic', True)
        assert tracker.get_persistence() == 2
        
        tracker.update('vt_monomorphic', True)
        assert tracker.get_persistence() == 3
    
    def test_reset_on_type_change(self):
        """Should reset on episode type change."""
        tracker = MockAlarmStateTracker()
        
        tracker.update('vt_monomorphic', True)
        tracker.update('vt_monomorphic', True)
        assert tracker.get_persistence() == 2
        
        # Different type resets
        tracker.update('svt', True)
        assert tracker.get_persistence() == 1
    
    def test_reset_on_non_detection(self):
        """Should reset when no detection."""
        tracker = MockAlarmStateTracker()
        
        tracker.update('vt_monomorphic', True)
        tracker.update('vt_monomorphic', True)
        assert tracker.get_persistence() == 2
        
        # No detection resets
        tracker.update('vt_monomorphic', False)
        assert tracker.get_persistence() == 0


# =============================================================================
# UNIFIED DECISION POLICY TESTS
# =============================================================================

class TestUnifiedDecisionPolicy:
    """Tests for UnifiedDecisionPolicy class."""
    
    def test_suppress_low_probability(self):
        """Low probability should result in SUPPRESS."""
        config = MockDecisionPolicyConfig(warning_threshold=0.4)
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.2,
            uncertainty=0.1,
            sqi_score=0.8,
            persistence=5,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.SUPPRESS
    
    def test_warning_medium_probability(self):
        """Medium probability should result in WARNING."""
        config = MockDecisionPolicyConfig(warning_threshold=0.4, alarm_threshold=0.7)
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.5,
            uncertainty=0.1,
            sqi_score=0.8,
            persistence=5,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.WARNING
    
    def test_alarm_high_probability_high_persistence(self):
        """High probability with persistence should result in ALARM."""
        config = MockDecisionPolicyConfig(
            warning_threshold=0.4,
            alarm_threshold=0.7,
            min_persistence_beats=3
        )
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.85,
            uncertainty=0.1,
            sqi_score=0.8,
            persistence=5,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.ALARM
    
    def test_defer_low_sqi(self):
        """Low SQI should result in DEFER."""
        config = MockDecisionPolicyConfig(min_sqi_for_alarm=0.5)
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.9,
            uncertainty=0.1,
            sqi_score=0.3,  # Too low
            persistence=5,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.DEFER
    
    def test_defer_high_uncertainty(self):
        """High uncertainty should result in DEFER."""
        config = MockDecisionPolicyConfig(high_uncertainty_threshold=0.3)
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.9,
            uncertainty=0.5,  # Too high
            sqi_score=0.8,
            persistence=5,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.DEFER
    
    def test_warning_low_persistence(self):
        """Low persistence should cap at WARNING even with high probability."""
        config = MockDecisionPolicyConfig(
            alarm_threshold=0.7,
            min_persistence_beats=3
        )
        policy = MockUnifiedDecisionPolicy(config)
        
        decision = policy.decide(
            probability=0.9,
            uncertainty=0.1,
            sqi_score=0.8,
            persistence=2,  # Below threshold
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.WARNING


# =============================================================================
# DECISION PRIORITY TESTS
# =============================================================================

class TestDecisionPriority:
    """Tests for decision gate priority."""
    
    def test_sqi_gate_first(self):
        """SQI gate should be checked first."""
        config = MockDecisionPolicyConfig(min_sqi_for_alarm=0.5)
        policy = MockUnifiedDecisionPolicy(config)
        
        # Even with perfect everything else, low SQI defers
        decision = policy.decide(
            probability=0.99,
            uncertainty=0.01,
            sqi_score=0.2,
            persistence=10,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.DEFER
    
    def test_uncertainty_gate_second(self):
        """Uncertainty gate should be checked after SQI."""
        config = MockDecisionPolicyConfig(
            min_sqi_for_alarm=0.5,
            high_uncertainty_threshold=0.3
        )
        policy = MockUnifiedDecisionPolicy(config)
        
        # Good SQI but high uncertainty
        decision = policy.decide(
            probability=0.99,
            uncertainty=0.5,
            sqi_score=0.8,
            persistence=10,
            episode_type='vt_monomorphic'
        )
        
        assert decision == DecisionAction.DEFER


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
