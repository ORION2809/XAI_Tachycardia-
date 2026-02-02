"""
Signal State Machine for XAI Tachycardia Detection.

v2.4: Formal state machine for artifact handling.

CRITICAL: Artifact handling is a formal requirement, not an afterthought.
The system must maintain explicit signal quality state and transition
between states with defined rules to prevent:
1. Alarm fatigue from artifact-triggered false alarms
2. Missing true arrhythmias during brief artifact
3. State "flapping" that causes inconsistent behavior

State Machine:
- GOOD: Normal operation, all alarms active
- MARGINAL: Reduced confidence, warnings only for non-critical
- SIGNAL_POOR: Artifact-dominated, suppress non-critical alarms
- LEADS_OFF: No signal, all alarms suppressed except technical

Transitions have minimum duration requirements to prevent flapping.
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import time


# =============================================================================
# SIGNAL STATE ENUM
# =============================================================================

class SignalState(Enum):
    """
    Signal quality state machine states.
    
    Each state has different alarm policies to balance
    sensitivity against false alarm reduction.
    """
    GOOD = "good"                   # Normal operation, all alarms active
    MARGINAL = "marginal"           # Reduced confidence, warnings only for non-critical
    SIGNAL_POOR = "signal_poor"     # Artifact-dominated, suppress non-critical alarms
    LEADS_OFF = "leads_off"         # No signal, all alarms suppressed except technical


# =============================================================================
# STATE TRANSITION RULES
# =============================================================================

@dataclass
class SignalStateTransition:
    """
    Rules for signal state transitions.
    
    Each transition has:
    - from_state: Starting state
    - to_state: Target state
    - sqi_threshold: SQI value that triggers this transition
    - min_duration_sec: Must persist for this long before transition completes
    """
    from_state: SignalState
    to_state: SignalState
    sqi_threshold: float            # SQI value that triggers transition
    comparison: str                 # "below" or "above"
    min_duration_sec: float         # Must persist for this long
    description: str = ""


# Standard transition rules
SIGNAL_STATE_TRANSITIONS = [
    # ========= Degradation Transitions =========
    
    # GOOD → MARGINAL (moderate degradation)
    SignalStateTransition(
        from_state=SignalState.GOOD,
        to_state=SignalState.MARGINAL,
        sqi_threshold=0.6,
        comparison="below",
        min_duration_sec=2.0,
        description="SQI below 0.6 for 2 seconds",
    ),
    
    # GOOD → SIGNAL_POOR (rapid degradation)
    SignalStateTransition(
        from_state=SignalState.GOOD,
        to_state=SignalState.SIGNAL_POOR,
        sqi_threshold=0.3,
        comparison="below",
        min_duration_sec=1.0,
        description="SQI below 0.3 for 1 second (rapid degradation)",
    ),
    
    # MARGINAL → SIGNAL_POOR (continued degradation)
    SignalStateTransition(
        from_state=SignalState.MARGINAL,
        to_state=SignalState.SIGNAL_POOR,
        sqi_threshold=0.4,
        comparison="below",
        min_duration_sec=2.0,
        description="SQI below 0.4 for 2 seconds",
    ),
    
    # ANY → LEADS_OFF (signal loss)
    SignalStateTransition(
        from_state=SignalState.GOOD,
        to_state=SignalState.LEADS_OFF,
        sqi_threshold=0.1,
        comparison="below",
        min_duration_sec=1.0,
        description="Near-zero signal for 1 second",
    ),
    SignalStateTransition(
        from_state=SignalState.MARGINAL,
        to_state=SignalState.LEADS_OFF,
        sqi_threshold=0.1,
        comparison="below",
        min_duration_sec=1.0,
        description="Near-zero signal for 1 second",
    ),
    SignalStateTransition(
        from_state=SignalState.SIGNAL_POOR,
        to_state=SignalState.LEADS_OFF,
        sqi_threshold=0.1,
        comparison="below",
        min_duration_sec=1.0,
        description="Near-zero signal for 1 second",
    ),
    
    # ========= Recovery Transitions =========
    # Recovery is SLOWER than degradation to prevent flapping
    
    # SIGNAL_POOR → MARGINAL (partial recovery)
    SignalStateTransition(
        from_state=SignalState.SIGNAL_POOR,
        to_state=SignalState.MARGINAL,
        sqi_threshold=0.5,
        comparison="above",
        min_duration_sec=3.0,
        description="SQI above 0.5 for 3 seconds (slow recovery)",
    ),
    
    # MARGINAL → GOOD (full recovery)
    SignalStateTransition(
        from_state=SignalState.MARGINAL,
        to_state=SignalState.GOOD,
        sqi_threshold=0.7,
        comparison="above",
        min_duration_sec=3.0,
        description="SQI above 0.7 for 3 seconds (slow recovery)",
    ),
    
    # LEADS_OFF → SIGNAL_POOR (signal return)
    SignalStateTransition(
        from_state=SignalState.LEADS_OFF,
        to_state=SignalState.SIGNAL_POOR,
        sqi_threshold=0.2,
        comparison="above",
        min_duration_sec=2.0,
        description="Signal returned (SQI > 0.2) for 2 seconds",
    ),
    
    # Direct LEADS_OFF → MARGINAL (fast recovery)
    SignalStateTransition(
        from_state=SignalState.LEADS_OFF,
        to_state=SignalState.MARGINAL,
        sqi_threshold=0.5,
        comparison="above",
        min_duration_sec=2.0,
        description="Good signal returned (SQI > 0.5) for 2 seconds",
    ),
]


# =============================================================================
# ALARM POLICY PER STATE
# =============================================================================

@dataclass
class AlarmPolicy:
    """Alarm policy for a given signal state."""
    vt_alarm_enabled: bool
    vfl_alarm_enabled: bool
    svt_alarm_enabled: bool
    sinus_tachy_enabled: bool
    confidence_penalty: float       # Reduce confidence by this amount
    defer_to_clinician: bool        # Force clinician review
    suppress_reason: Optional[str]  # If suppressed, why
    
    def is_any_alarm_enabled(self) -> bool:
        """Check if any alarm type is enabled."""
        return (self.vt_alarm_enabled or self.vfl_alarm_enabled or 
                self.svt_alarm_enabled or self.sinus_tachy_enabled)


# Alarm policies per state
ALARM_POLICIES: Dict[SignalState, AlarmPolicy] = {
    SignalState.GOOD: AlarmPolicy(
        vt_alarm_enabled=True,
        vfl_alarm_enabled=True,
        svt_alarm_enabled=True,
        sinus_tachy_enabled=True,
        confidence_penalty=0.0,
        defer_to_clinician=False,
        suppress_reason=None,
    ),
    SignalState.MARGINAL: AlarmPolicy(
        vt_alarm_enabled=True,      # VT always enabled
        vfl_alarm_enabled=True,     # VFL always enabled
        svt_alarm_enabled=True,     # SVT enabled but penalized
        sinus_tachy_enabled=False,  # Suppress low-priority
        confidence_penalty=0.15,
        defer_to_clinician=False,
        suppress_reason="sinus_tachy_suppressed_marginal_sqi",
    ),
    SignalState.SIGNAL_POOR: AlarmPolicy(
        vt_alarm_enabled=True,      # VT NEVER suppressed (goes to DEFER)
        vfl_alarm_enabled=True,     # VFL NEVER suppressed
        svt_alarm_enabled=False,    # Suppress
        sinus_tachy_enabled=False,  # Suppress
        confidence_penalty=0.30,
        defer_to_clinician=True,    # Clinician review for VT/VFL
        suppress_reason="non_critical_suppressed_poor_sqi",
    ),
    SignalState.LEADS_OFF: AlarmPolicy(
        vt_alarm_enabled=False,     # Cannot detect with no signal
        vfl_alarm_enabled=False,
        svt_alarm_enabled=False,
        sinus_tachy_enabled=False,
        confidence_penalty=1.0,
        defer_to_clinician=False,   # Generate technical alarm instead
        suppress_reason="leads_off",
    ),
}


# =============================================================================
# SQI RESULT (Lightweight for state machine)
# =============================================================================

@dataclass
class SQIInput:
    """
    Input to signal state manager.
    
    This is a lightweight structure - the full SQIResult is in quality/sqi.py.
    """
    overall_score: float            # 0-1 overall SQI
    is_usable: bool                 # Quick usability flag
    components: Dict[str, float] = field(default_factory=dict)
    
    # Optional specific components
    flatline_ratio: float = 0.0     # Fraction of flatline samples
    noise_power: float = 0.0        # Noise power estimate
    qrs_detectability: float = 1.0  # QRS detection confidence


# =============================================================================
# SIGNAL STATE MANAGER
# =============================================================================

class SignalStateManager:
    """
    Manage signal quality state transitions.
    
    v2.4: Formal state machine for artifact handling.
    
    This manager:
    1. Tracks current signal quality state
    2. Handles state transitions with hysteresis
    3. Provides alarm policies per state
    4. Logs state history for audit
    
    Usage:
        manager = SignalStateManager()
        
        # Update with each SQI computation
        current_state = manager.update(sqi_input, current_time)
        
        # Get alarm policy for current state
        policy = manager.get_alarm_policy()
    """
    
    def __init__(
        self,
        initial_state: SignalState = SignalState.GOOD,
        transitions: List[SignalStateTransition] = None,
        custom_policies: Dict[SignalState, AlarmPolicy] = None,
    ):
        """
        Initialize signal state manager.
        
        Args:
            initial_state: Starting state
            transitions: Custom transition rules (default: SIGNAL_STATE_TRANSITIONS)
            custom_policies: Custom alarm policies per state
        """
        self.transitions = transitions or SIGNAL_STATE_TRANSITIONS
        self.policies = custom_policies or ALARM_POLICIES
        
        # Current state
        self.current_state = initial_state
        self.state_entry_time: Optional[float] = None
        
        # Pending transition tracking
        self.pending_transition: Optional[SignalState] = None
        self.pending_transition_start: Optional[float] = None
        
        # History for audit
        self.state_history: List[Dict[str, Any]] = []
        self.max_history_length: int = 1000
    
    def update(
        self,
        sqi: SQIInput,
        current_time: float,
    ) -> SignalState:
        """
        Update state based on SQI.
        
        Handles transition logic with hysteresis to prevent flapping.
        
        Args:
            sqi: Current SQI input
            current_time: Current timestamp (seconds)
            
        Returns:
            Current signal state after update
        """
        # Determine target state based on SQI
        target_state = self._determine_target_state(sqi)
        
        # Handle state transition
        if target_state != self.current_state:
            # Check if we should transition
            should_transition, reason = self._check_transition(
                self.current_state, target_state, current_time
            )
            
            if should_transition:
                # Execute transition
                old_state = self.current_state
                self.current_state = target_state
                self.state_entry_time = current_time
                self.pending_transition = None
                self.pending_transition_start = None
                
                # Log transition
                self._log_transition(old_state, target_state, current_time, reason, sqi)
        else:
            # Stable state, clear pending
            self.pending_transition = None
            self.pending_transition_start = None
        
        return self.current_state
    
    def _determine_target_state(self, sqi: SQIInput) -> SignalState:
        """Determine target state based on SQI value."""
        # Check for leads off first (flatline)
        if sqi.flatline_ratio > 0.9 or sqi.overall_score < 0.1:
            return SignalState.LEADS_OFF
        
        # Thresholds for state determination
        if sqi.overall_score < 0.3:
            return SignalState.SIGNAL_POOR
        elif sqi.overall_score < 0.6:
            return SignalState.MARGINAL
        else:
            return SignalState.GOOD
    
    def _check_transition(
        self,
        from_state: SignalState,
        to_state: SignalState,
        current_time: float,
    ) -> Tuple[bool, str]:
        """
        Check if transition should complete.
        
        Returns:
            (should_transition, reason)
        """
        # Find applicable transition rule
        rule = self._get_transition_rule(from_state, to_state)
        
        if rule is None:
            # No direct rule, allow transition if moving to adjacent state
            return True, "no_rule_default_allow"
        
        # Check if this is a new pending transition
        if self.pending_transition != to_state:
            self.pending_transition = to_state
            self.pending_transition_start = current_time
            return False, f"transition_pending_{rule.min_duration_sec}s"
        
        # Check if pending transition has met duration requirement
        elapsed = current_time - self.pending_transition_start
        if elapsed >= rule.min_duration_sec:
            return True, f"duration_met_{elapsed:.1f}s"
        else:
            return False, f"waiting_{rule.min_duration_sec - elapsed:.1f}s"
    
    def _get_transition_rule(
        self,
        from_state: SignalState,
        to_state: SignalState,
    ) -> Optional[SignalStateTransition]:
        """Get transition rule for state pair."""
        for rule in self.transitions:
            if rule.from_state == from_state and rule.to_state == to_state:
                return rule
        return None
    
    def _log_transition(
        self,
        from_state: SignalState,
        to_state: SignalState,
        timestamp: float,
        reason: str,
        sqi: SQIInput,
    ):
        """Log state transition for audit."""
        entry = {
            "from_state": from_state.value,
            "to_state": to_state.value,
            "timestamp": timestamp,
            "reason": reason,
            "sqi_score": sqi.overall_score,
        }
        
        self.state_history.append(entry)
        
        # Trim history if too long
        if len(self.state_history) > self.max_history_length:
            self.state_history = self.state_history[-self.max_history_length:]
    
    def get_alarm_policy(self) -> AlarmPolicy:
        """Get alarm policy for current state."""
        return self.policies[self.current_state]
    
    def is_vt_enabled(self) -> bool:
        """Quick check if VT alarms are enabled."""
        return self.policies[self.current_state].vt_alarm_enabled
    
    def is_svt_enabled(self) -> bool:
        """Quick check if SVT alarms are enabled."""
        return self.policies[self.current_state].svt_alarm_enabled
    
    def get_confidence_penalty(self) -> float:
        """Get confidence penalty for current state."""
        return self.policies[self.current_state].confidence_penalty
    
    def should_defer_to_clinician(self) -> bool:
        """Check if current state requires clinician review."""
        return self.policies[self.current_state].defer_to_clinician
    
    def get_state_context(self, current_time: float) -> Dict[str, Any]:
        """Get full state context for decision input."""
        time_in_state = 0.0
        if self.state_entry_time is not None:
            time_in_state = current_time - self.state_entry_time
        
        policy = self.get_alarm_policy()
        
        return {
            "current_state": self.current_state.value,
            "time_in_state_sec": time_in_state,
            "pending_transition": self.pending_transition.value if self.pending_transition else None,
            "vt_enabled": policy.vt_alarm_enabled,
            "svt_enabled": policy.svt_alarm_enabled,
            "confidence_penalty": policy.confidence_penalty,
            "defer_to_clinician": policy.defer_to_clinician,
            "transition_history_length": len(self.state_history),
        }
    
    def get_recent_transitions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get last N state transitions."""
        return self.state_history[-n:]
    
    def get_state_duration_stats(self) -> Dict[str, float]:
        """Get statistics on time spent in each state."""
        stats = {state.value: 0.0 for state in SignalState}
        
        if len(self.state_history) < 2:
            return stats
        
        for i in range(len(self.state_history) - 1):
            entry = self.state_history[i]
            next_entry = self.state_history[i + 1]
            duration = next_entry["timestamp"] - entry["timestamp"]
            stats[entry["to_state"]] += duration
        
        return stats
    
    def reset(self, initial_state: SignalState = SignalState.GOOD):
        """Reset state machine to initial state."""
        self.current_state = initial_state
        self.state_entry_time = None
        self.pending_transition = None
        self.pending_transition_start = None
        self.state_history.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_state_manager(
    initial_state: SignalState = SignalState.GOOD,
) -> SignalStateManager:
    """Create a standard signal state manager."""
    return SignalStateManager(initial_state=initial_state)


def get_alarm_policy_for_state(state: SignalState) -> AlarmPolicy:
    """Get alarm policy for a specific state."""
    return ALARM_POLICIES[state]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIGNAL STATE MACHINE DEMO")
    print("=" * 60)
    
    # Create state manager
    manager = SignalStateManager()
    
    # Simulate signal quality degradation and recovery
    sqi_sequence = [
        (0.0, 0.9),   # Good signal
        (0.5, 0.85),  # Still good
        (1.0, 0.55),  # Degrading
        (1.5, 0.45),  # More degradation
        (2.0, 0.35),  # Poor
        (2.5, 0.25),  # Very poor
        (3.0, 0.20),  # Signal poor
        (3.5, 0.15),  # Near leads off
        (4.0, 0.08),  # Leads off
        (4.5, 0.05),  # Still off
        (5.0, 0.25),  # Signal returning
        (5.5, 0.45),  # Recovering
        (6.0, 0.55),  # Better
        (6.5, 0.65),  # Marginal
        (7.0, 0.75),  # Good
        (8.0, 0.80),  # Still good
        (9.0, 0.85),  # Stable good
        (10.0, 0.90), # Excellent
    ]
    
    print("\nSimulating signal quality changes:")
    print("-" * 60)
    print(f"{'Time':>6} {'SQI':>6} {'State':<15} {'VT':>4} {'SVT':>4} {'Penalty':>8}")
    print("-" * 60)
    
    for t, sqi_score in sqi_sequence:
        sqi_input = SQIInput(
            overall_score=sqi_score,
            is_usable=sqi_score > 0.3,
            flatline_ratio=0.95 if sqi_score < 0.1 else 0.0,
        )
        
        state = manager.update(sqi_input, t)
        policy = manager.get_alarm_policy()
        
        print(f"{t:>6.1f} {sqi_score:>6.2f} {state.value:<15} "
              f"{'✓' if policy.vt_alarm_enabled else '✗':>4} "
              f"{'✓' if policy.svt_alarm_enabled else '✗':>4} "
              f"{policy.confidence_penalty:>8.2f}")
    
    print("-" * 60)
    print("\nRecent transitions:")
    for t in manager.get_recent_transitions():
        print(f"  {t['from_state']} → {t['to_state']} at {t['timestamp']:.1f}s ({t['reason']})")
    
    print("\n" + "=" * 60)
