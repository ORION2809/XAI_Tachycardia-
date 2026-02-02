# Signal Quality module
# v2.4: Added signal state machine for formal artifact handling

from .sqi import SQISuite, SQIPolicy

from .signal_state import (
    SignalState,
    SignalStateTransition,
    SignalStateManager,
    AlarmPolicy,
    SQIInput,
    SIGNAL_STATE_TRANSITIONS,
    ALARM_POLICIES,
    create_state_manager,
    get_alarm_policy_for_state,
)

__all__ = [
    # SQI
    'SQISuite',
    'SQIPolicy',
    
    # Signal State Machine
    'SignalState',
    'SignalStateTransition',
    'SignalStateManager',
    'AlarmPolicy',
    'SQIInput',
    'SIGNAL_STATE_TRANSITIONS',
    'ALARM_POLICIES',
    'create_state_manager',
    'get_alarm_policy_for_state',
]
