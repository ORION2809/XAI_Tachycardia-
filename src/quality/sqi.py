"""
Signal Quality Index (SQI) Suite.

Multi-component signal quality assessment that replaces naive FFT-based checks.
Provides robust quality assessment across patient variation and electrode artifacts.

Components:
1. Baseline wander magnitude
2. Saturation / clipping detection
3. Kurtosis check
4. QRS detectability score
5. Powerline noise ratio
6. Flatline detection

Usage:
    sqi_suite = SQISuite()
    result = sqi_suite.compute_sqi(signal, fs=360)
    
    if not result.is_usable:
        # Suppress all alarms for this segment
        pass
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from scipy import ndimage as scipy_ndimage
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.contracts import SQIResult


class SQISuite:
    """
    Multi-component signal quality assessment.
    
    This replaces the naive FFT-based check with a comprehensive suite
    that is robust to patient variation and various artifact types.
    
    Attributes:
        baseline_wander_max_mv: Maximum acceptable baseline drift
        saturation_threshold: Fraction of max range indicating saturation
        min_kurtosis: Minimum acceptable kurtosis (too low = clipped)
        max_kurtosis: Maximum acceptable kurtosis (too high = spikes)
        min_qrs_detectability: Minimum fraction of expected beats detected
        powerline_noise_max_ratio: Maximum acceptable powerline noise ratio
        flatline_threshold: Maximum acceptable flatline ratio
    """
    
    def __init__(
        self,
        baseline_wander_max_mv: float = 0.5,
        saturation_threshold: float = 0.95,
        min_kurtosis: float = 2.0,
        max_kurtosis: float = 20.0,
        min_qrs_detectability: float = 0.6,
        powerline_noise_max_ratio: float = 0.1,
        flatline_threshold: float = 0.1,
    ):
        self.baseline_wander_max_mv = baseline_wander_max_mv
        self.saturation_threshold = saturation_threshold
        self.min_kurtosis = min_kurtosis
        self.max_kurtosis = max_kurtosis
        self.min_qrs_detectability = min_qrs_detectability
        self.powerline_noise_max_ratio = powerline_noise_max_ratio
        self.flatline_threshold = flatline_threshold
        
        # Component weights for overall score
        self.weights = {
            'baseline_wander': 0.15,
            'saturation': 0.20,
            'kurtosis': 0.10,
            'qrs_detectability': 0.30,  # Most important
            'powerline': 0.10,
            'flatline': 0.15,
        }
    
    def compute_sqi(self, signal: np.ndarray, fs: int = 360) -> SQIResult:
        """
        Compute comprehensive Signal Quality Index.
        
        Args:
            signal: ECG signal array (1D)
            fs: Sampling frequency in Hz
            
        Returns:
            SQIResult with overall score, usability flag, components, and recommendations
        """
        if len(signal) < fs:  # Less than 1 second
            return SQIResult(
                overall_score=0.0,
                is_usable=False,
                components={},
                recommendations=["Signal too short for quality assessment"]
            )
        
        components = {}
        recommendations = []
        
        # Normalize signal for consistent processing
        signal_normalized = self._normalize_signal(signal)
        
        # 1. Baseline wander magnitude
        try:
            baseline_score, baseline_rec = self._assess_baseline_wander(signal_normalized, fs)
            components['baseline_wander'] = baseline_score
            if baseline_rec:
                recommendations.append(baseline_rec)
        except Exception as e:
            components['baseline_wander'] = 0.5
            recommendations.append(f"Baseline assessment failed: {str(e)}")
        
        # 2. Saturation / clipping detection
        try:
            saturation_score, saturation_rec = self._assess_saturation(signal)
            components['saturation'] = saturation_score
            if saturation_rec:
                recommendations.append(saturation_rec)
        except Exception as e:
            components['saturation'] = 0.5
            recommendations.append(f"Saturation assessment failed: {str(e)}")
        
        # 3. Kurtosis check
        try:
            kurtosis_score, kurtosis_rec = self._assess_kurtosis(signal_normalized)
            components['kurtosis'] = kurtosis_score
            if kurtosis_rec:
                recommendations.append(kurtosis_rec)
        except Exception as e:
            components['kurtosis'] = 0.5
            recommendations.append(f"Kurtosis assessment failed: {str(e)}")
        
        # 4. QRS detectability
        try:
            qrs_score, qrs_rec, n_peaks = self._assess_qrs_detectability(signal_normalized, fs)
            components['qrs_detectability'] = qrs_score
            if qrs_rec:
                recommendations.append(qrs_rec)
        except Exception as e:
            components['qrs_detectability'] = 0.0
            n_peaks = 0
            recommendations.append(f"QRS detection failed: {str(e)}")
        
        # 5. Powerline noise
        try:
            powerline_score, powerline_rec = self._assess_powerline_noise(signal, fs)
            components['powerline'] = powerline_score
            if powerline_rec:
                recommendations.append(powerline_rec)
        except Exception as e:
            components['powerline'] = 0.5
            recommendations.append(f"Powerline assessment failed: {str(e)}")
        
        # 6. Flatline detection
        try:
            flatline_score, flatline_rec = self._assess_flatline(signal, fs)
            components['flatline'] = flatline_score
            if flatline_rec:
                recommendations.append(flatline_rec)
        except Exception as e:
            components['flatline'] = 0.5
            recommendations.append(f"Flatline assessment failed: {str(e)}")
        
        # Compute overall score (weighted combination)
        overall_score = sum(
            components.get(k, 0.5) * self.weights[k] 
            for k in self.weights
        )
        
        # Hard gate: if QRS detection fails badly or flatline is high, unusable
        is_usable = (
            components.get('qrs_detectability', 0) > 0.4 and 
            components.get('flatline', 0) > 0.5 and
            n_peaks >= 2  # Need at least 2 detected beats
        )
        
        return SQIResult(
            overall_score=overall_score,
            is_usable=is_usable,
            components=components,
            recommendations=recommendations
        )
    
    def _normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        mean = np.mean(signal)
        std = np.std(signal)
        if std < 1e-10:
            return signal - mean
        return (signal - mean) / std
    
    def _assess_baseline_wander(
        self, 
        signal: np.ndarray, 
        fs: int
    ) -> Tuple[float, Optional[str]]:
        """
        Assess baseline wander using median filter extraction.
        
        Returns:
            Tuple of (score, recommendation_if_any)
        """
        # Extract baseline using median filter
        window = int(0.6 * fs)  # 600ms window captures one cardiac cycle
        if window % 2 == 0:
            window += 1
        
        baseline = scipy_ndimage.median_filter(signal, size=window)
        
        # Measure baseline variation
        wander_magnitude = np.std(baseline)
        
        # Score: 1.0 if no wander, decreasing as wander increases
        score = 1.0 - min(wander_magnitude / self.baseline_wander_max_mv, 1.0)
        score = max(0.0, score)
        
        recommendation = None
        if score < 0.5:
            recommendation = f"High baseline wander detected (std={wander_magnitude:.3f})"
        
        return score, recommendation
    
    def _assess_saturation(
        self, 
        signal: np.ndarray
    ) -> Tuple[float, Optional[str]]:
        """
        Detect signal saturation/clipping.
        
        Saturation occurs when the signal hits the ADC limits.
        """
        signal_range = np.ptp(signal)
        if signal_range < 1e-10:
            return 0.0, "Signal has no amplitude variation (flat)"
        
        max_val = np.max(np.abs(signal))
        threshold = self.saturation_threshold * max_val
        
        # Count samples near saturation
        near_max = np.sum(np.abs(signal) > threshold)
        saturation_ratio = near_max / len(signal)
        
        # Score: 1.0 if no saturation, decreasing as more samples saturate
        # Scale up the ratio since even 5% saturation is concerning
        score = 1.0 - min(saturation_ratio * 10, 1.0)
        score = max(0.0, score)
        
        recommendation = None
        if saturation_ratio > 0.05:
            recommendation = f"Signal saturation detected ({saturation_ratio*100:.1f}% samples)"
        
        return score, recommendation
    
    def _assess_kurtosis(
        self, 
        signal: np.ndarray
    ) -> Tuple[float, Optional[str]]:
        """
        Assess signal kurtosis.
        
        - Low kurtosis: signal may be clipped/compressed
        - High kurtosis: signal may have spike artifacts
        - Normal ECG kurtosis: typically 3-10
        """
        kurtosis = scipy_stats.kurtosis(signal)
        
        recommendation = None
        
        if kurtosis < self.min_kurtosis:
            score = 0.3
            recommendation = f"Abnormally low kurtosis ({kurtosis:.1f}) - possible clipping"
        elif kurtosis > self.max_kurtosis:
            score = 0.5
            recommendation = f"High kurtosis ({kurtosis:.1f}) - possible spike artifacts"
        else:
            # Good kurtosis range
            # Optimal around 5-8 for ECG
            optimal = 6.0
            deviation = abs(kurtosis - optimal) / optimal
            score = max(0.5, 1.0 - deviation * 0.5)
        
        return score, recommendation
    
    def _assess_qrs_detectability(
        self, 
        signal: np.ndarray, 
        fs: int
    ) -> Tuple[float, Optional[str], int]:
        """
        Assess QRS detectability.
        
        If R-peaks can't be detected reliably, we can't trust classifications.
        
        Returns:
            Tuple of (score, recommendation, number_of_peaks_detected)
        """
        # Bandpass filter for QRS (5-15 Hz captures QRS energy)
        filtered = self._bandpass_filter(signal, fs, low=5, high=15)
        
        # Squared signal to emphasize QRS
        squared = filtered ** 2
        
        # Moving average to smooth
        window = int(0.1 * fs)  # 100ms window
        if window < 1:
            window = 1
        smoothed = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Find peaks
        height_threshold = np.percentile(smoothed, 90)
        min_distance = int(0.3 * fs)  # Minimum 300ms between beats (200 BPM max)
        
        try:
            peaks, _ = scipy_signal.find_peaks(
                smoothed, 
                height=height_threshold, 
                distance=min_distance
            )
        except Exception:
            peaks = np.array([])
        
        n_detected = len(peaks)
        
        # Expected beats based on reasonable HR range (40-180 BPM)
        duration_sec = len(signal) / fs
        min_expected = duration_sec * 40 / 60
        max_expected = duration_sec * 180 / 60
        
        recommendation = None
        
        if n_detected < min_expected * 0.5:
            # Far fewer beats than expected
            score = 0.2
            recommendation = f"Poor QRS detectability: only {n_detected} beats in {duration_sec:.1f}s"
        elif n_detected < min_expected:
            # Somewhat fewer
            detectability = n_detected / min_expected
            score = detectability
        elif n_detected > max_expected:
            # Too many detections - likely noise
            score = 0.4
            recommendation = f"Excessive peak detections ({n_detected}) - likely noise"
        else:
            # Reasonable range
            score = min(1.0, n_detected / (duration_sec * 60 / 60))  # Normalize to ~60 BPM
            score = min(1.0, score)
        
        # Ensure score meets minimum threshold check
        if score < self.min_qrs_detectability:
            recommendation = recommendation or f"Low QRS detectability score ({score:.2f})"
        
        return score, recommendation, n_detected
    
    def _assess_powerline_noise(
        self, 
        signal: np.ndarray, 
        fs: int
    ) -> Tuple[float, Optional[str]]:
        """
        Assess powerline interference (50/60 Hz).
        
        Uses Welch's method to estimate power spectral density.
        """
        # Compute PSD
        nperseg = min(len(signal), 1024)
        try:
            freqs, psd = scipy_signal.welch(signal, fs, nperseg=nperseg)
        except Exception:
            return 0.5, "Could not compute power spectrum"
        
        # Find power at 50 and 60 Hz bands (±2 Hz)
        mask_50 = (freqs >= 48) & (freqs <= 52)
        mask_60 = (freqs >= 58) & (freqs <= 62)
        
        powerline_power = np.sum(psd[mask_50]) + np.sum(psd[mask_60])
        total_power = np.sum(psd)
        
        if total_power < 1e-10:
            return 0.5, "No signal power detected"
        
        powerline_ratio = powerline_power / total_power
        
        # Score: 1.0 if no powerline noise, decreasing as ratio increases
        score = 1.0 - min(powerline_ratio / self.powerline_noise_max_ratio, 1.0)
        score = max(0.0, score)
        
        recommendation = None
        if powerline_ratio > self.powerline_noise_max_ratio:
            recommendation = f"High powerline interference ({powerline_ratio*100:.1f}% of power)"
        
        return score, recommendation
    
    def _assess_flatline(
        self, 
        signal: np.ndarray, 
        fs: int
    ) -> Tuple[float, Optional[str]]:
        """
        Detect flatline segments (electrode disconnect, lead-off).
        
        Flatline = very low variance over a window.
        """
        window = int(0.5 * fs)  # 500ms windows
        if window < 10:
            window = 10
        
        n_windows = len(signal) // window
        if n_windows == 0:
            return 1.0, None
        
        global_std = np.std(signal)
        if global_std < 1e-10:
            return 0.0, "Signal is completely flat"
        
        flatline_threshold = 0.01 * global_std
        flatline_count = 0
        
        for i in range(n_windows):
            segment = signal[i*window:(i+1)*window]
            if np.std(segment) < flatline_threshold:
                flatline_count += 1
        
        flatline_ratio = flatline_count / n_windows
        
        # Score: 1.0 if no flatlines, decreasing as more flatlines found
        score = 1.0 - flatline_ratio
        score = max(0.0, score)
        
        recommendation = None
        if flatline_ratio > self.flatline_threshold:
            recommendation = f"Flatline segments detected ({flatline_ratio*100:.1f}% of signal)"
        
        return score, recommendation
    
    def _bandpass_filter(
        self, 
        signal: np.ndarray, 
        fs: int, 
        low: float, 
        high: float,
        order: int = 2
    ) -> np.ndarray:
        """Apply bandpass Butterworth filter."""
        nyq = fs / 2
        low_norm = low / nyq
        high_norm = high / nyq
        
        # Ensure valid frequency range
        low_norm = max(0.001, min(low_norm, 0.99))
        high_norm = max(low_norm + 0.01, min(high_norm, 0.99))
        
        try:
            b, a = scipy_signal.butter(order, [low_norm, high_norm], btype='band')
            filtered = scipy_signal.filtfilt(b, a, signal)
            return filtered
        except Exception:
            return signal  # Return original if filtering fails


class SQIPolicy:
    """
    Policy for how to use SQI in the detection pipeline.
    
    Defines thresholds and actions based on signal quality scores.
    """
    
    def __init__(
        self,
        gate_threshold: float = 0.5,
        warn_threshold: float = 0.7,
        suppress_alarms_on_unusable: bool = True,
    ):
        """
        Initialize SQI policy.
        
        Args:
            gate_threshold: Below this, suppress all alarms
            warn_threshold: Below this, add uncertainty flag
            suppress_alarms_on_unusable: If True, suppress alarms when is_usable=False
        """
        self.gate_threshold = gate_threshold
        self.warn_threshold = warn_threshold
        self.suppress_alarms_on_unusable = suppress_alarms_on_unusable
    
    def apply_policy(
        self,
        prediction: Dict[str, Any],
        sqi: SQIResult
    ) -> Dict[str, Any]:
        """
        Apply SQI policy to a model prediction.
        
        Modifies the prediction dict based on signal quality.
        
        Args:
            prediction: Dictionary containing model predictions
            sqi: SQI result for the signal
            
        Returns:
            Modified prediction dict
        """
        # Make a copy to avoid modifying original
        result = prediction.copy()
        
        # Add SQI info
        result['sqi_score'] = sqi.overall_score
        result['sqi_components'] = sqi.components
        result['quality_level'] = sqi.get_quality_level()
        
        # Check if signal is usable
        if not sqi.is_usable and self.suppress_alarms_on_unusable:
            result['suppressed'] = True
            result['suppression_reason'] = 'signal_quality_unusable'
            result['alarm_type'] = None
            result['recommendations'] = sqi.recommendations
            return result
        
        # Check gate threshold
        if sqi.overall_score < self.gate_threshold:
            result['suppressed'] = True
            result['suppression_reason'] = 'low_signal_quality'
            result['alarm_type'] = None
            return result
        
        # Check warn threshold
        if sqi.overall_score < self.warn_threshold:
            result['quality_warning'] = True
            result['recommendations'] = sqi.recommendations
            
            # Reduce confidence based on SQI
            if 'confidence' in result:
                result['confidence'] *= sqi.overall_score
        else:
            result['quality_warning'] = False
        
        result['suppressed'] = False
        return result
    
    def should_suppress_alarm(self, sqi: SQIResult) -> Tuple[bool, str]:
        """
        Quick check if an alarm should be suppressed.
        
        Returns:
            Tuple of (should_suppress, reason)
        """
        if not sqi.is_usable:
            return True, "signal_unusable"
        
        if sqi.overall_score < self.gate_threshold:
            return True, f"low_quality_score_{sqi.overall_score:.2f}"
        
        return False, ""
    
    def get_confidence_adjustment(self, sqi: SQIResult) -> float:
        """
        Get confidence adjustment factor based on SQI.
        
        Returns a multiplier (0.0 to 1.0) for prediction confidence.
        """
        if not sqi.is_usable:
            return 0.0
        
        if sqi.overall_score >= self.warn_threshold:
            return 1.0
        
        if sqi.overall_score >= self.gate_threshold:
            # Linear interpolation between gate and warn thresholds
            return (sqi.overall_score - self.gate_threshold) / \
                   (self.warn_threshold - self.gate_threshold)
        
        return 0.0


# =============================================================================
# v2.3/v2.4: CLASS-CONDITIONAL SQI POLICY
# =============================================================================

class SQIPolicy:
    """
    v2.3: Class-conditional SQI application.
    
    CRITICAL: VF/VFL should NOT be suppressed even when QRS detectability is low,
    because VF has no clear QRS complexes by definition.
    """
    
    def __init__(
        self,
        suppress_threshold: float = 0.3,
        defer_threshold: float = 0.6,
        vf_spectral_check_enabled: bool = True,
    ):
        self.suppress_threshold = suppress_threshold
        self.defer_threshold = defer_threshold
        self.vf_spectral_check_enabled = vf_spectral_check_enabled
        self.disorganized_detector = DisorganizedRhythmDetector()
    
    def apply_policy(
        self,
        prediction: Dict[str, Any],
        sqi: SQIResult,
        model_probs: np.ndarray,
        signal: Optional[np.ndarray] = None,
        fs: int = 360,
    ) -> Dict[str, Any]:
        """
        Apply SQI-based policy with class-conditional logic.
        
        Args:
            prediction: Current prediction dict with episode_type, confidence
            sqi: SQI result
            model_probs: Model output probabilities per class
            signal: Raw signal (for spectral checks)
            fs: Sampling rate
            
        Returns:
            Modified prediction with applied policy
        """
        result = prediction.copy()
        episode_type = prediction.get('episode_type', '')
        
        # VT/VFL/VF special handling
        is_ventricular = episode_type in ('VT', 'VFL', 'VFIB', 'VT_MONOMORPHIC', 'VT_POLYMORPHIC')
        is_disorganized = episode_type in ('VFL', 'VFIB')
        
        # Get VFL/VF probability
        vf_prob = 0.0
        if len(model_probs) > 4:  # Assuming VFL is index 4
            vf_prob = model_probs[4]
        
        # Apply class-conditional logic
        if sqi.overall_score < self.suppress_threshold:
            # Very low quality
            if is_disorganized or vf_prob > 0.5:
                # DON'T suppress VF/VFL - check with spectral analysis
                if self.vf_spectral_check_enabled and signal is not None:
                    spectral_result = self.disorganized_detector.detect_vf_signature(signal, fs)
                    if spectral_result['is_vf_like']:
                        # Route to DEFER, not SUPPRESS
                        result['action'] = 'DEFER'
                        result['reason'] = 'vf_like_signal_low_sqi'
                        result['spectral_evidence'] = spectral_result
                        return result
                
                # Still DEFER for ventricular even without spectral check
                result['action'] = 'DEFER'
                result['reason'] = 'ventricular_low_sqi'
            else:
                # Suppress non-ventricular on very low quality
                result['action'] = 'SUPPRESS'
                result['reason'] = f'low_sqi_{sqi.overall_score:.2f}'
                result['episode_type'] = 'SUPPRESSED'
        
        elif sqi.overall_score < self.defer_threshold:
            # Marginal quality
            if is_ventricular:
                result['action'] = 'DEFER'
                result['reason'] = 'ventricular_marginal_sqi'
            else:
                result['action'] = 'WARN'
                result['confidence'] = prediction.get('confidence', 1.0) * 0.7
                result['reason'] = f'marginal_sqi_{sqi.overall_score:.2f}'
        
        else:
            # Good quality
            result['action'] = 'ALARM'
        
        return result


class DisorganizedRhythmDetector:
    """
    v2.3: Detect VF/VFL-like signals based on spectral characteristics.
    
    VF/VFL typically shows:
    - No clear QRS (by definition)
    - Dominant frequency 3-10 Hz
    - Low amplitude variability over time
    - High spectral entropy
    """
    
    def __init__(
        self,
        vf_freq_low: float = 3.0,
        vf_freq_high: float = 10.0,
        min_vf_power_ratio: float = 0.4,
        min_spectral_entropy: float = 0.7,
    ):
        self.vf_freq_low = vf_freq_low
        self.vf_freq_high = vf_freq_high
        self.min_vf_power_ratio = min_vf_power_ratio
        self.min_spectral_entropy = min_spectral_entropy
    
    def detect_vf_signature(
        self,
        signal: np.ndarray,
        fs: int = 360,
    ) -> Dict[str, Any]:
        """
        Detect VF/VFL spectral signature.
        
        Returns:
            Dict with is_vf_like, vf_power_ratio, spectral_entropy, 
            dominant_freq, amplitude_variability
        """
        result = {
            'is_vf_like': False,
            'vf_power_ratio': 0.0,
            'spectral_entropy': 0.0,
            'dominant_freq': 0.0,
            'amplitude_variability': 0.0,
        }
        
        if len(signal) < fs * 2:  # Need at least 2 seconds
            return result
        
        try:
            # Compute power spectral density
            from scipy.signal import welch
            freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs * 2))
            
            # VF frequency band power
            vf_mask = (freqs >= self.vf_freq_low) & (freqs <= self.vf_freq_high)
            total_power = np.sum(psd)
            vf_power = np.sum(psd[vf_mask])
            
            vf_power_ratio = vf_power / (total_power + 1e-10)
            result['vf_power_ratio'] = float(vf_power_ratio)
            
            # Dominant frequency
            result['dominant_freq'] = float(freqs[np.argmax(psd)])
            
            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-10)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
            spectral_entropy /= np.log2(len(psd))  # Normalize to 0-1
            result['spectral_entropy'] = float(spectral_entropy)
            
            # Amplitude variability (low in VF due to chaotic nature)
            window_size = int(fs * 0.5)  # 500ms windows
            n_windows = len(signal) // window_size
            if n_windows > 1:
                amplitudes = [
                    np.ptp(signal[i*window_size:(i+1)*window_size])
                    for i in range(n_windows)
                ]
                cv = np.std(amplitudes) / (np.mean(amplitudes) + 1e-10)
                result['amplitude_variability'] = float(cv)
            
            # Classify as VF-like
            is_vf_like = (
                vf_power_ratio >= self.min_vf_power_ratio and
                spectral_entropy >= self.min_spectral_entropy and
                self.vf_freq_low <= result['dominant_freq'] <= self.vf_freq_high
            )
            result['is_vf_like'] = is_vf_like
            
        except Exception as e:
            result['error'] = str(e)
        
        return result


# =============================================================================
# v2.4: SIGNAL STATE MACHINE
# =============================================================================

class SignalState:
    """Signal quality state machine states."""
    GOOD = "good"                   # Normal operation, all alarms active
    MARGINAL = "marginal"           # Reduced confidence, warnings only
    SIGNAL_POOR = "signal_poor"     # Artifact-dominated, suppress non-critical
    LEADS_OFF = "leads_off"         # No signal, all alarms suppressed


@dataclass
class SignalStateTransition:
    """Rules for signal state transitions."""
    from_state: str
    to_state: str
    condition: str
    min_duration_sec: float


SIGNAL_STATE_TRANSITIONS = [
    SignalStateTransition(SignalState.GOOD, SignalState.MARGINAL, 
                         "sqi < 0.6", 2.0),
    SignalStateTransition(SignalState.GOOD, SignalState.SIGNAL_POOR, 
                         "sqi < 0.3", 1.0),
    SignalStateTransition(SignalState.MARGINAL, SignalState.SIGNAL_POOR, 
                         "sqi < 0.4", 2.0),
    SignalStateTransition(SignalState.SIGNAL_POOR, SignalState.MARGINAL, 
                         "sqi >= 0.5", 3.0),  # Slower recovery
    SignalStateTransition(SignalState.MARGINAL, SignalState.GOOD, 
                         "sqi >= 0.7", 3.0),
]


class SignalStateManager:
    """
    v2.4: Manage signal quality state transitions with hysteresis.
    
    Prevents rapid flapping between states.
    """
    
    def __init__(
        self,
        suppress_threshold: float = 0.3,
        defer_threshold: float = 0.6,
        good_threshold: float = 0.7,
    ):
        self.suppress_threshold = suppress_threshold
        self.defer_threshold = defer_threshold
        self.good_threshold = good_threshold
        
        self.current_state = SignalState.GOOD
        self.state_entry_time: Optional[float] = None
        self.pending_transition: Optional[str] = None
        self.pending_transition_start: Optional[float] = None
    
    def update(
        self,
        sqi: SQIResult,
        current_time: float,
    ) -> str:
        """
        Update state based on SQI.
        
        Returns current state after update.
        """
        # Determine target state based on SQI
        flatline_score = sqi.components.get('flatline', 1.0)
        
        if flatline_score < 0.1:
            target_state = SignalState.LEADS_OFF
        elif sqi.overall_score < self.suppress_threshold:
            target_state = SignalState.SIGNAL_POOR
        elif sqi.overall_score < self.defer_threshold:
            target_state = SignalState.MARGINAL
        else:
            target_state = SignalState.GOOD
        
        # Handle state transition with hysteresis
        if target_state != self.current_state:
            if self.pending_transition != target_state:
                self.pending_transition = target_state
                self.pending_transition_start = current_time
            else:
                transition_rule = self._get_transition_rule(
                    self.current_state, target_state
                )
                if transition_rule:
                    elapsed = current_time - self.pending_transition_start
                    if elapsed >= transition_rule.min_duration_sec:
                        self.current_state = target_state
                        self.state_entry_time = current_time
                        self.pending_transition = None
        else:
            self.pending_transition = None
        
        return self.current_state
    
    def get_alarm_policy(self) -> Dict[str, Any]:
        """Get alarm policy for current state."""
        if self.current_state == SignalState.GOOD:
            return {
                "vt_alarm_enabled": True,
                "svt_alarm_enabled": True,
                "sinus_tachy_enabled": True,
                "confidence_penalty": 0.0,
            }
        elif self.current_state == SignalState.MARGINAL:
            return {
                "vt_alarm_enabled": True,
                "svt_alarm_enabled": True,
                "sinus_tachy_enabled": False,
                "confidence_penalty": 0.15,
            }
        elif self.current_state == SignalState.SIGNAL_POOR:
            return {
                "vt_alarm_enabled": True,  # VT NEVER suppressed (but DEFER)
                "svt_alarm_enabled": False,
                "sinus_tachy_enabled": False,
                "confidence_penalty": 0.30,
            }
        else:  # LEADS_OFF
            return {
                "vt_alarm_enabled": False,
                "svt_alarm_enabled": False,
                "sinus_tachy_enabled": False,
                "confidence_penalty": 1.0,
            }
    
    def _get_transition_rule(
        self, from_state: str, to_state: str
    ) -> Optional[SignalStateTransition]:
        for rule in SIGNAL_STATE_TRANSITIONS:
            if rule.from_state == from_state and rule.to_state == to_state:
                return rule
        return None
    
    def reset(self):
        """Reset to initial state."""
        self.current_state = SignalState.GOOD
        self.state_entry_time = None
        self.pending_transition = None
        self.pending_transition_start = None


def compute_segment_sqi(
    signal: np.ndarray,
    fs: int = 360,
    return_details: bool = False
) -> float:
    """
    Convenience function to compute SQI for a signal segment.
    
    Args:
        signal: ECG signal array
        fs: Sampling frequency
        return_details: If True, return full SQIResult instead of just score
        
    Returns:
        Overall SQI score (0.0 to 1.0) or full SQIResult if return_details=True
    """
    suite = SQISuite()
    result = suite.compute_sqi(signal, fs)
    
    if return_details:
        return result
    return result.overall_score


if __name__ == "__main__":
    # Demo / test
    import matplotlib.pyplot as plt
    
    # Generate synthetic ECG-like signal
    fs = 360
    duration = 10  # seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Simple synthetic ECG with QRS-like peaks
    ecg = np.zeros_like(t)
    hr = 75  # BPM
    beat_interval = 60 / hr  # seconds
    
    for i in range(int(duration / beat_interval)):
        beat_time = i * beat_interval
        beat_sample = int(beat_time * fs)
        if beat_sample + 50 < len(ecg):
            # Add QRS-like spike
            ecg[beat_sample:beat_sample+20] = np.sin(np.linspace(0, np.pi, 20)) * 1.0
            # Add T-wave
            if beat_sample + 100 < len(ecg):
                ecg[beat_sample+50:beat_sample+100] = np.sin(np.linspace(0, np.pi, 50)) * 0.3
    
    # Add some noise
    ecg += np.random.normal(0, 0.05, len(ecg))
    
    # Test SQI
    suite = SQISuite()
    result = suite.compute_sqi(ecg, fs)
    
    print("\n" + "="*60)
    print("SQI Assessment Results")
    print("="*60)
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"Quality Level: {result.get_quality_level()}")
    print(f"Is Usable: {result.is_usable}")
    print("\nComponent Scores:")
    for comp, score in result.components.items():
        print(f"  {comp}: {score:.3f}")
    if result.recommendations:
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
    print("="*60)
    
    # Test SQI Policy
    print("\n" + "="*60)
    print("SQI Policy Test")
    print("="*60)
    policy = SQIPolicy()
    prediction = {'episode_type': 'VFL', 'confidence': 0.8}
    probs = np.array([0.1, 0.0, 0.1, 0.1, 0.7])  # High VFL prob
    
    result_policy = policy.apply_policy(prediction, result, probs, ecg, fs)
    print(f"Policy result: {result_policy}")
    print("="*60)
