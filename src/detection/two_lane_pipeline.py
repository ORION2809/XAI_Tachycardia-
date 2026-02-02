"""
Two-Lane Detection Pipeline for Tachycardia Detection.

v2.4: Explicit separation of detection vs confirmation.

LANE 1 (Detection): Sensitivity-first, low threshold, fast
    - Goal: Don't miss any VT
    - Threshold: Lower (vt_detect_prob_threshold)
    - Duration: Short (vt_detect_min_duration_sec)
    - Output: CANDIDATE episodes

LANE 2 (Confirmation): Precision-focused, higher threshold, validated
    - Goal: Reduce false alarms
    - Requirements: Longer persistence + HR sanity + morphology
    - Duration: Long (vt_alarm_min_duration_sec)
    - Output: CONFIRMED episodes → routed to UnifiedDecisionPolicy

This separation allows:
- Early WARNING at detection threshold
- ALARM only after confirmation threshold
- Clear metrics: detection sensitivity vs alarm specificity
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# EPISODE TYPE (should align with evaluation.metrics)
# =============================================================================

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
    AFLUTTER = "aflutter"


# =============================================================================
# EPISODE LABEL (detection output)
# =============================================================================

@dataclass
class DetectedEpisode:
    """Detected episode from the pipeline."""
    start_sample: int
    end_sample: int
    start_time_sec: float = 0.0
    end_time_sec: float = 0.0
    episode_type: EpisodeType = EpisodeType.NORMAL
    severity: str = "detected"  # detected, confirmed
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample
    
    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


# =============================================================================
# TEMPORAL AND ALIGNMENT CONFIG
# =============================================================================

@dataclass
class TemporalConfig:
    """Temporal conversion utilities."""
    fs: int = 360
    downsample_factor: int = 8
    
    @property
    def timestep_samples(self) -> int:
        """Samples per model timestep."""
        return self.downsample_factor
    
    def seconds_to_windows(self, seconds: float) -> int:
        """Convert seconds to number of model timesteps."""
        samples = int(seconds * self.fs)
        return max(1, samples // self.downsample_factor)
    
    def windows_to_seconds(self, windows: int) -> float:
        """Convert model timesteps to seconds."""
        samples = windows * self.downsample_factor
        return samples / self.fs
    
    def samples_to_windows(self, samples: int) -> int:
        """Convert samples to model timesteps."""
        return samples // self.downsample_factor


@dataclass
class AlignmentConfig:
    """Alignment between model timesteps and signal samples."""
    fs: int = 360
    downsample_factor: int = 8
    padding_offset: int = 0  # For models with initial padding
    
    def timestep_to_sample_range(self, timestep: int) -> Tuple[int, int]:
        """Convert timestep index to sample range."""
        start = timestep * self.downsample_factor + self.padding_offset
        end = start + self.downsample_factor
        return (start, end)
    
    def sample_to_timestep(self, sample: int) -> int:
        """Convert sample index to timestep."""
        return max(0, (sample - self.padding_offset) // self.downsample_factor)


# =============================================================================
# DETECTION CONFIG
# =============================================================================

@dataclass
class DetectionConfig:
    """Configuration for detection pipeline."""
    
    # === Detection Lane (sensitivity-first) ===
    vt_detect_prob_threshold: float = 0.4
    vt_detect_min_duration_sec: float = 0.375  # ~3 beats at 180 BPM
    svt_detect_prob_threshold: float = 0.4
    svt_detect_min_duration_sec: float = 1.0
    
    # === Confirmation Lane (precision-focused) ===
    vt_alarm_prob_threshold: float = 0.7
    vt_alarm_min_duration_sec: float = 1.5
    svt_alarm_prob_threshold: float = 0.6
    svt_alarm_min_duration_sec: float = 3.0
    
    # === Additional gates ===
    alarm_requires_hr_sanity: bool = True
    alarm_requires_morphology: bool = True
    alarm_morphology_threshold: float = 0.5
    
    # === HR bounds ===
    vt_min_hr_bpm: float = 100.0
    vt_max_hr_bpm: float = 300.0
    svt_min_hr_bpm: float = 100.0
    svt_max_hr_bpm: float = 250.0
    
    # === SQI ===
    enable_sqi_gate: bool = True
    sqi_threshold: float = 0.5
    vf_sqi_bypass_threshold: float = 0.7  # VF bypass
    
    # === Smoothing ===
    smoothing_window_sec: float = 0.5
    
    def get_temporal_config(
        self,
        fs: int = 360,
        downsample_factor: int = 8
    ) -> TemporalConfig:
        """Create temporal config for this detection config."""
        return TemporalConfig(fs=fs, downsample_factor=downsample_factor)


# =============================================================================
# DETECTION LANE
# =============================================================================

@dataclass
class DetectionLane:
    """
    Lane 1: Sensitivity-first detection.
    Low threshold, short duration - don't miss anything.
    """
    prob_threshold: float
    min_duration_sec: float
    temporal: TemporalConfig
    alignment: AlignmentConfig
    
    def detect(
        self,
        probs: np.ndarray,
        signal: np.ndarray,
        r_peaks: Optional[np.ndarray] = None,
        episode_type: EpisodeType = EpisodeType.VT_MONOMORPHIC,
        class_indices: Tuple[int, ...] = (3, 4),  # VT, VFL
    ) -> List[DetectedEpisode]:
        """
        Detect candidate episodes (sensitivity-first).
        
        Args:
            probs: (seq_len, num_classes) probability array
            signal: Raw ECG signal
            r_peaks: R-peak locations (optional)
            episode_type: Type to assign to detected episodes
            class_indices: Which probability columns to use
        """
        min_windows = self.temporal.seconds_to_windows(self.min_duration_sec)
        
        # Combine relevant class probabilities
        if len(class_indices) == 1:
            target_probs = probs[:, class_indices[0]]
        else:
            target_probs = np.max(probs[:, list(class_indices)], axis=1)
        
        # Threshold
        detections = target_probs > self.prob_threshold
        
        # Find runs
        episodes = []
        runs = self._find_runs(detections)
        
        for start, end in runs:
            run_length = end - start
            if run_length >= min_windows:
                # Use alignment contract
                start_sample = self.alignment.timestep_to_sample_range(start)[0]
                end_sample = self.alignment.timestep_to_sample_range(end - 1)[1]
                
                confidence = float(np.mean(target_probs[start:end]))
                
                # Determine specific subtype if multiple classes
                if len(class_indices) > 1 and len(class_indices) == 2:
                    mean_c0 = float(np.mean(probs[start:end, class_indices[0]]))
                    mean_c1 = float(np.mean(probs[start:end, class_indices[1]]))
                    
                    actual_type = episode_type
                    if class_indices == (3, 4):  # VT, VFL
                        if mean_c1 > mean_c0:
                            actual_type = EpisodeType.VFL
                else:
                    actual_type = episode_type
                
                episodes.append(DetectedEpisode(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time_sec=start_sample / self.temporal.fs,
                    end_time_sec=end_sample / self.temporal.fs,
                    episode_type=actual_type,
                    severity="detected",
                    confidence=confidence,
                    evidence={
                        "lane": "detection",
                        "consecutive_windows": run_length,
                        "mean_prob": float(np.mean(target_probs[start:end])),
                    },
                ))
        
        return episodes
    
    def _find_runs(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        """Find start/end of True runs."""
        runs = []
        in_run = False
        start = 0
        for i, val in enumerate(binary):
            if val and not in_run:
                in_run = True
                start = i
            elif not val and in_run:
                in_run = False
                runs.append((start, i))
        if in_run:
            runs.append((start, len(binary)))
        return runs


# =============================================================================
# CONFIRMATION LANE
# =============================================================================

@dataclass
class ConfirmationLane:
    """
    Lane 2: Confirmation for ALARM eligibility.
    Higher threshold, longer duration, additional validation.
    """
    prob_threshold: float
    min_duration_sec: float
    requires_hr_sanity: bool
    requires_morphology: bool
    morphology_threshold: float
    temporal: TemporalConfig
    alignment: AlignmentConfig
    vt_min_hr: float = 100.0
    vt_max_hr: float = 300.0
    
    def confirm(
        self,
        probs: np.ndarray,
        signal: np.ndarray,
        r_peaks: Optional[np.ndarray],
        detected_episodes: List[DetectedEpisode],
        class_indices: Tuple[int, ...] = (3, 4),
    ) -> List[DetectedEpisode]:
        """
        Confirm detected episodes for ALARM eligibility.
        
        Requirements for confirmation:
        1. Probability stays above alarm threshold for min_duration
        2. HR is within clinical range (if required)
        3. Morphology is consistent with VT (if required)
        """
        confirmed = []
        min_windows = self.temporal.seconds_to_windows(self.min_duration_sec)
        
        for ep in detected_episodes:
            # Check 1: Duration at alarm threshold
            start_ts = self.alignment.sample_to_timestep(ep.start_sample)
            end_ts = self.alignment.sample_to_timestep(ep.end_sample)
            
            # Clamp to valid range
            start_ts = max(0, min(start_ts, len(probs) - 1))
            end_ts = max(start_ts + 1, min(end_ts, len(probs)))
            
            # Get probs for this episode
            if len(class_indices) == 1:
                ep_probs = probs[start_ts:end_ts, class_indices[0]]
            else:
                ep_probs = np.max(probs[start_ts:end_ts, list(class_indices)], axis=1)
            
            above_alarm = ep_probs > self.prob_threshold
            
            # Check if enough consecutive windows above alarm threshold
            max_consecutive = self._max_consecutive_true(above_alarm)
            if max_consecutive < min_windows:
                continue  # Not enough persistence at alarm level
            
            # Check 2: HR sanity
            if self.requires_hr_sanity and r_peaks is not None:
                hr_ok, hr_value = self._check_hr(ep, r_peaks)
                if not hr_ok:
                    continue
                ep.evidence['computed_hr_bpm'] = hr_value
            
            # Check 3: Morphology
            if self.requires_morphology and signal is not None:
                morph_score = self._check_morphology(ep, signal, r_peaks)
                if morph_score < self.morphology_threshold:
                    continue
                ep.evidence['morphology_score'] = morph_score
            
            # Passed all checks - confirmed for alarm
            ep.severity = "confirmed"
            ep.evidence['lane'] = "confirmation"
            ep.evidence['alarm_eligible'] = True
            ep.evidence['confirmation_windows'] = max_consecutive
            confirmed.append(ep)
        
        return confirmed
    
    def _max_consecutive_true(self, arr: np.ndarray) -> int:
        """Find maximum consecutive True values."""
        if len(arr) == 0:
            return 0
        max_len = 0
        current = 0
        for val in arr:
            if val:
                current += 1
                max_len = max(max_len, current)
            else:
                current = 0
        return max_len
    
    def _check_hr(
        self,
        episode: DetectedEpisode,
        r_peaks: np.ndarray,
    ) -> Tuple[bool, Optional[float]]:
        """Check if HR is in clinical VT range."""
        mask = (r_peaks >= episode.start_sample) & (r_peaks <= episode.end_sample)
        ep_peaks = r_peaks[mask]
        
        if len(ep_peaks) < 2:
            return False, None  # Can't compute HR
        
        rr_ms = np.diff(ep_peaks) / self.temporal.fs * 1000
        hr = 60000 / np.median(rr_ms)
        
        # VT range check
        return (self.vt_min_hr <= hr <= self.vt_max_hr), hr
    
    def _check_morphology(
        self,
        episode: DetectedEpisode,
        signal: np.ndarray,
        r_peaks: Optional[np.ndarray],
    ) -> float:
        """
        Get morphology score for episode (0 = narrow, 1 = wide/VT-like).
        
        Simple heuristic: QRS width estimation
        """
        if r_peaks is None:
            return 0.5  # Uncertain
        
        mask = (r_peaks >= episode.start_sample) & (r_peaks <= episode.end_sample)
        ep_peaks = r_peaks[mask].tolist()
        
        if len(ep_peaks) == 0:
            return 0.5
        
        # Simple QRS width estimation
        # Look at derivative around each peak
        widths = []
        half_window = int(0.06 * self.temporal.fs)  # 60ms half-window
        
        for peak in ep_peaks:
            peak = int(peak)
            start = max(0, peak - half_window)
            end = min(len(signal), peak + half_window)
            
            if end - start < 10:
                continue
            
            segment = signal[start:end]
            
            # Estimate width from threshold crossing
            threshold = 0.2 * np.max(np.abs(segment))
            above = np.abs(segment) > threshold
            width_samples = np.sum(above)
            width_ms = width_samples / self.temporal.fs * 1000
            widths.append(width_ms)
        
        if not widths:
            return 0.5
        
        mean_width = np.mean(widths)
        
        # VT typically has wide QRS (>120ms)
        # Map 80-160ms to 0-1 score
        score = (mean_width - 80) / 80
        return float(np.clip(score, 0, 1))


# =============================================================================
# TWO-LANE PIPELINE
# =============================================================================

class TwoLanePipeline:
    """
    v2.4: Explicit separation of detection vs confirmation.
    
    LANE 1 (Detection): Sensitivity-first, low threshold, fast
    LANE 2 (Confirmation): Precision-focused, higher threshold, validated
    
    Output goes to UnifiedDecisionPolicy for final alarm/warning decision.
    """
    
    def __init__(
        self,
        config: DetectionConfig,
        fs: int = 360,
        downsample_factor: int = 8,
    ):
        self.config = config
        self.fs = fs
        self.downsample_factor = downsample_factor
        
        self.temporal = TemporalConfig(fs=fs, downsample_factor=downsample_factor)
        self.alignment = AlignmentConfig(fs=fs, downsample_factor=downsample_factor)
        
        # Lane 1: Detection (sensitivity-first)
        self.vt_detection_lane = DetectionLane(
            prob_threshold=config.vt_detect_prob_threshold,
            min_duration_sec=config.vt_detect_min_duration_sec,
            temporal=self.temporal,
            alignment=self.alignment,
        )
        
        self.svt_detection_lane = DetectionLane(
            prob_threshold=config.svt_detect_prob_threshold,
            min_duration_sec=config.svt_detect_min_duration_sec,
            temporal=self.temporal,
            alignment=self.alignment,
        )
        
        # Lane 2: Confirmation (precision-focused)
        self.vt_confirmation_lane = ConfirmationLane(
            prob_threshold=config.vt_alarm_prob_threshold,
            min_duration_sec=config.vt_alarm_min_duration_sec,
            requires_hr_sanity=config.alarm_requires_hr_sanity,
            requires_morphology=config.alarm_requires_morphology,
            morphology_threshold=config.alarm_morphology_threshold,
            temporal=self.temporal,
            alignment=self.alignment,
            vt_min_hr=config.vt_min_hr_bpm,
            vt_max_hr=config.vt_max_hr_bpm,
        )
        
        self.svt_confirmation_lane = ConfirmationLane(
            prob_threshold=config.svt_alarm_prob_threshold,
            min_duration_sec=config.svt_alarm_min_duration_sec,
            requires_hr_sanity=config.alarm_requires_hr_sanity,
            requires_morphology=False,  # Morphology less relevant for SVT
            morphology_threshold=0.0,
            temporal=self.temporal,
            alignment=self.alignment,
            vt_min_hr=config.svt_min_hr_bpm,
            vt_max_hr=config.svt_max_hr_bpm,
        )
    
    def process(
        self,
        probs: np.ndarray,
        signal: np.ndarray,
        r_peaks: Optional[np.ndarray] = None,
        sqi_score: Optional[float] = None,
    ) -> Dict[str, List[DetectedEpisode]]:
        """
        Process through both lanes.
        
        Args:
            probs: (seq_len, num_classes) probability array
                   Class order: [NORMAL, SINUS_TACHY, SVT, VT, VFL]
            signal: Raw ECG signal
            r_peaks: R-peak locations (optional)
            sqi_score: Signal quality score (optional)
        
        Returns:
            Dict with:
                - 'detected': All detected episodes (sensitivity-first)
                - 'confirmed': Confirmed episodes ready for ALARM
                - 'warning_only': Detected but not yet confirmed
                - 'vt_detected': VT/VFL detected episodes
                - 'vt_confirmed': VT/VFL confirmed episodes
                - 'svt_detected': SVT detected episodes
                - 'svt_confirmed': SVT confirmed episodes
        """
        # SQI gate (if enabled)
        if self.config.enable_sqi_gate and sqi_score is not None:
            if sqi_score < self.config.sqi_threshold:
                # Check for VF bypass
                max_vf_prob = np.max(probs[:, 4]) if probs.shape[1] > 4 else 0
                if max_vf_prob < self.config.vf_sqi_bypass_threshold:
                    # Suppress all - low quality, no VF
                    return {
                        'detected': [],
                        'confirmed': [],
                        'warning_only': [],
                        'vt_detected': [],
                        'vt_confirmed': [],
                        'svt_detected': [],
                        'svt_confirmed': [],
                        'suppressed_reason': 'low_sqi',
                    }
        
        # Lane 1: VT/VFL Detection
        vt_detected = self.vt_detection_lane.detect(
            probs, signal, r_peaks,
            episode_type=EpisodeType.VT_MONOMORPHIC,
            class_indices=(3, 4),  # VT, VFL
        )
        
        # Lane 1: SVT Detection
        svt_detected = self.svt_detection_lane.detect(
            probs, signal, r_peaks,
            episode_type=EpisodeType.SVT,
            class_indices=(2,),  # SVT
        )
        
        # Lane 2: VT/VFL Confirmation
        vt_confirmed = self.vt_confirmation_lane.confirm(
            probs, signal, r_peaks, vt_detected,
            class_indices=(3, 4),
        )
        
        # Lane 2: SVT Confirmation
        svt_confirmed = self.svt_confirmation_lane.confirm(
            probs, signal, r_peaks, svt_detected,
            class_indices=(2,),
        )
        
        # Combine all detections and confirmations
        all_detected = vt_detected + svt_detected
        all_confirmed = vt_confirmed + svt_confirmed
        
        # Warning-only = detected but not confirmed
        confirmed_ids = {id(ep) for ep in all_confirmed}
        warning_only = [ep for ep in all_detected if id(ep) not in confirmed_ids]
        
        return {
            'detected': all_detected,
            'confirmed': all_confirmed,
            'warning_only': warning_only,
            'vt_detected': vt_detected,
            'vt_confirmed': vt_confirmed,
            'svt_detected': svt_detected,
            'svt_confirmed': svt_confirmed,
        }
    
    def get_detection_metrics(
        self,
        results: Dict[str, List[DetectedEpisode]],
    ) -> Dict[str, Any]:
        """Get summary metrics from processing results."""
        return {
            'n_vt_detected': len(results.get('vt_detected', [])),
            'n_vt_confirmed': len(results.get('vt_confirmed', [])),
            'n_svt_detected': len(results.get('svt_detected', [])),
            'n_svt_confirmed': len(results.get('svt_confirmed', [])),
            'vt_confirmation_rate': (
                len(results.get('vt_confirmed', [])) / 
                max(1, len(results.get('vt_detected', [])))
            ),
            'svt_confirmation_rate': (
                len(results.get('svt_confirmed', [])) / 
                max(1, len(results.get('svt_detected', [])))
            ),
        }


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Two-Lane Pipeline Demo (v2.4)")
    print("="*60)
    
    np.random.seed(42)
    
    # Create synthetic probabilities
    # Class order: [NORMAL, SINUS_TACHY, SVT, VT, VFL]
    n_timesteps = 100
    probs = np.zeros((n_timesteps, 5))
    probs[:, 0] = 0.8  # Mostly normal
    
    # Add VT episode from timestep 30-50
    probs[30:50, 0] = 0.1
    probs[30:50, 3] = 0.85  # High VT probability
    
    # Add SVT episode from timestep 70-85
    probs[70:85, 0] = 0.2
    probs[70:85, 2] = 0.65  # Medium SVT probability
    
    # Create synthetic signal
    fs = 360
    signal_length = n_timesteps * 8  # 8x downsample
    t = np.linspace(0, signal_length / fs, signal_length)
    signal = np.sin(2 * np.pi * 1.0 * t) * 500  # 1 Hz base signal
    
    # Create R-peaks (roughly 100 BPM during normal, 180 BPM during VT)
    r_peaks = []
    for i in range(0, signal_length, int(fs * 0.6)):  # ~100 BPM
        r_peaks.append(i)
    # Add faster peaks during VT region
    vt_start = 30 * 8
    vt_end = 50 * 8
    for i in range(vt_start, vt_end, int(fs * 0.33)):  # ~180 BPM
        r_peaks.append(i)
    r_peaks = np.array(sorted(set(r_peaks)))
    
    # Create pipeline
    config = DetectionConfig()
    pipeline = TwoLanePipeline(config, fs=fs)
    
    # Process
    results = pipeline.process(probs, signal, r_peaks, sqi_score=0.85)
    
    print("\n--- Detection Results ---")
    print(f"VT Detected: {len(results['vt_detected'])} episodes")
    for ep in results['vt_detected']:
        print(f"  {ep.episode_type.value}: {ep.start_time_sec:.2f}s - {ep.end_time_sec:.2f}s, "
              f"conf={ep.confidence:.2f}")
    
    print(f"\nVT Confirmed: {len(results['vt_confirmed'])} episodes")
    for ep in results['vt_confirmed']:
        print(f"  {ep.episode_type.value}: {ep.start_time_sec:.2f}s - {ep.end_time_sec:.2f}s, "
              f"conf={ep.confidence:.2f}, HR={ep.evidence.get('computed_hr_bpm', 'N/A')}")
    
    print(f"\nSVT Detected: {len(results['svt_detected'])} episodes")
    print(f"SVT Confirmed: {len(results['svt_confirmed'])} episodes")
    
    print(f"\nWarning Only: {len(results['warning_only'])} episodes")
    
    # Metrics
    metrics = pipeline.get_detection_metrics(results)
    print("\n--- Pipeline Metrics ---")
    print(f"VT Confirmation Rate: {metrics['vt_confirmation_rate']:.1%}")
    print(f"SVT Confirmation Rate: {metrics['svt_confirmation_rate']:.1%}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
