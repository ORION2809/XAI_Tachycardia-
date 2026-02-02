"""
Episode Detector for Tachycardia Detection.

v2.4: Convert dense per-timestep probabilities to episode detections.

This is the EXPLICIT prediction unit: dense probs → episode logic.
All temporal logic uses TemporalConfig for deterministic conversion.

Key features:
- Probability smoothing
- SQI gate with VF bypass
- HR sanity check
- Consecutive detection requirements
- Episode merging
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from scipy.ndimage import uniform_filter1d


# =============================================================================
# IMPORTS FROM TWO-LANE PIPELINE
# =============================================================================

from .two_lane_pipeline import (
    EpisodeType,
    DetectedEpisode,
    TemporalConfig,
    AlignmentConfig,
    DetectionConfig,
)


# =============================================================================
# EPISODE DETECTOR CONFIG
# =============================================================================

@dataclass
class EpisodeDetectorConfig:
    """Configuration for episode detection."""
    
    # Class probabilities
    num_classes: int = 5
    # Class order: [NORMAL, SINUS_TACHY, SVT, VT, VFL]
    vt_class_idx: int = 3
    vfl_class_idx: int = 4
    svt_class_idx: int = 2
    sinus_tachy_idx: int = 1
    
    # Probability thresholds
    vt_prob_threshold: float = 0.5
    vfl_prob_threshold: float = 0.5
    svt_prob_threshold: float = 0.5
    sinus_tachy_threshold: float = 0.6
    
    # Duration requirements (in seconds)
    vt_min_duration_sec: float = 0.5
    svt_min_duration_sec: float = 1.0
    sinus_tachy_min_duration_sec: float = 2.0
    
    # HR bounds
    vt_min_hr_bpm: float = 100.0
    vt_max_hr_bpm: float = 300.0
    svt_min_hr_bpm: float = 100.0
    svt_max_hr_bpm: float = 250.0
    sinus_tachy_min_hr_bpm: float = 100.0
    sinus_tachy_max_hr_bpm: float = 200.0
    
    # SQI settings
    enable_sqi_gate: bool = True
    sqi_threshold: float = 0.5
    vf_sqi_bypass_prob: float = 0.7  # VF bypass if prob > this
    
    # Smoothing
    smoothing_window_sec: float = 0.5
    
    # HR sanity
    enable_hr_sanity: bool = True
    
    # Merging
    merge_gap_sec: float = 0.5  # Merge episodes within this gap


# =============================================================================
# SQI RESULT (lightweight)
# =============================================================================

@dataclass
class SQIResult:
    """Lightweight SQI result for detector."""
    overall_score: float
    is_usable: bool
    components: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# EPISODE DETECTOR
# =============================================================================

class EpisodeDetector:
    """
    Convert dense per-timestep probabilities to episode detections.
    
    This is the EXPLICIT prediction unit: dense probs → episode logic.
    All temporal logic uses TemporalConfig for deterministic conversion.
    """
    
    def __init__(
        self,
        config: EpisodeDetectorConfig,
        fs: int = 360,
        downsample_factor: int = 8,
    ):
        self.config = config
        self.fs = fs
        self.downsample_factor = downsample_factor
        
        self.temporal = TemporalConfig(fs=fs, downsample_factor=downsample_factor)
        self.alignment = AlignmentConfig(fs=fs, downsample_factor=downsample_factor)
    
    def detect_episodes(
        self,
        probs: np.ndarray,
        signal: np.ndarray,
        fs: int,
        r_peaks: Optional[np.ndarray] = None,
        sqi: Optional[SQIResult] = None,
    ) -> List[DetectedEpisode]:
        """
        Main detection logic.
        
        Steps:
        1. Smooth probabilities (window size from seconds-based config)
        2. Apply SQI gate
        3. Detect VT episodes
        4. Detect SVT episodes
        5. Detect sinus tachycardia episodes
        6. Apply HR sanity check
        7. Merge overlapping detections
        8. Return episode list
        
        Args:
            probs: (seq_len, num_classes) probability array
            signal: Original signal for HR/SQI
            fs: Sampling frequency
            r_peaks: R-peak locations (optional)
            sqi: Signal quality result (optional)
        
        Returns:
            List of detected episodes
        """
        episodes = []
        
        # Update temporal config if fs differs
        if fs != self.fs:
            self.temporal = TemporalConfig(fs=fs, downsample_factor=self.downsample_factor)
            self.alignment = AlignmentConfig(fs=fs, downsample_factor=self.downsample_factor)
            self.fs = fs
        
        # 1. Smooth probabilities
        smoothing_windows = self.temporal.seconds_to_windows(self.config.smoothing_window_sec)
        smoothed = self._smooth_probs(probs, window_size=max(1, smoothing_windows))
        
        # 2. SQI gate
        if self.config.enable_sqi_gate and sqi is not None:
            if not sqi.is_usable or sqi.overall_score < self.config.sqi_threshold:
                # Check VF bypass
                max_vf_prob = float(np.max(smoothed[:, self.config.vfl_class_idx]))
                if max_vf_prob < self.config.vf_sqi_bypass_prob:
                    return []  # Suppress all detections
        
        # 3. Detect VT episodes
        vt_min_windows = self.temporal.seconds_to_windows(self.config.vt_min_duration_sec)
        
        vt_probs = np.maximum(
            smoothed[:, self.config.vt_class_idx],
            smoothed[:, self.config.vfl_class_idx]
        )
        
        vt_episodes = self._detect_class_episodes(
            vt_probs,
            threshold=self.config.vt_prob_threshold,
            min_consecutive=vt_min_windows,
            episode_type=EpisodeType.VT_MONOMORPHIC,
            fs=fs,
        )
        
        # Differentiate VT vs VFL
        for ep in vt_episodes:
            start_ts = self.alignment.sample_to_timestep(ep.start_sample)
            end_ts = self.alignment.sample_to_timestep(ep.end_sample)
            start_ts = max(0, min(start_ts, len(smoothed) - 1))
            end_ts = max(start_ts + 1, min(end_ts, len(smoothed)))
            
            mean_vt = float(np.mean(smoothed[start_ts:end_ts, self.config.vt_class_idx]))
            mean_vfl = float(np.mean(smoothed[start_ts:end_ts, self.config.vfl_class_idx]))
            
            if mean_vfl > mean_vt:
                ep.episode_type = EpisodeType.VFL
            
            ep.evidence['mean_vt_prob'] = mean_vt
            ep.evidence['mean_vfl_prob'] = mean_vfl
        
        # 4. Detect SVT episodes
        svt_min_windows = self.temporal.seconds_to_windows(self.config.svt_min_duration_sec)
        
        svt_episodes = self._detect_class_episodes(
            smoothed[:, self.config.svt_class_idx],
            threshold=self.config.svt_prob_threshold,
            min_consecutive=svt_min_windows,
            episode_type=EpisodeType.SVT,
            fs=fs,
        )
        
        # 5. Detect sinus tachycardia episodes
        sinus_min_windows = self.temporal.seconds_to_windows(
            self.config.sinus_tachy_min_duration_sec
        )
        
        sinus_episodes = self._detect_class_episodes(
            smoothed[:, self.config.sinus_tachy_idx],
            threshold=self.config.sinus_tachy_threshold,
            min_consecutive=sinus_min_windows,
            episode_type=EpisodeType.SINUS_TACHY,
            fs=fs,
        )
        
        # Combine
        episodes = vt_episodes + svt_episodes + sinus_episodes
        
        # 6. Apply HR sanity check
        if self.config.enable_hr_sanity and r_peaks is not None:
            episodes = self._apply_hr_sanity(episodes, r_peaks, fs)
        
        # 7. Merge overlapping episodes
        episodes = self._merge_overlapping(episodes, fs)
        
        return episodes
    
    def _smooth_probs(self, probs: np.ndarray, window_size: int) -> np.ndarray:
        """Smooth probabilities with moving average."""
        if window_size <= 1:
            return probs
        
        smoothed = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            smoothed[:, i] = uniform_filter1d(probs[:, i], window_size, mode='nearest')
        return smoothed
    
    def _detect_class_episodes(
        self,
        class_probs: np.ndarray,
        threshold: float,
        min_consecutive: int,
        episode_type: EpisodeType,
        fs: int,
    ) -> List[DetectedEpisode]:
        """Detect episodes for a single class."""
        episodes = []
        
        # Threshold
        detections = class_probs > threshold
        
        # Find runs of consecutive detections
        runs = self._find_runs(detections)
        
        for start, end in runs:
            run_length = end - start
            if run_length >= min_consecutive:
                # Use alignment contract for sample conversion
                start_range = self.alignment.timestep_to_sample_range(start)
                end_range = self.alignment.timestep_to_sample_range(end - 1)
                
                start_sample = start_range[0]
                end_sample = end_range[1]
                
                # Average probability for confidence
                confidence = float(np.mean(class_probs[start:end]))
                
                episodes.append(DetectedEpisode(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time_sec=start_sample / fs,
                    end_time_sec=end_sample / fs,
                    episode_type=episode_type,
                    severity="detected",
                    confidence=confidence,
                    evidence={
                        "consecutive_windows": run_length,
                        "alignment_verified": True,
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
    
    def _apply_hr_sanity(
        self,
        episodes: List[DetectedEpisode],
        r_peaks: np.ndarray,
        fs: int,
    ) -> List[DetectedEpisode]:
        """Filter episodes by HR sanity check."""
        filtered = []
        
        for ep in episodes:
            # Get HR bounds based on episode type
            if ep.episode_type in [EpisodeType.VT_MONOMORPHIC, EpisodeType.VT_POLYMORPHIC, 
                                   EpisodeType.VFL, EpisodeType.VFIB]:
                min_hr = self.config.vt_min_hr_bpm
                max_hr = self.config.vt_max_hr_bpm
            elif ep.episode_type == EpisodeType.SVT:
                min_hr = self.config.svt_min_hr_bpm
                max_hr = self.config.svt_max_hr_bpm
            else:  # Sinus tachy
                min_hr = self.config.sinus_tachy_min_hr_bpm
                max_hr = self.config.sinus_tachy_max_hr_bpm
            
            # Find R-peaks within episode
            mask = (r_peaks >= ep.start_sample) & (r_peaks <= ep.end_sample)
            episode_peaks = r_peaks[mask]
            
            if len(episode_peaks) >= 2:
                rr_intervals = np.diff(episode_peaks) / fs * 1000  # ms
                hr = 60000 / np.median(rr_intervals)
                
                if min_hr <= hr <= max_hr:
                    ep.evidence["computed_hr_bpm"] = hr
                    filtered.append(ep)
                else:
                    # HR outside expected range - reject
                    ep.evidence["hr_rejection"] = f"HR {hr:.0f} outside [{min_hr}, {max_hr}]"
            else:
                # Not enough beats to compute HR - keep with reduced confidence
                ep.confidence *= 0.7
                ep.evidence["hr_check"] = "insufficient_beats"
                filtered.append(ep)
        
        return filtered
    
    def _merge_overlapping(
        self,
        episodes: List[DetectedEpisode],
        fs: int,
    ) -> List[DetectedEpisode]:
        """Merge overlapping episodes of same type."""
        if not episodes:
            return []
        
        # Sort by start time
        episodes.sort(key=lambda e: e.start_sample)
        
        merge_gap_samples = int(self.config.merge_gap_sec * fs)
        
        merged = [episodes[0]]
        for ep in episodes[1:]:
            last = merged[-1]
            
            # Check if same type and within merge gap
            if (ep.episode_type == last.episode_type and 
                ep.start_sample <= last.end_sample + merge_gap_samples):
                # Merge
                last.end_sample = max(last.end_sample, ep.end_sample)
                last.end_time_sec = last.end_sample / fs
                last.confidence = max(last.confidence, ep.confidence)
                # Merge evidence
                if 'merged_count' in last.evidence:
                    last.evidence['merged_count'] += 1
                else:
                    last.evidence['merged_count'] = 2
            else:
                merged.append(ep)
        
        return merged


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Episode Detector Demo (v2.4)")
    print("="*60)
    
    np.random.seed(42)
    
    # Create synthetic probabilities
    n_timesteps = 150
    probs = np.zeros((n_timesteps, 5))
    probs[:, 0] = 0.8  # Mostly normal
    
    # Add VT episode from timestep 30-60
    probs[30:60, 0] = 0.1
    probs[30:60, 3] = 0.75  # VT probability
    
    # Add VFL episode from timestep 80-100
    probs[80:100, 0] = 0.1
    probs[80:100, 4] = 0.85  # VFL probability
    
    # Add SVT episode from timestep 120-140
    probs[120:140, 0] = 0.2
    probs[120:140, 2] = 0.65  # SVT probability
    
    # Create signal and R-peaks
    fs = 360
    signal_length = n_timesteps * 8
    signal = np.random.randn(signal_length) * 100
    
    # R-peaks at varying rates
    r_peaks = []
    # Normal region (60 BPM)
    for i in range(0, 30 * 8, fs):
        r_peaks.append(i)
    # VT region (180 BPM)
    for i in range(30 * 8, 60 * 8, int(fs * 0.33)):
        r_peaks.append(i)
    # Normal region
    for i in range(60 * 8, 80 * 8, fs):
        r_peaks.append(i)
    # VFL region (200 BPM)
    for i in range(80 * 8, 100 * 8, int(fs * 0.3)):
        r_peaks.append(i)
    # Normal region
    for i in range(100 * 8, 120 * 8, fs):
        r_peaks.append(i)
    # SVT region (150 BPM)
    for i in range(120 * 8, 140 * 8, int(fs * 0.4)):
        r_peaks.append(i)
    
    r_peaks = np.array(sorted(set(r_peaks)))
    
    # Create detector
    config = EpisodeDetectorConfig()
    detector = EpisodeDetector(config, fs=fs)
    
    # Detect
    episodes = detector.detect_episodes(
        probs, signal, fs, r_peaks,
        sqi=SQIResult(overall_score=0.85, is_usable=True)
    )
    
    print(f"\nDetected {len(episodes)} episodes:")
    for ep in episodes:
        hr = ep.evidence.get('computed_hr_bpm', 'N/A')
        if isinstance(hr, float):
            hr = f"{hr:.0f}"
        print(f"  {ep.episode_type.value}: {ep.start_time_sec:.2f}s - {ep.end_time_sec:.2f}s, "
              f"conf={ep.confidence:.2f}, HR={hr}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
