"""
Episode Label Generator for XAI Tachycardia Detection.

Exact rules for generating episode labels from beat annotations.
Based on BUILDABLE_SPEC.md Part 2.

Version: 2.4 (Deployment-Grade)

KEY FEATURES:
- CANDIDATE vs CONFIRMED VT tiers
- Morphology-aware VT confirmation
- Explicit handling of escape beats, fusion beats
- Clear edge case rules
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

import sys
sys.path.insert(0, '..')
from data.contracts import (
    BeatAnnotation,
    ECGSegment,
    EpisodeLabel,
    EpisodeType,
    VTLabelConfidence,
    LabelConfidenceTier,
)


@dataclass
class EpisodeLabelGeneratorConfig:
    """Configuration for episode label generation."""
    
    # VT Detection Rules
    vt_min_consecutive_v_beats: int = 3
    vt_min_hr_bpm: float = 100.0
    vt_max_hr_bpm: float = 300.0
    vt_sustained_duration_sec: float = 30.0
    
    # Morphology Criteria
    morphology_check_enabled: bool = True
    qrs_width_threshold_ms: float = 120.0
    template_similarity_threshold: float = 0.7
    
    # Run Continuity Rules
    fusion_breaks_run: bool = True
    escape_beats_as_vt: bool = False  # E beats NOT counted as VT by default
    max_gap_samples: int = 72  # ~200ms at 360Hz
    
    # HR Computation
    hr_window_beats: int = 4
    hr_method: str = "median"
    
    # Confidence
    min_confidence_for_label: float = 0.8


class EpisodeLabelGenerator:
    """
    Exact rules for generating episode labels from beat annotations.
    No ambiguity - every edge case has a defined behavior.
    
    KEY CHANGE from v2.1: Outputs CANDIDATE vs CONFIRMED VT separately.
    """
    
    def __init__(self, config: Optional[EpisodeLabelGeneratorConfig] = None):
        self.config = config or EpisodeLabelGeneratorConfig()
    
    def compute_instantaneous_hr(
        self, 
        rr_intervals_ms: List[float],
        method: str = "median"
    ) -> float:
        """
        Compute HR from RR intervals.
        
        Rule: Use median of last hr_window_beats intervals.
        If fewer beats available, use all available.
        If RR interval < 200ms or > 2000ms, exclude as artifact.
        """
        valid_rr = [rr for rr in rr_intervals_ms if 200 <= rr <= 2000]
        if len(valid_rr) == 0:
            return float('nan')
        
        window = self.config.hr_window_beats
        if method == "median":
            rr_representative = np.median(valid_rr[-window:])
        else:
            rr_representative = np.mean(valid_rr[-window:])
        
        return 60000.0 / rr_representative  # BPM
    
    def estimate_qrs_morphology(
        self,
        signal: np.ndarray,
        r_peak: int,
        fs: int
    ) -> Dict[str, float]:
        """
        Robust QRS morphology estimation using energy-based onset/offset detection.
        
        Returns:
            Dict with:
                - qrs_width_ms: Estimated width in milliseconds
                - morphology_confidence: 0-1 confidence in the measurement
                - width_category: 'narrow', 'borderline', 'wide', 'very_wide'
        """
        from scipy.signal import butter, filtfilt
        from scipy.ndimage import uniform_filter1d
        
        result = {
            'qrs_width_ms': 100.0,  # Default borderline
            'morphology_confidence': 0.0,
            'width_category': 'unknown',
            'onset_sample': 0,
            'offset_sample': 0,
        }
        
        # Window around R-peak (200ms each side for safety)
        window_samples = int(0.20 * fs)
        start = max(0, r_peak - window_samples)
        end = min(len(signal), r_peak + window_samples)
        
        if end - start < int(0.1 * fs):  # Need at least 100ms of signal
            result['morphology_confidence'] = 0.1
            return result
        
        segment = signal[start:end]
        r_peak_local = r_peak - start
        
        try:
            # Bandpass filter to isolate QRS energy (5-40 Hz)
            nyq = fs / 2
            low, high = 5 / nyq, min(40 / nyq, 0.99)
            b, a = butter(2, [low, high], btype='band')
            filtered = filtfilt(b, a, segment)
            
            # Compute energy envelope (squared + smoothed)
            energy = filtered ** 2
            smooth_window = max(3, int(0.01 * fs))  # 10ms smoothing
            if smooth_window % 2 == 0:
                smooth_window += 1
            energy_smooth = uniform_filter1d(energy, smooth_window)
            
            # Adaptive thresholds
            median_energy = np.median(energy_smooth)
            mad = np.median(np.abs(energy_smooth - median_energy))
            threshold = median_energy + 2.0 * mad
            
            # Find QRS region (above threshold)
            above_threshold = energy_smooth > threshold
            
            if not np.any(above_threshold):
                peak_energy = energy_smooth[r_peak_local]
                threshold = 0.1 * peak_energy
                above_threshold = energy_smooth > threshold
            
            if not np.any(above_threshold):
                result['morphology_confidence'] = 0.2
                return result
            
            # Find onset and offset
            crossings = np.where(above_threshold)[0]
            onset_local = crossings[0]
            offset_local = crossings[-1]
            
            # Compute width
            qrs_samples = offset_local - onset_local
            qrs_width_ms = qrs_samples / fs * 1000
            
            # Sanity bounds
            if qrs_width_ms < 40 or qrs_width_ms > 300:
                qrs_width_ms = np.clip(qrs_width_ms, 60, 200)
                confidence = 0.3
            else:
                peak_energy = energy_smooth[r_peak_local]
                snr = peak_energy / (median_energy + 1e-8)
                snr_factor = min(snr / 10, 1.0)
                width_plausibility = 1.0 - abs(qrs_width_ms - 100) / 100
                width_plausibility = max(width_plausibility, 0.3)
                confidence = 0.5 * snr_factor + 0.5 * width_plausibility
                confidence = np.clip(confidence, 0.2, 1.0)
            
            # Categorize width
            if qrs_width_ms < 100:
                width_category = 'narrow'
            elif qrs_width_ms < 120:
                width_category = 'borderline'
            elif qrs_width_ms < 160:
                width_category = 'wide'
            else:
                width_category = 'very_wide'
            
            result = {
                'qrs_width_ms': qrs_width_ms,
                'morphology_confidence': confidence,
                'width_category': width_category,
                'onset_sample': onset_local - r_peak_local,
                'offset_sample': offset_local - r_peak_local,
            }
            
        except Exception as e:
            result['morphology_confidence'] = 0.1
            result['error'] = str(e)
        
        return result
    
    def compute_run_morphology_score(
        self,
        signal: np.ndarray,
        r_peaks: List[int],
        fs: int
    ) -> Dict[str, float]:
        """
        Compute aggregate morphology score for a run of beats.
        
        Returns a SOFT morphology score (0-1) for use in DecisionPolicy.
        
        Returns:
            Dict with:
                - mean_width_ms: Average QRS width across run
                - morphology_score: 0 (narrow/SVT-like) to 1 (wide/VT-like)
                - consistency_score: How consistent are the widths
                - confidence: Overall confidence in morphology assessment
        """
        if len(r_peaks) == 0:
            return {
                'mean_width_ms': 100.0,
                'morphology_score': 0.5,
                'consistency_score': 0.0,
                'confidence': 0.0,
            }
        
        # Get morphology for each beat
        morphologies = [self.estimate_qrs_morphology(signal, p, fs) for p in r_peaks]
        
        # Filter by confidence
        confident = [m for m in morphologies if m['morphology_confidence'] > 0.3]
        
        if len(confident) == 0:
            return {
                'mean_width_ms': 100.0,
                'morphology_score': 0.5,
                'consistency_score': 0.0,
                'confidence': 0.1,
            }
        
        widths = [m['qrs_width_ms'] for m in confident]
        confidences = [m['morphology_confidence'] for m in confident]
        
        # Weighted mean width
        mean_width = np.average(widths, weights=confidences)
        
        # Consistency
        if len(widths) > 1:
            width_std = np.std(widths)
            consistency = 1.0 - min(width_std / 50, 1.0)
        else:
            consistency = 0.5
        
        # Morphology score: sigmoid centered at 120ms (VT threshold)
        morphology_score = 1 / (1 + np.exp(-(mean_width - 120) / 20))
        
        # Overall confidence
        mean_confidence = np.mean(confidences)
        coverage = len(confident) / len(r_peaks)
        overall_confidence = mean_confidence * coverage * (0.5 + 0.5 * consistency)
        
        return {
            'mean_width_ms': mean_width,
            'morphology_score': morphology_score,
            'consistency_score': consistency,
            'confidence': overall_confidence,
        }
    
    def compute_template_similarity(
        self,
        signal: np.ndarray,
        r_peaks: List[int],
        fs: int
    ) -> float:
        """
        Compute similarity between consecutive QRS complexes.
        
        High similarity = monomorphic (same origin)
        Low similarity = polymorphic (multiple foci)
        
        Returns: Mean pairwise correlation (0-1)
        """
        if len(r_peaks) < 2:
            return 1.0  # Assume monomorphic if can't assess
        
        template_width = int(0.12 * fs)  # 120ms
        templates = []
        
        for peak in r_peaks:
            start = peak - template_width // 2
            end = peak + template_width // 2
            if start >= 0 and end < len(signal):
                template = signal[start:end]
                template = (template - np.mean(template)) / (np.std(template) + 1e-8)
                templates.append(template)
        
        if len(templates) < 2:
            return 1.0
        
        # Compute pairwise correlations
        correlations = []
        for i in range(len(templates) - 1):
            corr = np.corrcoef(templates[i], templates[i+1])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 1.0
    
    def compute_rr_intervals(
        self,
        beat_annotations: List[BeatAnnotation],
        fs: int
    ) -> List[float]:
        """Compute RR intervals in milliseconds."""
        if len(beat_annotations) < 2:
            return []
        
        rr_intervals = []
        for i in range(1, len(beat_annotations)):
            delta_samples = (
                beat_annotations[i].sample_idx - 
                beat_annotations[i-1].sample_idx
            )
            rr_ms = delta_samples / fs * 1000
            rr_intervals.append(rr_ms)
        
        return rr_intervals
    
    def detect_vt_episodes(
        self,
        beat_annotations: List[BeatAnnotation],
        signal: Optional[np.ndarray] = None,
        fs: int = 360,
        record_id: str = "",
        source_dataset: str = "",
    ) -> Tuple[List[EpisodeLabel], List[EpisodeLabel]]:
        """
        Exact VT detection algorithm with CANDIDATE vs CONFIRMED tiers.
        
        Rule: VT episode = run of ≥3 consecutive ventricular beats
              with computed HR > 100 BPM and < 300 BPM.
        
        Returns:
            confirmed_vt: Episodes meeting ALL criteria (beat + HR + morphology)
            candidate_vt: Episodes meeting beat criteria only (need review)
        """
        confirmed_episodes = []
        candidate_episodes = []
        
        if len(beat_annotations) < self.config.vt_min_consecutive_v_beats:
            return confirmed_episodes, candidate_episodes
        
        # Compute RR intervals
        rr_intervals_ms = self.compute_rr_intervals(beat_annotations, fs)
        
        # Define ventricular types
        if self.config.escape_beats_as_vt:
            ventricular_types = {'V', 'E'}
        else:
            ventricular_types = {'V'}
        
        i = 0
        while i < len(beat_annotations):
            if beat_annotations[i].beat_type in ventricular_types:
                # Start of potential V run
                run_start = i
                run_end = i
                
                # Extend run with strict continuity rules
                while run_end + 1 < len(beat_annotations):
                    next_beat = beat_annotations[run_end + 1]
                    curr_beat = beat_annotations[run_end]
                    
                    gap_samples = next_beat.sample_idx - curr_beat.sample_idx
                    
                    if next_beat.beat_type in ventricular_types:
                        if gap_samples > self.config.max_gap_samples:
                            break
                        run_end += 1
                    elif next_beat.beat_type == 'F' and self.config.fusion_breaks_run:
                        break
                    elif next_beat.beat_type in {'U', '?'}:
                        break
                    else:
                        break
                
                run_length = run_end - run_start + 1
                
                if run_length >= self.config.vt_min_consecutive_v_beats:
                    # Compute HR for this run
                    run_rr = rr_intervals_ms[max(0, run_start-1):run_end]
                    hr = self.compute_instantaneous_hr(run_rr)
                    
                    # Duration calculation
                    start_sample = beat_annotations[run_start].sample_idx
                    end_sample = beat_annotations[run_end].sample_idx
                    duration_sec = (end_sample - start_sample) / fs
                    
                    # Sustained vs non-sustained
                    if duration_sec >= self.config.vt_sustained_duration_sec:
                        severity = "sustained"
                    else:
                        severity = "non-sustained"
                    
                    # Confirmation checks
                    is_confirmed = True
                    downgrade_reasons = []
                    
                    # Check 1: HR in valid range
                    if np.isnan(hr):
                        hr_status = "unknown"
                        downgrade_reasons.append("hr_unknown")
                        is_confirmed = False
                    elif hr < self.config.vt_min_hr_bpm:
                        hr_status = "below_threshold"
                        downgrade_reasons.append(f"hr_below_{self.config.vt_min_hr_bpm}")
                        is_confirmed = False
                    elif hr > self.config.vt_max_hr_bpm:
                        hr_status = "implausible"
                        downgrade_reasons.append(f"hr_above_{self.config.vt_max_hr_bpm}")
                        is_confirmed = False
                    else:
                        hr_status = "valid"
                    
                    # Check 2: Morphology
                    morphology_status = "not_checked"
                    episode_type = EpisodeType.VT_MONOMORPHIC
                    
                    if self.config.morphology_check_enabled and signal is not None:
                        run_peaks = [
                            beat_annotations[j].sample_idx 
                            for j in range(run_start, run_end + 1)
                        ]
                        
                        morphology_result = self.compute_run_morphology_score(
                            signal, run_peaks, fs
                        )
                        
                        morph_score = morphology_result['morphology_score']
                        morph_confidence = morphology_result['confidence']
                        
                        if morph_confidence > 0.5 and morph_score < 0.3:
                            morphology_status = "narrow_qrs"
                            downgrade_reasons.append("narrow_qrs_suspect_svt")
                            is_confirmed = False
                        elif morph_confidence > 0.5 and morph_score > 0.7:
                            morphology_status = "wide_qrs"
                        else:
                            morphology_status = "uncertain"
                        
                        # Monomorphic vs polymorphic
                        similarity = self.compute_template_similarity(
                            signal, run_peaks, fs
                        )
                        if similarity < self.config.template_similarity_threshold:
                            episode_type = EpisodeType.VT_POLYMORPHIC
                    
                    # Build evidence
                    evidence = {
                        "v_beat_count": run_length,
                        "computed_hr_bpm": hr,
                        "hr_status": hr_status,
                        "morphology_status": morphology_status,
                        "beat_indices": list(range(run_start, run_end + 1)),
                        "downgrade_reasons": downgrade_reasons,
                        "record_id": record_id,
                    }
                    
                    # Set confidence
                    if is_confirmed:
                        confidence = 1.0
                        vt_confidence = VTLabelConfidence.CONFIRMED
                        label_tier = LabelConfidenceTier.DERIVED_RHYTHM
                    else:
                        confidence = 0.6 - 0.1 * len(downgrade_reasons)
                        confidence = max(confidence, 0.3)
                        vt_confidence = VTLabelConfidence.CANDIDATE
                        label_tier = LabelConfidenceTier.HEURISTIC
                    
                    episode = EpisodeLabel(
                        start_sample=start_sample,
                        end_sample=end_sample,
                        start_time_sec=start_sample / fs,
                        end_time_sec=end_sample / fs,
                        episode_type=episode_type,
                        severity=severity,
                        confidence=confidence,
                        evidence=evidence,
                        label_tier=label_tier,
                        vt_confidence=vt_confidence,
                        source_dataset=source_dataset,
                        record_id=record_id,
                    )
                    
                    if is_confirmed:
                        confirmed_episodes.append(episode)
                    else:
                        candidate_episodes.append(episode)
                
                i = run_end + 1
            else:
                i += 1
        
        return confirmed_episodes, candidate_episodes
    
    def detect_svt_episodes(
        self,
        beat_annotations: List[BeatAnnotation],
        signal: Optional[np.ndarray] = None,
        fs: int = 360,
        record_id: str = "",
        source_dataset: str = "",
    ) -> List[EpisodeLabel]:
        """
        Detect SVT episodes from beat annotations.
        
        SVT is detected as runs of supraventricular ectopic beats with elevated HR.
        """
        episodes = []
        
        if len(beat_annotations) < 3:
            return episodes
        
        rr_intervals_ms = self.compute_rr_intervals(beat_annotations, fs)
        svt_types = {'A', 'a', 'S', 'J'}
        
        i = 0
        while i < len(beat_annotations):
            if beat_annotations[i].beat_type in svt_types:
                run_start = i
                run_end = i
                
                while run_end + 1 < len(beat_annotations):
                    next_beat = beat_annotations[run_end + 1]
                    curr_beat = beat_annotations[run_end]
                    gap_samples = next_beat.sample_idx - curr_beat.sample_idx
                    
                    if next_beat.beat_type in svt_types:
                        if gap_samples > self.config.max_gap_samples * 2:  # More lenient for SVT
                            break
                        run_end += 1
                    else:
                        break
                
                run_length = run_end - run_start + 1
                
                if run_length >= 3:  # Minimum 3 beats for SVT
                    run_rr = rr_intervals_ms[max(0, run_start-1):run_end]
                    hr = self.compute_instantaneous_hr(run_rr)
                    
                    if not np.isnan(hr) and hr >= 100.0:  # Tachycardia threshold
                        start_sample = beat_annotations[run_start].sample_idx
                        end_sample = beat_annotations[run_end].sample_idx
                        duration_sec = (end_sample - start_sample) / fs
                        
                        severity = "sustained" if duration_sec >= 30.0 else "non-sustained"
                        
                        episode = EpisodeLabel(
                            start_sample=start_sample,
                            end_sample=end_sample,
                            start_time_sec=start_sample / fs,
                            end_time_sec=end_sample / fs,
                            episode_type=EpisodeType.SVT,
                            severity=severity,
                            confidence=0.8,
                            evidence={
                                "svt_beat_count": run_length,
                                "computed_hr_bpm": hr,
                                "record_id": record_id,
                            },
                            label_tier=LabelConfidenceTier.DERIVED_RHYTHM,
                            source_dataset=source_dataset,
                            record_id=record_id,
                        )
                        episodes.append(episode)
                
                i = run_end + 1
            else:
                i += 1
        
        return episodes
    
    def detect_all_episodes(
        self,
        beat_annotations: List[BeatAnnotation],
        signal: Optional[np.ndarray] = None,
        fs: int = 360,
        record_id: str = "",
        source_dataset: str = "",
    ) -> Dict[str, List[EpisodeLabel]]:
        """
        Detect all episode types from beat annotations.
        
        Returns:
            Dict with keys: 'confirmed_vt', 'candidate_vt', 'svt', 'all'
        """
        confirmed_vt, candidate_vt = self.detect_vt_episodes(
            beat_annotations, signal, fs, record_id, source_dataset
        )
        
        svt_episodes = self.detect_svt_episodes(
            beat_annotations, signal, fs, record_id, source_dataset
        )
        
        all_episodes = confirmed_vt + candidate_vt + svt_episodes
        all_episodes.sort(key=lambda e: e.start_sample)
        
        return {
            'confirmed_vt': confirmed_vt,
            'candidate_vt': candidate_vt,
            'svt': svt_episodes,
            'all': all_episodes,
        }
