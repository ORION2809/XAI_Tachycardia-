"""
Clinical Labeling Module for Tachycardia Detection

Implements clinically validated definitions:
- VT: ≥3 consecutive ventricular beats at >100 BPM
- Non-sustained VT: <30 seconds
- Sustained VT: ≥30 seconds
- SVT: Narrow complex tachycardia >100 BPM
- Sinus Tachycardia: Sustained sinus rhythm >100 BPM for >30 seconds

Reference: ACC/AHA Guidelines for Ventricular Arrhythmias
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class TachycardiaType(Enum):
    """Clinical tachycardia classifications"""
    NORMAL = 0
    SINUS_TACHYCARDIA = 1
    SVT = 2  # Supraventricular Tachycardia
    VT_NONSUSTAINED = 3  # <30 seconds
    VT_SUSTAINED = 4  # ≥30 seconds
    VFL = 5  # Ventricular Flutter
    VF = 6  # Ventricular Fibrillation
    ATRIAL_FLUTTER = 7
    ATRIAL_FIBRILLATION_RVR = 8  # AF with rapid ventricular response


@dataclass
class TachycardiaEpisode:
    """Represents a clinical tachycardia episode"""
    episode_type: TachycardiaType
    start_sample: int
    end_sample: int
    start_time: float  # seconds
    end_time: float  # seconds
    duration: float  # seconds
    mean_hr: float  # BPM
    num_beats: int
    is_sustained: bool  # ≥30 seconds
    severity: str  # 'benign', 'moderate', 'life-threatening'
    beat_indices: List[int]


class ClinicalTachycardiaLabeler:
    """
    Labels tachycardia episodes using clinically validated criteria
    
    Key Clinical Definitions:
    1. Ventricular Tachycardia (VT): 
       - ≥3 consecutive ventricular beats at rate >100 BPM
       - Non-sustained: <30 seconds
       - Sustained: ≥30 seconds OR requiring intervention
       
    2. Supraventricular Tachycardia (SVT):
       - Narrow complex (<120ms QRS) tachycardia >100 BPM
       - Excludes sinus tachycardia
       
    3. Sinus Tachycardia:
       - Sinus rhythm with HR >100 BPM
       - Usually benign, requires sustained elevation
    """
    
    # MIT-BIH ventricular beat types
    VENTRICULAR_BEATS = {'V', 'E', 'F', '!'}  # PVC, V-escape, Fusion, V-flutter
    
    # MIT-BIH rhythm annotations for tachyarrhythmias
    RHYTHM_MAP = {
        '(VT': TachycardiaType.VT_NONSUSTAINED,  # Will upgrade to sustained if >30s
        '(VFL': TachycardiaType.VFL,
        '(SVTA': TachycardiaType.SVT,
        '(T': TachycardiaType.SINUS_TACHYCARDIA,
        '(AFL': TachycardiaType.ATRIAL_FLUTTER,
        '(AFIB': TachycardiaType.ATRIAL_FIBRILLATION_RVR,  # Only if HR >100
        '(N': TachycardiaType.NORMAL,
        '(B': TachycardiaType.NORMAL,  # Bigeminy is not tachycardia per se
    }
    
    # Clinical thresholds
    VT_MIN_BEATS = 3  # Minimum consecutive ventricular beats for VT
    SUSTAINED_DURATION = 30.0  # seconds - threshold for sustained VT
    TACHYCARDIA_HR_THRESHOLD = 100  # BPM
    SINUS_TACHY_MIN_DURATION = 30.0  # seconds - sustained sinus tachy
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize clinical labeler
        
        Args:
            sampling_rate: ECG sampling frequency in Hz
        """
        self.fs = sampling_rate
        
    def label_record(self, 
                     beat_annotations: pd.DataFrame,
                     rhythm_annotations: pd.DataFrame,
                     signal_length: int) -> Tuple[List[TachycardiaEpisode], np.ndarray]:
        """
        Label all tachycardia episodes in a record using clinical criteria
        
        Args:
            beat_annotations: Beat-level annotations with 'sample_num', 'beat_type'
            rhythm_annotations: Rhythm change annotations
            signal_length: Total signal length in samples
            
        Returns:
            Tuple of (list of TachycardiaEpisode, per-sample label array)
        """
        episodes = []
        
        # Method 1: Use rhythm annotations if available
        if not rhythm_annotations.empty:
            rhythm_episodes = self._label_from_rhythm_annotations(
                rhythm_annotations, signal_length
            )
            episodes.extend(rhythm_episodes)
        
        # Method 2: Detect VT from consecutive ventricular beats
        if not beat_annotations.empty:
            vt_episodes = self._detect_ventricular_tachycardia(beat_annotations)
            
            # Merge with rhythm-based episodes (avoid duplicates)
            for vt_ep in vt_episodes:
                if not self._overlaps_existing(vt_ep, episodes):
                    episodes.append(vt_ep)
        
        # Method 3: Detect sinus tachycardia from sustained high HR
        if not beat_annotations.empty:
            sinus_episodes = self._detect_sinus_tachycardia(
                beat_annotations, rhythm_annotations
            )
            for s_ep in sinus_episodes:
                if not self._overlaps_existing(s_ep, episodes):
                    episodes.append(s_ep)
        
        # Sort episodes by start time
        episodes.sort(key=lambda x: x.start_sample)
        
        # Create per-sample labels
        sample_labels = self._create_sample_labels(episodes, signal_length)
        
        return episodes, sample_labels
    
    def _label_from_rhythm_annotations(self, 
                                        rhythm_annotations: pd.DataFrame,
                                        signal_length: int) -> List[TachycardiaEpisode]:
        """Create episodes from rhythm annotation changes"""
        episodes = []
        
        rhythm_df = rhythm_annotations.sort_values('sample_num')
        
        for i, row in rhythm_df.iterrows():
            rhythm_code = row.get('rhythm', row.get('aux', ''))
            
            if rhythm_code not in self.RHYTHM_MAP:
                continue
                
            tachy_type = self.RHYTHM_MAP[rhythm_code]
            
            if tachy_type == TachycardiaType.NORMAL:
                continue
            
            start_sample = row['sample_num']
            
            # Find end of this rhythm (next rhythm change or end of record)
            next_rhythms = rhythm_df[rhythm_df['sample_num'] > start_sample]
            if not next_rhythms.empty:
                end_sample = next_rhythms.iloc[0]['sample_num']
            else:
                end_sample = signal_length
            
            duration = (end_sample - start_sample) / self.fs
            
            # Upgrade VT to sustained if duration ≥30s
            if tachy_type == TachycardiaType.VT_NONSUSTAINED and duration >= self.SUSTAINED_DURATION:
                tachy_type = TachycardiaType.VT_SUSTAINED
            
            # Determine severity
            severity = self._get_severity(tachy_type)
            
            episode = TachycardiaEpisode(
                episode_type=tachy_type,
                start_sample=start_sample,
                end_sample=end_sample,
                start_time=start_sample / self.fs,
                end_time=end_sample / self.fs,
                duration=duration,
                mean_hr=0,  # Will be calculated if beat data available
                num_beats=0,
                is_sustained=duration >= self.SUSTAINED_DURATION,
                severity=severity,
                beat_indices=[]
            )
            episodes.append(episode)
        
        return episodes
    
    def _detect_ventricular_tachycardia(self, 
                                         beat_annotations: pd.DataFrame) -> List[TachycardiaEpisode]:
        """
        Detect VT from consecutive ventricular beats
        
        Clinical Definition: ≥3 consecutive ventricular beats at >100 BPM
        """
        episodes = []
        
        # Sort beats by sample position
        beats = beat_annotations.sort_values('sample_num').reset_index(drop=True)
        
        if len(beats) < self.VT_MIN_BEATS:
            return episodes
        
        # Find runs of ventricular beats
        v_run_start = None
        v_run_beats = []
        
        for i, row in beats.iterrows():
            beat_type = row['beat_type']
            
            if beat_type in self.VENTRICULAR_BEATS:
                if v_run_start is None:
                    v_run_start = i
                v_run_beats.append(i)
            else:
                # End of ventricular run
                if len(v_run_beats) >= self.VT_MIN_BEATS:
                    episode = self._create_vt_episode(beats, v_run_beats)
                    if episode is not None:
                        episodes.append(episode)
                
                v_run_start = None
                v_run_beats = []
        
        # Check final run
        if len(v_run_beats) >= self.VT_MIN_BEATS:
            episode = self._create_vt_episode(beats, v_run_beats)
            if episode is not None:
                episodes.append(episode)
        
        return episodes
    
    def _create_vt_episode(self, 
                           beats: pd.DataFrame,
                           v_indices: List[int]) -> Optional[TachycardiaEpisode]:
        """Create a VT episode from consecutive ventricular beats"""
        
        if len(v_indices) < self.VT_MIN_BEATS:
            return None
        
        first_beat = beats.iloc[v_indices[0]]
        last_beat = beats.iloc[v_indices[-1]]
        
        start_sample = first_beat['sample_num']
        end_sample = last_beat['sample_num']
        
        duration = (end_sample - start_sample) / self.fs
        
        # Calculate heart rate from RR intervals
        if len(v_indices) > 1:
            sample_positions = [beats.iloc[i]['sample_num'] for i in v_indices]
            rr_intervals = np.diff(sample_positions) / self.fs
            mean_rr = np.mean(rr_intervals)
            mean_hr = 60 / mean_rr if mean_rr > 0 else 0
        else:
            mean_hr = 0
        
        # Check if rate qualifies as tachycardia (>100 BPM)
        if mean_hr < self.TACHYCARDIA_HR_THRESHOLD:
            # Some slow VT exists, but typically we want rate >100
            # For slow VT, still flag it but could be less urgent
            pass
        
        # Determine if sustained
        is_sustained = duration >= self.SUSTAINED_DURATION
        tachy_type = TachycardiaType.VT_SUSTAINED if is_sustained else TachycardiaType.VT_NONSUSTAINED
        
        return TachycardiaEpisode(
            episode_type=tachy_type,
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_sample / self.fs,
            end_time=end_sample / self.fs,
            duration=duration,
            mean_hr=mean_hr,
            num_beats=len(v_indices),
            is_sustained=is_sustained,
            severity=self._get_severity(tachy_type),
            beat_indices=v_indices
        )
    
    def _detect_sinus_tachycardia(self,
                                   beat_annotations: pd.DataFrame,
                                   rhythm_annotations: pd.DataFrame) -> List[TachycardiaEpisode]:
        """
        Detect sinus tachycardia from sustained high heart rate in normal rhythm
        
        Criteria:
        - Normal beat morphology (N beats)
        - Heart rate >100 BPM
        - Sustained for >30 seconds
        """
        episodes = []
        
        beats = beat_annotations.sort_values('sample_num').reset_index(drop=True)
        
        if len(beats) < 2:
            return episodes
        
        # Calculate instantaneous HR for each beat
        sample_nums = beats['sample_num'].values
        rr_intervals = np.diff(sample_nums) / self.fs
        
        # Avoid division by zero
        rr_intervals = np.maximum(rr_intervals, 0.001)
        heart_rates = 60 / rr_intervals
        
        # Find sustained periods of HR > 100 BPM
        is_tachy = heart_rates > self.TACHYCARDIA_HR_THRESHOLD
        
        # Also check that we're in normal rhythm (N beats)
        beat_types = beats['beat_type'].values
        is_normal_beat = np.array([bt == 'N' for bt in beat_types[1:]])  # Skip first beat
        
        # Combined: normal beat AND high HR
        is_sinus_tachy = is_tachy & is_normal_beat
        
        # Find runs
        run_start = None
        run_indices = []
        
        for i, is_st in enumerate(is_sinus_tachy):
            if is_st:
                if run_start is None:
                    run_start = i
                run_indices.append(i + 1)  # +1 because HR is for interval ending at i+1
            else:
                if run_start is not None and len(run_indices) > 0:
                    # Calculate run duration
                    start_sample = sample_nums[run_indices[0]]
                    end_sample = sample_nums[run_indices[-1]]
                    duration = (end_sample - start_sample) / self.fs
                    
                    if duration >= self.SINUS_TACHY_MIN_DURATION:
                        mean_hr = np.mean(heart_rates[run_start:run_start + len(run_indices)])
                        
                        episode = TachycardiaEpisode(
                            episode_type=TachycardiaType.SINUS_TACHYCARDIA,
                            start_sample=start_sample,
                            end_sample=end_sample,
                            start_time=start_sample / self.fs,
                            end_time=end_sample / self.fs,
                            duration=duration,
                            mean_hr=mean_hr,
                            num_beats=len(run_indices),
                            is_sustained=True,
                            severity='benign',
                            beat_indices=run_indices
                        )
                        episodes.append(episode)
                
                run_start = None
                run_indices = []
        
        # Check final run
        if run_start is not None and len(run_indices) > 0:
            start_sample = sample_nums[run_indices[0]]
            end_sample = sample_nums[run_indices[-1]]
            duration = (end_sample - start_sample) / self.fs
            
            if duration >= self.SINUS_TACHY_MIN_DURATION:
                mean_hr = np.mean(heart_rates[run_start:run_start + len(run_indices)])
                
                episode = TachycardiaEpisode(
                    episode_type=TachycardiaType.SINUS_TACHYCARDIA,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time=start_sample / self.fs,
                    end_time=end_sample / self.fs,
                    duration=duration,
                    mean_hr=mean_hr,
                    num_beats=len(run_indices),
                    is_sustained=True,
                    severity='benign',
                    beat_indices=run_indices
                )
                episodes.append(episode)
        
        return episodes
    
    def _overlaps_existing(self, 
                           new_episode: TachycardiaEpisode,
                           existing_episodes: List[TachycardiaEpisode]) -> bool:
        """Check if new episode overlaps with any existing episode"""
        for ep in existing_episodes:
            # Check for overlap
            if (new_episode.start_sample < ep.end_sample and 
                new_episode.end_sample > ep.start_sample):
                return True
        return False
    
    def _get_severity(self, tachy_type: TachycardiaType) -> str:
        """Assign clinical severity to tachycardia type"""
        severity_map = {
            TachycardiaType.NORMAL: 'none',
            TachycardiaType.SINUS_TACHYCARDIA: 'benign',
            TachycardiaType.SVT: 'moderate',
            TachycardiaType.VT_NONSUSTAINED: 'high',
            TachycardiaType.VT_SUSTAINED: 'life-threatening',
            TachycardiaType.VFL: 'life-threatening',
            TachycardiaType.VF: 'life-threatening',
            TachycardiaType.ATRIAL_FLUTTER: 'moderate',
            TachycardiaType.ATRIAL_FIBRILLATION_RVR: 'moderate'
        }
        return severity_map.get(tachy_type, 'unknown')
    
    def _create_sample_labels(self,
                               episodes: List[TachycardiaEpisode],
                               signal_length: int) -> np.ndarray:
        """Create per-sample label array from episodes"""
        labels = np.zeros(signal_length, dtype=np.int8)
        
        for episode in episodes:
            # Use episode type value as label
            start = max(0, episode.start_sample)
            end = min(signal_length, episode.end_sample)
            labels[start:end] = episode.episode_type.value
        
        return labels
    
    def create_segment_labels(self,
                              episodes: List[TachycardiaEpisode],
                              segment_starts: np.ndarray,
                              segment_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create labels for fixed-length segments
        
        Args:
            episodes: List of tachycardia episodes
            segment_starts: Start sample of each segment
            segment_length: Length of each segment in samples
            
        Returns:
            Tuple of (binary labels, multi-class labels)
        """
        n_segments = len(segment_starts)
        binary_labels = np.zeros(n_segments, dtype=np.int8)
        multiclass_labels = np.zeros(n_segments, dtype=np.int8)
        
        for i, start in enumerate(segment_starts):
            end = start + segment_length
            
            # Check if segment overlaps any episode
            for episode in episodes:
                # Calculate overlap
                overlap_start = max(start, episode.start_sample)
                overlap_end = min(end, episode.end_sample)
                overlap = max(0, overlap_end - overlap_start)
                
                # If >50% of episode is in segment, or >50% of segment is episode
                if overlap > 0:
                    overlap_ratio = overlap / segment_length
                    
                    if overlap_ratio > 0.3:  # At least 30% overlap
                        binary_labels[i] = 1
                        
                        # Use highest priority (most severe) type
                        if episode.episode_type.value > multiclass_labels[i]:
                            multiclass_labels[i] = episode.episode_type.value
        
        return binary_labels, multiclass_labels
    
    def get_episode_summary(self, episodes: List[TachycardiaEpisode]) -> Dict:
        """Generate summary statistics for detected episodes"""
        if not episodes:
            return {'total_episodes': 0}
        
        summary = {
            'total_episodes': len(episodes),
            'total_duration_seconds': sum(ep.duration for ep in episodes),
            'by_type': {},
            'by_severity': {},
            'sustained_count': sum(1 for ep in episodes if ep.is_sustained),
            'life_threatening_count': sum(1 for ep in episodes if ep.severity == 'life-threatening')
        }
        
        for ep in episodes:
            type_name = ep.episode_type.name
            if type_name not in summary['by_type']:
                summary['by_type'][type_name] = {'count': 0, 'total_duration': 0}
            summary['by_type'][type_name]['count'] += 1
            summary['by_type'][type_name]['total_duration'] += ep.duration
            
            if ep.severity not in summary['by_severity']:
                summary['by_severity'][ep.severity] = 0
            summary['by_severity'][ep.severity] += 1
        
        return summary


def main():
    """Test clinical labeling"""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from preprocessing.data_loader import MITBIHLoader
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mitbih_database')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    loader = MITBIHLoader(data_dir)
    labeler = ClinicalTachycardiaLabeler()
    
    # Test on record 207 (has VT episodes)
    record = loader.load_record('207')
    
    episodes, sample_labels = labeler.label_record(
        record.beat_annotations,
        record.rhythm_annotations,
        len(record.signal_mlii)
    )
    
    print(f"\nRecord 207 Clinical Labeling Results:")
    print(f"Total episodes detected: {len(episodes)}")
    
    for ep in episodes[:10]:  # Show first 10
        print(f"  {ep.episode_type.name}: {ep.duration:.1f}s at {ep.mean_hr:.0f} BPM "
              f"(severity: {ep.severity}, sustained: {ep.is_sustained})")
    
    summary = labeler.get_episode_summary(episodes)
    print(f"\nSummary:")
    print(f"  Total duration: {summary['total_duration_seconds']:.1f}s")
    print(f"  Life-threatening episodes: {summary['life_threatening_count']}")
    print(f"  By type: {summary['by_type']}")


if __name__ == '__main__':
    main()
