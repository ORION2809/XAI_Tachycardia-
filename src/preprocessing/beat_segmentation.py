"""
Beat Segmentation Module
Segments ECG signals into individual heartbeats for classification
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Optional
import pandas as pd
from .signal_processing import QRSDetector, SignalProcessor


class BeatSegmenter:
    """
    Segments ECG signal into individual beats centered on R-peaks
    
    Each beat window captures the P-wave, QRS complex, and T-wave.
    """
    
    def __init__(self, sampling_rate: int = 360,
                 pre_r_samples: int = 72,    # 200 ms before R-peak
                 post_r_samples: int = 144):  # 400 ms after R-peak
        """
        Initialize beat segmenter
        
        Args:
            sampling_rate: Sampling frequency in Hz
            pre_r_samples: Number of samples before R-peak to include
            post_r_samples: Number of samples after R-peak to include
        """
        self.fs = sampling_rate
        self.pre_r = pre_r_samples
        self.post_r = post_r_samples
        self.beat_length = pre_r_samples + post_r_samples
        
        self.qrs_detector = QRSDetector(sampling_rate)
        self.signal_processor = SignalProcessor(sampling_rate)
        
    def segment_beats(self, ecg_signal: np.ndarray,
                      r_peaks: Optional[np.ndarray] = None,
                      preprocess: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ECG signal into individual beats
        
        Args:
            ecg_signal: ECG signal (single lead)
            r_peaks: R-peak locations (if None, will detect automatically)
            preprocess: Whether to preprocess the signal first
            
        Returns:
            Tuple of (beats array [n_beats, beat_length], r_peak indices)
        """
        # Preprocess if requested
        if preprocess:
            ecg_signal = self.signal_processor.full_preprocessing(ecg_signal)
        
        # Detect R-peaks if not provided
        if r_peaks is None:
            r_peaks = self.qrs_detector.detect_r_peaks(ecg_signal)
        
        beats = []
        valid_peaks = []
        
        for r_peak in r_peaks:
            start = r_peak - self.pre_r
            end = r_peak + self.post_r
            
            # Skip beats at boundaries
            if start < 0 or end > len(ecg_signal):
                continue
            
            beat = ecg_signal[start:end]
            beats.append(beat)
            valid_peaks.append(r_peak)
        
        if len(beats) == 0:
            return np.array([]).reshape(0, self.beat_length), np.array([])
            
        return np.array(beats), np.array(valid_peaks)
    
    def segment_beats_with_labels(self, ecg_signal: np.ndarray,
                                   beat_annotations: pd.DataFrame,
                                   preprocess: bool = True) -> Dict:
        """
        Segment beats and assign labels from annotations
        
        Args:
            ecg_signal: ECG signal
            beat_annotations: DataFrame with 'sample_num', 'beat_type', etc.
            preprocess: Whether to preprocess signal
            
        Returns:
            Dictionary with beats, labels, and metadata
        """
        if preprocess:
            ecg_signal = self.signal_processor.full_preprocessing(ecg_signal)
        
        beats = []
        labels = []
        beat_types = []
        is_tachycardia = []
        sample_positions = []
        
        for _, row in beat_annotations.iterrows():
            r_peak = row['sample_num']
            start = r_peak - self.pre_r
            end = r_peak + self.post_r
            
            # Skip beats at boundaries
            if start < 0 or end > len(ecg_signal):
                continue
            
            beat = ecg_signal[start:end]
            beats.append(beat)
            beat_types.append(row['beat_type'])
            is_tachycardia.append(row.get('is_tachycardia_rhythm', False))
            sample_positions.append(r_peak)
        
        return {
            'beats': np.array(beats) if beats else np.array([]).reshape(0, self.beat_length),
            'beat_types': beat_types,
            'is_tachycardia': is_tachycardia,
            'sample_positions': sample_positions,
            'n_beats': len(beats)
        }
    
    def segment_with_context(self, ecg_signal: np.ndarray,
                              r_peaks: np.ndarray,
                              context_beats: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment beats with surrounding context beats
        
        Useful for capturing rhythm patterns (e.g., sequences of fast beats).
        
        Args:
            ecg_signal: ECG signal
            r_peaks: R-peak locations
            context_beats: Number of beats before and after to include
            
        Returns:
            Tuple of (sequence array, center indices)
        """
        # First get individual beats
        beats, valid_peaks = self.segment_beats(ecg_signal, r_peaks, preprocess=False)
        
        if len(beats) < 2 * context_beats + 1:
            return np.array([]), np.array([])
        
        sequences = []
        center_indices = []
        
        for i in range(context_beats, len(beats) - context_beats):
            sequence = beats[i - context_beats:i + context_beats + 1]
            sequences.append(sequence.flatten())
            center_indices.append(i)
        
        return np.array(sequences), np.array(center_indices)
    
    def extract_rr_features(self, r_peaks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract RR interval features for each beat
        
        Args:
            r_peaks: R-peak sample positions
            
        Returns:
            Dictionary of RR features per beat
        """
        rr_intervals = np.diff(r_peaks) / self.fs  # in seconds
        heart_rates = 60 / rr_intervals  # BPM
        
        # For each beat, compute local RR statistics
        features = {
            'rr_current': np.zeros(len(r_peaks)),
            'rr_previous': np.zeros(len(r_peaks)),
            'rr_ratio': np.zeros(len(r_peaks)),
            'hr_current': np.zeros(len(r_peaks)),
            'is_tachycardia_hr': np.zeros(len(r_peaks), dtype=bool)
        }
        
        for i in range(1, len(r_peaks)):
            features['rr_current'][i] = rr_intervals[i-1]
            features['hr_current'][i] = heart_rates[i-1]
            features['is_tachycardia_hr'][i] = heart_rates[i-1] > 100
            
            if i > 1:
                features['rr_previous'][i] = rr_intervals[i-2]
                if rr_intervals[i-2] > 0:
                    features['rr_ratio'][i] = rr_intervals[i-1] / rr_intervals[i-2]
        
        return features


class TachycardiaLabeler:
    """
    Labels beats and segments as tachycardia or normal
    
    Combines rhythm annotations and heart rate analysis.
    """
    
    TACHYCARDIA_RHYTHMS = {'(VT', '(VFL', '(SVTA', '(T', '(AFL'}
    HR_THRESHOLD = 100  # BPM
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize labeler
        
        Args:
            sampling_rate: Sampling frequency
        """
        self.fs = sampling_rate
        
    def label_beats(self, r_peaks: np.ndarray,
                    rhythm_annotations: pd.DataFrame,
                    signal_length: int) -> np.ndarray:
        """
        Label each beat as tachycardia (1) or normal (0)
        
        Uses rhythm annotations primarily, with HR as secondary indicator.
        
        Args:
            r_peaks: R-peak sample positions
            rhythm_annotations: DataFrame with rhythm change events
            signal_length: Total signal length
            
        Returns:
            Binary label array for each beat
        """
        labels = np.zeros(len(r_peaks), dtype=int)
        
        if rhythm_annotations.empty:
            # Fall back to HR-based labeling
            return self._label_by_heart_rate(r_peaks)
        
        # Create rhythm segments
        rhythm_changes = rhythm_annotations.sort_values('sample_num')
        
        for i, r_peak in enumerate(r_peaks):
            # Find which rhythm this beat belongs to
            preceding_rhythms = rhythm_changes[rhythm_changes['sample_num'] <= r_peak]
            
            if not preceding_rhythms.empty:
                current_rhythm = preceding_rhythms.iloc[-1]['rhythm']
                if current_rhythm in self.TACHYCARDIA_RHYTHMS:
                    labels[i] = 1
        
        return labels
    
    def _label_by_heart_rate(self, r_peaks: np.ndarray) -> np.ndarray:
        """Label based on instantaneous heart rate"""
        labels = np.zeros(len(r_peaks), dtype=int)
        
        if len(r_peaks) < 2:
            return labels
        
        rr_intervals = np.diff(r_peaks) / self.fs
        heart_rates = 60 / rr_intervals
        
        for i in range(1, len(r_peaks)):
            if heart_rates[i-1] > self.HR_THRESHOLD:
                labels[i] = 1
        
        return labels
    
    def get_tachycardia_type(self, r_peak: int,
                              rhythm_annotations: pd.DataFrame) -> str:
        """
        Get specific tachycardia type for a beat
        
        Args:
            r_peak: R-peak sample position
            rhythm_annotations: Rhythm annotation DataFrame
            
        Returns:
            Tachycardia type string or 'Normal'
        """
        if rhythm_annotations.empty:
            return 'Unknown'
        
        preceding = rhythm_annotations[rhythm_annotations['sample_num'] <= r_peak]
        
        if not preceding.empty:
            rhythm = preceding.iloc[-1]['rhythm']
            
            type_mapping = {
                '(VT': 'Ventricular Tachycardia',
                '(VFL': 'Ventricular Flutter',
                '(SVTA': 'Supraventricular Tachycardia',
                '(T': 'Sinus Tachycardia',
                '(AFL': 'Atrial Flutter',
                '(AFIB': 'Atrial Fibrillation',
                '(N': 'Normal Sinus Rhythm'
            }
            
            return type_mapping.get(rhythm, rhythm)
        
        return 'Unknown'
    
    def create_multiclass_labels(self, r_peaks: np.ndarray,
                                  rhythm_annotations: pd.DataFrame) -> np.ndarray:
        """
        Create multi-class labels for tachycardia types
        
        Classes:
        0 - Normal
        1 - Sinus Tachycardia (T)
        2 - Supraventricular Tachycardia (SVTA)
        3 - Ventricular Tachycardia (VT)
        4 - Ventricular Flutter (VFL)
        5 - Other arrhythmia
        
        Args:
            r_peaks: R-peak positions
            rhythm_annotations: Rhythm annotations
            
        Returns:
            Multi-class label array
        """
        class_mapping = {
            '(N': 0,
            '(T': 1,
            '(SVTA': 2,
            '(VT': 3,
            '(VFL': 4
        }
        
        labels = np.zeros(len(r_peaks), dtype=int)
        
        if rhythm_annotations.empty:
            return labels
        
        rhythm_changes = rhythm_annotations.sort_values('sample_num')
        
        for i, r_peak in enumerate(r_peaks):
            preceding = rhythm_changes[rhythm_changes['sample_num'] <= r_peak]
            
            if not preceding.empty:
                rhythm = preceding.iloc[-1]['rhythm']
                labels[i] = class_mapping.get(rhythm, 5)
        
        return labels


def create_dataset_from_records(records: Dict, 
                                output_dir: Optional[str] = None) -> Dict:
    """
    Create a complete dataset from all ECG records
    
    Args:
        records: Dictionary of ECGRecord objects
        output_dir: Optional directory to save processed data
        
    Returns:
        Dictionary containing processed beats and labels
    """
    segmenter = BeatSegmenter()
    labeler = TachycardiaLabeler()
    
    all_beats = []
    all_binary_labels = []
    all_multiclass_labels = []
    all_record_ids = []
    all_beat_types = []
    all_rr_features = []
    
    for record_id, record in records.items():
        print(f"Processing record {record_id}...")
        
        # Segment beats from MLII lead
        result = segmenter.segment_beats_with_labels(
            record.signal_mlii,
            record.beat_annotations
        )
        
        if result['n_beats'] == 0:
            continue
        
        # Get R-peaks from annotations
        r_peaks = np.array(result['sample_positions'])
        
        # Create labels
        binary_labels = labeler.label_beats(
            r_peaks, 
            record.rhythm_annotations,
            len(record.signal_mlii)
        )
        
        multiclass_labels = labeler.create_multiclass_labels(
            r_peaks,
            record.rhythm_annotations
        )
        
        # Extract RR features
        rr_features = segmenter.extract_rr_features(r_peaks)
        
        # Store
        all_beats.extend(result['beats'])
        all_binary_labels.extend(binary_labels)
        all_multiclass_labels.extend(multiclass_labels)
        all_record_ids.extend([record_id] * result['n_beats'])
        all_beat_types.extend(result['beat_types'])
        
        for key in rr_features:
            if key not in [f['name'] for f in all_rr_features]:
                all_rr_features.append({'name': key, 'values': []})
            for f in all_rr_features:
                if f['name'] == key:
                    f['values'].extend(rr_features[key])
    
    dataset = {
        'beats': np.array(all_beats),
        'binary_labels': np.array(all_binary_labels),
        'multiclass_labels': np.array(all_multiclass_labels),
        'record_ids': all_record_ids,
        'beat_types': all_beat_types,
        'rr_features': {f['name']: np.array(f['values']) for f in all_rr_features}
    }
    
    print(f"\nDataset Summary:")
    print(f"  Total beats: {len(dataset['beats'])}")
    print(f"  Tachycardia beats: {np.sum(dataset['binary_labels'])}")
    print(f"  Normal beats: {np.sum(dataset['binary_labels'] == 0)}")
    print(f"  Class distribution: {np.bincount(dataset['multiclass_labels'])}")
    
    return dataset


def main():
    """Test beat segmentation"""
    import os
    from .data_loader import MITBIHLoader
    
    # Load a test record
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mitbih_database')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    loader = MITBIHLoader(data_dir)
    record = loader.load_record('207')  # Has tachycardia episodes
    
    # Segment beats
    segmenter = BeatSegmenter()
    result = segmenter.segment_beats_with_labels(
        record.signal_mlii,
        record.beat_annotations
    )
    
    print(f"Record 207:")
    print(f"  Total beats segmented: {result['n_beats']}")
    print(f"  Beat shape: {result['beats'].shape if result['n_beats'] > 0 else 'N/A'}")
    print(f"  Tachycardia beats: {sum(result['is_tachycardia'])}")
    
    # Test labeling
    labeler = TachycardiaLabeler()
    r_peaks = np.array(result['sample_positions'])
    labels = labeler.label_beats(r_peaks, record.rhythm_annotations, len(record.signal_mlii))
    
    print(f"  Labeled as tachycardia: {np.sum(labels)}")
    print(f"  Labeled as normal: {np.sum(labels == 0)}")


if __name__ == '__main__':
    main()
