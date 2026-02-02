"""
MIT-BIH Database Loader
Loads ECG signals and annotations from the MIT-BIH Arrhythmia Database
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re


@dataclass
class ECGRecord:
    """Data class for storing ECG record information"""
    record_id: str
    signal_mlii: np.ndarray  # Lead MLII signal
    signal_v5: np.ndarray    # Lead V5 signal
    sampling_rate: int       # Hz
    beat_annotations: pd.DataFrame
    rhythm_annotations: pd.DataFrame
    duration_seconds: float


@dataclass
class Annotation:
    """Data class for beat/rhythm annotations"""
    time: str
    sample_num: int
    ann_type: str
    sub: int
    chan: int
    num: int
    aux: str


class MITBIHLoader:
    """
    Loader for MIT-BIH Arrhythmia Database
    
    The MIT-BIH database contains 48 half-hour excerpts of two-channel
    ambulatory ECG recordings from 47 subjects.
    """
    
    SAMPLING_RATE = 360  # Hz
    
    # Beat annotation symbols
    BEAT_ANNOTATIONS = {
        'N': 'Normal beat',
        'L': 'Left bundle branch block beat',
        'R': 'Right bundle branch block beat',
        'A': 'Atrial premature beat',
        'a': 'Aberrated atrial premature beat',
        'J': 'Nodal (junctional) premature beat',
        'S': 'Supraventricular premature beat',
        'V': 'Premature ventricular contraction',
        'F': 'Fusion of ventricular and normal beat',
        '/': 'Paced beat',
        'f': 'Fusion of paced and normal beat',
        'j': 'Nodal (junctional) escape beat',
        'E': 'Ventricular escape beat',
        'e': 'Atrial escape beat',
        'Q': 'Unclassifiable beat',
        '!': 'Ventricular flutter wave',
        '|': 'Isolated QRS-like artifact',
        'x': 'Non-conducted P-wave',
        '~': 'Change in signal quality'
    }
    
    # Rhythm annotation symbols (in Aux field)
    RHYTHM_ANNOTATIONS = {
        '(N': 'Normal sinus rhythm',
        '(AFIB': 'Atrial fibrillation',
        '(AFL': 'Atrial flutter',
        '(B': 'Ventricular bigeminy',
        '(T': 'Ventricular trigeminy / Sinus tachycardia',
        '(VT': 'Ventricular tachycardia',
        '(VFL': 'Ventricular flutter',
        '(SVTA': 'Supraventricular tachyarrhythmia',
        '(IVR': 'Idioventricular rhythm',
        '(NOD': 'Nodal rhythm',
        '(P': 'Paced rhythm',
        '(PREX': 'Pre-excitation',
        '(SBR': 'Sinus bradycardia',
        '(BII': 'Second degree heart block',
        '(AB': 'Atrial bigeminy'
    }
    
    # Tachycardia-related rhythms
    TACHYCARDIA_RHYTHMS = ['(VT', '(VFL', '(SVTA', '(T', '(AFL', '(AFIB']
    
    def __init__(self, data_dir: str):
        """
        Initialize the loader
        
        Args:
            data_dir: Path to the mitbih_database folder
        """
        self.data_dir = data_dir
        self.records = self._discover_records()
        
    def _discover_records(self) -> List[str]:
        """Discover all available record IDs"""
        records = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                record_id = file.replace('.csv', '')
                ann_file = f"{record_id}annotations.txt"
                if os.path.exists(os.path.join(self.data_dir, ann_file)):
                    records.append(record_id)
        return sorted(records)
    
    def load_signal(self, record_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ECG signal from CSV file
        
        Args:
            record_id: Record identifier (e.g., '100', '101')
            
        Returns:
            Tuple of (MLII signal, V5 signal) as numpy arrays
        """
        csv_path = os.path.join(self.data_dir, f"{record_id}.csv")
        
        # Read CSV with proper column handling
        df = pd.read_csv(csv_path)
        
        # Extract signals (columns are: sample #, MLII, V5)
        signal_mlii = df.iloc[:, 1].values.astype(np.float64)
        signal_v5 = df.iloc[:, 2].values.astype(np.float64)
        
        return signal_mlii, signal_v5
    
    def load_annotations(self, record_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and parse annotations from text file
        
        Args:
            record_id: Record identifier
            
        Returns:
            Tuple of (beat_annotations, rhythm_annotations) DataFrames
        """
        ann_path = os.path.join(self.data_dir, f"{record_id}annotations.txt")
        
        beat_annotations = []
        rhythm_annotations = []
        current_rhythm = '(N'  # Default to normal sinus rhythm
        
        with open(ann_path, 'r') as f:
            lines = f.readlines()
            
        # Skip header line
        for line in lines[1:]:
            parsed = self._parse_annotation_line(line)
            if parsed is None:
                continue
                
            time, sample_num, ann_type, sub, chan, num, aux = parsed
            
            # Check if this is a rhythm change annotation
            if aux and aux.startswith('('):
                current_rhythm = aux
                rhythm_annotations.append({
                    'time': time,
                    'sample_num': sample_num,
                    'rhythm': aux,
                    'rhythm_description': self.RHYTHM_ANNOTATIONS.get(aux, 'Unknown'),
                    'is_tachycardia': aux in self.TACHYCARDIA_RHYTHMS
                })
            
            # Beat annotations (single character types)
            if ann_type in self.BEAT_ANNOTATIONS:
                beat_annotations.append({
                    'time': time,
                    'sample_num': sample_num,
                    'beat_type': ann_type,
                    'beat_description': self.BEAT_ANNOTATIONS[ann_type],
                    'current_rhythm': current_rhythm,
                    'is_tachycardia_rhythm': current_rhythm in self.TACHYCARDIA_RHYTHMS
                })
        
        beat_df = pd.DataFrame(beat_annotations)
        rhythm_df = pd.DataFrame(rhythm_annotations)
        
        return beat_df, rhythm_df
    
    def _parse_annotation_line(self, line: str) -> Optional[Tuple]:
        """Parse a single annotation line"""
        line = line.strip()
        if not line:
            return None
            
        # Pattern: Time Sample# Type Sub Chan Num [Aux]
        # Example: "0:00.214       77     N    0    0    0"
        parts = line.split()
        if len(parts) < 6:
            return None
            
        try:
            time = parts[0]
            sample_num = int(parts[1])
            ann_type = parts[2]
            sub = int(parts[3])
            chan = int(parts[4])
            num = int(parts[5])
            aux = parts[6] if len(parts) > 6 else ''
            
            return time, sample_num, ann_type, sub, chan, num, aux
        except (ValueError, IndexError):
            return None
    
    def load_record(self, record_id: str) -> ECGRecord:
        """
        Load complete ECG record with signals and annotations
        
        Args:
            record_id: Record identifier
            
        Returns:
            ECGRecord object with all data
        """
        signal_mlii, signal_v5 = self.load_signal(record_id)
        beat_ann, rhythm_ann = self.load_annotations(record_id)
        
        duration = len(signal_mlii) / self.SAMPLING_RATE
        
        return ECGRecord(
            record_id=record_id,
            signal_mlii=signal_mlii,
            signal_v5=signal_v5,
            sampling_rate=self.SAMPLING_RATE,
            beat_annotations=beat_ann,
            rhythm_annotations=rhythm_ann,
            duration_seconds=duration
        )
    
    def load_all_records(self) -> Dict[str, ECGRecord]:
        """Load all available records"""
        records = {}
        for record_id in self.records:
            print(f"Loading record {record_id}...")
            records[record_id] = self.load_record(record_id)
        return records
    
    def get_tachycardia_segments(self, record: ECGRecord) -> List[Dict]:
        """
        Extract tachycardia segments from a record
        
        Args:
            record: ECGRecord object
            
        Returns:
            List of tachycardia segment dictionaries
        """
        segments = []
        rhythm_df = record.rhythm_annotations
        
        if rhythm_df.empty:
            return segments
            
        for i, row in rhythm_df.iterrows():
            if row['is_tachycardia']:
                start_sample = row['sample_num']
                
                # Find end of this rhythm (next rhythm change or end of record)
                next_rhythms = rhythm_df[rhythm_df['sample_num'] > start_sample]
                if not next_rhythms.empty:
                    end_sample = next_rhythms.iloc[0]['sample_num']
                else:
                    end_sample = len(record.signal_mlii)
                
                segments.append({
                    'record_id': record.record_id,
                    'rhythm': row['rhythm'],
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'duration_seconds': (end_sample - start_sample) / record.sampling_rate,
                    'signal_mlii': record.signal_mlii[start_sample:end_sample],
                    'signal_v5': record.signal_v5[start_sample:end_sample]
                })
                
        return segments
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary statistics for the entire dataset"""
        summary = []
        
        for record_id in self.records:
            record = self.load_record(record_id)
            beat_ann = record.beat_annotations
            rhythm_ann = record.rhythm_annotations
            
            # Count beats by type
            beat_counts = beat_ann['beat_type'].value_counts().to_dict()
            
            # Count tachycardia episodes
            tachy_episodes = rhythm_ann['is_tachycardia'].sum() if not rhythm_ann.empty else 0
            
            # Count beats during tachycardia
            tachy_beats = beat_ann['is_tachycardia_rhythm'].sum() if not beat_ann.empty else 0
            
            summary.append({
                'record_id': record_id,
                'duration_min': record.duration_seconds / 60,
                'total_beats': len(beat_ann),
                'normal_beats': beat_counts.get('N', 0),
                'pvc_beats': beat_counts.get('V', 0),
                'tachycardia_episodes': tachy_episodes,
                'beats_during_tachycardia': tachy_beats,
                **beat_counts
            })
            
        return pd.DataFrame(summary)


def main():
    """Test the data loader"""
    import sys
    
    # Get the data directory
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'mitbih_database')
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    loader = MITBIHLoader(data_dir)
    
    print(f"Found {len(loader.records)} records: {loader.records}")
    
    # Load first record as test
    record = loader.load_record(loader.records[0])
    print(f"\nRecord {record.record_id}:")
    print(f"  Duration: {record.duration_seconds:.1f} seconds")
    print(f"  Signal shape: {record.signal_mlii.shape}")
    print(f"  Total beats: {len(record.beat_annotations)}")
    print(f"  Beat types: {record.beat_annotations['beat_type'].value_counts().to_dict()}")
    
    # Get tachycardia segments
    tachy_segments = loader.get_tachycardia_segments(record)
    print(f"  Tachycardia segments: {len(tachy_segments)}")
    
    
if __name__ == '__main__':
    main()
