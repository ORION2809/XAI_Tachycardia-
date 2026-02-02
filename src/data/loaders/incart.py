"""
INCART Database Loader.

v2.4: Load and process INCART database with explicit contract compliance.

INCART Database Characteristics:
- 75 records from 32 patients
- 12-lead ECG, 30 minutes each
- Native sampling rate: 257 Hz
- Beat-level annotations available
- NO rhythm annotations

CRITICAL LIMITATION:
- Cannot derive TRUE VT from INCART (no rhythm annotations)
- Can only derive VENTRICULAR_RUN (≥3 consecutive V beats)
- Report as "V-run sensitivity", NOT "VT sensitivity"

Reference:
- PhysioNet: https://physionet.org/content/incartdb/1.0.0/
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from scipy.signal import resample_poly
from scipy.interpolate import interp1d
import logging

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# INCART CONTRACT (imported from harmonization)
# =============================================================================

# Re-export contract for convenience
INCART_NATIVE_FS = 257
INCART_TARGET_FS = 360
INCART_LEAD_TO_USE = "II"

# Beat label mapping (INCART → canonical)
INCART_BEAT_MAP = {
    'N': 'N',   # Normal
    'V': 'V',   # Ventricular ectopic
    'S': 'S',   # Supraventricular ectopic
    'F': 'F',   # Fusion
    # INCART also has:
    # 'Q': Unknown
    # 'n': Noise
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class INCARTRecord:
    """Single INCART record with all metadata."""
    record_id: str
    signal: np.ndarray          # Shape: (n_leads, n_samples)
    lead_names: List[str]
    fs: int                     # Sampling frequency (after resampling)
    
    # Annotations
    beat_samples: np.ndarray    # R-peak locations
    beat_labels: np.ndarray     # Beat type labels (str array)
    
    # Derived
    ventricular_runs: List[Tuple[int, int]] = field(default_factory=list)
    duration_sec: float = 0.0
    patient_id: Optional[str] = None
    
    def get_lead(self, lead_name: str) -> Optional[np.ndarray]:
        """Get signal for specific lead."""
        if lead_name in self.lead_names:
            idx = self.lead_names.index(lead_name)
            return self.signal[idx]
        return None
    
    def get_lead_ii(self) -> Optional[np.ndarray]:
        """Get Lead II (canonical lead)."""
        return self.get_lead("II")


@dataclass
class VentricularRun:
    """
    A ventricular run (NOT VT - we don't have rhythm confirmation).
    
    v2.3: Explicit that this is NOT a true VT episode.
    INCART lacks rhythm annotations, so we can only detect V-runs.
    """
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    n_beats: int
    hr_bpm: Optional[float] = None
    
    # Explicit flag
    is_confirmed_vt: bool = False  # Always False for INCART
    confidence_tier: str = "heuristic"  # From beat labels only
    
    @property
    def duration_sec(self) -> float:
        return self.end_time_sec - self.start_time_sec


# =============================================================================
# INCART LOADER
# =============================================================================

class INCARTLoader:
    """
    Load and process INCART database.
    
    CRITICAL: This loader explicitly handles INCART's limitations:
    - Can detect ventricular RUNS (≥3 V beats)
    - Cannot confirm VT (no rhythm annotations)
    - All "VT-like" detections are labeled as VENTRICULAR_RUN
    
    Usage:
        loader = INCARTLoader(data_dir="/path/to/incart")
        records = loader.load_all()
        
        # Get V-runs for evaluation
        for record in records:
            v_runs = loader.detect_ventricular_runs(record)
    """
    
    # Standard INCART lead order
    LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", 
                  "V1", "V2", "V3", "V4", "V5", "V6"]
    
    def __init__(
        self,
        data_dir: str,
        target_fs: int = 360,
        lead_to_use: str = "II",
        resample: bool = True,
    ):
        """
        Initialize INCART loader.
        
        Args:
            data_dir: Path to INCART database directory
            target_fs: Target sampling frequency (default: 360 Hz for MIT-BIH compat)
            lead_to_use: Primary lead to use (default: II)
            resample: Whether to resample to target_fs
        """
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.lead_to_use = lead_to_use
        self.resample = resample
        
        # Validate directory
        if not self.data_dir.exists():
            logger.warning(f"INCART data directory not found: {data_dir}")
    
    def _get_record_files(self) -> List[str]:
        """Get list of record IDs in the database."""
        # INCART records are numbered I01-I75
        record_ids = []
        
        # Check for .dat files (WFDB format)
        dat_files = list(self.data_dir.glob("I*.dat"))
        for f in dat_files:
            record_id = f.stem
            record_ids.append(record_id)
        
        # Also check for CSV format (alternative storage)
        csv_files = list(self.data_dir.glob("I*.csv"))
        for f in csv_files:
            record_id = f.stem
            if record_id not in record_ids:
                record_ids.append(record_id)
        
        return sorted(record_ids)
    
    def _resample_signal(
        self,
        signal: np.ndarray,
        orig_fs: int,
        target_fs: int,
    ) -> np.ndarray:
        """
        Resample signal from orig_fs to target_fs.
        
        Uses polyphase resampling for efficiency.
        """
        if orig_fs == target_fs:
            return signal
        
        # For 257 → 360, we use interpolation
        # GCD(257, 360) = 1, so use scipy resample_poly with up=360, down=257
        try:
            # resample_poly is more efficient for large signals
            resampled = resample_poly(signal, target_fs, orig_fs, axis=-1)
        except Exception:
            # Fallback to interpolation
            n_samples_orig = signal.shape[-1]
            n_samples_new = int(n_samples_orig * target_fs / orig_fs)
            t_orig = np.linspace(0, 1, n_samples_orig)
            t_new = np.linspace(0, 1, n_samples_new)
            
            if signal.ndim == 1:
                f = interp1d(t_orig, signal, kind='linear')
                resampled = f(t_new)
            else:
                resampled = np.zeros((signal.shape[0], n_samples_new))
                for i in range(signal.shape[0]):
                    f = interp1d(t_orig, signal[i], kind='linear')
                    resampled[i] = f(t_new)
        
        return resampled
    
    def _resample_annotations(
        self,
        samples: np.ndarray,
        orig_fs: int,
        target_fs: int,
        orig_length: int,
        new_length: int,
    ) -> np.ndarray:
        """Resample annotation sample indices."""
        if orig_fs == target_fs:
            return samples
        
        # Scale sample indices proportionally
        scale = new_length / orig_length
        return (samples * scale).astype(int)
    
    def _load_wfdb_record(self, record_id: str) -> Optional[INCARTRecord]:
        """Load record from WFDB format (if wfdb library available)."""
        try:
            import wfdb
        except ImportError:
            logger.warning("wfdb library not installed. Install with: pip install wfdb")
            return None
        
        try:
            record_path = str(self.data_dir / record_id)
            record = wfdb.rdrecord(record_path)
            ann = wfdb.rdann(record_path, 'atr')
            
            # Extract signal (transpose to [leads, samples])
            signal = record.p_signal.T
            lead_names = record.sig_name
            orig_fs = record.fs
            orig_length = signal.shape[1]
            
            # Resample if needed
            if self.resample and orig_fs != self.target_fs:
                signal = self._resample_signal(signal, orig_fs, self.target_fs)
                beat_samples = self._resample_annotations(
                    ann.sample, orig_fs, self.target_fs, 
                    orig_length, signal.shape[1]
                )
                fs = self.target_fs
            else:
                beat_samples = ann.sample
                fs = orig_fs
            
            # Map beat labels
            beat_labels = np.array([
                INCART_BEAT_MAP.get(s, 'U') for s in ann.symbol
            ])
            
            return INCARTRecord(
                record_id=record_id,
                signal=signal,
                lead_names=lead_names,
                fs=fs,
                beat_samples=beat_samples,
                beat_labels=beat_labels,
                duration_sec=signal.shape[1] / fs,
            )
            
        except Exception as e:
            logger.error(f"Error loading WFDB record {record_id}: {e}")
            return None
    
    def _load_csv_record(self, record_id: str) -> Optional[INCARTRecord]:
        """Load record from CSV format (alternative storage)."""
        # CSV format: first column is time, rest are leads
        csv_path = self.data_dir / f"{record_id}.csv"
        ann_path = self.data_dir / f"{record_id}annotations.txt"
        
        if not csv_path.exists():
            return None
        
        try:
            # Load signal
            data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
            signal = data[:, 1:].T  # Transpose to [leads, samples]
            
            # Assume standard INCART lead order if not specified
            lead_names = self.LEAD_ORDER[:signal.shape[0]]
            orig_fs = INCART_NATIVE_FS
            orig_length = signal.shape[1]
            
            # Load annotations if available
            beat_samples = np.array([])
            beat_labels = np.array([])
            
            if ann_path.exists():
                with open(ann_path, 'r') as f:
                    lines = f.readlines()
                
                samples_list = []
                labels_list = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            sample = int(parts[0])
                            label = parts[1]
                            samples_list.append(sample)
                            labels_list.append(INCART_BEAT_MAP.get(label, 'U'))
                        except ValueError:
                            continue
                
                beat_samples = np.array(samples_list)
                beat_labels = np.array(labels_list)
            
            # Resample if needed
            if self.resample and orig_fs != self.target_fs:
                signal = self._resample_signal(signal, orig_fs, self.target_fs)
                if len(beat_samples) > 0:
                    beat_samples = self._resample_annotations(
                        beat_samples, orig_fs, self.target_fs,
                        orig_length, signal.shape[1]
                    )
                fs = self.target_fs
            else:
                fs = orig_fs
            
            return INCARTRecord(
                record_id=record_id,
                signal=signal,
                lead_names=list(lead_names),
                fs=fs,
                beat_samples=beat_samples,
                beat_labels=beat_labels,
                duration_sec=signal.shape[1] / fs,
            )
            
        except Exception as e:
            logger.error(f"Error loading CSV record {record_id}: {e}")
            return None
    
    def load_record(self, record_id: str) -> Optional[INCARTRecord]:
        """
        Load a single INCART record.
        
        Tries WFDB format first, then CSV.
        """
        # Try WFDB first
        record = self._load_wfdb_record(record_id)
        if record is not None:
            return record
        
        # Fall back to CSV
        return self._load_csv_record(record_id)
    
    def load_all(
        self,
        max_records: Optional[int] = None,
        record_ids: Optional[List[str]] = None,
    ) -> List[INCARTRecord]:
        """
        Load all INCART records.
        
        Args:
            max_records: Maximum number of records to load
            record_ids: Specific record IDs to load (None = all)
        
        Returns:
            List of INCARTRecord objects
        """
        if record_ids is None:
            record_ids = self._get_record_files()
        
        if max_records is not None:
            record_ids = record_ids[:max_records]
        
        records = []
        for rid in record_ids:
            record = self.load_record(rid)
            if record is not None:
                records.append(record)
                logger.info(f"Loaded INCART record {rid}")
            else:
                logger.warning(f"Failed to load INCART record {rid}")
        
        return records
    
    def detect_ventricular_runs(
        self,
        record: INCARTRecord,
        min_consecutive: int = 3,
        min_hr_bpm: float = 100.0,
    ) -> List[VentricularRun]:
        """
        Detect ventricular runs from beat annotations.
        
        CRITICAL: These are V-runs, NOT confirmed VT.
        INCART lacks rhythm annotations, so we cannot confirm VT.
        
        Args:
            record: INCART record with beat annotations
            min_consecutive: Minimum consecutive V beats (default: 3)
            min_hr_bpm: Minimum heart rate for run (default: 100 BPM)
        
        Returns:
            List of VentricularRun objects (NOT VT episodes)
        """
        v_runs = []
        
        if len(record.beat_samples) == 0:
            return v_runs
        
        # Find runs of V beats
        is_v_beat = record.beat_labels == 'V'
        
        run_start = None
        run_count = 0
        run_samples = []
        
        for i, (sample, is_v) in enumerate(zip(record.beat_samples, is_v_beat)):
            if is_v:
                if run_start is None:
                    run_start = i
                run_count += 1
                run_samples.append(sample)
            else:
                # End of potential run
                if run_count >= min_consecutive:
                    # Check HR criterion
                    if len(run_samples) >= 2:
                        rr_intervals = np.diff(run_samples) / record.fs
                        mean_rr = np.mean(rr_intervals)
                        hr_bpm = 60.0 / mean_rr if mean_rr > 0 else 0
                        
                        if hr_bpm >= min_hr_bpm:
                            start_sample = run_samples[0]
                            end_sample = run_samples[-1]
                            
                            v_runs.append(VentricularRun(
                                start_sample=start_sample,
                                end_sample=end_sample,
                                start_time_sec=start_sample / record.fs,
                                end_time_sec=end_sample / record.fs,
                                n_beats=run_count,
                                hr_bpm=hr_bpm,
                                is_confirmed_vt=False,  # NEVER True for INCART
                                confidence_tier="heuristic",
                            ))
                
                # Reset
                run_start = None
                run_count = 0
                run_samples = []
        
        # Check final run
        if run_count >= min_consecutive and len(run_samples) >= 2:
            rr_intervals = np.diff(run_samples) / record.fs
            mean_rr = np.mean(rr_intervals)
            hr_bpm = 60.0 / mean_rr if mean_rr > 0 else 0
            
            if hr_bpm >= min_hr_bpm:
                start_sample = run_samples[0]
                end_sample = run_samples[-1]
                
                v_runs.append(VentricularRun(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time_sec=start_sample / record.fs,
                    end_time_sec=end_sample / record.fs,
                    n_beats=run_count,
                    hr_bpm=hr_bpm,
                    is_confirmed_vt=False,
                    confidence_tier="heuristic",
                ))
        
        return v_runs
    
    def get_evaluation_summary(
        self,
        records: List[INCARTRecord],
    ) -> Dict[str, Any]:
        """
        Get summary statistics for evaluation.
        
        Returns metrics labeled as V-RUN sensitivity (NOT VT).
        """
        total_duration_sec = 0
        total_v_beats = 0
        total_v_runs = 0
        
        for record in records:
            total_duration_sec += record.duration_sec
            total_v_beats += np.sum(record.beat_labels == 'V')
            v_runs = self.detect_ventricular_runs(record)
            total_v_runs += len(v_runs)
        
        return {
            "dataset": "INCART",
            "n_records": len(records),
            "total_duration_hours": total_duration_sec / 3600,
            "total_v_beats": int(total_v_beats),
            "total_v_runs": total_v_runs,
            "v_runs_per_hour": total_v_runs / (total_duration_sec / 3600) if total_duration_sec > 0 else 0,
            # EXPLICIT: These are V-runs, NOT VT
            "note": "V-runs derived from beat labels. NOT confirmed VT (no rhythm annotations).",
            "report_as": "V-run sensitivity",
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def load_incart_dataset(
    data_dir: str,
    max_records: Optional[int] = None,
    detect_v_runs: bool = True,
) -> Tuple[List[INCARTRecord], Dict[str, Any]]:
    """
    Convenience function to load INCART dataset.
    
    Args:
        data_dir: Path to INCART database
        max_records: Maximum records to load (None = all)
        detect_v_runs: Whether to detect V-runs for each record
    
    Returns:
        (records, summary) tuple
    """
    loader = INCARTLoader(data_dir)
    records = loader.load_all(max_records=max_records)
    
    if detect_v_runs:
        for record in records:
            record.ventricular_runs = [
                (vr.start_sample, vr.end_sample) 
                for vr in loader.detect_ventricular_runs(record)
            ]
    
    summary = loader.get_evaluation_summary(records)
    return records, summary


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("INCART LOADER DEMO")
    print("=" * 60)
    
    # Demo without actual data
    print("\nINCARTLoader capabilities:")
    print("- Load 75 INCART records (12-lead, 30 min each)")
    print("- Resample 257 Hz → 360 Hz (MIT-BIH compatible)")
    print("- Detect ventricular runs (≥3 consecutive V beats)")
    print()
    print("CRITICAL LIMITATION:")
    print("- INCART has NO rhythm annotations")
    print("- Cannot confirm TRUE VT episodes")
    print("- Report as 'V-run sensitivity', NOT 'VT sensitivity'")
    print()
    
    # Show contract
    print("Dataset Contract:")
    print(f"  Native FS: {INCART_NATIVE_FS} Hz")
    print(f"  Target FS: {INCART_TARGET_FS} Hz")
    print(f"  Lead to use: {INCART_LEAD_TO_USE}")
    print(f"  Beat labels: {list(INCART_BEAT_MAP.keys())}")
    print()
    print("=" * 60)
