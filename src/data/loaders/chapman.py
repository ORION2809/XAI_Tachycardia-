"""
Chapman-Shaoxing Database Loader.

v2.4: Load and process Chapman-Shaoxing database with explicit contract compliance.

Chapman-Shaoxing Database Characteristics:
- 10,646 12-lead ECGs from patients at Chapman University and Shaoxing Hospital
- 10-second recordings at 500 Hz
- Rhythm labels from automated software + cardiologist validation
- NO beat-level annotations

CRITICAL LIMITATIONS:
- NO beat-level annotations
- VT cases are EXTREMELY RARE in this dataset
- Use ONLY for SVT/AFib validation, NOT VT

Reference:
- PhysioNet: https://physionet.org/content/ecg-arrhythmia/1.0.0/
- Paper: Zheng et al., "A 12-lead electrocardiogram database for arrhythmia research"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from scipy.signal import resample_poly
from scipy.io import loadmat
import logging

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# CHAPMAN-SHAOXING CONTRACT
# =============================================================================

CHAPMAN_NATIVE_FS = 500
CHAPMAN_TARGET_FS = 360
CHAPMAN_LEAD_TO_USE = "II"

# Rhythm label mapping (Chapman → canonical)
CHAPMAN_RHYTHM_MAP = {
    # Supraventricular tachycardia
    'SVT': 'SVT',
    'AVNRT': 'SVT',
    'AVRT': 'SVT',
    'AT': 'SVT',           # Atrial tachycardia
    
    # Atrial fibrillation/flutter
    'AF': 'AFIB_RVR',
    'AFIB': 'AFIB_RVR',
    'AFL': 'AFLUTTER',
    'AFLT': 'AFLUTTER',
    
    # Sinus rhythms
    'STACH': 'SINUS_TACHY',
    'ST': 'SINUS_TACHY',    # Sinus tachycardia
    'SR': 'NORMAL',         # Sinus rhythm
    'SB': 'NORMAL',         # Sinus bradycardia
    'NSR': 'NORMAL',        # Normal sinus rhythm
    
    # Others (not tachycardia-related)
    'SA': 'NORMAL',         # Sinus arrhythmia
    'PAC': 'NORMAL',        # Premature atrial contraction (isolated)
    'PVC': 'NORMAL',        # Premature ventricular contraction (isolated)
}

# Standard 12-lead order
CHAPMAN_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", 
                      "V1", "V2", "V3", "V4", "V5", "V6"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ChapmanRecord:
    """Single Chapman-Shaoxing record with all metadata."""
    record_id: str
    signal: np.ndarray          # Shape: (n_leads, n_samples)
    lead_names: List[str]
    fs: int                     # Sampling frequency (after resampling)
    
    # Labels
    rhythm_labels: List[str]    # Original rhythm labels
    canonical_rhythms: List[str]  # Mapped to canonical types
    
    # Metadata
    age: Optional[int] = None
    sex: Optional[str] = None
    duration_sec: float = 10.0
    
    def get_lead(self, lead_name: str) -> Optional[np.ndarray]:
        """Get signal for specific lead."""
        if lead_name in self.lead_names:
            idx = self.lead_names.index(lead_name)
            return self.signal[idx]
        return None
    
    def get_lead_ii(self) -> Optional[np.ndarray]:
        """Get Lead II (canonical lead)."""
        return self.get_lead("II")
    
    def has_svt(self) -> bool:
        """Check if record has any SVT label."""
        return any(r in ['SVT', 'AFIB_RVR', 'AFLUTTER'] for r in self.canonical_rhythms)
    
    def has_afib(self) -> bool:
        """Check if record has AFib label."""
        return 'AFIB_RVR' in self.canonical_rhythms
    
    def has_sinus_tachy(self) -> bool:
        """Check if record has sinus tachycardia."""
        return 'SINUS_TACHY' in self.canonical_rhythms


# =============================================================================
# CHAPMAN LOADER
# =============================================================================

class ChapmanLoader:
    """
    Load and process Chapman-Shaoxing database.
    
    CRITICAL: This loader explicitly handles Chapman's limitations:
    - Can identify SVT, AFib from rhythm labels
    - CANNOT reliably identify VT (no beat annotations, extremely rare)
    - Use ONLY for SVT/AFib validation
    
    The Chapman-Shaoxing database provides ECGs in multiple formats:
    - WFDB format (.dat/.hea files)
    - MATLAB format (.mat files)
    - CSV format
    
    Usage:
        loader = ChapmanLoader(data_dir="/path/to/chapman")
        records = loader.load_all()
        
        # Filter for AFib records
        afib_records = [r for r in records if r.has_afib()]
    """
    
    def __init__(
        self,
        data_dir: str,
        target_fs: int = 360,
        lead_to_use: str = "II",
        resample: bool = True,
    ):
        """
        Initialize Chapman loader.
        
        Args:
            data_dir: Path to Chapman-Shaoxing database directory
            target_fs: Target sampling frequency (default: 360 Hz)
            lead_to_use: Primary lead to use (default: II)
            resample: Whether to resample to target_fs
        """
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.lead_to_use = lead_to_use
        self.resample = resample
        
        # Paths (Chapman has specific structure)
        self.ecg_dir = self.data_dir / "ECGDataDenoised"
        self.diagnostics_file = self.data_dir / "Diagnostics.xlsx"
        self.attributes_file = self.data_dir / "AttributesDictionary.xlsx"
        
        # Also check alternative structures
        if not self.ecg_dir.exists():
            self.ecg_dir = self.data_dir
        
        # Cache
        self._diagnostics_df = None
        
        if not self.data_dir.exists():
            logger.warning(f"Chapman data directory not found: {data_dir}")
    
    def _load_diagnostics(self) -> Optional[Any]:
        """Load diagnostics/labels file."""
        if self._diagnostics_df is not None:
            return self._diagnostics_df
        
        try:
            import pandas as pd
            
            # Try Excel format first
            if self.diagnostics_file.exists():
                self._diagnostics_df = pd.read_excel(self.diagnostics_file)
            else:
                # Try CSV alternative
                csv_path = self.data_dir / "Diagnostics.csv"
                if csv_path.exists():
                    self._diagnostics_df = pd.read_csv(csv_path)
                else:
                    # Try conditions file (alternative naming)
                    cond_path = self.data_dir / "conditions.csv"
                    if cond_path.exists():
                        self._diagnostics_df = pd.read_csv(cond_path)
            
            return self._diagnostics_df
            
        except ImportError:
            logger.warning("pandas not installed. Install with: pip install pandas")
            return None
        except Exception as e:
            logger.error(f"Error loading Chapman diagnostics: {e}")
            return None
    
    def _resample_signal(
        self,
        signal: np.ndarray,
        orig_fs: int,
        target_fs: int,
    ) -> np.ndarray:
        """Resample signal to target frequency."""
        if orig_fs == target_fs:
            return signal
        
        try:
            from math import gcd
            g = gcd(target_fs, orig_fs)
            up = target_fs // g
            down = orig_fs // g
            resampled = resample_poly(signal, up, down, axis=-1)
        except Exception:
            from scipy.interpolate import interp1d
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
    
    def _extract_rhythm_labels(self, label_str: str) -> Tuple[List[str], List[str]]:
        """
        Extract rhythm labels from diagnostic string.
        
        Returns:
            (raw_labels, canonical_rhythms)
        """
        raw_labels = []
        canonical = []
        
        if not label_str or not isinstance(label_str, str):
            return raw_labels, canonical
        
        # Labels may be comma-separated or semi-colon separated
        parts = label_str.replace(';', ',').split(',')
        
        for part in parts:
            label = part.strip().upper()
            if label:
                raw_labels.append(label)
                if label in CHAPMAN_RHYTHM_MAP:
                    canonical.append(CHAPMAN_RHYTHM_MAP[label])
        
        return raw_labels, canonical
    
    def _load_mat_signal(self, record_path: Path) -> Optional[np.ndarray]:
        """Load signal from MATLAB .mat file."""
        try:
            mat_data = loadmat(str(record_path))
            
            # Chapman .mat files typically have 'val' key
            if 'val' in mat_data:
                signal = mat_data['val']
            elif 'ECG' in mat_data:
                signal = mat_data['ECG']
            else:
                # Try first array-like key
                for key in mat_data:
                    if not key.startswith('_'):
                        val = mat_data[key]
                        if isinstance(val, np.ndarray) and val.ndim >= 1:
                            signal = val
                            break
                else:
                    return None
            
            # Ensure shape is [leads, samples]
            if signal.ndim == 2:
                if signal.shape[0] > signal.shape[1]:
                    signal = signal.T
            
            return signal.astype(np.float32)
            
        except Exception as e:
            logger.debug(f"Could not load MAT file {record_path}: {e}")
            return None
    
    def _load_wfdb_signal(self, record_id: str) -> Optional[np.ndarray]:
        """Load signal from WFDB format."""
        try:
            import wfdb
            record_path = str(self.ecg_dir / record_id)
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal.T
            return signal
        except ImportError:
            return None
        except Exception as e:
            logger.debug(f"Could not load WFDB record {record_id}: {e}")
            return None
    
    def _load_csv_signal(self, record_path: Path) -> Optional[np.ndarray]:
        """Load signal from CSV format."""
        try:
            data = np.loadtxt(record_path, delimiter=',', skiprows=1)
            # Assume columns are leads
            signal = data.T
            return signal.astype(np.float32)
        except Exception as e:
            logger.debug(f"Could not load CSV file {record_path}: {e}")
            return None
    
    def _get_record_files(self) -> List[Tuple[str, Path]]:
        """Get list of record IDs and their file paths."""
        records = []
        
        # Check for different file formats
        for ext in ['*.mat', '*.dat', '*.csv']:
            for f in self.ecg_dir.glob(ext):
                record_id = f.stem
                if not record_id.startswith('.'):
                    records.append((record_id, f))
        
        return records
    
    def load_record(self, record_id: str) -> Optional[ChapmanRecord]:
        """Load a single Chapman record."""
        # Find the file
        signal = None
        
        # Try different formats
        mat_path = self.ecg_dir / f"{record_id}.mat"
        if mat_path.exists():
            signal = self._load_mat_signal(mat_path)
        
        if signal is None:
            signal = self._load_wfdb_signal(record_id)
        
        if signal is None:
            csv_path = self.ecg_dir / f"{record_id}.csv"
            if csv_path.exists():
                signal = self._load_csv_signal(csv_path)
        
        if signal is None:
            logger.warning(f"Could not load signal for record {record_id}")
            return None
        
        # Resample if needed
        if self.resample and CHAPMAN_NATIVE_FS != self.target_fs:
            signal = self._resample_signal(signal, CHAPMAN_NATIVE_FS, self.target_fs)
            fs = self.target_fs
        else:
            fs = CHAPMAN_NATIVE_FS
        
        # Get labels from diagnostics
        diagnostics = self._load_diagnostics()
        raw_rhythms = []
        canonical_rhythms = []
        age = None
        sex = None
        
        if diagnostics is not None:
            # Find this record's labels
            id_col = None
            for col in ['FileName', 'filename', 'ECG_ID', 'ecg_id', 'ID', 'id']:
                if col in diagnostics.columns:
                    id_col = col
                    break
            
            if id_col is not None:
                row = diagnostics[diagnostics[id_col] == record_id]
                if not row.empty:
                    row = row.iloc[0]
                    
                    # Get rhythm labels
                    for col in ['Rhythm', 'rhythm', 'Diagnosis', 'diagnosis']:
                        if col in diagnostics.columns:
                            raw_rhythms, canonical_rhythms = self._extract_rhythm_labels(
                                str(row.get(col, ''))
                            )
                            break
                    
                    # Get demographics
                    age = row.get('Age', row.get('age'))
                    if age and not np.isnan(age):
                        age = int(age)
                    else:
                        age = None
                    
                    sex = row.get('Sex', row.get('sex', row.get('Gender', row.get('gender'))))
        
        return ChapmanRecord(
            record_id=record_id,
            signal=signal,
            lead_names=CHAPMAN_LEAD_ORDER[:signal.shape[0]],
            fs=fs,
            rhythm_labels=raw_rhythms,
            canonical_rhythms=canonical_rhythms,
            age=age,
            sex=str(sex) if sex else None,
            duration_sec=signal.shape[1] / fs,
        )
    
    def load_all(
        self,
        max_records: Optional[int] = None,
        filter_rhythm: Optional[str] = None,
    ) -> List[ChapmanRecord]:
        """
        Load Chapman records.
        
        Args:
            max_records: Maximum number of records to load
            filter_rhythm: Only load records with this canonical rhythm
        
        Returns:
            List of ChapmanRecord objects
        """
        record_files = self._get_record_files()
        
        if max_records is not None:
            record_files = record_files[:max_records * 2]  # Load extra in case of filtering
        
        records = []
        for record_id, _ in record_files:
            if max_records is not None and len(records) >= max_records:
                break
            
            record = self.load_record(record_id)
            if record is None:
                continue
            
            # Filter by rhythm if specified
            if filter_rhythm is not None:
                if filter_rhythm not in record.canonical_rhythms:
                    continue
            
            records.append(record)
        
        logger.info(f"Loaded {len(records)} Chapman records")
        return records
    
    def get_svt_records(self, max_records: Optional[int] = None) -> List[ChapmanRecord]:
        """Get records with SVT labels (SVT, AFib, AFL)."""
        records = self.load_all(max_records=None)
        svt_records = [r for r in records if r.has_svt()]
        
        if max_records is not None:
            svt_records = svt_records[:max_records]
        
        return svt_records
    
    def get_afib_records(self, max_records: Optional[int] = None) -> List[ChapmanRecord]:
        """Get records with AFib labels."""
        return self.load_all(max_records=max_records, filter_rhythm='AFIB_RVR')
    
    def get_evaluation_summary(
        self,
        records: List[ChapmanRecord],
    ) -> Dict[str, Any]:
        """Get summary statistics for evaluation."""
        rhythm_counts = {
            'SVT': 0,
            'AFIB_RVR': 0,
            'AFLUTTER': 0,
            'SINUS_TACHY': 0,
            'NORMAL': 0,
        }
        
        for record in records:
            for rhythm in record.canonical_rhythms:
                if rhythm in rhythm_counts:
                    rhythm_counts[rhythm] += 1
        
        return {
            "dataset": "Chapman-Shaoxing",
            "n_records": len(records),
            "total_duration_hours": sum(r.duration_sec for r in records) / 3600,
            "rhythm_counts": rhythm_counts,
            "n_svt": rhythm_counts['SVT'] + rhythm_counts['AFIB_RVR'] + rhythm_counts['AFLUTTER'],
            "n_afib": rhythm_counts['AFIB_RVR'],
            "n_sinus_tachy": rhythm_counts['SINUS_TACHY'],
            # EXPLICIT: Do NOT use for VT evaluation
            "note": "Chapman has NO beat annotations. VT cases extremely rare. Use for SVT only.",
            "vt_labeling_supported": False,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def load_chapman_dataset(
    data_dir: str,
    max_records: Optional[int] = None,
    svt_only: bool = False,
) -> Tuple[List[ChapmanRecord], Dict[str, Any]]:
    """
    Convenience function to load Chapman-Shaoxing dataset.
    
    Args:
        data_dir: Path to Chapman database
        max_records: Maximum records to load
        svt_only: Only load records with SVT labels
    
    Returns:
        (records, summary) tuple
    """
    loader = ChapmanLoader(data_dir)
    
    if svt_only:
        records = loader.get_svt_records(max_records=max_records)
    else:
        records = loader.load_all(max_records=max_records)
    
    summary = loader.get_evaluation_summary(records)
    return records, summary


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CHAPMAN-SHAOXING LOADER DEMO")
    print("=" * 60)
    
    print("\nChapmanLoader capabilities:")
    print("- Load 10,646 Chapman-Shaoxing records (12-lead, 10-second)")
    print("- Resample 500 Hz → 360 Hz (MIT-BIH compatible)")
    print("- Support MAT, WFDB, and CSV formats")
    print("- Filter by rhythm type (SVT, AFib)")
    print()
    print("CRITICAL LIMITATIONS:")
    print("- Chapman has NO beat-level annotations")
    print("- VT cases are EXTREMELY RARE in this dataset")
    print("- Use ONLY for SVT/AFib validation, NOT VT")
    print()
    
    # Show rhythm mapping
    print("Rhythm Label Mapping:")
    for native, canonical in sorted(CHAPMAN_RHYTHM_MAP.items()):
        print(f"  {native} → {canonical}")
    print()
    print("=" * 60)
