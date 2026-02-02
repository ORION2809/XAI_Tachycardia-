"""
PTB-XL Database Loader.

v2.4: Load and process PTB-XL database with explicit contract compliance.

PTB-XL Database Characteristics:
- 21,837 clinical 12-lead ECGs from 18,885 patients
- 10-second recordings at 500 Hz (100 Hz version also available)
- Statement-level rhythm labels (not beat-level)
- NO beat annotations

CRITICAL LIMITATIONS:
- NO beat-level annotations
- Cannot apply clinical VT criteria (≥3 V beats)
- VT labels are EXTREMELY RARE in this dataset
- Use ONLY for SVT/AFib validation, NOT VT

Reference:
- PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/
- Paper: Wagner et al., "PTB-XL, a large publicly available electrocardiography dataset"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from scipy.signal import resample_poly
import logging
import ast

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# PTB-XL CONTRACT
# =============================================================================

PTBXL_NATIVE_FS = 500
PTBXL_TARGET_FS = 360
PTBXL_LEAD_TO_USE = "II"

# Rhythm label mapping (PTB-XL statements → canonical)
PTBXL_RHYTHM_MAP = {
    # Supraventricular
    'SVTAC': 'SVT',
    'PSVT': 'SVT',
    'AT': 'SVT',       # Atrial tachycardia
    'AVNRT': 'SVT',    # AV nodal reentrant tachycardia
    'AVRT': 'SVT',     # AV reentrant tachycardia
    
    # Atrial fibrillation/flutter
    'AFIB': 'AFIB_RVR',
    'AFLT': 'AFLUTTER',
    
    # Sinus tachycardia
    'STACH': 'SINUS_TACHY',
    'SR': 'NORMAL',     # Sinus rhythm
    'SBRAD': 'NORMAL',  # Sinus bradycardia (not tachycardia)
    
    # NOTE: VT labels exist but are extremely rare
    # Do NOT use PTB-XL for VT evaluation
}

# Standard 12-lead order
PTBXL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", 
                    "V1", "V2", "V3", "V4", "V5", "V6"]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PTBXLRecord:
    """Single PTB-XL record with all metadata."""
    ecg_id: int
    patient_id: int
    signal: np.ndarray          # Shape: (n_leads, n_samples)
    lead_names: List[str]
    fs: int                     # Sampling frequency (after resampling)
    
    # Labels
    rhythm_labels: List[str]    # Rhythm statement labels
    canonical_rhythms: List[str]  # Mapped to canonical types
    diagnostic_labels: List[str]  # Diagnostic statements
    
    # Metadata
    age: Optional[int] = None
    sex: Optional[str] = None
    duration_sec: float = 10.0
    
    # Quality
    signal_quality: Optional[str] = None  # "clean", "noisy", etc.
    validated_by_human: bool = False
    
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


@dataclass
class PTBXLSplit:
    """Train/val/test split information."""
    train_ids: List[int]
    val_ids: List[int]
    test_ids: List[int]
    fold: int = 1  # PTB-XL provides 10-fold splits


# =============================================================================
# PTB-XL LOADER
# =============================================================================

class PTBXLLoader:
    """
    Load and process PTB-XL database.
    
    CRITICAL: This loader explicitly handles PTB-XL's limitations:
    - Can identify SVT, AFib, AFL from rhythm statements
    - CANNOT reliably identify VT (no beat annotations)
    - Use ONLY for SVT/AFib validation
    
    Usage:
        loader = PTBXLLoader(data_dir="/path/to/ptb-xl")
        records = loader.load_all()
        
        # Filter for SVT records
        svt_records = [r for r in records if r.has_svt()]
    """
    
    def __init__(
        self,
        data_dir: str,
        target_fs: int = 360,
        use_100hz: bool = False,  # Use 100 Hz version (smaller files)
        lead_to_use: str = "II",
        resample: bool = True,
    ):
        """
        Initialize PTB-XL loader.
        
        Args:
            data_dir: Path to PTB-XL database directory
            target_fs: Target sampling frequency (default: 360 Hz)
            use_100hz: Use 100 Hz version instead of 500 Hz
            lead_to_use: Primary lead to use (default: II)
            resample: Whether to resample to target_fs
        """
        self.data_dir = Path(data_dir)
        self.target_fs = target_fs
        self.use_100hz = use_100hz
        self.native_fs = 100 if use_100hz else PTBXL_NATIVE_FS
        self.lead_to_use = lead_to_use
        self.resample = resample
        
        # Paths
        self.records_dir = self.data_dir / ("records100" if use_100hz else "records500")
        self.metadata_path = self.data_dir / "ptbxl_database.csv"
        self.scp_statements_path = self.data_dir / "scp_statements.csv"
        
        # Cache
        self._metadata_df = None
        self._scp_statements = None
        
        if not self.data_dir.exists():
            logger.warning(f"PTB-XL data directory not found: {data_dir}")
    
    def _load_metadata(self) -> Optional[Any]:
        """Load PTB-XL metadata CSV."""
        if self._metadata_df is not None:
            return self._metadata_df
        
        try:
            import pandas as pd
            self._metadata_df = pd.read_csv(self.metadata_path, index_col='ecg_id')
            # Parse scp_codes column (stored as string representation of dict)
            self._metadata_df['scp_codes'] = self._metadata_df['scp_codes'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
            )
            return self._metadata_df
        except ImportError:
            logger.warning("pandas not installed. Install with: pip install pandas")
            return None
        except Exception as e:
            logger.error(f"Error loading PTB-XL metadata: {e}")
            return None
    
    def _load_scp_statements(self) -> Dict[str, Dict]:
        """Load SCP statement descriptions."""
        if self._scp_statements is not None:
            return self._scp_statements
        
        try:
            import pandas as pd
            df = pd.read_csv(self.scp_statements_path, index_col=0)
            self._scp_statements = df.to_dict('index')
            return self._scp_statements
        except Exception as e:
            logger.warning(f"Could not load SCP statements: {e}")
            return {}
    
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
            # For 500 → 360, use resample_poly
            from math import gcd
            g = gcd(target_fs, orig_fs)
            up = target_fs // g
            down = orig_fs // g
            resampled = resample_poly(signal, up, down, axis=-1)
        except Exception:
            # Fallback to interpolation
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
    
    def _extract_rhythm_labels(self, scp_codes: Dict) -> Tuple[List[str], List[str]]:
        """
        Extract rhythm labels from SCP codes.
        
        Returns:
            (raw_rhythm_labels, canonical_rhythms)
        """
        raw_labels = []
        canonical = []
        
        scp_statements = self._load_scp_statements()
        
        for code, likelihood in scp_codes.items():
            # Only consider codes with likelihood > 0
            if likelihood <= 0:
                continue
            
            # Check if this is a rhythm statement
            if code in scp_statements:
                stmt = scp_statements[code]
                if stmt.get('diagnostic_class') == 'RHYTHM' or code in PTBXL_RHYTHM_MAP:
                    raw_labels.append(code)
                    if code in PTBXL_RHYTHM_MAP:
                        canonical.append(PTBXL_RHYTHM_MAP[code])
            elif code in PTBXL_RHYTHM_MAP:
                raw_labels.append(code)
                canonical.append(PTBXL_RHYTHM_MAP[code])
        
        return raw_labels, canonical
    
    def _load_wfdb_signal(self, ecg_id: int, folder: str) -> Optional[np.ndarray]:
        """Load signal from WFDB format."""
        try:
            import wfdb
            record_path = str(self.records_dir / folder / f"{ecg_id:05d}_hr")
            if not Path(record_path + ".dat").exists():
                record_path = str(self.records_dir / folder / f"{ecg_id:05d}_lr")
            
            record = wfdb.rdrecord(record_path)
            signal = record.p_signal.T  # Transpose to [leads, samples]
            return signal
        except ImportError:
            logger.warning("wfdb library not installed")
            return None
        except Exception as e:
            logger.debug(f"Could not load WFDB record {ecg_id}: {e}")
            return None
    
    def _load_npy_signal(self, ecg_id: int, folder: str) -> Optional[np.ndarray]:
        """Load signal from numpy format (if pre-processed)."""
        npy_path = self.records_dir / folder / f"{ecg_id:05d}.npy"
        if npy_path.exists():
            try:
                signal = np.load(npy_path)
                if signal.ndim == 2:
                    return signal.T if signal.shape[0] > signal.shape[1] else signal
                return signal
            except Exception as e:
                logger.debug(f"Could not load numpy record {ecg_id}: {e}")
        return None
    
    def load_record(self, ecg_id: int) -> Optional[PTBXLRecord]:
        """Load a single PTB-XL record."""
        metadata = self._load_metadata()
        if metadata is None or ecg_id not in metadata.index:
            return None
        
        row = metadata.loc[ecg_id]
        
        # Get folder path (PTB-XL organizes records into folders)
        folder = f"{(ecg_id // 1000):02d}000"
        
        # Try to load signal
        signal = self._load_wfdb_signal(ecg_id, folder)
        if signal is None:
            signal = self._load_npy_signal(ecg_id, folder)
        
        if signal is None:
            logger.warning(f"Could not load signal for record {ecg_id}")
            return None
        
        # Resample if needed
        if self.resample and self.native_fs != self.target_fs:
            signal = self._resample_signal(signal, self.native_fs, self.target_fs)
            fs = self.target_fs
        else:
            fs = self.native_fs
        
        # Extract labels
        scp_codes = row.get('scp_codes', {})
        raw_rhythms, canonical_rhythms = self._extract_rhythm_labels(scp_codes)
        
        # Get diagnostic labels (all non-rhythm)
        diagnostic_labels = [k for k in scp_codes.keys() if k not in raw_rhythms]
        
        return PTBXLRecord(
            ecg_id=ecg_id,
            patient_id=int(row.get('patient_id', 0)),
            signal=signal,
            lead_names=PTBXL_LEAD_ORDER,
            fs=fs,
            rhythm_labels=raw_rhythms,
            canonical_rhythms=canonical_rhythms,
            diagnostic_labels=diagnostic_labels,
            age=int(row.get('age', 0)) if row.get('age') else None,
            sex=row.get('sex'),
            duration_sec=signal.shape[1] / fs,
            validated_by_human=row.get('validated_by_human', False),
        )
    
    def load_all(
        self,
        max_records: Optional[int] = None,
        filter_rhythm: Optional[str] = None,
        split: Optional[str] = None,  # "train", "val", "test"
        fold: int = 1,
    ) -> List[PTBXLRecord]:
        """
        Load PTB-XL records.
        
        Args:
            max_records: Maximum number of records to load
            filter_rhythm: Only load records with this canonical rhythm
            split: Which split to load ("train", "val", "test")
            fold: Which fold to use (1-10, PTB-XL provides stratified folds)
        
        Returns:
            List of PTBXLRecord objects
        """
        metadata = self._load_metadata()
        if metadata is None:
            return []
        
        # Get ecg_ids to load
        if split is not None:
            # Use stratified fold column
            fold_col = f"strat_fold"
            if fold_col in metadata.columns:
                if split == "test":
                    ecg_ids = metadata[metadata[fold_col] == fold].index.tolist()
                elif split == "val":
                    val_fold = fold + 1 if fold < 10 else 1
                    ecg_ids = metadata[metadata[fold_col] == val_fold].index.tolist()
                else:  # train
                    ecg_ids = metadata[~metadata[fold_col].isin([fold, (fold+1) if fold < 10 else 1])].index.tolist()
            else:
                ecg_ids = metadata.index.tolist()
        else:
            ecg_ids = metadata.index.tolist()
        
        # Filter by rhythm if specified
        if filter_rhythm is not None:
            filtered_ids = []
            for ecg_id in ecg_ids:
                row = metadata.loc[ecg_id]
                scp_codes = row.get('scp_codes', {})
                _, canonical = self._extract_rhythm_labels(scp_codes)
                if filter_rhythm in canonical:
                    filtered_ids.append(ecg_id)
            ecg_ids = filtered_ids
        
        # Limit records
        if max_records is not None:
            ecg_ids = ecg_ids[:max_records]
        
        # Load records
        records = []
        for ecg_id in ecg_ids:
            record = self.load_record(ecg_id)
            if record is not None:
                records.append(record)
        
        logger.info(f"Loaded {len(records)} PTB-XL records")
        return records
    
    def get_svt_records(self, max_records: Optional[int] = None) -> List[PTBXLRecord]:
        """Get records with SVT labels (SVT, AFib, AFL)."""
        records = self.load_all(max_records=None)  # Load all first
        svt_records = [r for r in records if r.has_svt()]
        
        if max_records is not None:
            svt_records = svt_records[:max_records]
        
        return svt_records
    
    def get_afib_records(self, max_records: Optional[int] = None) -> List[PTBXLRecord]:
        """Get records with AFib labels."""
        return self.load_all(max_records=max_records, filter_rhythm='AFIB_RVR')
    
    def get_evaluation_summary(
        self,
        records: List[PTBXLRecord],
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
            "dataset": "PTB-XL",
            "n_records": len(records),
            "total_duration_hours": sum(r.duration_sec for r in records) / 3600,
            "rhythm_counts": rhythm_counts,
            "n_svt": rhythm_counts['SVT'] + rhythm_counts['AFIB_RVR'] + rhythm_counts['AFLUTTER'],
            "n_afib": rhythm_counts['AFIB_RVR'],
            "n_sinus_tachy": rhythm_counts['SINUS_TACHY'],
            # EXPLICIT: Do NOT use for VT evaluation
            "note": "PTB-XL has NO beat annotations. Use for SVT/AFib validation ONLY.",
            "vt_labeling_supported": False,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def load_ptbxl_dataset(
    data_dir: str,
    max_records: Optional[int] = None,
    svt_only: bool = False,
) -> Tuple[List[PTBXLRecord], Dict[str, Any]]:
    """
    Convenience function to load PTB-XL dataset.
    
    Args:
        data_dir: Path to PTB-XL database
        max_records: Maximum records to load
        svt_only: Only load records with SVT labels
    
    Returns:
        (records, summary) tuple
    """
    loader = PTBXLLoader(data_dir)
    
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
    print("PTB-XL LOADER DEMO")
    print("=" * 60)
    
    print("\nPTBXLLoader capabilities:")
    print("- Load 21,837 PTB-XL records (12-lead, 10-second)")
    print("- Resample 500 Hz → 360 Hz (MIT-BIH compatible)")
    print("- Filter by rhythm type (SVT, AFib, Sinus Tachy)")
    print("- Support stratified train/val/test splits")
    print()
    print("CRITICAL LIMITATIONS:")
    print("- PTB-XL has NO beat-level annotations")
    print("- Cannot apply clinical VT criteria")
    print("- VT labels are EXTREMELY RARE")
    print("- Use ONLY for SVT/AFib validation, NOT VT")
    print()
    
    # Show rhythm mapping
    print("Rhythm Label Mapping:")
    for native, canonical in PTBXL_RHYTHM_MAP.items():
        print(f"  {native} → {canonical}")
    print()
    print("=" * 60)
