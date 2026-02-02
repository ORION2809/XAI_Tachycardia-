"""
Dataset Harmonization Contracts.

Each external dataset requires an explicit mapping table that defines:
- How to extract and resample signals
- Which lead to use as canonical "II-like"
- How to map native labels to canonical labels
- What labeling is actually supported

CRITICAL: Do NOT promise VT labeling for datasets that don't support it.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


@dataclass
class DatasetContract:
    """
    Contract defining how to use a specific dataset.
    
    This contract ensures that we don't make assumptions about what
    a dataset can provide. Each dataset has explicit capabilities
    and limitations.
    
    Attributes:
        name: Dataset identifier
        native_fs: Original sampling frequency
        available_leads: All leads available in the dataset
        lead_to_use: Which lead maps to canonical "II-like"
        has_beat_annotations: Whether beat-level annotations exist
        has_rhythm_annotations: Whether rhythm-level annotations exist
        beat_label_map: Native beat label → canonical label
        rhythm_label_map: Native rhythm → canonical episode type
        vt_labeling_supported: Can we derive TRUE VT episodes?
        svt_labeling_supported: Can we derive SVT episodes?
        ventricular_run_supported: Can derive V-runs from beat labels? (v2.3)
        known_limitations: Documented issues with the dataset
    """
    name: str
    native_fs: int
    available_leads: List[str]
    lead_to_use: str
    has_beat_annotations: bool
    has_rhythm_annotations: bool
    beat_label_map: Dict[str, str]
    rhythm_label_map: Dict[str, str]
    vt_labeling_supported: bool
    svt_labeling_supported: bool
    # v2.3: Distinguish V-run detection from true VT detection
    ventricular_run_supported: bool = False
    known_limitations: List[str] = field(default_factory=list)
    
    # Resampling configuration
    target_fs: int = 360  # Canonical sampling rate
    
    def needs_resampling(self) -> bool:
        """Check if this dataset needs resampling."""
        return self.native_fs != self.target_fs
    
    def get_resample_ratio(self) -> float:
        """Get the resampling ratio."""
        return self.target_fs / self.native_fs
    
    def map_beat_label(self, native_label: str) -> str:
        """Map native beat label to canonical label."""
        return self.beat_label_map.get(native_label, 'U')  # Default to Unknown
    
    def map_rhythm_label(self, native_rhythm: str) -> Optional[str]:
        """Map native rhythm label to canonical episode type."""
        return self.rhythm_label_map.get(native_rhythm)
    
    def validate_for_vt_evaluation(self) -> bool:
        """
        Check if this dataset can be used for VT evaluation.
        
        VT evaluation requires either:
        1. Rhythm annotations that include VT
        2. Beat annotations that allow VT derivation (≥3 consecutive V)
        """
        if not self.vt_labeling_supported:
            return False
        
        # Must have beat annotations for clinical VT criteria
        if not self.has_beat_annotations:
            # Can only use rhythm annotations directly
            return 'VT' in self.rhythm_label_map.values() or \
                   'VT_MONOMORPHIC' in self.rhythm_label_map.values()
        
        return True
    
    def get_usage_recommendation(self) -> str:
        """Get recommendation for how to use this dataset."""
        uses = []
        
        if self.vt_labeling_supported:
            uses.append("VT detection")
        if self.svt_labeling_supported:
            uses.append("SVT/AFib detection")
        if not uses:
            uses.append("Signal quality / preprocessing only")
        
        return f"Recommended for: {', '.join(uses)}"


# =============================================================================
# Concrete Dataset Contracts
# =============================================================================

MIT_BIH_CONTRACT = DatasetContract(
    name="MIT-BIH",
    native_fs=360,
    available_leads=["MLII", "V5", "V1", "V2", "V4"],
    lead_to_use="MLII",  # Most common, II-like
    has_beat_annotations=True,
    has_rhythm_annotations=True,
    beat_label_map={
        # Normal and normal variants
        'N': 'N',   # Normal beat
        'L': 'N',   # Left bundle branch block (treat as normal variant)
        'R': 'N',   # Right bundle branch block (treat as normal variant)
        'e': 'N',   # Atrial escape
        'j': 'N',   # Nodal (junctional) escape
        
        # Supraventricular ectopic
        'A': 'A',   # Atrial premature
        'a': 'A',   # Aberrated atrial premature
        'S': 'S',   # Supraventricular premature
        'J': 'A',   # Nodal (junctional) premature
        
        # Ventricular ectopic
        'V': 'V',   # Premature ventricular contraction
        'E': 'V',   # Ventricular escape
        
        # Fusion
        'F': 'F',   # Fusion of ventricular and normal
        
        # Paced
        '/': 'P',   # Paced beat
        'f': 'P',   # Fusion of paced and normal
        
        # Unknown / artifact
        'Q': 'U',   # Unclassifiable
        '?': 'U',   # Beat not classified
        '|': 'U',   # Isolated QRS-like artifact
        '~': 'U',   # Change in signal quality
    },
    rhythm_label_map={
        # Ventricular rhythms
        '(VT': 'VT_MONOMORPHIC',
        '(VFL': 'VFL',
        
        # Supraventricular rhythms
        '(SVTA': 'SVT',
        '(AFIB': 'AFIB_RVR',
        '(AFL': 'AFLUTTER',
        
        # Normal rhythms
        '(N': 'NORMAL',
        '(SBR': 'NORMAL',       # Sinus bradycardia (not tachycardia)
        '(B': 'NORMAL',         # Ventricular bigeminy
        '(T': 'NORMAL',         # Ventricular trigeminy
        
        # Other
        '(AB': 'NORMAL',        # Atrial bigeminy
        '(IVR': 'NORMAL',       # Idioventricular rhythm
        '(NOD': 'NORMAL',       # Nodal rhythm
        '(P': 'NORMAL',         # Paced rhythm
        '(PREX': 'NORMAL',      # Pre-excitation
    },
    vt_labeling_supported=True,
    svt_labeling_supported=True,
    known_limitations=[
        "Only 2 leads per record",
        "VT episodes are short (mostly non-sustained)",
        "Limited patient diversity (47 patients)",
        "Some records have significant noise",
        "Beat annotations may miss some beats in noisy sections",
    ]
)

INCART_CONTRACT = DatasetContract(
    name="INCART",
    native_fs=257,
    available_leads=["I", "II", "III", "aVR", "aVL", "aVF", 
                     "V1", "V2", "V3", "V4", "V5", "V6"],
    lead_to_use="II",
    has_beat_annotations=True,
    has_rhythm_annotations=False,  # No rhythm annotations!
    beat_label_map={
        'N': 'N',   # Normal
        'V': 'V',   # Ventricular ectopic
        'S': 'S',   # Supraventricular ectopic
        'F': 'F',   # Fusion
        'Q': 'U',   # Unknown
    },
    rhythm_label_map={},  # Cannot derive rhythm labels directly
    # v2.3 FIX: Cannot derive TRUE VT without rhythm annotations
    vt_labeling_supported=False,
    svt_labeling_supported=False, # Cannot reliably derive SVT
    # v2.3: Can derive V-runs from beat labels (but NOT true VT)
    ventricular_run_supported=True,
    known_limitations=[
        "No rhythm annotations - can only detect VENTRICULAR_RUN, not true VT",
        "v2.3: Report 'V-run sensitivity', NOT 'VT sensitivity'",
        "Need resampling 257→360 Hz (introduces interpolation artifacts)",
        "SVT labeling not supported - cannot reliably identify SVT",
        "Smaller dataset than MIT-BIH",
    ]
)

PTB_XL_CONTRACT = DatasetContract(
    name="PTB-XL",
    native_fs=500,
    available_leads=["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"],
    lead_to_use="II",
    has_beat_annotations=False,   # No beat-level annotations!
    has_rhythm_annotations=True,  # Has statement-level rhythm labels
    beat_label_map={},            # No beat labels available
    rhythm_label_map={
        # Using SNOMED-CT codes / SCP-ECG codes
        'SVTAC': 'SVT',
        'AFIB': 'AFIB_RVR',
        'AFLT': 'AFLUTTER',
        'STACH': 'SINUS_TACHYCARDIA',
        # Note: PTB-XL has very few VT labels
    },
    vt_labeling_supported=False,  # No beat labels to derive VT
    svt_labeling_supported=True,
    known_limitations=[
        "No beat annotations - cannot apply clinical VT criteria",
        "Use only for SVT/AFib validation, NOT VT",
        "Need resampling 500→360 Hz",
        "Recording length is only 10 seconds",
        "VT cases extremely rare in this dataset",
    ]
)

CHAPMAN_SHAOXING_CONTRACT = DatasetContract(
    name="Chapman-Shaoxing",
    native_fs=500,
    available_leads=["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "V6"],
    lead_to_use="II",
    has_beat_annotations=False,
    has_rhythm_annotations=True,
    beat_label_map={},
    rhythm_label_map={
        'SVT': 'SVT',
        'AT': 'SVT',        # Atrial tachycardia
        'AF': 'AFIB_RVR',   # Atrial fibrillation
        'AFL': 'AFLUTTER',  # Atrial flutter
        'ST': 'SINUS_TACHYCARDIA',
    },
    vt_labeling_supported=False,
    svt_labeling_supported=True,
    known_limitations=[
        "No beat annotations",
        "VT cases extremely rare in this dataset",
        "Use for SVT/AFib only",
        "Need resampling 500→360 Hz",
        "12-lead ECG - need to select single lead for consistency",
    ]
)

AFDB_CONTRACT = DatasetContract(
    name="MIT-BIH-AFIB",
    native_fs=250,
    available_leads=["ECG1", "ECG2"],
    lead_to_use="ECG1",  # First available lead
    has_beat_annotations=True,
    has_rhythm_annotations=True,
    beat_label_map={
        'N': 'N',
        'V': 'V',
        'S': 'S',
        'F': 'F',
    },
    rhythm_label_map={
        '(AFIB': 'AFIB_RVR',
        '(AFL': 'AFLUTTER',
        '(N': 'NORMAL',
    },
    vt_labeling_supported=True,   # Has beat labels
    svt_labeling_supported=True,
    known_limitations=[
        "Focused on atrial fibrillation - limited VT examples",
        "Need resampling 250→360 Hz",
        "25 long-term recordings",
    ]
)


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY: Dict[str, DatasetContract] = {
    'MIT-BIH': MIT_BIH_CONTRACT,
    'INCART': INCART_CONTRACT,
    'PTB-XL': PTB_XL_CONTRACT,
    'Chapman-Shaoxing': CHAPMAN_SHAOXING_CONTRACT,
    'MIT-BIH-AFIB': AFDB_CONTRACT,
}


def get_contract(dataset_name: str) -> DatasetContract:
    """Get the contract for a dataset by name."""
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


def get_vt_validation_datasets() -> List[str]:
    """Get list of datasets that can be used for VT validation."""
    return [
        name for name, contract in DATASET_REGISTRY.items()
        if contract.validate_for_vt_evaluation()
    ]


def get_svt_validation_datasets() -> List[str]:
    """Get list of datasets that can be used for SVT validation."""
    return [
        name for name, contract in DATASET_REGISTRY.items()
        if contract.svt_labeling_supported
    ]


def print_dataset_summary():
    """Print summary of all dataset capabilities."""
    print("\n" + "="*80)
    print("DATASET HARMONIZATION SUMMARY")
    print("="*80)
    
    for name, contract in DATASET_REGISTRY.items():
        print(f"\n{name}")
        print("-" * len(name))
        print(f"  Sampling rate: {contract.native_fs} Hz → {contract.target_fs} Hz")
        print(f"  Lead used: {contract.lead_to_use}")
        print(f"  Beat annotations: {'✓' if contract.has_beat_annotations else '✗'}")
        print(f"  Rhythm annotations: {'✓' if contract.has_rhythm_annotations else '✗'}")
        print(f"  VT labeling: {'✓ SUPPORTED' if contract.vt_labeling_supported else '✗ NOT SUPPORTED'}")
        print(f"  SVT labeling: {'✓ SUPPORTED' if contract.svt_labeling_supported else '✗ NOT SUPPORTED'}")
        print(f"  {contract.get_usage_recommendation()}")
        if contract.known_limitations:
            print(f"  Limitations:")
            for lim in contract.known_limitations[:3]:  # Show first 3
                print(f"    - {lim}")
    
    print("\n" + "="*80)
    print(f"VT validation datasets: {get_vt_validation_datasets()}")
    print(f"SVT validation datasets: {get_svt_validation_datasets()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    print_dataset_summary()
