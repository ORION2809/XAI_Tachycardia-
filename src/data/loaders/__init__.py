"""
External Dataset Loaders.

v2.4: Unified loaders for external ECG datasets with explicit contracts.

Supported datasets:
- INCART: 75 records, 12-lead, beat annotations (V-run detection)
- PTB-XL: 21,837 records, 12-lead, rhythm labels (SVT/AFib only)
- Chapman-Shaoxing: 10,646 records, 12-lead, rhythm labels (SVT only)

CRITICAL: Each loader respects the dataset contract's limitations.
Do NOT use INCART for "VT sensitivity" - use "V-run sensitivity" instead.
"""

from .incart import INCARTLoader, load_incart_dataset
from .ptbxl import PTBXLLoader, load_ptbxl_dataset
from .chapman import ChapmanLoader, load_chapman_dataset

__all__ = [
    # INCART
    'INCARTLoader',
    'load_incart_dataset',
    
    # PTB-XL
    'PTBXLLoader',
    'load_ptbxl_dataset',
    
    # Chapman-Shaoxing
    'ChapmanLoader',
    'load_chapman_dataset',
]
