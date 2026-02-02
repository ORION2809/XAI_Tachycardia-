# Data contracts and harmonization module
# v2.4: Added external dataset loaders

from .contracts import (
    ECGSegment,
    BeatAnnotation,
    EpisodeLabel,
    EpisodeType,
    SQIResult,
)
from .harmonization import (
    DatasetContract,
    MIT_BIH_CONTRACT,
    INCART_CONTRACT,
    PTB_XL_CONTRACT,
    CHAPMAN_SHAOXING_CONTRACT,
)

# External dataset loaders
from .loaders import (
    # INCART
    INCARTLoader,
    load_incart_dataset,
    
    # PTB-XL
    PTBXLLoader,
    load_ptbxl_dataset,
    
    # Chapman-Shaoxing
    ChapmanLoader,
    load_chapman_dataset,
)

__all__ = [
    # Contracts
    'ECGSegment',
    'BeatAnnotation',
    'EpisodeLabel',
    'EpisodeType',
    'SQIResult',
    'DatasetContract',
    'MIT_BIH_CONTRACT',
    'INCART_CONTRACT',
    'PTB_XL_CONTRACT',
    'CHAPMAN_SHAOXING_CONTRACT',
    
    # Loaders
    'INCARTLoader',
    'load_incart_dataset',
    'PTBXLLoader',
    'load_ptbxl_dataset',
    'ChapmanLoader',
    'load_chapman_dataset',
]
