"""
Core data contracts for the XAI Tachycardia Detection system.

All data flows through standardized formats regardless of source dataset.
These contracts ensure type safety and consistency across the pipeline.

Version: 2.4 (Deployment-Grade)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np


class EpisodeType(Enum):
    """
    Exhaustive episode taxonomy for tachycardia classification.
    
    This enum defines ALL possible episode types that the system can detect
    or label. Using an enum ensures type safety and prevents string typos.
    """
    NORMAL = "normal"
    SINUS_TACHYCARDIA = "sinus_tachy"
    SVT = "svt"                     # Supraventricular tachycardia (generic)
    AFIB_RVR = "afib_rvr"           # AFib with rapid ventricular response
    AFLUTTER = "aflutter"           # Atrial flutter
    VT_MONOMORPHIC = "vt_mono"      # Monomorphic ventricular tachycardia
    VT_POLYMORPHIC = "vt_poly"      # Polymorphic VT / Torsades
    VFL = "vfl"                     # Ventricular flutter
    VFIB = "vfib"                   # Ventricular fibrillation
    UNKNOWN = "unknown"             # Cannot determine
    ARTIFACT = "artifact"           # Signal quality too poor to label
    
    @classmethod
    def ventricular_types(cls) -> set:
        """Return set of ventricular tachycardia types."""
        return {cls.VT_MONOMORPHIC, cls.VT_POLYMORPHIC, cls.VFL, cls.VFIB}
    
    @classmethod
    def supraventricular_types(cls) -> set:
        """Return set of supraventricular tachycardia types."""
        return {cls.SVT, cls.AFIB_RVR, cls.AFLUTTER}
    
    @classmethod
    def tachycardia_types(cls) -> set:
        """Return all tachycardia types (VT + SVT + sinus)."""
        return cls.ventricular_types() | cls.supraventricular_types() | {cls.SINUS_TACHYCARDIA}
    
    def is_ventricular(self) -> bool:
        """Check if this episode type is ventricular."""
        return self in self.ventricular_types()
    
    def is_supraventricular(self) -> bool:
        """Check if this episode type is supraventricular."""
        return self in self.supraventricular_types()
    
    def is_life_threatening(self) -> bool:
        """Check if this episode type is potentially life-threatening."""
        return self in {
            EpisodeType.VT_MONOMORPHIC,
            EpisodeType.VT_POLYMORPHIC,
            EpisodeType.VFL,
            EpisodeType.VFIB,
        }


class VTLabelConfidence(Enum):
    """
    VT label confidence tiers - explicit truth hierarchy.
    
    v2.3: Distinguishes confirmed VT from V-beat runs that shouldn't be called VT.
    """
    CONFIRMED = "confirmed"         # Meets all criteria (beat sequence + HR + morphology)
    CANDIDATE = "candidate"         # Meets beat criteria, missing morphology confirmation
    HEURISTIC = "heuristic"         # Derived from beat labels only (no rhythm annotation)
    RHYTHM_DERIVED = "rhythm"       # From rhythm annotations (highest trust if expert)
    VENTRICULAR_RUN = "v_run"       # Consecutive V beats - proxy, NOT true VT


class LabelConfidenceTier(Enum):
    """
    Truth hierarchy for labels - higher = more trustworthy.
    
    Used for confidence-weighted metrics and stratified reporting.
    """
    EXPERT_RHYTHM = "expert_rhythm"     # Cardiologist-annotated rhythm labels
    DERIVED_RHYTHM = "derived_rhythm"   # Algorithm-derived from beat labels
    HEURISTIC = "heuristic"             # Beat-sequence heuristic only
    UNCERTAIN = "uncertain"             # Ambiguous or conflicting evidence


@dataclass
class VTLabelCriteria:
    """
    Operational criteria for VT labeling.
    
    Defines the exact thresholds used to determine VT labels.
    """
    # Core criteria (clinical definition)
    min_consecutive_v_beats: int = 3
    min_hr_bpm: float = 100.0
    max_hr_bpm: float = 300.0       # Above this = implausible, likely artifact
    
    # Morphology criteria (for confirmation)
    min_qrs_width_ms: float = 120.0  # Wide QRS = ventricular origin
    max_qrs_template_variance: float = 0.3  # Low variance = monomorphic
    
    # Continuity criteria
    max_gap_for_run_ms: float = 200.0  # Gap > this splits runs
    fusion_breaks_run: bool = True      # F beat interrupts V run
    
    # Special beat handling
    include_escape_beats: bool = False  # 'E' often != true VT
    artifact_splits_run: bool = True    # 'U'/'?' splits runs
    
    # Sustained VT threshold
    sustained_duration_sec: float = 30.0


@dataclass
class ECGSegment:
    """
    Canonical ECG segment - the atomic unit for all processing.
    
    All data from any source dataset is converted to this format before
    being processed by the pipeline. This ensures consistency.
    
    Attributes:
        signal: Single-lead ECG signal, normalized to mV scale
        fs: Sampling frequency (canonical: 360 Hz)
        lead_name: Name of the lead (canonical: "II")
        record_id: Source record identifier
        patient_id: Patient identifier (critical for train/test split integrity)
        start_time_sec: Absolute start time in the original recording
        duration_sec: Segment duration in seconds
        source_dataset: Origin dataset name
    """
    signal: np.ndarray              # Shape: (n_samples,) - single lead, normalized
    fs: int = 360                   # Canonical sampling rate (Hz)
    lead_name: str = "II"           # Canonical lead (II-like)
    record_id: str = ""             # Source record identifier
    patient_id: str = ""            # Patient identifier (for split integrity)
    start_time_sec: float = 0.0     # Absolute start time in recording
    duration_sec: float = 0.0       # Segment duration
    source_dataset: str = ""        # "MIT-BIH", "PTB-XL", "INCART", etc.
    
    def __post_init__(self):
        """Validate and compute derived fields."""
        if len(self.signal) > 0 and self.duration_sec == 0:
            self.duration_sec = len(self.signal) / self.fs
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the segment."""
        return len(self.signal)
    
    @property
    def end_time_sec(self) -> float:
        """End time of the segment."""
        return self.start_time_sec + self.duration_sec
    
    def get_sample_at_time(self, time_sec: float) -> int:
        """Convert absolute time to sample index within this segment."""
        relative_time = time_sec - self.start_time_sec
        return int(relative_time * self.fs)
    
    def get_time_at_sample(self, sample_idx: int) -> float:
        """Convert sample index to absolute time."""
        return self.start_time_sec + sample_idx / self.fs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'signal': self.signal.tolist(),
            'fs': self.fs,
            'lead_name': self.lead_name,
            'record_id': self.record_id,
            'patient_id': self.patient_id,
            'start_time_sec': self.start_time_sec,
            'duration_sec': self.duration_sec,
            'source_dataset': self.source_dataset,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ECGSegment':
        """Create from dictionary."""
        data['signal'] = np.array(data['signal'])
        return cls(**data)


@dataclass
class BeatAnnotation:
    """
    Single beat annotation.
    
    Represents the annotation for a single heartbeat, including its location,
    type, and confidence level.
    
    Attributes:
        sample_idx: R-peak sample index in segment
        beat_type: Canonical beat type ('N', 'V', 'A', 'S', 'F', 'P', 'U')
        confidence: Annotation confidence (1.0 = expert, <1.0 = derived/uncertain)
        original_label: Original label from source dataset (for debugging)
    """
    sample_idx: int                 # R-peak sample index in segment
    beat_type: str                  # 'N', 'V', 'A', 'S', 'F', 'P', 'U'
    confidence: float = 1.0         # Annotation confidence
    original_label: str = ""        # Original label before mapping
    
    # Canonical beat type definitions
    NORMAL_TYPES = {'N', 'L', 'R', 'e', 'j'}     # Normal and normal variants
    VENTRICULAR_TYPES = {'V', 'E'}               # Ventricular ectopic
    SUPRAVENTRICULAR_TYPES = {'A', 'a', 'S', 'J'} # Supraventricular ectopic
    FUSION_TYPES = {'F'}                          # Fusion beats
    PACED_TYPES = {'/', 'f', 'P'}                # Paced beats
    UNKNOWN_TYPES = {'Q', '?', 'U'}              # Unknown/unclassifiable
    
    def is_ventricular(self) -> bool:
        """Check if this is a ventricular beat."""
        return self.beat_type in self.VENTRICULAR_TYPES
    
    def is_normal(self) -> bool:
        """Check if this is a normal beat."""
        return self.beat_type in self.NORMAL_TYPES or self.beat_type == 'N'
    
    def is_supraventricular(self) -> bool:
        """Check if this is a supraventricular ectopic."""
        return self.beat_type in self.SUPRAVENTRICULAR_TYPES
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'sample_idx': self.sample_idx,
            'beat_type': self.beat_type,
            'confidence': self.confidence,
            'original_label': self.original_label,
        }


@dataclass
class EpisodeLabel:
    """
    Ground truth or predicted episode.
    
    An episode represents a continuous period of a specific rhythm type.
    This is the primary unit for evaluation (episode-level metrics).
    
    Attributes:
        start_sample: Start sample index
        end_sample: End sample index
        start_time_sec: Start time in seconds
        end_time_sec: End time in seconds
        episode_type: Type of episode (from EpisodeType enum)
        severity: 'sustained' (≥30s), 'non-sustained', or 'unknown'
        confidence: Label confidence (0.0 to 1.0)
        evidence: Supporting evidence (beat sequence, HR values, etc.)
    """
    start_sample: int
    end_sample: int
    start_time_sec: float
    end_time_sec: float
    episode_type: EpisodeType
    severity: str = "unknown"           # 'sustained', 'non-sustained', 'unknown'
    confidence: float = 1.0             # Label confidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    # v2.3/v2.4: Label provenance tracking
    label_tier: Optional[LabelConfidenceTier] = None
    vt_confidence: Optional[VTLabelConfidence] = None
    source_dataset: str = ""
    record_id: str = ""
    
    # Severity thresholds
    SUSTAINED_DURATION_SEC = 30.0
    
    @property
    def duration_sec(self) -> float:
        """Duration of the episode in seconds."""
        return self.end_time_sec - self.start_time_sec
    
    @property
    def duration_samples(self) -> int:
        """Duration of the episode in samples."""
        return self.end_sample - self.start_sample
    
    def is_sustained(self) -> bool:
        """Check if episode is sustained (≥30 seconds)."""
        return self.duration_sec >= self.SUSTAINED_DURATION_SEC
    
    def overlaps_with(self, other: 'EpisodeLabel') -> bool:
        """Check if this episode overlaps with another."""
        return not (self.end_sample <= other.start_sample or 
                   other.end_sample <= self.start_sample)
    
    def compute_iou(self, other: 'EpisodeLabel') -> float:
        """Compute Intersection over Union with another episode."""
        if not self.overlaps_with(other):
            return 0.0
        
        intersection_start = max(self.start_sample, other.start_sample)
        intersection_end = min(self.end_sample, other.end_sample)
        intersection = intersection_end - intersection_start
        
        union = self.duration_samples + other.duration_samples - intersection
        return intersection / union if union > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_sample': self.start_sample,
            'end_sample': self.end_sample,
            'start_time_sec': self.start_time_sec,
            'end_time_sec': self.end_time_sec,
            'episode_type': self.episode_type.value,
            'severity': self.severity,
            'confidence': self.confidence,
            'evidence': self.evidence,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodeLabel':
        """Create from dictionary."""
        data['episode_type'] = EpisodeType(data['episode_type'])
        return cls(**data)
    
    def __repr__(self) -> str:
        return (f"EpisodeLabel({self.episode_type.value}, "
                f"{self.start_time_sec:.2f}s-{self.end_time_sec:.2f}s, "
                f"{self.severity}, conf={self.confidence:.2f})")


@dataclass
class SQIResult:
    """
    Signal Quality Index assessment result.
    
    Comprehensive signal quality assessment including overall score,
    usability flag, component scores, and recommendations.
    
    Attributes:
        overall_score: Weighted combination of components (0.0 to 1.0)
        is_usable: Hard gate - can we trust classifications from this signal?
        components: Individual quality component scores
        recommendations: List of quality issues detected
    """
    overall_score: float            # 0.0 (unusable) to 1.0 (excellent)
    is_usable: bool                 # Hard gate: can we trust classifications?
    components: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Thresholds
    USABLE_THRESHOLD = 0.5
    GOOD_THRESHOLD = 0.7
    EXCELLENT_THRESHOLD = 0.9
    
    def get_quality_level(self) -> str:
        """Get human-readable quality level."""
        if not self.is_usable:
            return "unusable"
        if self.overall_score >= self.EXCELLENT_THRESHOLD:
            return "excellent"
        if self.overall_score >= self.GOOD_THRESHOLD:
            return "good"
        if self.overall_score >= self.USABLE_THRESHOLD:
            return "acceptable"
        return "poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'overall_score': self.overall_score,
            'is_usable': self.is_usable,
            'components': self.components,
            'recommendations': self.recommendations,
            'quality_level': self.get_quality_level(),
        }


@dataclass
class PredictionResult:
    """
    Model prediction result with uncertainty and explanations.
    
    Encapsulates all outputs from the detection pipeline for a single segment.
    """
    # Core predictions
    episodes: List[EpisodeLabel]
    dense_probabilities: np.ndarray     # (seq_len, num_classes) per-timestep probs
    
    # Quality and uncertainty
    sqi: SQIResult
    uncertainty: np.ndarray             # (seq_len,) per-timestep uncertainty
    calibrated: bool = False
    
    # Alarm state
    alarm_tier: Optional[str] = None    # None, 'warning', 'alarm'
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    
    # XAI (optional, computed on demand)
    attributions: Optional[np.ndarray] = None
    clinical_explanation: Optional[Dict[str, Any]] = None
    
    def get_primary_prediction(self) -> Optional[EpisodeLabel]:
        """Get the highest-confidence episode prediction."""
        if not self.episodes:
            return None
        return max(self.episodes, key=lambda e: e.confidence)
    
    def has_ventricular_tachycardia(self) -> bool:
        """Check if any VT episode was detected."""
        return any(ep.episode_type.is_ventricular() for ep in self.episodes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'episodes': [ep.to_dict() for ep in self.episodes],
            'sqi': self.sqi.to_dict(),
            'alarm_tier': self.alarm_tier,
            'suppressed': self.suppressed,
            'suppression_reason': self.suppression_reason,
            'mean_uncertainty': float(np.mean(self.uncertainty)) if self.uncertainty is not None else None,
        }
