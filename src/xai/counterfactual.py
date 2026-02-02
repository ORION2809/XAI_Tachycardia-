"""
Clinical Counterfactual Explanations.

Rule-based explanations from the clinical decision layer.
Answers "What would need to change for this NOT to be [diagnosis]?"

This is complementary to saliency-based XAI and often more
useful for clinical users who think in terms of diagnostic criteria.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.contracts import EpisodeLabel, EpisodeType


@dataclass
class CounterfactualFactor:
    """A factor contributing to a diagnosis."""
    name: str                       # Factor name
    current_value: float            # Current value
    threshold: float                # Threshold for this criterion
    importance: str                 # 'critical', 'high', 'medium', 'low'
    direction: str                  # 'above' or 'below' threshold
    unit: str = ""                  # Unit of measurement
    
    @property
    def margin(self) -> float:
        """How far above/below threshold."""
        if self.direction == 'above':
            return self.current_value - self.threshold
        else:
            return self.threshold - self.current_value
    
    @property
    def is_met(self) -> bool:
        """Whether this criterion is currently met."""
        return self.margin > 0
    
    def get_counterfactual(self) -> str:
        """Get counterfactual statement for this factor."""
        if self.direction == 'above':
            needed = self.threshold - 1
            return f"If {self.name} was ≤{needed:.0f}{self.unit}, this criterion would not be met"
        else:
            needed = self.threshold + 1
            return f"If {self.name} was ≥{needed:.0f}{self.unit}, this criterion would not be met"


@dataclass
class CounterfactualExplanation:
    """Complete counterfactual explanation for a diagnosis."""
    episode_type: EpisodeType
    factors: List[CounterfactualFactor]
    counterfactuals: List[str]
    summary: str
    confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_type': self.episode_type.value,
            'factors': [
                {
                    'name': f.name,
                    'value': f.current_value,
                    'threshold': f.threshold,
                    'importance': f.importance,
                    'direction': f.direction,
                    'unit': f.unit,
                    'is_met': f.is_met,
                    'margin': f.margin,
                }
                for f in self.factors
            ],
            'counterfactuals': self.counterfactuals,
            'summary': self.summary,
            'confidence': self.confidence,
        }
    
    def get_critical_factors(self) -> List[CounterfactualFactor]:
        """Get factors marked as critical."""
        return [f for f in self.factors if f.importance == 'critical']
    
    def get_easiest_counterfactual(self) -> Optional[CounterfactualFactor]:
        """
        Get the factor closest to its threshold.
        
        This represents the "easiest" change that would alter the diagnosis.
        """
        met_factors = [f for f in self.factors if f.is_met]
        if not met_factors:
            return None
        return min(met_factors, key=lambda f: abs(f.margin))


class ClinicalCounterfactual:
    """
    Generate clinical counterfactual explanations.
    
    These explanations are based on the clinical diagnostic criteria
    and explain WHY a particular diagnosis was made in terms
    clinicians understand.
    
    VT Criteria:
    - ≥3 consecutive ventricular beats
    - Heart rate > 100 BPM
    - Duration determines sustained (≥30s) vs non-sustained
    
    SVT Criteria:
    - Narrow QRS complex
    - Heart rate > 100 BPM
    - Regular or irregular rhythm
    """
    
    # Clinical thresholds
    VT_MIN_V_BEATS = 3
    VT_MIN_HR = 100.0
    VT_SUSTAINED_DURATION = 30.0
    SVT_MIN_HR = 100.0
    SINUS_TACHY_MIN_HR = 100.0
    
    def __init__(self):
        pass
    
    def explain_vt(
        self,
        episode: EpisodeLabel,
        signal: Optional[np.ndarray] = None,
        r_peaks: Optional[np.ndarray] = None,
        fs: int = 360,
    ) -> CounterfactualExplanation:
        """
        Generate counterfactual explanation for VT diagnosis.
        
        Args:
            episode: The VT episode label
            signal: Optional ECG signal
            r_peaks: Optional R-peak locations
            fs: Sampling frequency
            
        Returns:
            CounterfactualExplanation
        """
        factors = []
        counterfactuals = []
        
        # Factor 1: Consecutive ventricular beats
        v_count = episode.evidence.get('v_beat_count', 0)
        if v_count == 0:
            # Estimate from duration
            duration_sec = episode.duration_sec
            v_count = max(3, int(duration_sec * 2.5))  # Rough estimate
        
        factors.append(CounterfactualFactor(
            name="Consecutive V beats",
            current_value=v_count,
            threshold=self.VT_MIN_V_BEATS,
            importance="critical",
            direction="above",
            unit=" beats",
        ))
        
        if v_count >= self.VT_MIN_V_BEATS:
            counterfactuals.append(
                f"If there were only {self.VT_MIN_V_BEATS - 1} consecutive ventricular beats, "
                f"this would NOT meet VT criteria (currently {v_count} beats)"
            )
        
        # Factor 2: Heart rate
        hr = episode.evidence.get('computed_hr_bpm', 0)
        if hr == 0 and r_peaks is not None and len(r_peaks) >= 2:
            # Compute from R-peaks in episode
            ep_peaks = r_peaks[(r_peaks >= episode.start_sample) & 
                              (r_peaks <= episode.end_sample)]
            if len(ep_peaks) >= 2:
                rr = np.diff(ep_peaks) / fs * 1000  # ms
                hr = 60000 / np.median(rr)
        
        if hr > 0:
            factors.append(CounterfactualFactor(
                name="Heart rate",
                current_value=hr,
                threshold=self.VT_MIN_HR,
                importance="critical",
                direction="above",
                unit=" BPM",
            ))
            
            if hr >= self.VT_MIN_HR:
                counterfactuals.append(
                    f"If heart rate was <{self.VT_MIN_HR:.0f} BPM, "
                    f"this would NOT meet VT rate criterion (currently {hr:.0f} BPM)"
                )
        
        # Factor 3: Duration (for sustained classification)
        duration = episode.duration_sec
        factors.append(CounterfactualFactor(
            name="Episode duration",
            current_value=duration,
            threshold=self.VT_SUSTAINED_DURATION,
            importance="high" if episode.severity == "sustained" else "medium",
            direction="above",
            unit=" seconds",
        ))
        
        if episode.severity == "sustained":
            counterfactuals.append(
                f"If duration was <{self.VT_SUSTAINED_DURATION:.0f}s, "
                f"this would be classified as NON-SUSTAINED VT (currently {duration:.1f}s)"
            )
        else:
            counterfactuals.append(
                f"This is non-sustained VT ({duration:.1f}s). "
                f"Sustained VT requires ≥{self.VT_SUSTAINED_DURATION:.0f}s duration"
            )
        
        # Summary
        if episode.severity == "sustained":
            summary = (
                f"Sustained Ventricular Tachycardia detected: "
                f"{v_count} consecutive ventricular beats at {hr:.0f} BPM "
                f"lasting {duration:.1f} seconds"
            )
        else:
            summary = (
                f"Non-sustained Ventricular Tachycardia detected: "
                f"{v_count} consecutive ventricular beats at {hr:.0f} BPM "
                f"lasting {duration:.1f} seconds"
            )
        
        return CounterfactualExplanation(
            episode_type=episode.episode_type,
            factors=factors,
            counterfactuals=counterfactuals,
            summary=summary,
            confidence=episode.confidence,
        )
    
    def explain_svt(
        self,
        episode: EpisodeLabel,
        signal: Optional[np.ndarray] = None,
        r_peaks: Optional[np.ndarray] = None,
        fs: int = 360,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation for SVT diagnosis."""
        factors = []
        counterfactuals = []
        
        # Factor 1: Heart rate
        hr = episode.evidence.get('computed_hr_bpm', 0)
        if hr > 0:
            factors.append(CounterfactualFactor(
                name="Heart rate",
                current_value=hr,
                threshold=self.SVT_MIN_HR,
                importance="critical",
                direction="above",
                unit=" BPM",
            ))
            
            if hr >= self.SVT_MIN_HR:
                counterfactuals.append(
                    f"If heart rate was <{self.SVT_MIN_HR:.0f} BPM, "
                    f"this would NOT be classified as tachycardia (currently {hr:.0f} BPM)"
                )
        
        # Factor 2: Duration
        duration = episode.duration_sec
        factors.append(CounterfactualFactor(
            name="Episode duration",
            current_value=duration,
            threshold=1.0,  # Arbitrary minimum for significance
            importance="medium",
            direction="above",
            unit=" seconds",
        ))
        
        # Factor 3: QRS width (if available)
        qrs_width = episode.evidence.get('qrs_width_ms', 0)
        if qrs_width > 0:
            factors.append(CounterfactualFactor(
                name="QRS width",
                current_value=qrs_width,
                threshold=120,  # Wide QRS threshold
                importance="high",
                direction="below",  # SVT has narrow QRS
                unit=" ms",
            ))
            
            if qrs_width < 120:
                counterfactuals.append(
                    f"QRS width is narrow ({qrs_width:.0f} ms), consistent with SVT. "
                    f"If QRS was ≥120 ms, would consider VT or aberrant conduction"
                )
        
        # Summary based on SVT subtype
        if episode.episode_type == EpisodeType.AFIB_RVR:
            summary = f"Atrial Fibrillation with Rapid Ventricular Response at {hr:.0f} BPM"
        elif episode.episode_type == EpisodeType.AFLUTTER:
            summary = f"Atrial Flutter with ventricular rate {hr:.0f} BPM"
        else:
            summary = f"Supraventricular Tachycardia at {hr:.0f} BPM"
        
        return CounterfactualExplanation(
            episode_type=episode.episode_type,
            factors=factors,
            counterfactuals=counterfactuals,
            summary=summary,
            confidence=episode.confidence,
        )
    
    def explain_sinus_tachycardia(
        self,
        episode: EpisodeLabel,
        r_peaks: Optional[np.ndarray] = None,
        fs: int = 360,
    ) -> CounterfactualExplanation:
        """Generate counterfactual explanation for sinus tachycardia."""
        factors = []
        counterfactuals = []
        
        hr = episode.evidence.get('computed_hr_bpm', 100)
        
        factors.append(CounterfactualFactor(
            name="Heart rate",
            current_value=hr,
            threshold=self.SINUS_TACHY_MIN_HR,
            importance="critical",
            direction="above",
            unit=" BPM",
        ))
        
        counterfactuals.append(
            f"If heart rate was <{self.SINUS_TACHY_MIN_HR:.0f} BPM, "
            f"this would be normal sinus rhythm (currently {hr:.0f} BPM)"
        )
        
        summary = f"Sinus Tachycardia at {hr:.0f} BPM with normal P-wave morphology"
        
        return CounterfactualExplanation(
            episode_type=episode.episode_type,
            factors=factors,
            counterfactuals=counterfactuals,
            summary=summary,
            confidence=episode.confidence,
        )
    
    def explain(
        self,
        episode: EpisodeLabel,
        signal: Optional[np.ndarray] = None,
        r_peaks: Optional[np.ndarray] = None,
        fs: int = 360,
    ) -> CounterfactualExplanation:
        """
        Generate appropriate counterfactual explanation based on episode type.
        
        Args:
            episode: Episode to explain
            signal: Optional ECG signal
            r_peaks: Optional R-peak locations
            fs: Sampling frequency
            
        Returns:
            CounterfactualExplanation
        """
        if episode.episode_type.is_ventricular():
            return self.explain_vt(episode, signal, r_peaks, fs)
        elif episode.episode_type.is_supraventricular():
            return self.explain_svt(episode, signal, r_peaks, fs)
        elif episode.episode_type == EpisodeType.SINUS_TACHYCARDIA:
            return self.explain_sinus_tachycardia(episode, r_peaks, fs)
        else:
            # Generic explanation
            return CounterfactualExplanation(
                episode_type=episode.episode_type,
                factors=[],
                counterfactuals=[],
                summary=f"Episode type: {episode.episode_type.value}",
                confidence=episode.confidence,
            )
    
    def format_for_clinician(
        self,
        explanation: CounterfactualExplanation,
    ) -> str:
        """
        Format explanation as human-readable text for clinicians.
        
        Returns a structured text explanation.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("CLINICAL EXPLANATION")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"DIAGNOSIS: {explanation.summary}")
        lines.append(f"Confidence: {explanation.confidence * 100:.0f}%")
        lines.append("")
        
        if explanation.factors:
            lines.append("DIAGNOSTIC CRITERIA:")
            lines.append("-" * 40)
            for factor in explanation.factors:
                status = "✓" if factor.is_met else "✗"
                lines.append(
                    f"  {status} {factor.name}: {factor.current_value:.1f}{factor.unit} "
                    f"(threshold: {'>' if factor.direction == 'above' else '<'}"
                    f"{factor.threshold:.1f}{factor.unit})"
                )
            lines.append("")
        
        if explanation.counterfactuals:
            lines.append("WHAT WOULD CHANGE THE DIAGNOSIS:")
            lines.append("-" * 40)
            for cf in explanation.counterfactuals:
                lines.append(f"  • {cf}")
            lines.append("")
        
        # Easiest counterfactual
        easiest = explanation.get_easiest_counterfactual()
        if easiest:
            lines.append("CLOSEST CRITERION:")
            lines.append(f"  {easiest.name} is {abs(easiest.margin):.1f}{easiest.unit} "
                        f"{'above' if easiest.direction == 'above' else 'below'} threshold")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


if __name__ == "__main__":
    print("Clinical Counterfactual Demo")
    print("=" * 60)
    
    # Create a sample VT episode
    vt_episode = EpisodeLabel(
        start_sample=0,
        end_sample=3600,
        start_time_sec=0.0,
        end_time_sec=10.0,
        episode_type=EpisodeType.VT_MONOMORPHIC,
        severity="non-sustained",
        confidence=0.92,
        evidence={
            'v_beat_count': 15,
            'computed_hr_bpm': 180,
        }
    )
    
    explainer = ClinicalCounterfactual()
    explanation = explainer.explain(vt_episode)
    
    print(explainer.format_for_clinician(explanation))
    
    print("\nAs dictionary:")
    import json
    print(json.dumps(explanation.to_dict(), indent=2))
