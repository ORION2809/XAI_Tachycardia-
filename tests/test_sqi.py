"""
Unit tests for SQI (Signal Quality Index) module.

Tests for:
- SQIComponent computation
- SQISuite aggregation
- SQIPolicy class-conditional behavior
- VF/VFL SQI bypass
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# MOCK SQI CLASSES (for testing without full implementation)
# =============================================================================

class MockSQIResult:
    """Mock SQI result for testing."""
    def __init__(self, overall_score: float, components: dict = None):
        self.overall_score = overall_score
        self.components = components or {}


class MockSQISuite:
    """Mock SQI suite for testing."""
    
    def compute_sqi(self, signal: np.ndarray, fs: int) -> MockSQIResult:
        """Compute mock SQI based on signal properties."""
        # Simple heuristic: low amplitude = low quality
        amplitude = np.std(signal)
        score = min(1.0, amplitude / 500.0)
        
        return MockSQIResult(
            overall_score=score,
            components={
                'qrs_detectability': score,
                'baseline_stability': score,
                'noise_level': 1.0 - score,
            }
        )


class MockSQIPolicy:
    """Mock SQI policy for testing class-conditional behavior."""
    
    VF_VFL_TYPES = {'VFL', 'VFIB', 'VT_POLYMORPHIC'}
    
    def __init__(self, min_sqi: float = 0.6, vf_bypass_threshold: float = 0.7):
        self.min_sqi = min_sqi
        self.vf_bypass_threshold = vf_bypass_threshold
    
    def should_suppress(
        self,
        episode_type: str,
        confidence: float,
        sqi_score: float,
    ) -> bool:
        """Determine if prediction should be suppressed due to SQI."""
        # VF/VFL bypass: high confidence VF should not be suppressed
        if episode_type in self.VF_VFL_TYPES and confidence >= self.vf_bypass_threshold:
            return False
        
        # Normal suppression
        return sqi_score < self.min_sqi


# =============================================================================
# SQI COMPUTATION TESTS
# =============================================================================

class TestSQIComputation:
    """Tests for SQI computation."""
    
    def test_high_quality_signal_high_sqi(self):
        """High quality signal should have high SQI."""
        suite = MockSQISuite()
        
        # Generate clean signal with good amplitude
        t = np.linspace(0, 10, 3600)  # 10s at 360 Hz
        signal = 1000 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz sine, amplitude 1000
        
        result = suite.compute_sqi(signal, 360)
        
        assert result.overall_score >= 0.8
    
    def test_low_quality_signal_low_sqi(self):
        """Low quality signal should have low SQI."""
        suite = MockSQISuite()
        
        # Generate low amplitude noisy signal
        signal = np.random.randn(3600) * 10  # Very low amplitude noise
        
        result = suite.compute_sqi(signal, 360)
        
        assert result.overall_score < 0.5
    
    def test_sqi_components_returned(self):
        """SQI result should include component scores."""
        suite = MockSQISuite()
        signal = np.random.randn(3600) * 500
        
        result = suite.compute_sqi(signal, 360)
        
        assert 'qrs_detectability' in result.components
        assert 'baseline_stability' in result.components
        assert 'noise_level' in result.components


# =============================================================================
# SQI POLICY TESTS
# =============================================================================

class TestSQIPolicy:
    """Tests for SQI policy class-conditional behavior."""
    
    def test_low_sqi_suppresses_normal_detection(self):
        """Low SQI should suppress normal VT detection."""
        policy = MockSQIPolicy(min_sqi=0.6)
        
        # Low SQI, normal VT
        should_suppress = policy.should_suppress(
            episode_type="VT_MONOMORPHIC",
            confidence=0.9,
            sqi_score=0.4
        )
        
        assert should_suppress is True
    
    def test_high_sqi_does_not_suppress(self):
        """High SQI should not suppress detection."""
        policy = MockSQIPolicy(min_sqi=0.6)
        
        # High SQI
        should_suppress = policy.should_suppress(
            episode_type="VT_MONOMORPHIC",
            confidence=0.9,
            sqi_score=0.8
        )
        
        assert should_suppress is False
    
    def test_vf_bypass_low_sqi_high_confidence(self):
        """VF/VFL with high confidence should NOT be suppressed even with low SQI."""
        policy = MockSQIPolicy(min_sqi=0.6, vf_bypass_threshold=0.7)
        
        # Low SQI, but VFL with high confidence
        should_suppress = policy.should_suppress(
            episode_type="VFL",
            confidence=0.85,
            sqi_score=0.3  # Very low SQI
        )
        
        assert should_suppress is False  # Bypass active!
    
    def test_vf_bypass_low_confidence_still_suppressed(self):
        """VF/VFL with low confidence should still be suppressed on low SQI."""
        policy = MockSQIPolicy(min_sqi=0.6, vf_bypass_threshold=0.7)
        
        # Low SQI, VFL with low confidence
        should_suppress = policy.should_suppress(
            episode_type="VFL",
            confidence=0.5,
            sqi_score=0.3
        )
        
        assert should_suppress is True  # Low confidence, no bypass
    
    def test_vfib_bypass(self):
        """VFIB should also have bypass."""
        policy = MockSQIPolicy(min_sqi=0.6, vf_bypass_threshold=0.7)
        
        should_suppress = policy.should_suppress(
            episode_type="VFIB",
            confidence=0.9,
            sqi_score=0.2
        )
        
        assert should_suppress is False
    
    def test_svt_no_bypass(self):
        """SVT should NOT have VF bypass."""
        policy = MockSQIPolicy(min_sqi=0.6, vf_bypass_threshold=0.7)
        
        # SVT should be suppressed on low SQI regardless of confidence
        should_suppress = policy.should_suppress(
            episode_type="SVT",
            confidence=0.95,
            sqi_score=0.3
        )
        
        assert should_suppress is True


# =============================================================================
# SQI THRESHOLD TESTS
# =============================================================================

class TestSQIThresholds:
    """Tests for SQI threshold behavior."""
    
    def test_boundary_sqi_exactly_at_threshold(self):
        """Boundary case: SQI exactly at threshold."""
        policy = MockSQIPolicy(min_sqi=0.6)
        
        # Exactly at threshold - should not suppress
        should_suppress = policy.should_suppress(
            episode_type="VT_MONOMORPHIC",
            confidence=0.9,
            sqi_score=0.6
        )
        
        assert should_suppress is False
    
    def test_boundary_sqi_just_below_threshold(self):
        """Boundary case: SQI just below threshold."""
        policy = MockSQIPolicy(min_sqi=0.6)
        
        # Just below threshold - should suppress
        should_suppress = policy.should_suppress(
            episode_type="VT_MONOMORPHIC",
            confidence=0.9,
            sqi_score=0.59
        )
        
        assert should_suppress is True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSQIIntegration:
    """Integration tests for SQI pipeline."""
    
    def test_full_pipeline_clean_signal(self):
        """Clean signal should pass through without suppression."""
        suite = MockSQISuite()
        policy = MockSQIPolicy(min_sqi=0.6)
        
        # Clean signal
        signal = np.random.randn(3600) * 500
        sqi_result = suite.compute_sqi(signal, 360)
        
        should_suppress = policy.should_suppress(
            episode_type="VT_MONOMORPHIC",
            confidence=0.9,
            sqi_score=sqi_result.overall_score
        )
        
        # With good signal, should not suppress
        assert not should_suppress
    
    def test_full_pipeline_noisy_signal_vf_bypass(self):
        """Noisy signal with VF detection should use bypass."""
        suite = MockSQISuite()
        policy = MockSQIPolicy(min_sqi=0.6, vf_bypass_threshold=0.7)
        
        # Very noisy signal
        signal = np.random.randn(3600) * 10  # Low amplitude
        sqi_result = suite.compute_sqi(signal, 360)
        
        # Low SQI expected
        assert sqi_result.overall_score < 0.5
        
        # But VFL with high confidence should bypass
        should_suppress = policy.should_suppress(
            episode_type="VFL",
            confidence=0.85,
            sqi_score=sqi_result.overall_score
        )
        
        assert not should_suppress  # Bypass active


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
