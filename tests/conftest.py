"""
pytest configuration and shared fixtures.

This module provides shared fixtures and configuration for all tests.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# =============================================================================
# CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_ecg_signal():
    """Generate a sample ECG-like signal."""
    fs = 360
    duration_sec = 10
    n_samples = fs * duration_sec
    
    t = np.linspace(0, duration_sec, n_samples)
    
    # Simple ECG approximation: QRS complexes
    signal = np.zeros(n_samples)
    
    # Add QRS complexes at ~75 bpm (every 0.8 seconds)
    rr_interval = int(0.8 * fs)  # ~288 samples
    
    for i in range(0, n_samples - 50, rr_interval):
        # Simple QRS shape
        signal[i:i+10] = 0.1 * np.sin(np.linspace(0, np.pi, 10))
        signal[i+10:i+20] = np.linspace(0.1, 1.0, 10)
        signal[i+20:i+25] = np.linspace(1.0, -0.3, 5)
        signal[i+25:i+35] = np.linspace(-0.3, 0.0, 10)
    
    # Add some noise
    signal += np.random.randn(n_samples) * 0.05
    
    return signal, fs


@pytest.fixture
def sample_vt_signal():
    """Generate a sample VT-like signal (wide QRS, fast rate)."""
    fs = 360
    duration_sec = 10
    n_samples = fs * duration_sec
    
    t = np.linspace(0, duration_sec, n_samples)
    
    signal = np.zeros(n_samples)
    
    # VT at ~180 bpm (every 0.33 seconds)
    rr_interval = int(0.33 * fs)  # ~120 samples
    
    for i in range(0, n_samples - 60, rr_interval):
        # Wide QRS shape
        signal[i:i+30] = np.sin(np.linspace(0, np.pi, 30))
        signal[i+30:i+50] = -0.5 * np.sin(np.linspace(0, np.pi, 20))
    
    # Add noise
    signal += np.random.randn(n_samples) * 0.1
    
    return signal, fs


@pytest.fixture
def sample_noisy_signal():
    """Generate a noisy/artifacted signal."""
    fs = 360
    duration_sec = 10
    n_samples = fs * duration_sec
    
    # High amplitude noise
    signal = np.random.randn(n_samples) * 0.5
    
    # Add baseline wander
    t = np.linspace(0, duration_sec, n_samples)
    signal += 0.3 * np.sin(2 * np.pi * 0.3 * t)
    
    return signal, fs


# =============================================================================
# MODEL FIXTURES
# =============================================================================

@pytest.fixture
def mock_model_output():
    """Mock model probability outputs."""
    # 5 classes: Normal, Sinus Tachy, SVT, VT, VFL
    n_samples = 100
    
    probs = np.random.rand(n_samples, 5)
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    return probs


@pytest.fixture
def mock_uncertainty():
    """Mock uncertainty estimates."""
    n_samples = 100
    return np.random.rand(n_samples) * 0.3  # 0 to 0.3 uncertainty


# =============================================================================
# EPISODE FIXTURES
# =============================================================================

@pytest.fixture
def sample_episode_labels():
    """Sample episode labels for testing."""
    from evaluation.metrics import EpisodeLabel, EpisodeType, LabelConfidenceTier
    
    return [
        EpisodeLabel(
            EpisodeType.VT_MONOMORPHIC, 
            start_sample=1000, 
            end_sample=2000,
            start_time_sec=2.78,
            end_time_sec=5.56,
            label_tier=LabelConfidenceTier.EXPERT_RHYTHM,
            patient_id="P001"
        ),
        EpisodeLabel(
            EpisodeType.SVT,
            start_sample=5000,
            end_sample=7000,
            start_time_sec=13.89,
            end_time_sec=19.44,
            label_tier=LabelConfidenceTier.DERIVED_RHYTHM,
            patient_id="P002"
        ),
    ]


# =============================================================================
# PATIENT FIXTURES
# =============================================================================

@pytest.fixture
def sample_patient_metadata():
    """Sample patient metadata for testing."""
    from evaluation.validation import PatientMetadata
    
    return {
        "P001": PatientMetadata("P001", age=65, is_paced=False, mean_sqi=0.85),
        "P002": PatientMetadata("P002", age=78, is_paced=True, mean_sqi=0.55),
        "P003": PatientMetadata("P003", age=45, has_bbb=True, mean_sqi=0.90),
    }


@pytest.fixture
def sample_monitoring_hours():
    """Sample monitoring hours per patient."""
    return {"P001": 2.0, "P002": 1.5, "P003": 1.0}
