"""
Feature Extraction Module for ECG Beats
Extracts morphological, statistical, and frequency-domain features
"""

import numpy as np
from scipy import signal, stats
from scipy.fft import fft
from typing import Dict, List, Tuple, Optional
import warnings


class FeatureExtractor:
    """
    Comprehensive ECG beat feature extractor
    
    Extracts features that are:
    - Clinically meaningful (for interpretability)
    - Discriminative for tachycardia detection
    - Suitable for XAI explanations
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize feature extractor
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        self.feature_names = []
        
    def extract_all_features(self, beat: np.ndarray,
                              rr_current: float = 0.0,
                              rr_previous: float = 0.0,
                              hr: float = 0.0) -> np.ndarray:
        """
        Extract all features from a single beat
        
        Args:
            beat: Single ECG beat waveform
            rr_current: Current RR interval (seconds)
            rr_previous: Previous RR interval (seconds)
            hr: Heart rate (BPM)
            
        Returns:
            Feature vector
        """
        features = []
        self.feature_names = []
        
        # 1. Statistical features
        stat_features, stat_names = self._extract_statistical_features(beat)
        features.extend(stat_features)
        self.feature_names.extend(stat_names)
        
        # 2. Morphological features
        morph_features, morph_names = self._extract_morphological_features(beat)
        features.extend(morph_features)
        self.feature_names.extend(morph_names)
        
        # 3. Frequency domain features
        freq_features, freq_names = self._extract_frequency_features(beat)
        features.extend(freq_features)
        self.feature_names.extend(freq_names)
        
        # 4. Wavelet features
        wavelet_features, wavelet_names = self._extract_wavelet_features(beat)
        features.extend(wavelet_features)
        self.feature_names.extend(wavelet_names)
        
        # 5. RR interval features
        rr_features, rr_names = self._extract_rr_features(rr_current, rr_previous, hr)
        features.extend(rr_features)
        self.feature_names.extend(rr_names)
        
        return np.array(features)
    
    def extract_features_batch(self, beats: np.ndarray,
                                rr_features: Optional[Dict] = None) -> np.ndarray:
        """
        Extract features from multiple beats
        
        Args:
            beats: Array of beats [n_beats, beat_length]
            rr_features: Dictionary with RR interval features per beat
            
        Returns:
            Feature matrix [n_beats, n_features]
        """
        n_beats = len(beats)
        
        if n_beats == 0:
            return np.array([])
        
        # Initialize with first beat to get feature count
        first_features = self.extract_all_features(
            beats[0],
            rr_features.get('rr_current', np.zeros(n_beats))[0] if rr_features else 0,
            rr_features.get('rr_previous', np.zeros(n_beats))[0] if rr_features else 0,
            rr_features.get('hr_current', np.zeros(n_beats))[0] if rr_features else 0
        )
        
        n_features = len(first_features)
        feature_matrix = np.zeros((n_beats, n_features))
        feature_matrix[0] = first_features
        
        for i in range(1, n_beats):
            rr_curr = rr_features.get('rr_current', np.zeros(n_beats))[i] if rr_features else 0
            rr_prev = rr_features.get('rr_previous', np.zeros(n_beats))[i] if rr_features else 0
            hr = rr_features.get('hr_current', np.zeros(n_beats))[i] if rr_features else 0
            
            feature_matrix[i] = self.extract_all_features(beats[i], rr_curr, rr_prev, hr)
        
        return feature_matrix
    
    def _extract_statistical_features(self, beat: np.ndarray) -> Tuple[List, List]:
        """Extract statistical features from beat"""
        features = []
        names = []
        
        # Basic statistics
        features.append(np.mean(beat))
        names.append('mean')
        
        features.append(np.std(beat))
        names.append('std')
        
        features.append(np.var(beat))
        names.append('variance')
        
        features.append(np.median(beat))
        names.append('median')
        
        features.append(np.min(beat))
        names.append('min')
        
        features.append(np.max(beat))
        names.append('max')
        
        features.append(np.ptp(beat))  # Peak-to-peak
        names.append('peak_to_peak')
        
        # Higher-order statistics
        features.append(stats.skew(beat))
        names.append('skewness')
        
        features.append(stats.kurtosis(beat))
        names.append('kurtosis')
        
        # Root mean square
        features.append(np.sqrt(np.mean(beat**2)))
        names.append('rms')
        
        # Energy
        features.append(np.sum(beat**2))
        names.append('energy')
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.signbit(beat)))
        features.append(zero_crossings / len(beat))
        names.append('zero_crossing_rate')
        
        # Percentiles
        for p in [5, 25, 75, 95]:
            features.append(np.percentile(beat, p))
            names.append(f'percentile_{p}')
        
        # IQR
        features.append(np.percentile(beat, 75) - np.percentile(beat, 25))
        names.append('iqr')
        
        return features, names
    
    def _extract_morphological_features(self, beat: np.ndarray) -> Tuple[List, List]:
        """Extract morphological features from beat waveform"""
        features = []
        names = []
        
        beat_len = len(beat)
        
        # Find R-peak (assumed to be around center)
        center = beat_len // 3  # Pre-R is 1/3 of beat
        search_start = max(0, center - 20)
        search_end = min(beat_len, center + 40)
        r_peak_idx = search_start + np.argmax(beat[search_start:search_end])
        
        # R-peak amplitude
        r_amplitude = beat[r_peak_idx]
        features.append(r_amplitude)
        names.append('r_amplitude')
        
        # R-peak position (normalized)
        features.append(r_peak_idx / beat_len)
        names.append('r_position_normalized')
        
        # Find Q and S points (local minima around R)
        q_search_start = max(0, r_peak_idx - 30)
        q_search_end = r_peak_idx
        if q_search_end > q_search_start:
            q_idx = q_search_start + np.argmin(beat[q_search_start:q_search_end])
            q_amplitude = beat[q_idx]
        else:
            q_idx = r_peak_idx
            q_amplitude = r_amplitude
        
        features.append(q_amplitude)
        names.append('q_amplitude')
        
        s_search_start = r_peak_idx
        s_search_end = min(beat_len, r_peak_idx + 30)
        if s_search_end > s_search_start:
            s_idx = s_search_start + np.argmin(beat[s_search_start:s_search_end])
            s_amplitude = beat[s_idx]
        else:
            s_idx = r_peak_idx
            s_amplitude = r_amplitude
        
        features.append(s_amplitude)
        names.append('s_amplitude')
        
        # QRS complex features
        qrs_duration = (s_idx - q_idx) / self.fs * 1000  # in ms
        features.append(qrs_duration)
        names.append('qrs_duration_ms')
        
        qrs_amplitude = r_amplitude - min(q_amplitude, s_amplitude)
        features.append(qrs_amplitude)
        names.append('qrs_amplitude')
        
        # QRS area (integral)
        if s_idx > q_idx:
            qrs_area = np.trapz(np.abs(beat[q_idx:s_idx+1]))
            features.append(qrs_area)
        else:
            features.append(0)
        names.append('qrs_area')
        
        # T-wave features (after S point)
        t_search_start = s_idx + 10
        t_search_end = min(beat_len, s_idx + 100)
        
        if t_search_end > t_search_start:
            t_segment = beat[t_search_start:t_search_end]
            t_peak_local = np.argmax(np.abs(t_segment))
            t_amplitude = t_segment[t_peak_local]
            
            features.append(t_amplitude)
            names.append('t_amplitude')
            
            features.append(np.mean(t_segment))
            names.append('t_wave_mean')
        else:
            features.append(0)
            names.append('t_amplitude')
            features.append(0)
            names.append('t_wave_mean')
        
        # P-wave features (before Q point)
        p_search_start = 0
        p_search_end = max(0, q_idx - 10)
        
        if p_search_end > p_search_start:
            p_segment = beat[p_search_start:p_search_end]
            p_peak_local = np.argmax(p_segment)
            p_amplitude = p_segment[p_peak_local]
            
            features.append(p_amplitude)
            names.append('p_amplitude')
            
            features.append(np.mean(p_segment))
            names.append('p_wave_mean')
        else:
            features.append(0)
            names.append('p_amplitude')
            features.append(0)
            names.append('p_wave_mean')
        
        # Slope features
        if len(beat) > 1:
            derivative = np.diff(beat)
            features.append(np.max(derivative))
            names.append('max_upslope')
            
            features.append(np.min(derivative))
            names.append('max_downslope')
            
            features.append(np.mean(np.abs(derivative)))
            names.append('mean_abs_slope')
        else:
            features.extend([0, 0, 0])
            names.extend(['max_upslope', 'max_downslope', 'mean_abs_slope'])
        
        # Symmetry features
        mid = beat_len // 2
        first_half = beat[:mid]
        second_half = beat[mid:2*mid] if len(beat) >= 2*mid else beat[mid:]
        
        if len(second_half) == len(first_half) and len(first_half) > 0:
            correlation = np.corrcoef(first_half, second_half[::-1])[0, 1]
            features.append(correlation if not np.isnan(correlation) else 0)
        else:
            features.append(0)
        names.append('beat_symmetry')
        
        return features, names
    
    def _extract_frequency_features(self, beat: np.ndarray) -> Tuple[List, List]:
        """Extract frequency domain features"""
        features = []
        names = []
        
        # FFT
        n = len(beat)
        fft_vals = np.abs(fft(beat))[:n//2]
        freqs = np.fft.fftfreq(n, 1/self.fs)[:n//2]
        
        # Power spectral density
        psd = fft_vals ** 2
        total_power = np.sum(psd)
        
        if total_power > 0:
            # Normalized PSD
            psd_norm = psd / total_power
            
            # Spectral centroid (center of mass of spectrum)
            spectral_centroid = np.sum(freqs * psd_norm)
            features.append(spectral_centroid)
            names.append('spectral_centroid')
            
            # Spectral spread (standard deviation of spectrum)
            spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))
            features.append(spectral_spread)
            names.append('spectral_spread')
            
            # Spectral entropy
            psd_norm_nonzero = psd_norm[psd_norm > 0]
            spectral_entropy = -np.sum(psd_norm_nonzero * np.log2(psd_norm_nonzero))
            features.append(spectral_entropy)
            names.append('spectral_entropy')
            
            # Band powers
            # Low frequency (0-5 Hz)
            lf_mask = (freqs >= 0) & (freqs < 5)
            lf_power = np.sum(psd[lf_mask]) / total_power
            features.append(lf_power)
            names.append('lf_power_ratio')
            
            # Mid frequency (5-15 Hz) - QRS complex
            mf_mask = (freqs >= 5) & (freqs < 15)
            mf_power = np.sum(psd[mf_mask]) / total_power
            features.append(mf_power)
            names.append('mf_power_ratio')
            
            # High frequency (15-40 Hz)
            hf_mask = (freqs >= 15) & (freqs < 40)
            hf_power = np.sum(psd[hf_mask]) / total_power
            features.append(hf_power)
            names.append('hf_power_ratio')
            
            # Dominant frequency
            dominant_freq_idx = np.argmax(psd)
            features.append(freqs[dominant_freq_idx])
            names.append('dominant_frequency')
            
            # Spectral edge frequency (95% of power)
            cumsum_psd = np.cumsum(psd_norm)
            edge_idx = np.argmax(cumsum_psd >= 0.95)
            features.append(freqs[edge_idx])
            names.append('spectral_edge_freq_95')
        else:
            features.extend([0] * 9)
            names.extend([
                'spectral_centroid', 'spectral_spread', 'spectral_entropy',
                'lf_power_ratio', 'mf_power_ratio', 'hf_power_ratio',
                'dominant_frequency', 'spectral_edge_freq_95', 'total_power'
            ])
        
        features.append(total_power)
        names.append('total_power')
        
        return features, names
    
    def _extract_wavelet_features(self, beat: np.ndarray, 
                                   levels: int = 4) -> Tuple[List, List]:
        """Extract wavelet-based features using simple implementation"""
        features = []
        names = []
        
        # Simple multi-resolution analysis using downsampling
        current_signal = beat.copy()
        
        for level in range(1, levels + 1):
            # Approximate wavelet decomposition
            # Low-pass (approximation)
            if len(current_signal) < 4:
                break
                
            # Simple low-pass filter
            kernel = np.array([0.25, 0.5, 0.25])
            approx = np.convolve(current_signal, kernel, mode='same')[::2]
            
            # High-pass (detail) - difference
            detail = current_signal[:-1] - current_signal[1:]
            
            # Extract statistics from coefficients
            features.append(np.mean(np.abs(detail)))
            names.append(f'wavelet_detail_{level}_mean')
            
            features.append(np.std(detail))
            names.append(f'wavelet_detail_{level}_std')
            
            features.append(np.sum(detail**2))
            names.append(f'wavelet_detail_{level}_energy')
            
            current_signal = approx
        
        # Final approximation statistics
        if len(current_signal) > 0:
            features.append(np.mean(current_signal))
            names.append('wavelet_approx_mean')
            
            features.append(np.std(current_signal))
            names.append('wavelet_approx_std')
        else:
            features.extend([0, 0])
            names.extend(['wavelet_approx_mean', 'wavelet_approx_std'])
        
        return features, names
    
    def _extract_rr_features(self, rr_current: float, 
                              rr_previous: float, 
                              hr: float) -> Tuple[List, List]:
        """Extract RR interval related features"""
        features = []
        names = []
        
        # Current RR interval
        features.append(rr_current)
        names.append('rr_interval')
        
        # Previous RR interval
        features.append(rr_previous)
        names.append('rr_previous')
        
        # Heart rate
        features.append(hr)
        names.append('heart_rate')
        
        # RR ratio (current / previous)
        if rr_previous > 0:
            rr_ratio = rr_current / rr_previous
        else:
            rr_ratio = 1.0
        features.append(rr_ratio)
        names.append('rr_ratio')
        
        # RR difference
        features.append(rr_current - rr_previous)
        names.append('rr_diff')
        
        # Is tachycardia (HR > 100)
        features.append(1 if hr > 100 else 0)
        names.append('is_tachycardia_hr')
        
        # Is bradycardia (HR < 60)
        features.append(1 if hr < 60 else 0)
        names.append('is_bradycardia_hr')
        
        return features, names
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (after calling extract_all_features)"""
        return self.feature_names
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Group features by category for XAI explanations
        
        Returns:
            Dictionary mapping category to feature names
        """
        groups = {
            'RR_Interval': [n for n in self.feature_names if 'rr_' in n or 'heart_rate' in n],
            'QRS_Complex': [n for n in self.feature_names if 'qrs' in n or n in ['r_amplitude', 'q_amplitude', 's_amplitude']],
            'Waveform_Statistics': [n for n in self.feature_names if n in ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'rms', 'energy']],
            'P_Wave': [n for n in self.feature_names if 'p_' in n],
            'T_Wave': [n for n in self.feature_names if 't_' in n],
            'Frequency': [n for n in self.feature_names if 'spectral' in n or 'power' in n or 'frequency' in n],
            'Wavelet': [n for n in self.feature_names if 'wavelet' in n],
            'Morphology': [n for n in self.feature_names if 'slope' in n or 'symmetry' in n]
        }
        return groups


def main():
    """Test feature extraction"""
    # Create synthetic beat
    np.random.seed(42)
    t = np.linspace(0, 0.6, 216)  # 600ms beat at 360 Hz
    
    # Simple synthetic ECG beat
    beat = np.zeros_like(t)
    # P-wave
    beat += 0.1 * np.exp(-((t - 0.1) / 0.02) ** 2)
    # QRS complex
    beat += -0.2 * np.exp(-((t - 0.18) / 0.01) ** 2)  # Q
    beat += 1.0 * np.exp(-((t - 0.2) / 0.015) ** 2)   # R
    beat += -0.3 * np.exp(-((t - 0.22) / 0.01) ** 2)  # S
    # T-wave
    beat += 0.3 * np.exp(-((t - 0.35) / 0.04) ** 2)
    
    # Add noise
    beat += 0.02 * np.random.randn(len(beat))
    
    # Extract features
    extractor = FeatureExtractor(sampling_rate=360)
    features = extractor.extract_all_features(beat, rr_current=0.6, rr_previous=0.65, hr=100)
    
    print(f"Extracted {len(features)} features:")
    for name, value in zip(extractor.get_feature_names(), features):
        print(f"  {name}: {value:.4f}")
    
    print("\nFeature groups:")
    groups = extractor.get_feature_importance_groups()
    for group, feature_names in groups.items():
        print(f"  {group}: {len(feature_names)} features")


if __name__ == '__main__':
    main()
