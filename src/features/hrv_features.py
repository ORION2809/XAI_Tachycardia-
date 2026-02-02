"""
Heart Rate Variability (HRV) Feature Extraction
Extracts HRV features for tachycardia characterization
"""

import numpy as np
from scipy import signal, interpolate
from typing import Dict, List, Tuple


class HRVFeatureExtractor:
    """
    Extracts Heart Rate Variability features from RR interval series
    
    HRV features are crucial for:
    - Distinguishing sinus tachycardia from pathological tachycardia
    - Detecting autonomic dysfunction
    - Assessing arrhythmia severity
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize HRV feature extractor
        
        Args:
            sampling_rate: ECG sampling rate in Hz
        """
        self.fs = sampling_rate
        
    def extract_time_domain_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract time-domain HRV features
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            
        Returns:
            Dictionary of time-domain HRV features
        """
        if len(rr_intervals) < 2:
            return self._empty_time_domain_features()
        
        # Convert to milliseconds for standard HRV metrics
        rr_ms = rr_intervals * 1000
        
        features = {}
        
        # Mean RR interval
        features['mean_rr'] = np.mean(rr_ms)
        
        # Mean heart rate
        features['mean_hr'] = 60000 / features['mean_rr'] if features['mean_rr'] > 0 else 0
        
        # SDNN - Standard deviation of NN intervals
        features['sdnn'] = np.std(rr_ms, ddof=1)
        
        # RMSSD - Root mean square of successive differences
        diff_rr = np.diff(rr_ms)
        features['rmssd'] = np.sqrt(np.mean(diff_rr ** 2))
        
        # pNN50 - Percentage of successive differences > 50ms
        nn50 = np.sum(np.abs(diff_rr) > 50)
        features['pnn50'] = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # pNN20 - Percentage of successive differences > 20ms
        nn20 = np.sum(np.abs(diff_rr) > 20)
        features['pnn20'] = (nn20 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
        
        # SDSD - Standard deviation of successive differences
        features['sdsd'] = np.std(diff_rr, ddof=1) if len(diff_rr) > 1 else 0
        
        # CV - Coefficient of variation
        features['cv'] = (features['sdnn'] / features['mean_rr']) * 100 if features['mean_rr'] > 0 else 0
        
        # Range
        features['rr_range'] = np.max(rr_ms) - np.min(rr_ms)
        
        # Median absolute deviation
        features['mad'] = np.median(np.abs(rr_ms - np.median(rr_ms)))
        
        # Triangular index (histogram-based)
        hist, bin_edges = np.histogram(rr_ms, bins=int(features['rr_range'] / 7.8125) + 1)
        features['triangular_index'] = len(rr_ms) / np.max(hist) if np.max(hist) > 0 else 0
        
        return features
    
    def _empty_time_domain_features(self) -> Dict[str, float]:
        """Return empty time domain features"""
        return {
            'mean_rr': 0, 'mean_hr': 0, 'sdnn': 0, 'rmssd': 0,
            'pnn50': 0, 'pnn20': 0, 'sdsd': 0, 'cv': 0,
            'rr_range': 0, 'mad': 0, 'triangular_index': 0
        }
    
    def extract_frequency_domain_features(self, rr_intervals: np.ndarray,
                                           r_peak_times: np.ndarray = None) -> Dict[str, float]:
        """
        Extract frequency-domain HRV features using Welch's method
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            r_peak_times: Time points of R-peaks (for interpolation)
            
        Returns:
            Dictionary of frequency-domain HRV features
        """
        if len(rr_intervals) < 10:
            return self._empty_frequency_domain_features()
        
        # If r_peak_times not provided, create from cumulative sum
        if r_peak_times is None:
            r_peak_times = np.cumsum(rr_intervals)
            r_peak_times = np.insert(r_peak_times, 0, 0)
        
        # Resample to uniform 4 Hz
        resample_rate = 4  # Hz
        
        # Interpolate RR intervals
        try:
            f_interp = interpolate.interp1d(r_peak_times[:-1], rr_intervals,
                                            kind='cubic', fill_value='extrapolate')
            
            t_uniform = np.arange(r_peak_times[0], r_peak_times[-2], 1/resample_rate)
            rr_resampled = f_interp(t_uniform)
        except Exception:
            return self._empty_frequency_domain_features()
        
        # Remove mean (detrend)
        rr_detrended = rr_resampled - np.mean(rr_resampled)
        
        # Compute power spectral density using Welch's method
        nperseg = min(256, len(rr_detrended))
        
        try:
            freqs, psd = signal.welch(rr_detrended, fs=resample_rate, nperseg=nperseg)
        except Exception:
            return self._empty_frequency_domain_features()
        
        features = {}
        
        # Define frequency bands
        vlf_band = (0.003, 0.04)   # Very low frequency
        lf_band = (0.04, 0.15)     # Low frequency
        hf_band = (0.15, 0.4)      # High frequency
        
        # Calculate band powers
        vlf_mask = (freqs >= vlf_band[0]) & (freqs < vlf_band[1])
        lf_mask = (freqs >= lf_band[0]) & (freqs < lf_band[1])
        hf_mask = (freqs >= hf_band[0]) & (freqs < hf_band[1])
        
        # Integrate power in each band
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1
        
        vlf_power = np.trapz(psd[vlf_mask], dx=df) if np.any(vlf_mask) else 0
        lf_power = np.trapz(psd[lf_mask], dx=df) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], dx=df) if np.any(hf_mask) else 0
        
        total_power = vlf_power + lf_power + hf_power
        
        features['vlf_power'] = vlf_power * 1e6  # Convert to ms^2
        features['lf_power'] = lf_power * 1e6
        features['hf_power'] = hf_power * 1e6
        features['total_power'] = total_power * 1e6
        
        # Normalized units (excluding VLF)
        lf_hf_total = lf_power + hf_power
        features['lf_nu'] = (lf_power / lf_hf_total) * 100 if lf_hf_total > 0 else 0
        features['hf_nu'] = (hf_power / lf_hf_total) * 100 if lf_hf_total > 0 else 0
        
        # LF/HF ratio - important for autonomic balance
        features['lf_hf_ratio'] = lf_power / hf_power if hf_power > 0 else 0
        
        # Peak frequencies
        if np.any(lf_mask) and np.sum(psd[lf_mask]) > 0:
            lf_peak_idx = np.argmax(psd[lf_mask])
            features['lf_peak_freq'] = freqs[lf_mask][lf_peak_idx]
        else:
            features['lf_peak_freq'] = 0
            
        if np.any(hf_mask) and np.sum(psd[hf_mask]) > 0:
            hf_peak_idx = np.argmax(psd[hf_mask])
            features['hf_peak_freq'] = freqs[hf_mask][hf_peak_idx]
        else:
            features['hf_peak_freq'] = 0
        
        return features
    
    def _empty_frequency_domain_features(self) -> Dict[str, float]:
        """Return empty frequency domain features"""
        return {
            'vlf_power': 0, 'lf_power': 0, 'hf_power': 0, 'total_power': 0,
            'lf_nu': 0, 'hf_nu': 0, 'lf_hf_ratio': 0,
            'lf_peak_freq': 0, 'hf_peak_freq': 0
        }
    
    def extract_nonlinear_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """
        Extract nonlinear HRV features
        
        Args:
            rr_intervals: Array of RR intervals in seconds
            
        Returns:
            Dictionary of nonlinear HRV features
        """
        if len(rr_intervals) < 10:
            return self._empty_nonlinear_features()
        
        rr_ms = rr_intervals * 1000
        
        features = {}
        
        # Poincaré plot features (SD1, SD2)
        rr_n = rr_ms[:-1]  # RR(n)
        rr_n1 = rr_ms[1:]  # RR(n+1)
        
        # SD1: Short-term variability (perpendicular to identity line)
        diff_rr = rr_n1 - rr_n
        features['sd1'] = np.std(diff_rr, ddof=1) / np.sqrt(2)
        
        # SD2: Long-term variability (along identity line)
        sum_rr = rr_n1 + rr_n
        features['sd2'] = np.sqrt(2 * np.var(rr_ms, ddof=1) - features['sd1']**2)
        
        # SD1/SD2 ratio
        features['sd1_sd2_ratio'] = features['sd1'] / features['sd2'] if features['sd2'] > 0 else 0
        
        # Sample entropy (approximate)
        features['sample_entropy'] = self._sample_entropy(rr_ms, m=2, r=0.2)
        
        # Approximate entropy
        features['approx_entropy'] = self._approximate_entropy(rr_ms, m=2, r=0.2)
        
        # Detrended fluctuation analysis (simplified)
        features['dfa_alpha1'] = self._dfa_alpha(rr_ms, scale_range=(4, 16))
        features['dfa_alpha2'] = self._dfa_alpha(rr_ms, scale_range=(16, 64))
        
        return features
    
    def _empty_nonlinear_features(self) -> Dict[str, float]:
        """Return empty nonlinear features"""
        return {
            'sd1': 0, 'sd2': 0, 'sd1_sd2_ratio': 0,
            'sample_entropy': 0, 'approx_entropy': 0,
            'dfa_alpha1': 0, 'dfa_alpha2': 0
        }
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy"""
        n = len(data)
        if n < m + 2:
            return 0
        
        r_threshold = r * np.std(data)
        
        def count_matches(m_val):
            templates = np.array([data[i:i+m_val] for i in range(n - m_val)])
            count = 0
            for i in range(len(templates)):
                for j in range(i + 1, len(templates)):
                    if np.max(np.abs(templates[i] - templates[j])) < r_threshold:
                        count += 1
            return count
        
        A = count_matches(m + 1)
        B = count_matches(m)
        
        if B == 0:
            return 0
        
        return -np.log(A / B) if A > 0 else 0
    
    def _approximate_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate approximate entropy"""
        n = len(data)
        if n < m + 2:
            return 0
        
        r_threshold = r * np.std(data)
        
        def phi(m_val):
            templates = np.array([data[i:i+m_val] for i in range(n - m_val + 1)])
            C = np.zeros(len(templates))
            
            for i in range(len(templates)):
                matches = np.sum(np.max(np.abs(templates - templates[i]), axis=1) < r_threshold)
                C[i] = matches / len(templates)
            
            return np.mean(np.log(C[C > 0]))
        
        return phi(m) - phi(m + 1)
    
    def _dfa_alpha(self, data: np.ndarray, scale_range: Tuple[int, int] = (4, 16)) -> float:
        """
        Simplified Detrended Fluctuation Analysis
        Returns the scaling exponent alpha
        """
        n = len(data)
        if n < scale_range[1]:
            return 0
        
        # Integrate the series
        y = np.cumsum(data - np.mean(data))
        
        scales = np.arange(scale_range[0], min(scale_range[1], n // 4) + 1)
        fluctuations = []
        
        for scale in scales:
            n_segments = n // scale
            if n_segments < 1:
                continue
            
            F_n = 0
            for i in range(n_segments):
                segment = y[i * scale:(i + 1) * scale]
                
                # Linear detrending
                t = np.arange(len(segment))
                if len(t) > 1:
                    coeffs = np.polyfit(t, segment, 1)
                    trend = np.polyval(coeffs, t)
                    F_n += np.sum((segment - trend) ** 2)
            
            F_n = np.sqrt(F_n / (n_segments * scale))
            fluctuations.append(F_n)
        
        if len(fluctuations) < 2:
            return 0
        
        # Linear fit in log-log space
        log_scales = np.log(scales[:len(fluctuations)])
        log_fluct = np.log(np.array(fluctuations) + 1e-10)
        
        try:
            coeffs = np.polyfit(log_scales, log_fluct, 1)
            return coeffs[0]  # Alpha is the slope
        except Exception:
            return 0
    
    def extract_all_features(self, rr_intervals: np.ndarray,
                              r_peak_times: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all HRV features
        
        Args:
            rr_intervals: RR intervals in seconds
            r_peak_times: Optional R-peak time points
            
        Returns:
            Dictionary with all HRV features
        """
        features = {}
        
        # Time domain
        time_features = self.extract_time_domain_features(rr_intervals)
        features.update(time_features)
        
        # Frequency domain
        freq_features = self.extract_frequency_domain_features(rr_intervals, r_peak_times)
        features.update(freq_features)
        
        # Nonlinear
        nonlinear_features = self.extract_nonlinear_features(rr_intervals)
        features.update(nonlinear_features)
        
        return features
    
    def extract_sliding_window_features(self, rr_intervals: np.ndarray,
                                         window_size: int = 30,
                                         step_size: int = 1) -> List[Dict[str, float]]:
        """
        Extract HRV features using sliding window
        
        Args:
            rr_intervals: Full RR interval series
            window_size: Number of beats in window
            step_size: Step between windows
            
        Returns:
            List of feature dictionaries for each window
        """
        features_list = []
        
        for i in range(0, len(rr_intervals) - window_size + 1, step_size):
            window = rr_intervals[i:i + window_size]
            features = self.extract_all_features(window)
            features['window_start'] = i
            features['window_end'] = i + window_size
            features_list.append(features)
        
        return features_list


def main():
    """Test HRV feature extraction"""
    # Create synthetic RR intervals
    np.random.seed(42)
    
    # Normal sinus rhythm (~70 BPM, ~857ms RR)
    normal_rr = 0.857 + 0.05 * np.random.randn(100)
    
    # Tachycardia (~120 BPM, ~500ms RR)
    tachy_rr = 0.500 + 0.02 * np.random.randn(100)
    
    extractor = HRVFeatureExtractor()
    
    print("Normal Rhythm HRV Features:")
    normal_features = extractor.extract_all_features(normal_rr)
    for key, value in normal_features.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTachycardia HRV Features:")
    tachy_features = extractor.extract_all_features(tachy_rr)
    for key, value in tachy_features.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n=== Key Differences ===")
    print(f"Mean HR: Normal={normal_features['mean_hr']:.1f}, Tachy={tachy_features['mean_hr']:.1f}")
    print(f"RMSSD: Normal={normal_features['rmssd']:.2f}, Tachy={tachy_features['rmssd']:.2f}")
    print(f"LF/HF: Normal={normal_features['lf_hf_ratio']:.2f}, Tachy={tachy_features['lf_hf_ratio']:.2f}")


if __name__ == '__main__':
    main()
