"""
ECG Signal Processing Module
Implements filtering, normalization, and quality enhancement for ECG signals
"""

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from typing import Tuple, Optional
import warnings


class SignalProcessor:
    """
    ECG Signal Processor for noise removal and quality enhancement
    
    Implements:
    - Baseline wander removal (high-pass filtering)
    - Powerline interference removal (notch filter)
    - High-frequency noise removal (low-pass filtering)
    - Signal normalization
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize the signal processor
        
        Args:
            sampling_rate: Sampling frequency in Hz (default 360 for MIT-BIH)
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2
        
    def remove_baseline_wander(self, ecg_signal: np.ndarray, 
                                cutoff: float = 0.5) -> np.ndarray:
        """
        Remove baseline wander using high-pass filter
        
        Baseline wander is typically < 0.5 Hz caused by respiration
        and patient movement.
        
        Args:
            ecg_signal: Raw ECG signal
            cutoff: High-pass cutoff frequency in Hz
            
        Returns:
            Filtered signal with baseline removed
        """
        # Design Butterworth high-pass filter
        order = 2
        normalized_cutoff = cutoff / self.nyquist
        
        # Ensure cutoff is valid
        if normalized_cutoff >= 1:
            warnings.warn(f"Cutoff frequency {cutoff} Hz too high for Nyquist {self.nyquist} Hz")
            return ecg_signal
            
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        
        # Apply zero-phase filtering (forward-backward)
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        return filtered
    
    def remove_powerline_interference(self, ecg_signal: np.ndarray,
                                       powerline_freq: float = 60.0,
                                       quality_factor: float = 30.0) -> np.ndarray:
        """
        Remove powerline interference using notch filter
        
        Args:
            ecg_signal: ECG signal
            powerline_freq: Powerline frequency (60 Hz in US, 50 Hz in EU)
            quality_factor: Quality factor for notch filter
            
        Returns:
            Signal with powerline interference removed
        """
        # Design notch filter
        normalized_freq = powerline_freq / self.nyquist
        
        if normalized_freq >= 1:
            warnings.warn(f"Powerline frequency {powerline_freq} Hz >= Nyquist")
            return ecg_signal
            
        b, a = signal.iirnotch(normalized_freq, quality_factor)
        
        # Apply filter
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        return filtered
    
    def remove_high_frequency_noise(self, ecg_signal: np.ndarray,
                                     cutoff: float = 40.0) -> np.ndarray:
        """
        Remove high-frequency noise using low-pass filter
        
        ECG signal content is typically below 40 Hz for diagnostic purposes.
        
        Args:
            ecg_signal: ECG signal
            cutoff: Low-pass cutoff frequency in Hz
            
        Returns:
            Low-pass filtered signal
        """
        order = 4
        normalized_cutoff = cutoff / self.nyquist
        
        if normalized_cutoff >= 1:
            warnings.warn(f"Cutoff frequency {cutoff} Hz too high")
            return ecg_signal
            
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        return filtered
    
    def bandpass_filter(self, ecg_signal: np.ndarray,
                        low_cutoff: float = 0.5,
                        high_cutoff: float = 40.0) -> np.ndarray:
        """
        Apply bandpass filter to ECG signal
        
        Combines baseline wander removal and high-frequency noise removal.
        
        Args:
            ecg_signal: Raw ECG signal
            low_cutoff: High-pass cutoff (removes baseline wander)
            high_cutoff: Low-pass cutoff (removes high-freq noise)
            
        Returns:
            Bandpass filtered signal
        """
        order = 4
        low_norm = low_cutoff / self.nyquist
        high_norm = high_cutoff / self.nyquist
        
        # Ensure valid frequency range
        if low_norm >= 1 or high_norm >= 1 or low_norm >= high_norm:
            warnings.warn("Invalid cutoff frequencies")
            return ecg_signal
            
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        return filtered
    
    def remove_spike_artifacts(self, ecg_signal: np.ndarray,
                                kernel_size: int = 3) -> np.ndarray:
        """
        Remove spike artifacts using median filter
        
        Args:
            ecg_signal: ECG signal
            kernel_size: Median filter kernel size (odd number)
            
        Returns:
            Signal with spikes removed
        """
        return median_filter(ecg_signal, size=kernel_size)
    
    def normalize_signal(self, ecg_signal: np.ndarray,
                         method: str = 'zscore') -> np.ndarray:
        """
        Normalize ECG signal
        
        Args:
            ecg_signal: ECG signal
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized signal
        """
        if method == 'zscore':
            # Z-score normalization
            mean = np.mean(ecg_signal)
            std = np.std(ecg_signal)
            if std == 0:
                return ecg_signal - mean
            return (ecg_signal - mean) / std
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_val = np.min(ecg_signal)
            max_val = np.max(ecg_signal)
            if max_val == min_val:
                return np.zeros_like(ecg_signal)
            return (ecg_signal - min_val) / (max_val - min_val)
            
        elif method == 'robust':
            # Robust scaling using median and IQR
            median = np.median(ecg_signal)
            q75, q25 = np.percentile(ecg_signal, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return ecg_signal - median
            return (ecg_signal - median) / iqr
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def full_preprocessing(self, ecg_signal: np.ndarray,
                           remove_baseline: bool = True,
                           remove_powerline: bool = True,
                           remove_hf_noise: bool = True,
                           normalize: bool = True,
                           norm_method: str = 'zscore') -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            ecg_signal: Raw ECG signal
            remove_baseline: Whether to remove baseline wander
            remove_powerline: Whether to remove powerline interference
            remove_hf_noise: Whether to remove high-frequency noise
            normalize: Whether to normalize the signal
            norm_method: Normalization method
            
        Returns:
            Fully preprocessed signal
        """
        processed = ecg_signal.copy()
        
        if remove_baseline:
            processed = self.remove_baseline_wander(processed)
            
        if remove_powerline:
            processed = self.remove_powerline_interference(processed)
            
        if remove_hf_noise:
            processed = self.remove_high_frequency_noise(processed)
            
        if normalize:
            processed = self.normalize_signal(processed, method=norm_method)
            
        return processed
    
    def compute_signal_quality(self, ecg_signal: np.ndarray) -> dict:
        """
        Compute signal quality metrics
        
        Args:
            ecg_signal: ECG signal (preferably preprocessed)
            
        Returns:
            Dictionary with quality metrics
        """
        # Signal-to-noise ratio estimation
        # Using median absolute deviation as noise estimate
        median = np.median(ecg_signal)
        mad = np.median(np.abs(ecg_signal - median))
        noise_estimate = 1.4826 * mad  # Scale factor for Gaussian
        
        signal_power = np.var(ecg_signal)
        noise_power = noise_estimate ** 2
        
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
            
        # Flatline detection
        diff = np.diff(ecg_signal)
        flatline_ratio = np.sum(np.abs(diff) < 1e-6) / len(diff)
        
        # Clipping detection
        max_val = np.max(np.abs(ecg_signal))
        clipping_threshold = 0.99 * max_val
        clipping_ratio = np.sum(np.abs(ecg_signal) > clipping_threshold) / len(ecg_signal)
        
        return {
            'snr_db': snr_db,
            'flatline_ratio': flatline_ratio,
            'clipping_ratio': clipping_ratio,
            'signal_range': np.ptp(ecg_signal),
            'mean': np.mean(ecg_signal),
            'std': np.std(ecg_signal),
            'is_good_quality': snr_db > 10 and flatline_ratio < 0.1 and clipping_ratio < 0.01
        }


class QRSDetector:
    """
    QRS Complex Detector using Pan-Tompkins algorithm
    
    Detects R-peaks in ECG signal for beat segmentation.
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize QRS detector
        
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.fs = sampling_rate
        
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using simplified Pan-Tompkins algorithm
        
        Args:
            ecg_signal: Preprocessed ECG signal
            
        Returns:
            Array of R-peak sample indices
        """
        # Step 1: Bandpass filter (5-15 Hz for QRS detection)
        nyquist = self.fs / 2
        low = 5 / nyquist
        high = 15 / nyquist
        b, a = signal.butter(1, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, ecg_signal)
        
        # Step 2: Derivative (emphasizes QRS slope)
        derivative = np.diff(filtered)
        
        # Step 3: Squaring (makes all values positive, emphasizes large values)
        squared = derivative ** 2
        
        # Step 4: Moving window integration
        window_size = int(0.150 * self.fs)  # 150 ms window
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        
        # Step 5: Find peaks
        # Adaptive threshold
        threshold = 0.3 * np.max(integrated)
        
        # Minimum distance between peaks (200 ms = 72 samples at 360 Hz)
        min_distance = int(0.2 * self.fs)
        
        # Find peaks above threshold
        peaks, _ = signal.find_peaks(integrated, height=threshold, distance=min_distance)
        
        # Step 6: Refine R-peak locations by finding local maximum in original signal
        refined_peaks = []
        search_window = int(0.05 * self.fs)  # 50 ms window
        
        for peak in peaks:
            start = max(0, peak - search_window)
            end = min(len(ecg_signal), peak + search_window)
            
            # Find maximum in window
            local_max_idx = start + np.argmax(ecg_signal[start:end])
            refined_peaks.append(local_max_idx)
            
        return np.array(refined_peaks)
    
    def compute_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Compute RR intervals from R-peaks
        
        Args:
            r_peaks: Array of R-peak sample indices
            
        Returns:
            Array of RR intervals in seconds
        """
        rr_samples = np.diff(r_peaks)
        rr_seconds = rr_samples / self.fs
        return rr_seconds
    
    def compute_heart_rate(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous heart rate from R-peaks
        
        Args:
            r_peaks: Array of R-peak sample indices
            
        Returns:
            Array of heart rates in BPM
        """
        rr_intervals = self.compute_rr_intervals(r_peaks)
        # Avoid division by zero
        rr_intervals = np.maximum(rr_intervals, 0.001)
        heart_rate = 60 / rr_intervals
        return heart_rate
    
    def detect_tachycardia_by_hr(self, r_peaks: np.ndarray,
                                  threshold_bpm: float = 100.0) -> np.ndarray:
        """
        Detect tachycardia based on heart rate threshold
        
        Args:
            r_peaks: Array of R-peak sample indices
            threshold_bpm: HR threshold for tachycardia (default 100 BPM)
            
        Returns:
            Boolean array indicating tachycardia for each beat
        """
        heart_rates = self.compute_heart_rate(r_peaks)
        is_tachycardia = heart_rates > threshold_bpm
        
        # Pad to match r_peaks length (first beat has no preceding RR)
        is_tachycardia = np.concatenate([[False], is_tachycardia])
        
        return is_tachycardia


def main():
    """Test signal processing functions"""
    # Create a synthetic ECG-like signal for testing
    fs = 360
    t = np.linspace(0, 10, fs * 10)
    
    # Simulate ECG with noise
    ecg = np.sin(2 * np.pi * 1 * t)  # 1 Hz heartbeat
    ecg += 0.5 * np.sin(2 * np.pi * 0.1 * t)  # Baseline wander
    ecg += 0.2 * np.sin(2 * np.pi * 60 * t)  # Powerline noise
    ecg += 0.1 * np.random.randn(len(t))  # Random noise
    
    # Process
    processor = SignalProcessor(fs)
    
    # Test individual filters
    no_baseline = processor.remove_baseline_wander(ecg)
    no_powerline = processor.remove_powerline_interference(ecg)
    no_hf = processor.remove_high_frequency_noise(ecg)
    
    # Full preprocessing
    clean = processor.full_preprocessing(ecg)
    
    # Quality metrics
    quality = processor.compute_signal_quality(clean)
    print("Signal Quality Metrics:")
    for k, v in quality.items():
        print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
