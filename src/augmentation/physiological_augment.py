"""
Physiological ECG Data Augmentation

Implements clinically realistic augmentation techniques instead of SMOTE.
These augmentations preserve the physiological characteristics of ECG signals
while introducing natural variations seen in real-world recordings.

Reference: "Data Augmentation for Deep Learning Based ECG Classification"
"""

import numpy as np
from scipy import signal, interpolate
from typing import Tuple, Optional, List
import warnings


class PhysiologicalAugmenter:
    """
    ECG-specific data augmentation using physiologically plausible transformations
    
    Unlike SMOTE (which interpolates between samples creating unrealistic morphologies),
    these augmentations introduce variations that occur naturally in ECG recordings:
    - Heart rate variability
    - Baseline wander from respiration
    - Muscle artifact noise
    - Electrode movement
    - Amplitude variations from lead placement
    """
    
    def __init__(self, sampling_rate: int = 360, random_seed: Optional[int] = None):
        """
        Initialize augmenter
        
        Args:
            sampling_rate: ECG sampling frequency in Hz
            random_seed: Random seed for reproducibility
        """
        self.fs = sampling_rate
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment(self, 
                ecg_segment: np.ndarray,
                augmentations: Optional[List[str]] = None,
                intensity: str = 'medium') -> np.ndarray:
        """
        Apply random augmentations to ECG segment
        
        Args:
            ecg_segment: Input ECG segment
            augmentations: List of augmentation types to apply (None = random selection)
            intensity: 'light', 'medium', or 'strong'
            
        Returns:
            Augmented ECG segment
        """
        available_augmentations = [
            'time_warp',
            'amplitude_scale',
            'baseline_wander',
            'gaussian_noise',
            'powerline_noise',
            'amplitude_shift',
            'random_crop_pad',
            'time_shift',
            'spike_artifact'
        ]
        
        if augmentations is None:
            # Randomly select 2-4 augmentations
            n_augs = np.random.randint(2, 5)
            augmentations = np.random.choice(available_augmentations, n_augs, replace=False)
        
        # Set intensity parameters
        intensity_params = self._get_intensity_params(intensity)
        
        augmented = ecg_segment.copy()
        
        for aug in augmentations:
            if aug == 'time_warp':
                augmented = self.time_warp(augmented, **intensity_params['time_warp'])
            elif aug == 'amplitude_scale':
                augmented = self.amplitude_scale(augmented, **intensity_params['amplitude_scale'])
            elif aug == 'baseline_wander':
                augmented = self.add_baseline_wander(augmented, **intensity_params['baseline_wander'])
            elif aug == 'gaussian_noise':
                augmented = self.add_gaussian_noise(augmented, **intensity_params['gaussian_noise'])
            elif aug == 'powerline_noise':
                augmented = self.add_powerline_noise(augmented, **intensity_params['powerline_noise'])
            elif aug == 'amplitude_shift':
                augmented = self.amplitude_shift(augmented, **intensity_params['amplitude_shift'])
            elif aug == 'random_crop_pad':
                augmented = self.random_crop_pad(augmented, **intensity_params['random_crop_pad'])
            elif aug == 'time_shift':
                augmented = self.time_shift(augmented, **intensity_params['time_shift'])
            elif aug == 'spike_artifact':
                augmented = self.add_spike_artifact(augmented, **intensity_params['spike_artifact'])
        
        return augmented
    
    def _get_intensity_params(self, intensity: str) -> dict:
        """Get augmentation parameters based on intensity level"""
        if intensity == 'light':
            return {
                'time_warp': {'warp_factor_range': (0.95, 1.05)},
                'amplitude_scale': {'scale_range': (0.9, 1.1)},
                'baseline_wander': {'amplitude_range': (0.01, 0.05), 'freq_range': (0.1, 0.3)},
                'gaussian_noise': {'snr_range': (30, 40)},
                'powerline_noise': {'amplitude_range': (0.01, 0.03)},
                'amplitude_shift': {'shift_range': (-0.05, 0.05)},
                'random_crop_pad': {'max_crop_ratio': 0.05},
                'time_shift': {'max_shift_ratio': 0.05},
                'spike_artifact': {'n_spikes': 1, 'amplitude_range': (0.1, 0.2)}
            }
        elif intensity == 'strong':
            return {
                'time_warp': {'warp_factor_range': (0.85, 1.15)},
                'amplitude_scale': {'scale_range': (0.7, 1.3)},
                'baseline_wander': {'amplitude_range': (0.1, 0.3), 'freq_range': (0.05, 0.5)},
                'gaussian_noise': {'snr_range': (15, 25)},
                'powerline_noise': {'amplitude_range': (0.05, 0.15)},
                'amplitude_shift': {'shift_range': (-0.2, 0.2)},
                'random_crop_pad': {'max_crop_ratio': 0.15},
                'time_shift': {'max_shift_ratio': 0.15},
                'spike_artifact': {'n_spikes': 3, 'amplitude_range': (0.3, 0.5)}
            }
        else:  # medium
            return {
                'time_warp': {'warp_factor_range': (0.9, 1.1)},
                'amplitude_scale': {'scale_range': (0.8, 1.2)},
                'baseline_wander': {'amplitude_range': (0.03, 0.1), 'freq_range': (0.1, 0.4)},
                'gaussian_noise': {'snr_range': (20, 35)},
                'powerline_noise': {'amplitude_range': (0.02, 0.08)},
                'amplitude_shift': {'shift_range': (-0.1, 0.1)},
                'random_crop_pad': {'max_crop_ratio': 0.1},
                'time_shift': {'max_shift_ratio': 0.1},
                'spike_artifact': {'n_spikes': 2, 'amplitude_range': (0.2, 0.4)}
            }
    
    def time_warp(self, 
                  ecg: np.ndarray,
                  warp_factor_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Time-domain warping to simulate heart rate variability
        
        This stretches or compresses the signal to simulate natural HR variations.
        Clinically relevant: HR naturally varies with respiration (RSA) and activity.
        
        Args:
            ecg: Input ECG segment
            warp_factor_range: Range of time warping factors (1.0 = no change)
            
        Returns:
            Time-warped ECG (resampled to original length)
        """
        warp_factor = np.random.uniform(*warp_factor_range)
        
        original_length = len(ecg)
        new_length = int(original_length * warp_factor)
        
        if new_length == original_length:
            return ecg
        
        # Create interpolation function
        x_original = np.linspace(0, 1, original_length)
        x_new = np.linspace(0, 1, new_length)
        
        # Interpolate
        f = interpolate.interp1d(x_original, ecg, kind='cubic')
        warped = f(x_new)
        
        # Resample back to original length
        f_resample = interpolate.interp1d(np.linspace(0, 1, len(warped)), warped, kind='cubic')
        result = f_resample(np.linspace(0, 1, original_length))
        
        return result
    
    def amplitude_scale(self,
                        ecg: np.ndarray,
                        scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Scale amplitude to simulate lead placement variations
        
        Clinically relevant: ECG amplitude varies with electrode placement,
        patient body habitus, and skin impedance.
        
        Args:
            ecg: Input ECG segment
            scale_range: Range of scaling factors
            
        Returns:
            Amplitude-scaled ECG
        """
        scale = np.random.uniform(*scale_range)
        return ecg * scale
    
    def add_baseline_wander(self,
                            ecg: np.ndarray,
                            amplitude_range: Tuple[float, float] = (0.05, 0.15),
                            freq_range: Tuple[float, float] = (0.1, 0.5)) -> np.ndarray:
        """
        Add baseline wander artifact
        
        Clinically relevant: Caused by patient respiration (0.1-0.5 Hz)
        and patient movement. Very common in ambulatory recordings.
        
        Args:
            ecg: Input ECG segment
            amplitude_range: Amplitude of baseline wander (relative to signal std)
            freq_range: Frequency range of wander (Hz)
            
        Returns:
            ECG with baseline wander
        """
        n_samples = len(ecg)
        t = np.arange(n_samples) / self.fs
        
        # Random frequency and phase
        freq = np.random.uniform(*freq_range)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(*amplitude_range) * np.std(ecg)
        
        # Create sinusoidal baseline wander
        wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
        
        # Optionally add a second harmonic for more realistic wander
        if np.random.random() > 0.5:
            freq2 = freq * np.random.uniform(1.5, 2.5)
            phase2 = np.random.uniform(0, 2 * np.pi)
            amp2 = amplitude * np.random.uniform(0.3, 0.6)
            wander += amp2 * np.sin(2 * np.pi * freq2 * t + phase2)
        
        return ecg + wander
    
    def add_gaussian_noise(self,
                           ecg: np.ndarray,
                           snr_range: Tuple[float, float] = (20, 35)) -> np.ndarray:
        """
        Add Gaussian white noise
        
        Clinically relevant: Represents thermal noise in amplifiers,
        and some EMG artifact from muscle tremor.
        
        Args:
            ecg: Input ECG segment
            snr_range: Signal-to-noise ratio range in dB
            
        Returns:
            ECG with added noise
        """
        snr_db = np.random.uniform(*snr_range)
        
        # Calculate signal power
        signal_power = np.mean(ecg ** 2)
        
        # Calculate noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate noise
        noise = np.sqrt(noise_power) * np.random.randn(len(ecg))
        
        return ecg + noise
    
    def add_powerline_noise(self,
                            ecg: np.ndarray,
                            amplitude_range: Tuple[float, float] = (0.02, 0.1),
                            frequency: float = 60.0) -> np.ndarray:
        """
        Add powerline interference (50 or 60 Hz)
        
        Clinically relevant: Very common artifact from electrical mains.
        Typically 50 Hz (Europe) or 60 Hz (Americas).
        
        Args:
            ecg: Input ECG segment
            amplitude_range: Amplitude relative to signal std
            frequency: Powerline frequency (50 or 60 Hz)
            
        Returns:
            ECG with powerline noise
        """
        n_samples = len(ecg)
        t = np.arange(n_samples) / self.fs
        
        # Add some frequency variation (real powerline isn't exactly 50/60 Hz)
        actual_freq = frequency + np.random.uniform(-0.1, 0.1)
        
        amplitude = np.random.uniform(*amplitude_range) * np.std(ecg)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Powerline noise
        noise = amplitude * np.sin(2 * np.pi * actual_freq * t + phase)
        
        # Sometimes add harmonics (especially 2nd and 3rd)
        if np.random.random() > 0.7:
            noise += (amplitude * 0.3) * np.sin(2 * np.pi * 2 * actual_freq * t + phase)
        
        return ecg + noise
    
    def amplitude_shift(self,
                        ecg: np.ndarray,
                        shift_range: Tuple[float, float] = (-0.1, 0.1)) -> np.ndarray:
        """
        Shift signal amplitude (DC offset)
        
        Clinically relevant: Different baseline levels due to 
        electrode-skin interface potentials.
        
        Args:
            ecg: Input ECG segment
            shift_range: Range of shift relative to signal std
            
        Returns:
            Amplitude-shifted ECG
        """
        shift = np.random.uniform(*shift_range) * np.std(ecg)
        return ecg + shift
    
    def random_crop_pad(self,
                        ecg: np.ndarray,
                        max_crop_ratio: float = 0.1) -> np.ndarray:
        """
        Randomly crop and pad to simulate slight timing variations
        
        Clinically relevant: Exact timing of recording start/end varies.
        
        Args:
            ecg: Input ECG segment
            max_crop_ratio: Maximum ratio of signal to crop
            
        Returns:
            Cropped and padded ECG (same length as input)
        """
        original_length = len(ecg)
        max_crop = int(original_length * max_crop_ratio)
        
        if max_crop < 1:
            return ecg
        
        crop_amount = np.random.randint(0, max_crop + 1)
        
        if crop_amount == 0:
            return ecg
        
        # Randomly crop from start or end
        if np.random.random() > 0.5:
            # Crop from start, pad at end
            cropped = ecg[crop_amount:]
            padded = np.concatenate([cropped, ecg[-crop_amount:]])
        else:
            # Crop from end, pad at start
            cropped = ecg[:-crop_amount]
            padded = np.concatenate([ecg[:crop_amount], cropped])
        
        return padded
    
    def time_shift(self,
                   ecg: np.ndarray,
                   max_shift_ratio: float = 0.1) -> np.ndarray:
        """
        Circular time shift of the signal
        
        Args:
            ecg: Input ECG segment
            max_shift_ratio: Maximum shift as ratio of signal length
            
        Returns:
            Time-shifted ECG
        """
        max_shift = int(len(ecg) * max_shift_ratio)
        
        if max_shift < 1:
            return ecg
        
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(ecg, shift)
    
    def add_spike_artifact(self,
                           ecg: np.ndarray,
                           n_spikes: int = 2,
                           amplitude_range: Tuple[float, float] = (0.2, 0.5)) -> np.ndarray:
        """
        Add random spike artifacts (electrode movement)
        
        Clinically relevant: Momentary electrode contact issues
        cause brief spikes in the recording.
        
        Args:
            ecg: Input ECG segment
            n_spikes: Maximum number of spikes to add
            amplitude_range: Spike amplitude relative to signal std
            
        Returns:
            ECG with spike artifacts
        """
        result = ecg.copy()
        
        actual_n_spikes = np.random.randint(0, n_spikes + 1)
        
        for _ in range(actual_n_spikes):
            # Random position (avoid very edges)
            pos = np.random.randint(5, len(ecg) - 5)
            
            # Random amplitude and sign
            amplitude = np.random.uniform(*amplitude_range) * np.std(ecg)
            sign = np.random.choice([-1, 1])
            
            # Create brief spike (1-3 samples)
            spike_width = np.random.randint(1, 4)
            for i in range(-spike_width//2, spike_width//2 + 1):
                if 0 <= pos + i < len(result):
                    result[pos + i] += sign * amplitude * np.exp(-abs(i))
        
        return result


class TachycardiaAugmentationPipeline:
    """
    Specialized augmentation pipeline for tachycardia data
    
    Applies stronger augmentation to minority class (tachycardia)
    while maintaining physiological realism.
    """
    
    def __init__(self, sampling_rate: int = 360):
        """
        Initialize pipeline
        
        Args:
            sampling_rate: ECG sampling frequency
        """
        self.fs = sampling_rate
        self.augmenter = PhysiologicalAugmenter(sampling_rate)
    
    def augment_dataset(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        target_ratio: float = 1.0,
                        max_augments_per_sample: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset to balance classes
        
        Args:
            X: Feature matrix or ECG segments [n_samples, n_features]
            y: Binary labels
            target_ratio: Target ratio of minority to majority (1.0 = balanced)
            max_augments_per_sample: Max augmented copies per original sample
            
        Returns:
            Tuple of (augmented X, augmented y)
        """
        # Count classes
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if n_positive == 0:
            warnings.warn("No positive samples to augment")
            return X, y
        
        # Calculate how many augmented samples needed
        target_positive = int(n_negative * target_ratio)
        n_augments_needed = max(0, target_positive - n_positive)
        
        if n_augments_needed == 0:
            print("Classes already balanced or positive class is majority")
            return X, y
        
        # Get positive samples
        positive_indices = np.where(y == 1)[0]
        positive_samples = X[positive_indices]
        
        # Generate augmented samples
        augmented_samples = []
        augments_per_sample = min(
            max_augments_per_sample,
            int(np.ceil(n_augments_needed / n_positive))
        )
        
        print(f"Augmenting {n_positive} positive samples with {augments_per_sample} variations each...")
        
        for sample in positive_samples:
            for i in range(augments_per_sample):
                if len(augmented_samples) >= n_augments_needed:
                    break
                
                # Vary intensity
                intensities = ['light', 'medium', 'strong']
                intensity = intensities[i % len(intensities)]
                
                aug_sample = self.augmenter.augment(sample, intensity=intensity)
                augmented_samples.append(aug_sample)
            
            if len(augmented_samples) >= n_augments_needed:
                break
        
        # Combine original and augmented
        augmented_X = np.vstack([X, np.array(augmented_samples)])
        augmented_y = np.concatenate([y, np.ones(len(augmented_samples))])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(augmented_y))
        augmented_X = augmented_X[shuffle_idx]
        augmented_y = augmented_y[shuffle_idx]
        
        print(f"Final class distribution: {np.sum(augmented_y == 0)} normal, {np.sum(augmented_y == 1)} tachycardia")
        
        return augmented_X, augmented_y


def main():
    """Test augmentation functions"""
    import matplotlib.pyplot as plt
    
    # Create synthetic ECG beat
    np.random.seed(42)
    t = np.linspace(0, 0.6, 216)  # 600ms beat at 360 Hz
    
    # Synthetic beat
    beat = np.zeros_like(t)
    beat += 0.1 * np.exp(-((t - 0.1) / 0.02) ** 2)  # P-wave
    beat += -0.2 * np.exp(-((t - 0.18) / 0.01) ** 2)  # Q
    beat += 1.0 * np.exp(-((t - 0.2) / 0.015) ** 2)   # R
    beat += -0.3 * np.exp(-((t - 0.22) / 0.01) ** 2)  # S
    beat += 0.3 * np.exp(-((t - 0.35) / 0.04) ** 2)   # T-wave
    
    augmenter = PhysiologicalAugmenter(sampling_rate=360)
    
    # Apply different augmentations
    print("Testing augmentations:")
    
    aug_types = ['time_warp', 'amplitude_scale', 'baseline_wander', 
                 'gaussian_noise', 'powerline_noise']
    
    for aug_type in aug_types:
        augmented = augmenter.augment(beat, augmentations=[aug_type], intensity='medium')
        diff = np.mean(np.abs(augmented - beat))
        print(f"  {aug_type}: Mean change = {diff:.4f}")
    
    # Test combined augmentation
    combined = augmenter.augment(beat, intensity='strong')
    print(f"\nCombined (strong): Mean change = {np.mean(np.abs(combined - beat)):.4f}")


if __name__ == '__main__':
    main()
