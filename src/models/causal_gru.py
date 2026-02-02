"""
Causal GRU Tachycardia Detector.

Primary model for streaming deployment. Uses unidirectional GRU
(NO future context) to ensure real-time feasibility.

Key Features:
- CNN feature extractor for morphological patterns
- Causal (unidirectional) GRU for temporal context
- Per-timestep classification (dense output)
- MC Dropout for uncertainty estimation

Output: Dense per-timestep probabilities, NOT window labels.
This enables fine-grained episode detection and accurate latency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Union
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for CausalTachycardiaDetector."""
    # Input configuration
    input_channels: int = 1
    
    # CNN configuration
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    
    # GRU configuration
    gru_hidden: int = 128
    gru_layers: int = 2
    
    # Classification configuration
    num_classes: int = 5  # NORMAL, SINUS_TACHY, SVT, VT, VFL
    
    # Regularization
    dropout: float = 0.3
    
    # Uncertainty
    mc_dropout_samples: int = 10
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [7, 5, 3]


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor for ECG signals.
    
    Uses temporal convolutions with progressive downsampling
    to extract morphological features.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        channel_sizes: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [7, 5, 3],
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.downsample_factor = 2 ** len(channel_sizes)  # Total downsampling
        
        layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(channel_sizes, kernel_sizes)):
            padding = kernel_size // 2
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),  # Downsample 2x
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.output_channels = channel_sizes[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)
            
        Returns:
            features: (batch, output_channels, seq_len // downsample_factor)
        """
        return self.conv_layers(x)


class CausalGRU(nn.Module):
    """
    Causal (unidirectional) GRU for temporal modeling.
    
    CRITICAL: Uses unidirectional GRU only - no future context.
    This is essential for streaming/real-time deployment.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # CRITICAL: Causal only
            dropout=dropout if num_layers > 1 else 0,
        )
        self.hidden_size = hidden_size
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, features)
            hidden: Optional initial hidden state
            
        Returns:
            output: (batch, seq_len, hidden_size)
            hidden: Final hidden state
        """
        return self.gru(x, hidden)


class PerTimestepClassifier(nn.Module):
    """
    Per-timestep classification head.
    
    Produces dense predictions (one per timestep) rather than
    a single window-level prediction. This enables:
    - Fine-grained episode boundary detection
    - Accurate latency measurement
    - Smooth probability curves for decision logic
    """
    
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, input_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_size // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            
        Returns:
            logits: (batch, seq_len, num_classes)
        """
        return self.classifier(x)


class CausalTachycardiaDetector(nn.Module):
    """
    Primary model: Causal (unidirectional) CNN-GRU for streaming deployment.
    
    Output: Dense per-timestep probability (not window label).
    
    Architecture:
        1. CNN feature extractor (temporal convolutions)
        2. Causal GRU (unidirectional - NO future context)
        3. Per-timestep classifier
    
    This model is designed for real-time streaming inference:
        - No future context required
        - Can process incrementally
        - Supports hidden state caching for efficiency
    
    Example:
        model = CausalTachycardiaDetector()
        logits = model(x)  # x: (batch, 1, seq_len)
        probs = F.softmax(logits, dim=-1)
    """
    
    # Class indices
    CLASS_NORMAL = 0
    CLASS_SINUS_TACHY = 1
    CLASS_SVT = 2
    CLASS_VT = 3
    CLASS_VFL = 4
    
    CLASS_NAMES = ['NORMAL', 'SINUS_TACHY', 'SVT', 'VT', 'VFL']
    
    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: List[int] = None,
        gru_hidden: int = 128,
        gru_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_channels=input_channels,
            channel_sizes=cnn_channels,
            dropout=dropout * 0.5,  # Less dropout in CNN
        )
        self.downsample_factor = self.cnn.downsample_factor
        
        # Causal GRU
        self.gru = CausalGRU(
            input_size=cnn_channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=dropout,
        )
        
        # Per-timestep classifier
        self.classifier = PerTimestepClassifier(
            input_size=gru_hidden,
            num_classes=num_classes,
            dropout=dropout,
        )
        
        # For MC Dropout during inference
        self.mc_dropout = nn.Dropout(dropout)
        
        # Hidden state cache for streaming
        self._cached_hidden = None
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: (batch, 1, seq_len) raw ECG signal
            hidden: Optional GRU hidden state for streaming
            return_features: If True, also return intermediate features
            
        Returns:
            logits: (batch, seq_len // downsample, num_classes) per-timestep logits
            features: (batch, seq_len // downsample, gru_hidden) if return_features
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, channels, seq_len // 8)
        
        # Reshape for GRU: (batch, seq_len, channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # Causal GRU
        gru_out, final_hidden = self.gru(cnn_out, hidden)  # (batch, seq, hidden)
        
        # Per-timestep classification
        logits = self.classifier(gru_out)  # (batch, seq, num_classes)
        
        if return_features:
            return logits, gru_out
        return logits
    
    def predict(
        self, 
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibrated probabilities.
        
        Args:
            x: Input signal
            temperature: Temperature for scaling (1.0 = no scaling)
            
        Returns:
            probs: (batch, seq, num_classes) probabilities
            predictions: (batch, seq) argmax predictions
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            logits_scaled = logits / temperature
            probs = F.softmax(logits_scaled, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        return probs, predictions
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation.
        
        Runs multiple forward passes with dropout enabled to
        estimate prediction uncertainty.
        
        Args:
            x: Input signal
            n_samples: Number of MC samples
            
        Returns:
            mean_probs: (batch, seq, num_classes) mean probabilities
            uncertainty: (batch, seq) entropy-based uncertainty (0-1 normalized)
        """
        self.train()  # Enable dropout
        
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = self.forward(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)
        
        # Stack samples: (n_samples, batch, seq, classes)
        samples = torch.stack(samples, dim=0)
        
        # Mean probabilities
        mean_probs = samples.mean(dim=0)
        
        # Entropy-based uncertainty
        # Normalized so 0 = certain, 1 = maximally uncertain
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        max_entropy = np.log(self.num_classes)
        uncertainty = entropy / max_entropy
        
        self.eval()
        return mean_probs, uncertainty
    
    def predict_streaming(
        self,
        x_chunk: torch.Tensor,
        reset_hidden: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming prediction with hidden state caching.
        
        Maintains GRU hidden state between calls for efficient
        incremental processing.
        
        Args:
            x_chunk: (batch, 1, chunk_len) ECG chunk
            reset_hidden: If True, reset hidden state
            
        Returns:
            probs: Probabilities for this chunk
            predictions: Argmax predictions
        """
        if reset_hidden:
            self._cached_hidden = None
        
        self.eval()
        with torch.no_grad():
            # CNN features
            cnn_out = self.cnn(x_chunk)
            cnn_out = cnn_out.permute(0, 2, 1)
            
            # GRU with cached hidden
            gru_out, self._cached_hidden = self.gru(cnn_out, self._cached_hidden)
            
            # Classify
            logits = self.classifier(gru_out)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        return probs, predictions
    
    def get_output_length(self, input_length: int) -> int:
        """Calculate output sequence length for a given input length."""
        return input_length // self.downsample_factor
    
    def get_receptive_field(self) -> int:
        """
        Estimate receptive field in input samples.
        
        This indicates how much input context affects each output timestep.
        """
        # CNN receptive field
        # Each layer: kernel_size + (kernel_size - 1) * (dilation - 1)
        # With pooling, effective RF doubles each layer
        cnn_rf = 7 + 5 + 3  # Sum of kernel sizes
        cnn_rf *= self.downsample_factor  # Pooling effect
        
        # GRU has theoretically infinite receptive field
        # but practically limited by gradient flow
        # Use downsample_factor as multiplier
        return cnn_rf * 2
    
    @classmethod
    def from_config(cls, config: ModelConfig) -> 'CausalTachycardiaDetector':
        """Create model from config dataclass."""
        return cls(
            input_channels=config.input_channels,
            cnn_channels=config.cnn_channels,
            gru_hidden=config.gru_hidden,
            gru_layers=config.gru_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by module."""
        counts = {
            'cnn': sum(p.numel() for p in self.cnn.parameters()),
            'gru': sum(p.numel() for p in self.gru.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }
        counts['total'] = sum(counts.values())
        counts['trainable'] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts


class BiLSTMBaseline(nn.Module):
    """
    Bi-LSTM baseline for offline research comparison.
    
    NOT deploy-realistic due to future context requirement.
    Use only for comparing performance upper bound.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        cnn_channels: List[int] = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        self.num_classes = num_classes
        
        # Same CNN
        self.cnn = CNNFeatureExtractor(
            input_channels=input_channels,
            channel_sizes=cnn_channels,
            dropout=dropout * 0.5,
        )
        
        # Bidirectional LSTM (uses future context!)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,  # Uses future context
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # Classifier takes 2x hidden (bidirectional)
        self.classifier = PerTimestepClassifier(
            input_size=lstm_hidden * 2,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, seq_len)
            
        Returns:
            logits: (batch, seq_len // 8, num_classes)
        """
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        logits = self.classifier(lstm_out)
        return logits


def create_model(
    model_type: str = 'causal_gru',
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'causal_gru' (default) or 'bilstm'
        **kwargs: Model configuration
        
    Returns:
        Model instance
    """
    if model_type == 'causal_gru':
        return CausalTachycardiaDetector(**kwargs)
    elif model_type == 'bilstm':
        return BiLSTMBaseline(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# v2.3: LATENCY SPECIFICATION
# =============================================================================

@dataclass
class LatencySpec:
    """Explicit latency contract for deployment."""
    
    # Detection latency: time from onset to detection
    target_detection_latency_sec: float = 3.0
    max_detection_latency_sec: float = 5.0
    
    # Processing latency: time to process one segment
    max_processing_latency_ms: float = 100.0
    
    # Alarm latency: time from detection to alarm
    target_alarm_latency_sec: float = 5.0
    
    def validate_detection_latency(self, actual_latency_sec: float) -> bool:
        """Check if detection latency meets requirements."""
        return actual_latency_sec <= self.max_detection_latency_sec
    
    def validate_processing_latency(self, actual_latency_ms: float) -> bool:
        """Check if processing latency meets requirements."""
        return actual_latency_ms <= self.max_processing_latency_ms


# =============================================================================
# v2.3: SELECTIVE UNCERTAINTY ESTIMATION
# =============================================================================

class SelectiveUncertaintyEstimator:
    """
    v2.3: Selective MC Dropout + Boundary Uncertainty.
    
    Problems with always-on MC Dropout:
    1. Expensive: 10x forward passes per window
    2. Unstable: variance between samples can mask real changes
    
    Solution: Only run MC Dropout when:
    - Score is near decision thresholds
    - Episode boundary is being evaluated
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 10,
        threshold_margin: float = 0.15,
        fast_n_samples: int = 3,
    ):
        self.model = model
        self.n_samples = n_samples
        self.threshold_margin = threshold_margin
        self.fast_n_samples = fast_n_samples
    
    def predict_selective(
        self,
        x: torch.Tensor,
        thresholds: Dict[str, float],
    ) -> Dict[str, any]:
        """
        Selective uncertainty estimation.
        
        Returns:
            Dict with:
                - mean_probs: mean probabilities
                - uncertainty: entropy-based uncertainty or None
                - boundary_uncertainty: onset time variance
                - mc_triggered: whether full MC was run
                - fast_probs: quick single-pass probs
        """
        # Step 1: Fast single-pass prediction
        self.model.eval()
        with torch.no_grad():
            fast_logits = self.model.forward(x)
            fast_probs = F.softmax(fast_logits, dim=-1)
        
        # Step 2: Check if any timestep is near threshold
        needs_full_mc = self._check_threshold_proximity(fast_probs, thresholds)
        
        result = {
            'fast_probs': fast_probs,
            'mc_triggered': needs_full_mc,
        }
        
        if needs_full_mc:
            mean_probs, uncertainty, boundary_uncertainty = \
                self._run_mc_with_boundary(x, thresholds)
            
            result['mean_probs'] = mean_probs
            result['uncertainty'] = uncertainty
            result['boundary_uncertainty'] = boundary_uncertainty
        else:
            result['mean_probs'] = fast_probs
            result['uncertainty'] = None
            result['boundary_uncertainty'] = None
        
        return result
    
    def _check_threshold_proximity(
        self,
        probs: torch.Tensor,
        thresholds: Dict[str, float],
    ) -> bool:
        """Check if any class probability is near its threshold."""
        class_indices = {'VT': 3, 'VFL': 4, 'SVT': 2, 'SINUS_TACHY': 1}
        
        for class_name, threshold in thresholds.items():
            if class_name not in class_indices:
                continue
            idx = class_indices[class_name]
            if idx < probs.shape[-1]:
                class_probs = probs[:, :, idx]
                near_threshold = torch.abs(class_probs - threshold) < self.threshold_margin
                if near_threshold.any():
                    return True
        
        return False
    
    def _run_mc_with_boundary(
        self,
        x: torch.Tensor,
        thresholds: Dict[str, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Full MC Dropout with boundary uncertainty computation.
        
        Boundary uncertainty = variance in detected onset time across MC samples.
        """
        self.model.train()  # Enable dropout
        
        samples = []
        onset_times = {cls: [] for cls in thresholds.keys()}
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                logits = self.model.forward(x)
                probs = F.softmax(logits, dim=-1)
                samples.append(probs)
                
                # Track onset time for each sample
                for cls, thresh in thresholds.items():
                    cls_idx = {'VT': 3, 'VFL': 4, 'SVT': 2, 'SINUS_TACHY': 1}.get(cls)
                    if cls_idx is None or cls_idx >= probs.shape[-1]:
                        continue
                    
                    cls_probs = probs[0, :, cls_idx].cpu().numpy()
                    above_thresh = cls_probs > thresh
                    if above_thresh.any():
                        onset_idx = np.argmax(above_thresh)
                        onset_times[cls].append(onset_idx)
        
        samples = torch.stack(samples, dim=0)
        mean_probs = samples.mean(dim=0)
        
        # Standard entropy-based uncertainty
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        max_entropy = np.log(mean_probs.shape[-1])
        uncertainty = entropy / max_entropy
        
        # Boundary uncertainty: std of onset times across samples
        boundary_uncertainty = {}
        for cls, times in onset_times.items():
            if len(times) >= 2:
                boundary_uncertainty[cls] = float(np.std(times))
            else:
                boundary_uncertainty[cls] = float('inf')
        
        self.model.eval()
        return mean_probs, uncertainty, boundary_uncertainty
    
    def get_boundary_confidence(
        self,
        boundary_uncertainty: Dict[str, float],
        max_acceptable_std: float = 3.0,
    ) -> Dict[str, float]:
        """Convert boundary uncertainty to confidence score."""
        confidences = {}
        for cls, std in boundary_uncertainty.items():
            if std == float('inf'):
                confidences[cls] = 0.0
            else:
                confidences[cls] = max(0, 1.0 - std / max_acceptable_std)
        return confidences


# =============================================================================
# v2.3: SENSITIVITY-FIRST TRAINING
# =============================================================================

@dataclass
class SensitivityFirstConfig:
    """
    Training configuration that prioritizes VT sensitivity.
    
    Philosophy: Missing VT is catastrophic. False alarms are merely annoying.
    We tune for high sensitivity FIRST, then minimize false alarms.
    """
    # Class weights: penalize FN heavily
    # [NORMAL, SINUS_TACHY, SVT, VT, VFL]
    class_weights: List[float] = None
    
    # Focal loss parameters
    use_focal_loss: bool = True
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    
    # Sensitivity floor for model selection
    min_vt_sensitivity: float = 0.90
    
    # Threshold tuning strategy
    threshold_tuning_strategy: str = "sensitivity_first"
    
    def __post_init__(self):
        if self.class_weights is None:
            self.class_weights = [1.0, 2.0, 3.0, 10.0, 10.0]


class SensitivityFirstLoss(nn.Module):
    """
    Custom loss that penalizes false negatives heavily for VT/VFL.
    
    Options:
    1. Weighted Cross-Entropy (simple, effective)
    2. Focal Loss (handles class imbalance better)
    3. Combined (focal + asymmetric weight)
    """
    
    def __init__(self, config: SensitivityFirstConfig = None):
        super().__init__()
        if config is None:
            config = SensitivityFirstConfig()
        self.config = config
        self.class_weights = torch.tensor(config.class_weights)
        
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, seq_len, num_classes)
            targets: (batch, seq_len) integer labels
            valid_mask: (batch, seq_len) which positions to include
            
        Returns:
            Scalar loss
        """
        device = logits.device
        weights = self.class_weights.to(device)
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        if valid_mask is not None:
            mask_flat = valid_mask.view(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        if self.config.use_focal_loss:
            return self._focal_loss(logits_flat, targets_flat, weights)
        else:
            return F.cross_entropy(logits_flat, targets_flat, weight=weights)
    
    def _focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Focal loss: -alpha * (1 - p)^gamma * log(p)
        
        Focuses on hard examples (low p for correct class).
        """
        probs = F.softmax(logits, dim=-1)
        
        # Get prob of correct class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_correct = (probs * targets_one_hot.float()).sum(dim=-1)
        
        # Focal weight: (1 - p)^gamma
        focal_weight = (1 - p_correct) ** self.config.focal_gamma
        
        # Class weight for each sample
        sample_weights = weights[targets]
        
        # Combined loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        focal_loss = self.config.focal_alpha * focal_weight * sample_weights * ce_loss
        
        return focal_loss.mean()


class ThresholdTuner:
    """
    Tune detection thresholds with sensitivity-first strategy.
    
    Strategy:
    1. Find ALL thresholds that achieve >= min_vt_sensitivity
    2. Among those, pick threshold with lowest FA/hr
    3. If no threshold achieves sensitivity floor, pick highest sensitivity
    """
    
    def __init__(self, config: SensitivityFirstConfig = None):
        if config is None:
            config = SensitivityFirstConfig()
        self.config = config
        
    def find_optimal_threshold(
        self,
        probs: np.ndarray,
        ground_truth: np.ndarray,
        hours_monitored: float,
        threshold_grid: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Find optimal threshold using sensitivity-first strategy.
        
        Args:
            probs: (N,) VT probabilities
            ground_truth: (N,) binary VT labels
            hours_monitored: Total monitoring hours
            threshold_grid: Thresholds to try
            
        Returns:
            Dict with optimal_threshold, sensitivity, fa_per_hour, all_results
        """
        if threshold_grid is None:
            threshold_grid = np.arange(0.1, 0.95, 0.05)
        
        results = []
        for thresh in threshold_grid:
            predictions = (probs >= thresh).astype(int)
            
            tp = np.sum((predictions == 1) & (ground_truth == 1))
            fn = np.sum((predictions == 0) & (ground_truth == 1))
            fp = np.sum((predictions == 1) & (ground_truth == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fa_per_hour = fp / hours_monitored if hours_monitored > 0 else float('inf')
            
            results.append({
                'threshold': thresh,
                'sensitivity': sensitivity,
                'fa_per_hour': fa_per_hour,
                'tp': tp,
                'fn': fn,
                'fp': fp,
            })
        
        # Sensitivity-first selection
        # Step 1: Filter to thresholds meeting sensitivity floor
        valid_results = [
            r for r in results 
            if r['sensitivity'] >= self.config.min_vt_sensitivity
        ]
        
        if valid_results:
            # Step 2: Among valid, pick lowest FA/hr
            best = min(valid_results, key=lambda r: r['fa_per_hour'])
        else:
            # Fallback: pick highest sensitivity
            best = max(results, key=lambda r: r['sensitivity'])
        
        return {
            'optimal_threshold': best['threshold'],
            'sensitivity': best['sensitivity'],
            'fa_per_hour': best['fa_per_hour'],
            'meets_sensitivity_floor': best['sensitivity'] >= self.config.min_vt_sensitivity,
            'all_results': results,
        }


# =============================================================================
# v2.4: EXTENDED CLASS SUPPORT
# =============================================================================

class ExtendedClassConfig:
    """Extended class configuration for v2.4 multi-episode support."""
    
    # Standard 5 classes
    CLASS_NORMAL = 0
    CLASS_SINUS_TACHY = 1
    CLASS_SVT = 2
    CLASS_VT = 3
    CLASS_VFL = 4
    
    # Extended classes (v2.4)
    CLASS_VT_MONO = 5      # Monomorphic VT
    CLASS_VT_POLY = 6      # Polymorphic VT
    CLASS_AFIB_RVR = 7     # AFib with RVR
    
    CLASS_NAMES = [
        'NORMAL', 'SINUS_TACHY', 'SVT', 'VT', 'VFL',
        'VT_MONO', 'VT_POLY', 'AFIB_RVR'
    ]
    
    # Mapping for backward compatibility
    EXTENDED_TO_BASE = {
        5: 3,  # VT_MONO → VT
        6: 3,  # VT_POLY → VT
        7: 2,  # AFIB_RVR → SVT
    }
    
    @classmethod
    def collapse_to_base(cls, extended_class: int) -> int:
        """Map extended class to base 5-class scheme."""
        return cls.EXTENDED_TO_BASE.get(extended_class, extended_class)


if __name__ == "__main__":
    # Demo
    print("="*60)
    print("Causal Tachycardia Detector Demo (v2.4)")
    print("="*60)
    
    # Create model
    model = CausalTachycardiaDetector(
        num_classes=5,
        gru_hidden=128,
        gru_layers=2,
    )
    
    # Print architecture
    print(f"\nArchitecture:")
    print(f"  Downsample factor: {model.downsample_factor}x")
    print(f"  Receptive field: ~{model.get_receptive_field()} samples")
    
    # Parameter counts
    params = model.count_parameters()
    print(f"\nParameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 3600  # 10 seconds at 360 Hz
    x = torch.randn(batch_size, 1, seq_len)
    
    print(f"\nInput shape: {x.shape}")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Output length: {model.get_output_length(seq_len)}")
    
    # Test uncertainty estimation
    print("\nTesting MC Dropout uncertainty...")
    mean_probs, uncertainty = model.predict_with_uncertainty(x, n_samples=5)
    print(f"Probabilities shape: {mean_probs.shape}")
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    
    # Test streaming
    print("\nTesting streaming inference...")
    chunk_len = 360  # 1 second chunks
    for i in range(3):
        chunk = torch.randn(1, 1, chunk_len)
        probs, preds = model.predict_streaming(chunk, reset_hidden=(i==0))
        print(f"  Chunk {i+1}: output shape {probs.shape}")
    
    # Test Selective Uncertainty Estimator
    print("\n" + "-"*60)
    print("Testing SelectiveUncertaintyEstimator (v2.3)...")
    estimator = SelectiveUncertaintyEstimator(model, n_samples=5)
    thresholds = {'VT': 0.7, 'VFL': 0.6, 'SVT': 0.5}
    result = estimator.predict_selective(x[:1], thresholds)
    print(f"  MC triggered: {result['mc_triggered']}")
    print(f"  Fast probs shape: {result['fast_probs'].shape}")
    
    # Test Sensitivity-First Loss
    print("\n" + "-"*60)
    print("Testing SensitivityFirstLoss (v2.3)...")
    loss_fn = SensitivityFirstLoss()
    targets = torch.randint(0, 5, (batch_size, logits.shape[1]))
    loss = loss_fn(logits, targets)
    print(f"  Loss value: {loss.item():.4f}")
    
    # Test Threshold Tuner
    print("\n" + "-"*60)
    print("Testing ThresholdTuner (v2.3)...")
    tuner = ThresholdTuner()
    probs_test = np.random.rand(1000)
    gt_test = (np.random.rand(1000) > 0.9).astype(int)
    result = tuner.find_optimal_threshold(probs_test, gt_test, hours_monitored=10.0)
    print(f"  Optimal threshold: {result['optimal_threshold']:.2f}")
    print(f"  Sensitivity: {result['sensitivity']:.3f}")
    print(f"  FA/hr: {result['fa_per_hour']:.2f}")
    print(f"  Meets floor: {result['meets_sensitivity_floor']}")
    
    # Test Latency Spec
    print("\n" + "-"*60)
    print("Testing LatencySpec (v2.3)...")
    latency_spec = LatencySpec()
    print(f"  Target detection latency: {latency_spec.target_detection_latency_sec}s")
    print(f"  Max processing latency: {latency_spec.max_processing_latency_ms}ms")
    print(f"  Validate 2s detection: {latency_spec.validate_detection_latency(2.0)}")
    print(f"  Validate 50ms processing: {latency_spec.validate_processing_latency(50.0)}")
    
    # Test Extended Class Config
    print("\n" + "-"*60)
    print("Testing ExtendedClassConfig (v2.4)...")
    print(f"  VT_MONO (5) → base: {ExtendedClassConfig.collapse_to_base(5)}")
    print(f"  VT_POLY (6) → base: {ExtendedClassConfig.collapse_to_base(6)}")
    print(f"  AFIB_RVR (7) → base: {ExtendedClassConfig.collapse_to_base(7)}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print("="*60)
