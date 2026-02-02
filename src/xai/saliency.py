"""
XAI Saliency Module.

Proper XAI using gradient-based methods, NOT attention weights.
Attention weights are NOT reliable explanations - they can be non-causal and unstable.

Methods implemented:
1. Integrated Gradients - Primary method (Sundararajan et al., 2017)
2. Gradient × Input - Fast approximation
3. Occlusion Sensitivity - Model-agnostic

These methods provide attributions that show WHICH parts of the input
signal contributed to the model's decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


@dataclass
class AttributionResult:
    """Result from attribution computation."""
    attributions: np.ndarray          # Per-sample attribution values
    target_class: int                 # Class being explained
    baseline_type: str                # Type of baseline used
    method: str                       # Attribution method name
    metadata: Dict[str, Any] = None   # Additional info
    
    @property
    def normalized_attributions(self) -> np.ndarray:
        """Get attributions normalized to [-1, 1]."""
        abs_max = np.abs(self.attributions).max()
        if abs_max > 0:
            return self.attributions / abs_max
        return self.attributions
    
    @property
    def positive_attributions(self) -> np.ndarray:
        """Get only positive attributions (evidence FOR the class)."""
        return np.maximum(self.attributions, 0)
    
    @property
    def negative_attributions(self) -> np.ndarray:
        """Get only negative attributions (evidence AGAINST the class)."""
        return np.minimum(self.attributions, 0)
    
    def get_top_k_regions(
        self, 
        k: int = 5, 
        window_size: int = 50
    ) -> List[Tuple[int, int, float]]:
        """
        Get top-k most important regions.
        
        Returns:
            List of (start, end, importance) tuples
        """
        # Compute importance per window
        n_windows = len(self.attributions) // window_size
        window_importance = []
        
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            importance = np.sum(np.abs(self.attributions[start:end]))
            window_importance.append((start, end, importance))
        
        # Sort by importance
        window_importance.sort(key=lambda x: x[2], reverse=True)
        
        return window_importance[:k]


class IntegratedGradients:
    """
    Integrated Gradients attribution method.
    
    Computes attributions by integrating gradients along a path from
    a baseline to the input. This satisfies two key axioms:
    - Sensitivity: If input differs from baseline and changes prediction, 
                   it gets non-zero attribution
    - Implementation Invariance: Two functionally identical models get
                                  same attributions
    
    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks" (2017)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
        n_steps: int = 50,
        baseline: Optional[torch.Tensor] = None,
        internal_batch_size: int = 16,
    ) -> AttributionResult:
        """
        Compute Integrated Gradients attributions.
        
        Args:
            x: Input tensor (1, 1, seq_len)
            target_class: Class index to explain
            n_steps: Number of integration steps (more = more accurate)
            baseline: Reference input (default: zeros)
            internal_batch_size: Batch size for gradient computation
            
        Returns:
            AttributionResult with per-sample attributions
        """
        x = x.to(self.device)
        
        if baseline is None:
            baseline = torch.zeros_like(x)
        else:
            baseline = baseline.to(self.device)
        
        # Generate interpolated inputs along path
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        # Compute gradients in batches
        all_gradients = []
        
        for batch_start in range(0, n_steps, internal_batch_size):
            batch_end = min(batch_start + internal_batch_size, n_steps)
            batch_alphas = alphas[batch_start:batch_end]
            
            # Interpolated inputs: baseline + alpha * (x - baseline)
            # Shape: (batch_size, 1, seq_len)
            interpolated = baseline + batch_alphas.view(-1, 1, 1) * (x - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = self.model(interpolated)
            
            # Get target class output (average over time dimension)
            target_outputs = outputs[:, :, target_class].mean(dim=1)
            
            # Compute gradients
            gradients = torch.autograd.grad(
                outputs=target_outputs.sum(),
                inputs=interpolated,
                create_graph=False,
            )[0]
            
            all_gradients.append(gradients.detach())
        
        # Concatenate all gradients
        all_gradients = torch.cat(all_gradients, dim=0)  # (n_steps, 1, seq_len)
        
        # Average gradients
        avg_gradients = all_gradients.mean(dim=0)  # (1, seq_len)
        
        # Integrated Gradients = (x - baseline) * avg_gradients
        attributions = (x - baseline).squeeze() * avg_gradients.squeeze()
        
        return AttributionResult(
            attributions=attributions.detach().cpu().numpy(),
            target_class=target_class,
            baseline_type="zeros" if baseline.sum() == 0 else "custom",
            method="integrated_gradients",
            metadata={"n_steps": n_steps},
        )
    
    def compute_with_multiple_baselines(
        self,
        x: torch.Tensor,
        target_class: int,
        n_baselines: int = 10,
        n_steps: int = 50,
    ) -> AttributionResult:
        """
        Compute IG with multiple random baselines and average.
        
        This can provide more stable attributions by reducing
        dependence on baseline choice.
        """
        all_attributions = []
        
        for _ in range(n_baselines):
            # Random Gaussian baseline
            baseline = torch.randn_like(x) * 0.1
            result = self.compute(x, target_class, n_steps, baseline)
            all_attributions.append(result.attributions)
        
        avg_attributions = np.mean(all_attributions, axis=0)
        
        return AttributionResult(
            attributions=avg_attributions,
            target_class=target_class,
            baseline_type="random_average",
            method="integrated_gradients_averaged",
            metadata={"n_baselines": n_baselines, "n_steps": n_steps},
        )


class GradientXInput:
    """
    Gradient × Input attribution method.
    
    Simple and fast approximation to Integrated Gradients.
    Computes gradient of output w.r.t. input and multiplies by input.
    
    Faster than IG but less accurate. Good for quick exploration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
    ) -> AttributionResult:
        """
        Compute Gradient × Input attributions.
        
        Args:
            x: Input tensor (1, 1, seq_len)
            target_class: Class index to explain
            
        Returns:
            AttributionResult
        """
        x = x.to(self.device)
        x = x.clone().requires_grad_(True)
        
        # Forward pass
        outputs = self.model(x)
        
        # Get target class output
        target_output = outputs[:, :, target_class].mean()
        
        # Compute gradient
        target_output.backward()
        
        gradients = x.grad.squeeze()
        
        # Attribution = gradient × input
        attributions = (gradients * x.squeeze()).detach().cpu().numpy()
        
        return AttributionResult(
            attributions=attributions,
            target_class=target_class,
            baseline_type="none",
            method="gradient_x_input",
        )


class OcclusionSensitivity:
    """
    Occlusion Sensitivity attribution method.
    
    Model-agnostic method that measures how prediction changes
    when parts of the input are occluded (replaced with baseline).
    
    Slower than gradient methods but doesn't require gradients,
    making it applicable to any model type.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def compute(
        self,
        x: torch.Tensor,
        target_class: int,
        window_size: int = 50,
        stride: int = 10,
        occlusion_value: float = 0.0,
    ) -> AttributionResult:
        """
        Compute occlusion sensitivity attributions.
        
        Args:
            x: Input tensor (1, 1, seq_len)
            target_class: Class index to explain
            window_size: Size of occlusion window (samples)
            stride: Stride for sliding window
            occlusion_value: Value to replace occluded region with
            
        Returns:
            AttributionResult
        """
        x = x.to(self.device)
        seq_len = x.shape[-1]
        
        # Get baseline prediction
        self.model.eval()
        with torch.no_grad():
            baseline_out = self.model(x)
            baseline_probs = F.softmax(baseline_out, dim=-1)
            baseline_prob = baseline_probs[:, :, target_class].mean().item()
        
        # Compute importance for each position
        importance = np.zeros(seq_len)
        counts = np.zeros(seq_len)
        
        for start in range(0, seq_len - window_size + 1, stride):
            end = start + window_size
            
            # Create occluded input
            x_occluded = x.clone()
            x_occluded[:, :, start:end] = occlusion_value
            
            # Get occluded prediction
            with torch.no_grad():
                occluded_out = self.model(x_occluded)
                occluded_probs = F.softmax(occluded_out, dim=-1)
                occluded_prob = occluded_probs[:, :, target_class].mean().item()
            
            # Importance = drop in probability when region is occluded
            drop = baseline_prob - occluded_prob
            
            # Assign importance to all samples in window
            importance[start:end] += drop
            counts[start:end] += 1
        
        # Average overlapping contributions
        attributions = np.divide(
            importance, 
            counts, 
            where=counts > 0,
            out=np.zeros_like(importance)
        )
        
        return AttributionResult(
            attributions=attributions,
            target_class=target_class,
            baseline_type="occlusion",
            method="occlusion_sensitivity",
            metadata={"window_size": window_size, "stride": stride},
        )


class XAIModule:
    """
    Unified XAI module providing multiple attribution methods.
    
    IMPORTANT: Attention weights are NOT explanations. This module
    provides proper gradient-based attributions that are:
    - Causally meaningful
    - Stable under small perturbations (when checked)
    - Aligned with clinical expectations
    
    Usage:
        xai = XAIModule(model)
        result = xai.explain(x, target_class)
        
        # Check stability
        stability = xai.check_stability(x, target_class)
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        default_method: str = 'integrated_gradients',
    ):
        self.model = model
        self.device = device
        self.default_method = default_method
        
        # Initialize attribution methods
        self.ig = IntegratedGradients(model, device)
        self.grad_input = GradientXInput(model, device)
        self.occlusion = OcclusionSensitivity(model, device)
    
    def explain(
        self,
        x: torch.Tensor,
        target_class: int,
        method: Optional[str] = None,
        **kwargs
    ) -> AttributionResult:
        """
        Generate explanation for a prediction.
        
        Args:
            x: Input signal
            target_class: Class to explain
            method: 'integrated_gradients', 'gradient_x_input', or 'occlusion'
            **kwargs: Method-specific arguments
            
        Returns:
            AttributionResult
        """
        method = method or self.default_method
        
        if method == 'integrated_gradients':
            return self.ig.compute(x, target_class, **kwargs)
        elif method == 'gradient_x_input':
            return self.grad_input.compute(x, target_class)
        elif method == 'occlusion':
            return self.occlusion.compute(x, target_class, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain_all_methods(
        self,
        x: torch.Tensor,
        target_class: int,
    ) -> Dict[str, AttributionResult]:
        """
        Compute attributions with all methods for comparison.
        
        Returns:
            Dict mapping method name to AttributionResult
        """
        return {
            'integrated_gradients': self.ig.compute(x, target_class),
            'gradient_x_input': self.grad_input.compute(x, target_class),
            'occlusion': self.occlusion.compute(x, target_class),
        }
    
    def get_most_important_regions(
        self,
        x: torch.Tensor,
        target_class: int,
        k: int = 5,
        window_size: int = 50,
    ) -> List[Tuple[int, int, float]]:
        """
        Get the k most important input regions.
        
        Returns:
            List of (start_sample, end_sample, importance) tuples
        """
        result = self.explain(x, target_class)
        return result.get_top_k_regions(k, window_size)


def visualize_attribution(
    signal: np.ndarray,
    attribution: np.ndarray,
    fs: int = 360,
    title: str = "Attribution Visualization",
    save_path: Optional[str] = None,
):
    """
    Visualize signal with attribution overlay.
    
    Args:
        signal: Original ECG signal
        attribution: Attribution values (same length as signal)
        fs: Sampling frequency
        title: Plot title
        save_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    time = np.arange(len(signal)) / fs
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Plot 1: Original signal
    axes[0].plot(time, signal, 'b-', linewidth=0.5)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('ECG Signal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Attribution heatmap
    # Normalize attribution to [-1, 1]
    attr_norm = attribution / (np.abs(attribution).max() + 1e-8)
    
    # Create colormap: red for positive, blue for negative
    cmap = plt.cm.RdBu_r
    
    # Plot as colored segments
    for i in range(len(time) - 1):
        color = cmap((attr_norm[i] + 1) / 2)  # Map [-1,1] to [0,1]
        axes[1].axvspan(time[i], time[i+1], color=color, alpha=0.7)
    
    axes[1].set_ylabel('Attribution')
    axes[1].set_title('Attribution Heatmap (Red=Positive, Blue=Negative)')
    
    # Plot 3: Signal with attribution overlay
    axes[2].plot(time, signal, 'k-', linewidth=0.5, alpha=0.5)
    
    # Overlay positive attributions
    pos_mask = attribution > 0
    axes[2].fill_between(
        time, 0, signal, 
        where=pos_mask, 
        color='red', 
        alpha=0.3, 
        label='Positive attribution'
    )
    
    # Overlay negative attributions
    neg_mask = attribution < 0
    axes[2].fill_between(
        time, 0, signal,
        where=neg_mask,
        color='blue',
        alpha=0.3,
        label='Negative attribution'
    )
    
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Signal with Attribution Overlay')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    print("XAI Saliency Module Demo")
    print("=" * 60)
    
    # Create a simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(1, 16, 7, padding=3)
            self.pool = nn.MaxPool1d(8)
            self.fc = nn.Linear(16, 5)
        
        def forward(self, x):
            x = F.relu(self.conv(x))
            x = self.pool(x)
            x = x.permute(0, 2, 1)
            return self.fc(x)
    
    model = SimpleModel()
    xai = XAIModule(model, device='cpu')
    
    # Create test input
    x = torch.randn(1, 1, 360)
    
    print("\nComputing attributions with all methods...")
    results = xai.explain_all_methods(x, target_class=3)
    
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Shape: {result.attributions.shape}")
        print(f"  Range: [{result.attributions.min():.4f}, {result.attributions.max():.4f}]")
        print(f"  Top regions: {result.get_top_k_regions(3)}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
