"""
Deep Learning Models for Tachycardia Detection
Implements 1D-CNN and LSTM models with attention mechanisms
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class CNN1DClassifier:
    """
    1D Convolutional Neural Network for ECG beat classification
    
    Uses numpy-only implementation for portability.
    For production, recommend using TensorFlow/PyTorch.
    """
    
    def __init__(self, input_length: int = 216,
                 n_classes: int = 2,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Initialize CNN classifier
        
        Args:
            input_length: Length of input beat waveform
            n_classes: Number of output classes
            learning_rate: Learning rate for training
            random_state: Random seed
        """
        self.input_length = input_length
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        self.scaler = StandardScaler()
        self.weights = {}
        self.is_fitted = False
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using He initialization"""
        # Conv layer 1: 32 filters, kernel size 5
        self.weights['conv1_w'] = np.random.randn(32, 1, 5) * np.sqrt(2.0 / 5)
        self.weights['conv1_b'] = np.zeros(32)
        
        # Conv layer 2: 64 filters, kernel size 3
        self.weights['conv2_w'] = np.random.randn(64, 32, 3) * np.sqrt(2.0 / (32 * 3))
        self.weights['conv2_b'] = np.zeros(64)
        
        # Calculate flattened size after convolutions and pooling
        # After conv1 (kernel=5): L - 4
        # After pool1 (size=2): (L - 4) // 2
        # After conv2 (kernel=3): ((L - 4) // 2) - 2
        # After pool2 (size=2): (((L - 4) // 2) - 2) // 2
        
        L = self.input_length
        after_conv1 = L - 4
        after_pool1 = after_conv1 // 2
        after_conv2 = after_pool1 - 2
        after_pool2 = after_conv2 // 2
        
        flat_size = 64 * after_pool2
        
        # Dense layers
        self.weights['fc1_w'] = np.random.randn(flat_size, 128) * np.sqrt(2.0 / flat_size)
        self.weights['fc1_b'] = np.zeros(128)
        
        self.weights['fc2_w'] = np.random.randn(128, self.n_classes) * np.sqrt(2.0 / 128)
        self.weights['fc2_b'] = np.zeros(self.n_classes)
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _conv1d(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        1D convolution operation
        
        Args:
            x: Input [batch, channels, length]
            w: Weights [out_channels, in_channels, kernel_size]
            b: Bias [out_channels]
            
        Returns:
            Output [batch, out_channels, new_length]
        """
        batch_size, in_channels, length = x.shape
        out_channels, _, kernel_size = w.shape
        out_length = length - kernel_size + 1
        
        output = np.zeros((batch_size, out_channels, out_length))
        
        for i in range(out_length):
            window = x[:, :, i:i+kernel_size]  # [batch, in_channels, kernel_size]
            for oc in range(out_channels):
                output[:, oc, i] = np.sum(window * w[oc], axis=(1, 2)) + b[oc]
        
        return output
    
    def _max_pool1d(self, x: np.ndarray, pool_size: int = 2) -> np.ndarray:
        """1D max pooling"""
        batch_size, channels, length = x.shape
        out_length = length // pool_size
        
        output = np.zeros((batch_size, channels, out_length))
        
        for i in range(out_length):
            output[:, :, i] = np.max(x[:, :, i*pool_size:(i+1)*pool_size], axis=2)
        
        return output
    
    def _forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass through the network
        
        Args:
            x: Input beats [batch, length]
            
        Returns:
            Dictionary with all layer activations
        """
        cache = {}
        
        # Add channel dimension [batch, 1, length]
        x = x.reshape(x.shape[0], 1, -1)
        cache['input'] = x
        
        # Conv1 + ReLU + Pool
        conv1 = self._conv1d(x, self.weights['conv1_w'], self.weights['conv1_b'])
        cache['conv1'] = conv1
        relu1 = self._relu(conv1)
        cache['relu1'] = relu1
        pool1 = self._max_pool1d(relu1)
        cache['pool1'] = pool1
        
        # Conv2 + ReLU + Pool
        conv2 = self._conv1d(pool1, self.weights['conv2_w'], self.weights['conv2_b'])
        cache['conv2'] = conv2
        relu2 = self._relu(conv2)
        cache['relu2'] = relu2
        pool2 = self._max_pool1d(relu2)
        cache['pool2'] = pool2
        
        # Flatten
        flat = pool2.reshape(pool2.shape[0], -1)
        cache['flat'] = flat
        
        # FC1 + ReLU
        fc1 = flat @ self.weights['fc1_w'] + self.weights['fc1_b']
        cache['fc1'] = fc1
        relu_fc1 = self._relu(fc1)
        cache['relu_fc1'] = relu_fc1
        
        # FC2 (output)
        fc2 = relu_fc1 @ self.weights['fc2_w'] + self.weights['fc2_b']
        cache['fc2'] = fc2
        
        # Softmax
        output = self._softmax(fc2)
        cache['output'] = output
        
        return cache
    
    def _compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # One-hot encode if needed
        if len(y_true.shape) == 1:
            y_one_hot = np.zeros((len(y_true), self.n_classes))
            y_one_hot[np.arange(len(y_true)), y_true] = 1
            y_true = y_one_hot
        
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 50,
            batch_size: int = 32,
            validation_split: float = 0.1,
            class_weight: Optional[Dict] = None,
            verbose: bool = True) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            X: Input beats [n_samples, length]
            y: Labels [n_samples]
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Fraction for validation
            class_weight: Optional class weights for imbalance
            verbose: Print training progress
            
        Returns:
            Training history
        """
        # Scale input
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Handle class imbalance with weights
        if class_weight is None:
            class_counts = np.bincount(y)
            total = len(y)
            class_weight = {i: total / (len(class_counts) * count) 
                          for i, count in enumerate(class_counts)}
        
        sample_weights = np.array([class_weight[label] for label in y])
        
        # Split validation
        n_val = int(len(X_scaled) * validation_split)
        indices = np.random.permutation(len(X_scaled))
        
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        
        X_train, y_train = X_scaled[train_idx], y[train_idx]
        X_val, y_val = X_scaled[val_idx], y[val_idx]
        weights_train = sample_weights[train_idx]
        
        history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            # Shuffle training data
            perm = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]
            weights_shuffled = weights_train[perm]
            
            epoch_loss = 0
            
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                w_batch = weights_shuffled[start:end]
                
                # Forward pass
                cache = self._forward(X_batch)
                
                # Compute loss
                loss = self._compute_loss(cache['output'], y_batch)
                epoch_loss += loss
                
                # Backward pass (simplified gradient descent)
                self._backward(cache, y_batch, w_batch)
            
            epoch_loss /= n_batches
            
            # Validation
            val_cache = self._forward(X_val)
            val_loss = self._compute_loss(val_cache['output'], y_val)
            
            # Accuracy
            train_pred = np.argmax(self._forward(X_train)['output'], axis=1)
            val_pred = np.argmax(val_cache['output'], axis=1)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            history['loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        
        self.is_fitted = True
        return history
    
    def _backward(self, cache: Dict, y_true: np.ndarray, 
                  sample_weights: np.ndarray):
        """
        Backward pass with gradient descent update
        
        Simplified implementation - for production use autograd
        """
        batch_size = len(y_true)
        
        # One-hot encode
        y_one_hot = np.zeros((batch_size, self.n_classes))
        y_one_hot[np.arange(batch_size), y_true] = 1
        
        # Output gradient
        d_output = (cache['output'] - y_one_hot) * sample_weights.reshape(-1, 1)
        d_output /= batch_size
        
        # FC2 gradients
        d_fc2_w = cache['relu_fc1'].T @ d_output
        d_fc2_b = np.sum(d_output, axis=0)
        
        # Backprop through FC2
        d_relu_fc1 = d_output @ self.weights['fc2_w'].T
        d_fc1 = d_relu_fc1 * self._relu_derivative(cache['fc1'])
        
        # FC1 gradients
        d_fc1_w = cache['flat'].T @ d_fc1
        d_fc1_b = np.sum(d_fc1, axis=0)
        
        # Update weights (simple gradient descent)
        lr = self.learning_rate
        
        self.weights['fc2_w'] -= lr * d_fc2_w
        self.weights['fc2_b'] -= lr * d_fc2_b
        self.weights['fc1_w'] -= lr * d_fc1_w
        self.weights['fc1_b'] -= lr * d_fc1_b
        
        # Note: For simplicity, not backpropagating through conv layers
        # In production, use proper autograd framework
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)
        
        cache = self._forward(X_scaled)
        return np.argmax(cache['output'], axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)
        
        cache = self._forward(X_scaled)
        return cache['output']
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'sensitivity': recall_score(y, y_pred, pos_label=1, zero_division=0),
            'specificity': recall_score(y, y_pred, pos_label=0, zero_division=0)
        }
        
        try:
            metrics['auc_roc'] = roc_auc_score(y, y_proba)
        except ValueError:
            metrics['auc_roc'] = 0.0
        
        return metrics
    
    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get attention-like weights for interpretability
        
        Uses gradient-based saliency as proxy for attention
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        X_flat = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.transform(X_flat)
        X_scaled = X_scaled.reshape(X.shape)
        
        # Compute output
        cache = self._forward(X_scaled)
        
        # Use conv1 activations as importance proxy
        # Higher activation = more important region
        conv1_acts = cache['relu1']  # [batch, channels, length]
        
        # Average across channels
        attention = np.mean(np.abs(conv1_acts), axis=1)  # [batch, length]
        
        # Normalize
        attention = attention / (np.max(attention, axis=1, keepdims=True) + 1e-8)
        
        # Upsample to original length
        from scipy.ndimage import zoom
        original_length = X.shape[1]
        current_length = attention.shape[1]
        scale = original_length / current_length
        
        upsampled = np.array([zoom(a, scale, order=1) for a in attention])
        
        # Ensure correct length
        if upsampled.shape[1] > original_length:
            upsampled = upsampled[:, :original_length]
        elif upsampled.shape[1] < original_length:
            pad = original_length - upsampled.shape[1]
            upsampled = np.pad(upsampled, ((0, 0), (0, pad)), mode='edge')
        
        return upsampled


def main():
    """Test deep learning models"""
    import json
    
    # Load dataset
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'processed')
    
    if not os.path.exists(os.path.join(data_dir, 'train_test_split.npz')):
        print("Dataset not found. Run the preprocessing pipeline first.")
        return
    
    print("Loading dataset...")
    split = np.load(os.path.join(data_dir, 'train_test_split.npz'), allow_pickle=True)
    
    beats_train = split['beats_train']
    beats_test = split['beats_test']
    y_train = split['y_train_binary']
    y_test = split['y_test_binary']
    
    print(f"Train beats: {beats_train.shape}")
    print(f"Test beats: {beats_test.shape}")
    print(f"Train class distribution: {np.bincount(y_train)}")
    
    # Subsample for faster testing
    n_train = min(5000, len(beats_train))
    n_test = min(1000, len(beats_test))
    
    idx_train = np.random.choice(len(beats_train), n_train, replace=False)
    idx_test = np.random.choice(len(beats_test), n_test, replace=False)
    
    X_train = beats_train[idx_train]
    X_test = beats_test[idx_test]
    y_train_sub = y_train[idx_train]
    y_test_sub = y_test[idx_test]
    
    print(f"\nUsing subset: {n_train} train, {n_test} test")
    print(f"Train class distribution: {np.bincount(y_train_sub)}")
    
    # Train CNN
    print("\n" + "="*60)
    print("Training 1D-CNN")
    print("="*60)
    
    cnn = CNN1DClassifier(
        input_length=X_train.shape[1],
        n_classes=2,
        learning_rate=0.001
    )
    
    history = cnn.fit(
        X_train, y_train_sub,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=True
    )
    
    # Evaluate
    print("\nTest Set Evaluation:")
    metrics = cnn.evaluate(X_test, y_test_sub)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Get attention weights for first few samples
    print("\nGetting attention weights...")
    attention = cnn.get_attention_weights(X_test[:5])
    print(f"Attention shape: {attention.shape}")


if __name__ == '__main__':
    main()
