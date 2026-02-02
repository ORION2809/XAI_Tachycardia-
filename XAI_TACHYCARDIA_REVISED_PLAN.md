# XAI Tachycardia Detection - REVISED Implementation Plan

## Based on Critical Analysis of Initial Approach

This revised plan addresses the fundamental shortcomings identified in the critique, focusing on **clinical validity**, **generalizability**, **sensitivity optimization**, and **meaningful explainability**.

---

## 🔴 Critical Issues to Address

| Issue | Current State | Required Fix | Priority |
|-------|---------------|--------------|----------|
| **Class Imbalance** | 25.9:1 ratio, SMOTE planned | Physiological augmentation + class weights | 🔴 Critical |
| **Clinical Labeling** | HR-based threshold only | VT ≥3 beats @ >100bpm, sustained criteria | 🔴 Critical |
| **Sequence Context** | Beat-by-beat classification | CNN+LSTM hybrid for episode detection | 🔴 Critical |
| **Dataset Limitations** | 48 patients, 1980s data | External datasets + transfer learning | 🔴 Critical |
| **Evaluation Metrics** | F1/Accuracy focused | Sensitivity, False Alarm/Hour, PPV | 🔴 Critical |
| **XAI Approach** | SHAP on features | Saliency maps on ECG waveform | 🟡 High |
| **Threshold Tuning** | Default 0.5 | Optimized for max sensitivity | 🟡 High |

---

## Revised Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REVISED XAI TACHYCARDIA DETECTION SYSTEM                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────────┐     ┌─────────────────────────┐  │
│  │  Raw ECG    │────▶│ Clinical-Grade   │────▶│ Episode-Level           │  │
│  │  Segments   │     │ Preprocessing    │     │ Segmentation            │  │
│  │  (5-10 sec) │     │ (Noise-aware)    │     │ (Not beat-by-beat)      │  │
│  └─────────────┘     └──────────────────┘     └───────────┬─────────────┘  │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    HYBRID CNN-LSTM MODEL                            │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌────────────────────┐    │   │
│  │  │ 1D-CNN       │───▶│ Bi-LSTM      │───▶│ Attention          │    │   │
│  │  │ (Morphology) │    │ (Temporal)   │    │ (Saliency for XAI) │    │   │
│  │  └──────────────┘    └──────────────┘    └────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    CLINICAL DECISION LAYER                          │   │
│  │  • VT Detection: ≥3 consecutive ventricular beats @ >100 BPM       │   │
│  │  • SVT Detection: Narrow complex tachy, sustained >30 sec          │   │
│  │  • Sinus Tachy: Sustained HR >100 BPM in normal rhythm             │   │
│  │  • Episode Grouping: Consecutive detections required               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                           │                 │
│                                                           ▼                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    XAI OUTPUT (Saliency-Based)                      │   │
│  │  • Highlighted ECG segment showing tachycardia region              │   │
│  │  • Confidence score with uncertainty quantification                │   │
│  │  • Clinical reasoning: "7 consecutive wide-complex beats @ 150bpm" │   │
│  │  • Episode duration and characteristics                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Pipeline REVISION (Week 1-2)

### 1.1 Clinical Labeling Criteria (CRITICAL FIX)

**Current Problem:** Simple HR > 100 BPM threshold doesn't match clinical definitions.

**Revised Criteria:**

| Tachycardia Type | Clinical Definition | Implementation |
|------------------|---------------------|----------------|
| **Ventricular Tachycardia (VT)** | ≥3 consecutive ventricular beats @ >100 BPM | Detect V-beat runs, check rate |
| **Non-Sustained VT** | VT lasting <30 seconds | Episode duration < 30s |
| **Sustained VT** | VT lasting ≥30 seconds OR requiring intervention | Episode duration ≥ 30s |
| **Ventricular Flutter (VFL)** | Rapid VT ~300 BPM, sine-wave pattern | Use rhythm annotation |
| **SVT (SVTA)** | Narrow complex tachy >100 BPM, not sinus | Use rhythm annotation |
| **Sinus Tachycardia** | Sustained sinus rhythm >100 BPM for >30 sec | HR threshold + duration |

### 1.2 Episode-Level Segmentation (CRITICAL FIX)

**Current Problem:** Beat-by-beat classification loses temporal context.

**Revised Approach:**
```
Episode Detection Strategy:
1. Identify rhythm change events from annotations
2. Extract 5-10 second windows around events
3. Label ENTIRE segment as tachycardia or normal
4. For continuous monitoring: Use sliding window with episode grouping
```

### 1.3 Physiological Data Augmentation (Replace SMOTE)

**Current Problem:** SMOTE creates unrealistic interpolated ECGs.

**Revised Augmentation:**

| Technique | Description | Physiological Basis |
|-----------|-------------|---------------------|
| **Time Warping** | ±5-15% stretch/compress | Natural HR variability |
| **Amplitude Scaling** | ±10-20% scaling | Lead placement variation |
| **Baseline Wander** | Add low-freq sinusoid (0.1-0.5 Hz) | Respiration artifact |
| **Gaussian Noise** | SNR 20-40 dB | EMG/electrode noise |
| **Powerline Noise** | 50/60 Hz sinusoid | Common interference |
| **Beat Morphology Variation** | Slight QRS width changes | Normal variation |
| **Heart Rate Jitter** | Small RR interval variations | Natural variability |

**NOT Using:**
- SMOTE (interpolation creates unrealistic morphologies)
- GAN-generated ECGs (unless validated by cardiologists)

### 1.4 Class Weighting Strategy

```python
# Instead of oversampling, use cost-sensitive learning
class_weights = {
    0: 1.0,      # Normal
    1: 10.0,     # Sinus Tachycardia (less dangerous)
    2: 25.0,     # SVT (moderate risk)
    3: 50.0,     # VT (high risk - must detect)
    4: 100.0     # VFL (life-threatening - maximum weight)
}
```

### 1.5 External Dataset Integration

**Required for Generalization:**

| Dataset | Size | Tachycardia Content | Use |
|---------|------|---------------------|-----|
| **PTB-XL** | 21,837 records | Sinus tachy, SVT | Primary augmentation |
| **INCART** | 75 records | VT episodes | VT enhancement |
| **AFDB** | 25 records | AF with RVR | Atrial tachy |
| **CU Ventricular** | 35 records | VT/VF | Critical arrhythmia |
| **Chapman-Shaoxing** | 10,646 records | Mixed | Validation |

---

## Phase 2: Model Architecture (Week 3-4)

### 2.1 Hybrid CNN-LSTM with Attention

**Why This Architecture:**
- **CNN**: Captures local morphology (QRS width, shape)
- **Bi-LSTM**: Captures temporal patterns (consecutive beats, episode duration)
- **Attention**: Provides interpretable saliency for XAI

```
Model Architecture:
├── Input: 10-second ECG window (3600 samples @ 360 Hz)
├── CNN Block 1: Conv1D(64, k=5) → BatchNorm → ReLU → MaxPool
├── CNN Block 2: Conv1D(128, k=5) → BatchNorm → ReLU → MaxPool
├── CNN Block 3: Conv1D(256, k=3) → BatchNorm → ReLU → MaxPool
├── Bi-LSTM: 128 units (captures forward/backward temporal patterns)
├── Attention Layer: Self-attention for saliency maps
├── Dense: 64 → Dropout(0.5) → 32
├── Output: 
│   ├── Binary: Tachycardia (0/1)
│   ├── Multi-class: Normal/SinusTachy/SVT/VT/VFL
│   └── Confidence: Calibrated probability
```

### 2.2 Regularization (Prevent Overfitting on 48 Patients)

| Technique | Setting | Purpose |
|-----------|---------|---------|
| Dropout | 0.5 on dense layers | Prevent co-adaptation |
| Spatial Dropout | 0.2 on CNN | Prevent filter overfitting |
| L2 Regularization | 1e-4 | Weight decay |
| Early Stopping | patience=10 | Stop before overfit |
| Data Augmentation | On-the-fly | Implicit regularization |
| Label Smoothing | 0.1 | Reduce overconfidence |

### 2.3 Transfer Learning Strategy

```
Pre-training Options:
1. Self-supervised on large unlabeled ECG data (contrastive learning)
2. Pre-train on PTB-XL general arrhythmia task
3. Use pre-trained ECG encoder (if available from literature)

Fine-tuning:
- Freeze CNN layers initially
- Train LSTM + attention on tachycardia task
- Gradually unfreeze CNN for fine-tuning
```

### 2.4 Ensemble for Robustness

```
Ensemble Strategy:
├── Model 1: CNN-LSTM (morphology + temporal)
├── Model 2: Transformer-based (long-range dependencies)
├── Model 3: Random Forest on engineered features (interpretable baseline)
│
└── Voting: Weighted average (weights learned on validation set)
    - If 2/3 models agree: High confidence
    - If only 1 model fires: Lower confidence, possible false alarm
```

---

## Phase 3: Clinical Evaluation Protocol (Week 5-6)

### 3.1 Inter-Patient Split (MANDATORY)

```
Split Strategy:
- Training: 38 patients (80%)
- Validation: 5 patients (10%) - for threshold tuning
- Test: 5 patients (10%) - final evaluation only

Stratification: Ensure each split has tachycardia episodes
Cross-validation: 5-fold patient-level CV for robust estimates
```

### 3.2 Clinically Relevant Metrics

**Primary Metrics (Report These):**

| Metric | Target | Clinical Meaning |
|--------|--------|------------------|
| **Sensitivity (Recall)** | ≥95% for VT/VFL | Don't miss life-threatening arrhythmias |
| **False Alarms per Hour** | <1/hour | Avoid alarm fatigue |
| **Positive Predictive Value** | ≥80% | Most alarms are real |
| **Episode Detection Rate** | ≥90% | Detect actual episodes, not just beats |

**Secondary Metrics:**
- Specificity, F1-score, AUC-ROC, AUC-PR
- Per-record sensitivity (ensure no patient is completely missed)
- Latency to detection (how quickly after episode starts)

### 3.3 Threshold Optimization

```python
# Optimize threshold for clinical requirements
def find_optimal_threshold(y_true, y_proba, min_sensitivity=0.95):
    """Find threshold that achieves minimum sensitivity while maximizing PPV"""
    thresholds = np.linspace(0.01, 0.99, 99)
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        sens = recall_score(y_true, y_pred)
        
        if sens >= min_sensitivity:
            ppv = precision_score(y_true, y_pred)
            return thresh, sens, ppv
    
    # If can't achieve target, use lowest threshold
    return 0.01, 1.0, precision_score(y_true, (y_proba >= 0.01))
```

### 3.4 External Validation (CRITICAL)

```
External Validation Protocol:
1. Train model on MIT-BIH only
2. Test on INCART database (never seen during training)
3. Test on subset of PTB-XL (held out entirely)
4. Report performance drop (expect ~5-15% drop is acceptable)
5. If drop >20%, model is overfitting to MIT-BIH
```

### 3.5 Episode-Level Evaluation

```
Episode-Level Metrics:
- True Positive Episode: ≥50% of episode correctly detected
- False Positive Episode: Alert during non-tachycardia period
- Missed Episode: <50% of episode detected

Report:
- Episode Sensitivity: # detected episodes / # total episodes
- Episode PPV: # true alarm episodes / # total alarms
```

---

## Phase 4: False Alarm Reduction (Week 7)

### 4.1 Consecutive Detection Requirement

```python
class EpisodeDetector:
    """Require consecutive positive predictions to reduce false alarms"""
    
    def __init__(self, min_consecutive_beats=3, min_duration_sec=1.0):
        self.min_beats = min_consecutive_beats
        self.min_duration = min_duration_sec
    
    def detect_episodes(self, beat_predictions, beat_times):
        """
        Only flag tachycardia if:
        1. At least 3 consecutive beats are positive
        2. Duration is at least 1 second
        3. HR in that segment is actually >100 BPM
        """
        episodes = []
        current_episode = []
        
        for i, (pred, time) in enumerate(zip(beat_predictions, beat_times)):
            if pred == 1:
                current_episode.append((i, time))
            else:
                if len(current_episode) >= self.min_beats:
                    duration = current_episode[-1][1] - current_episode[0][1]
                    if duration >= self.min_duration:
                        episodes.append(current_episode)
                current_episode = []
        
        return episodes
```

### 4.2 Signal Quality Check

```python
def is_signal_quality_acceptable(segment, threshold=0.3):
    """
    Reject predictions during noisy segments to reduce false alarms
    """
    # Check for flatline
    if np.std(segment) < 0.01:
        return False
    
    # Check for excessive noise (high-frequency content)
    fft = np.fft.fft(segment)
    high_freq_power = np.sum(np.abs(fft[len(fft)//4:]))
    total_power = np.sum(np.abs(fft))
    
    if high_freq_power / total_power > threshold:
        return False  # Too noisy
    
    return True
```

### 4.3 Heart Rate Sanity Check

```python
def validate_tachycardia_detection(segment, fs=360):
    """
    Confirm HR is actually elevated before raising alarm
    Even if model predicts tachycardia, verify with simple HR check
    """
    # Detect R-peaks
    r_peaks = detect_r_peaks(segment, fs)
    
    if len(r_peaks) < 2:
        return False
    
    # Calculate heart rate
    rr_intervals = np.diff(r_peaks) / fs
    hr = 60 / np.mean(rr_intervals)
    
    # Tachycardia requires HR > 100 BPM
    return hr > 100
```

---

## Phase 5: Saliency-Based XAI (Week 8-9)

### 5.1 Replace Feature-Based with Waveform Saliency

**Current Problem:** SHAP on extracted features is cluttered and not intuitive for clinicians.

**Revised Approach:**

| Method | Use Case | Output |
|--------|----------|--------|
| **Grad-CAM** | Highlight CNN activation regions | Heatmap on ECG |
| **Integrated Gradients** | Precise attribution per sample | Continuous saliency |
| **Attention Weights** | Built-in interpretability | Attention heatmap |
| **Temporal Importance** | Show confidence over time | Rolling prediction curve |

### 5.2 Saliency Visualization

```python
def create_clinical_explanation(ecg_segment, prediction, saliency_map, hr):
    """
    Generate clinician-friendly explanation
    """
    explanation = {
        'ecg_plot': plot_ecg_with_saliency(ecg_segment, saliency_map),
        'prediction': 'Ventricular Tachycardia' if prediction == 3 else ...,
        'confidence': f'{prediction_proba:.1%}',
        'reasoning': generate_clinical_text(prediction, hr, saliency_map),
        'highlighted_region': get_peak_saliency_region(saliency_map),
        'episode_duration': calculate_episode_duration(),
        'heart_rate': f'{hr:.0f} BPM'
    }
    return explanation

def generate_clinical_text(prediction, hr, saliency_map):
    """
    Example output:
    "ALERT: Ventricular Tachycardia detected
     - 7 consecutive wide-complex beats identified
     - Heart rate: 152 BPM
     - Episode duration: 4.2 seconds
     - See highlighted region on ECG"
    """
```

### 5.3 XAI Validation Protocol

```
Validation Steps:
1. Take 50 true positive detections
2. Show ECG + saliency to cardiologist (blinded)
3. Ask: "Does the highlighted region match what you consider the arrhythmia?"
4. Compute agreement rate (target: >80%)

5. Take 50 false positives
6. Analyze saliency: Is model looking at noise? Artifacts?
7. Use insights to improve preprocessing or training

8. Ensure saliency is STABLE:
   - Same input should produce same explanation
   - Small perturbations should not drastically change explanation
```

### 5.4 Counterfactual Explanations

```python
def generate_counterfactual(tachycardia_segment):
    """
    Show: "What would need to change for this NOT to be tachycardia?"
    
    Example output:
    "This segment was classified as VT because:
     - Heart rate is 145 BPM (>100 BPM threshold)
     - QRS complexes are wide (>120ms)
     - 5 consecutive abnormal beats detected
     
     For normal classification, would need:
     - Heart rate <100 BPM, OR
     - Fewer than 3 consecutive abnormal beats"
    """
```

---

## Phase 6: Integration & Testing (Week 10)

### 6.1 End-to-End Pipeline

```
Complete Pipeline:
1. Input: Raw ECG segment (5-10 seconds)
2. Quality Check: Reject if too noisy
3. Preprocessing: Filter, normalize (using TRAINING stats only)
4. Model Inference: CNN-LSTM prediction
5. Episode Logic: Group consecutive detections
6. HR Validation: Confirm rate >100 BPM
7. Output: 
   - Alert (if episode detected)
   - Confidence score
   - Saliency visualization
   - Clinical explanation text
```

### 6.2 Performance Benchmarks

| Metric | MIT-BIH Target | External Validation Target |
|--------|----------------|---------------------------|
| VT/VFL Sensitivity | ≥98% | ≥90% |
| Overall Sensitivity | ≥95% | ≥85% |
| False Alarms/Hour | <0.5 | <2.0 |
| Episode Detection | ≥95% | ≥85% |
| PPV | ≥85% | ≥70% |

### 6.3 Failure Mode Analysis

```
Analyze Every False Negative:
- Was it a labeling error?
- Was it at segment boundary?
- Was signal quality poor?
- Was it an unusual morphology not in training?

Analyze False Positives:
- Noise-related?
- Ectopic beat misclassified?
- Sinus tachycardia false alarm?
- Model uncertainty (check confidence)?
```

---

## Revised File Structure

```
project/
├── data/
│   ├── raw/                      # Original MIT-BIH
│   ├── external/                 # PTB-XL, INCART, etc.
│   ├── processed/                # Cleaned segments
│   └── augmented/                # Augmented training data
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py        # Multi-dataset loader
│   │   ├── signal_processing.py  # Clinical-grade filtering
│   │   ├── quality_check.py      # Signal quality assessment
│   │   └── clinical_labeling.py  # Proper VT/SVT criteria
│   ├── augmentation/
│   │   ├── physiological_augment.py  # Time warp, noise, etc.
│   │   └── augmentation_pipeline.py
│   ├── models/
│   │   ├── cnn_lstm_attention.py # Hybrid model
│   │   ├── ensemble.py           # Model ensemble
│   │   └── episode_detector.py   # Consecutive detection logic
│   ├── xai/
│   │   ├── saliency_maps.py      # Grad-CAM, IntGrad
│   │   ├── attention_viz.py      # Attention visualization
│   │   └── clinical_explanations.py  # Text generation
│   ├── evaluation/
│   │   ├── clinical_metrics.py   # Sensitivity, FA/hour
│   │   ├── episode_metrics.py    # Episode-level eval
│   │   └── external_validation.py
│   └── utils/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_augmentation_validation.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_xai_analysis.ipynb
│   └── 05_clinical_evaluation.ipynb
├── models/                       # Saved models
├── results/                      # Evaluation results
└── XAI_TACHYCARDIA_REVISED_PLAN.md
```

---

## Priority Action Items

| # | Action | Week | Impact |
|---|--------|------|--------|
| 1 | Fix clinical labeling (VT ≥3 beats criteria) | 1 | 🔴 Critical |
| 2 | Implement physiological augmentation (not SMOTE) | 1 | 🔴 Critical |
| 3 | Add class weighting to loss function | 2 | 🔴 Critical |
| 4 | Build CNN-LSTM-Attention model | 3-4 | 🔴 Critical |
| 5 | Implement episode-level detection logic | 4 | 🔴 Critical |
| 6 | Add signal quality checking | 4 | 🟡 High |
| 7 | Create proper evaluation with clinical metrics | 5 | 🔴 Critical |
| 8 | External dataset validation (INCART/PTB-XL) | 6 | 🔴 Critical |
| 9 | Implement saliency-based XAI | 7-8 | 🟡 High |
| 10 | Threshold optimization for sensitivity | 8 | 🟡 High |
| 11 | False alarm reduction (consecutive detection) | 9 | 🟡 High |
| 12 | Clinical explanation generation | 9 | 🟡 High |
| 13 | End-to-end integration testing | 10 | 🟢 Medium |

---

## Key Differences from Original Plan

| Aspect | Original Plan | Revised Plan |
|--------|---------------|--------------|
| **Labeling** | HR > 100 BPM | Clinical criteria (VT ≥3 beats) |
| **Oversampling** | SMOTE | Physiological augmentation + class weights |
| **Model** | Multiple options | CNN-LSTM-Attention hybrid |
| **Input** | Beat-by-beat | Episode segments (5-10 sec) |
| **XAI** | SHAP on features | Saliency maps on ECG waveform |
| **Metrics** | F1, Accuracy | Sensitivity, FA/hour, Episode detection |
| **Validation** | MIT-BIH only | External datasets required |
| **Output** | Classification | Clinical explanation + highlighted ECG |

---

## Expected Outcomes

With these revisions:

- **Sensitivity for VT/VFL**: 95-98% (critical arrhythmias rarely missed)
- **False Alarm Rate**: <1 per hour (clinically acceptable)
- **Generalization**: <15% performance drop on external data
- **XAI Quality**: >80% agreement with cardiologist assessment
- **Clinical Readiness**: Suitable for clinical validation study

---

## References for Implementation

1. **Clinical VT Definition**: ACC/AHA Guidelines - VT = ≥3 consecutive ventricular beats @ >100 BPM
2. **CNN-LSTM for ECG**: Hannun et al., Nature Medicine 2019
3. **Attention for ECG**: Mousavi et al., "Inter- and Intra-Patient ECG Heartbeat Classification"
4. **Grad-CAM for Time Series**: Saliency methods adapted for 1D signals
5. **MIT-BIH Limitations**: Luz et al., "ECG arrhythmia classification based on optimum-path forest"
