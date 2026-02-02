# XAI for Tachycardia Detection - Implementation Plan

## Project Overview
Building an Explainable AI (XAI) system for tachycardia detection using the MIT-BIH Arrhythmia Database.

---

## Dataset Summary

### Structure
| Aspect | Details |
|--------|---------|
| **Records** | 48 patient recordings |
| **ECG Leads** | 2 channels (MLII and V5) |
| **Sampling Rate** | 360 Hz |
| **Duration per Record** | ~30 minutes (650,000 samples) |
| **Total Samples** | ~31.2 million samples |

### Beat-Level Annotations (112,210 total beats)
| Symbol | Count | Description |
|--------|-------|-------------|
| N | 75,052 | Normal beat |
| L | 8,075 | Left bundle branch block |
| R | 7,259 | Right bundle branch block |
| V | 7,130 | Premature ventricular contraction |
| / | 7,028 | Paced beat |
| A | 2,696 | Atrial premature beat |
| f | 1,785 | Fusion of paced and normal |
| ! | 472 | Ventricular flutter wave |
| j | 312 | Nodal (junctional) escape |
| E | 122 | Ventricular escape |

### Tachycardia-Relevant Rhythm Annotations
| Rhythm | Episodes | Clinical Significance |
|--------|----------|----------------------|
| **VT** | 61 | Ventricular Tachycardia ⚠️ |
| **VFL** | 6 | Ventricular Flutter ⚠️ |
| **SVTA** | 26 | Supraventricular Tachyarrhythmia ⚠️ |
| **T** | 83 | Sinus Tachycardia ⚠️ |
| **AFL** | 45 | Atrial Flutter |
| **AFIB** | 107 | Atrial Fibrillation |

---

## Dataset Quality Assessment

### Strengths
- ✅ Gold-standard expert annotations
- ✅ Consistent data (all records have 650,001 samples)
- ✅ Two-lead redundancy (MLII and V5)
- ✅ Rich rhythm variety
- ✅ High sampling rate (360 Hz)

### Challenges
| Issue | Impact | Severity |
|-------|--------|----------|
| Class Imbalance | Normal (67%) vs Tachycardia (~0.15%) | Critical |
| Limited Tachycardia Episodes | Only 176 rhythm changes | Critical |
| Small Patient Pool | 48 patients | Moderate |
| Signal Noise | Baseline wander, artifacts | Moderate |

---

## Implementation Phases

### Phase 1: Data Pipeline (Week 1-2) ✅ COMPLETE
- [x] Load and parse all 48 records
- [x] Implement signal preprocessing (filtering, normalization)
- [x] R-peak detection and beat segmentation
- [x] Feature extraction (62 features: temporal, morphological, statistical, frequency, wavelet)
- [x] Create unified dataset with tachycardia labels
- [x] Train/test split (patient-wise to avoid data leakage)

#### Phase 1 Results:
| Metric | Value |
|--------|-------|
| Total Beats | 110,872 |
| Normal Beats | 106,743 (96.3%) |
| Tachycardia Beats | 4,129 (3.7%) |
| Class Imbalance | 25.9:1 |
| Features Extracted | 62 |
| Beat Window | 216 samples (600ms) |

| Tachycardia Type | Count |
|------------------|-------|
| Sinus Tachycardia | 1,370 |
| Atrial Flutter | 1,416 |
| SVT | 469 |
| VT | 402 |
| VFL | 472 |

### Phase 2: Baseline Models (Week 3-4)
- [ ] Train traditional ML models (Random Forest, XGBoost, SVM)
- [ ] Implement 1D-CNN baseline
- [ ] LSTM for sequence modeling
- [ ] Evaluate with stratified k-fold cross-validation
- [ ] Hyperparameter tuning

### Phase 3: XAI Integration (Week 5-6)
- [ ] Implement SHAP explanations
- [ ] Add LIME for local explanations
- [ ] Create attention-based neural network
- [ ] Develop Grad-CAM visualization
- [ ] Prototype-based explanations

### Phase 4: Explanation Quality (Week 7-8)
- [ ] Generate natural language explanations
- [ ] Create clinical decision support interface
- [ ] Validate explanations with domain knowledge
- [ ] Implement counterfactual explanations

### Phase 5: Deployment & Evaluation (Week 9-10)
- [ ] Model optimization and compression
- [ ] Real-time inference pipeline
- [ ] Comprehensive evaluation metrics
- [ ] API development
- [ ] Documentation

---

## XAI Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    XAI TACHYCARDIA DETECTION                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw ECG ──▶ Preprocessing ──▶ Feature Extraction               │
│                                       │                         │
│         ┌─────────────────────────────┴─────────────────┐      │
│         ▼                                               ▼      │
│  ┌──────────────────┐                    ┌──────────────────┐  │
│  │ Interpretable    │                    │ Deep Learning    │  │
│  │ Models           │                    │ + Post-hoc XAI   │  │
│  │ (RF, XGBoost)    │                    │ (CNN, LSTM)      │  │
│  └────────┬─────────┘                    └────────┬─────────┘  │
│           │                                       │            │
│           └───────────────────┬───────────────────┘            │
│                               ▼                                │
│                    ┌──────────────────┐                        │
│                    │ Ensemble + XAI   │                        │
│                    │ Explanations     │                        │
│                    └────────┬─────────┘                        │
│                             ▼                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ OUTPUT: Prediction + Confidence + Feature Importance +   │  │
│  │         ECG Highlighting + Natural Language Explanation  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tachycardia Labeling Strategy

### Definition
- **Tachycardia**: Heart rate > 100 bpm (RR interval < 600ms at 360Hz)
- **Pathological Tachycardia**: VT, VFL, SVTA rhythm episodes

### Classification Levels
```
Level 1: Binary
├── Normal
└── Tachycardia

Level 2: Multi-class
├── Sinus Tachycardia (T)
├── Supraventricular Tachycardia (SVTA)
├── Ventricular Tachycardia (VT)
└── Ventricular Flutter (VFL)
```

---

## Data Augmentation Strategy

### For Class Imbalance
1. **SMOTE** - Synthetic Minority Oversampling
2. **Time Warping** - ±10% temporal stretching
3. **Amplitude Scaling** - 0.9-1.1x scaling
4. **Noise Injection** - Realistic ECG noise
5. **GAN-based Synthesis** - Generate realistic tachycardia beats

### External Data Sources
- PTB-XL Database (21,837 records)
- PhysioNet 2017 AF Challenge
- ICBEB 2018 (6,877 12-lead ECGs)
- Chapman-Shaoxing (10,646 records)

---

## Evaluation Metrics

### Model Performance
| Metric | Target | Priority |
|--------|--------|----------|
| Sensitivity | > 95% | Critical |
| Specificity | > 90% | Important |
| F1-Score | > 0.90 | Important |
| AUC-ROC | > 0.95 | Important |

### XAI Quality
| Metric | Description |
|--------|-------------|
| Fidelity | Explanation matches model behavior |
| Consistency | Same inputs produce same explanations |
| Comprehensibility | Humans can understand |
| Clinical Validity | Makes medical sense |

---

## File Structure

```
project/
├── data/
│   ├── raw/                    # Original MIT-BIH data
│   ├── processed/              # Cleaned and segmented data
│   └── features/               # Extracted features
├── src/
│   ├── preprocessing/          # Signal processing modules
│   ├── features/               # Feature extraction
│   ├── models/                 # ML/DL models
│   ├── xai/                    # Explainability modules
│   └── utils/                  # Helper functions
├── notebooks/                  # Jupyter notebooks for analysis
├── models/                     # Saved trained models
├── results/                    # Evaluation results
└── docs/                       # Documentation
```

---

## Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
tensorflow>=2.8.0 or pytorch>=1.10.0
shap>=0.40.0
lime>=0.2.0
matplotlib>=3.4.0
seaborn>=0.11.0
biosppy>=0.8.0
neurokit2>=0.1.0
wfdb>=3.4.0
imbalanced-learn>=0.9.0
```

---

## Contact & References

### MIT-BIH Database
- Source: PhysioNet (https://physionet.org/content/mitdb/1.0.0/)
- Citation: Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. IEEE Eng in Med and Biol 20(3):45-50 (May-June 2001)

### Key Papers
1. "Explainable AI for Healthcare" - Nature Medicine 2020
2. "SHAP: A Unified Approach to Interpreting Model Predictions"
3. "Attention-based Deep Learning for ECG Classification"
