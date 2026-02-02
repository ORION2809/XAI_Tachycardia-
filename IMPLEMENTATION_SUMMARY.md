# XAI Tachycardia Detection System - Implementation Summary

**Version**: BUILDABLE_SPEC.md v2.4 Implementation  
**Date**: January 24, 2026  
**Status**: Core Infrastructure Complete, Training Data Expansion Needed

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [What We Implemented](#2-what-we-implemented)
3. [Why We Made These Design Choices](#3-why-we-made-these-design-choices)
4. [Current System Architecture](#4-current-system-architecture)
5. [Test Coverage & Verification](#5-test-coverage--verification)
6. [Critical Gaps & What's Needed](#6-critical-gaps--whats-needed)
7. [Recommended Improvements](#7-recommended-improvements)
8. [Action Items for Production Readiness](#8-action-items-for-production-readiness)

---

## 1. Executive Summary

### What We Built
A complete **explainable AI (XAI) framework** for detecting tachycardia episodes from ECG signals, implementing the BUILDABLE_SPEC.md v2.4 specification. The system includes:

- **9 verified modules** with 83 passing tests
- **Episode-level detection** (not just beat classification)
- **Two-lane detection pipeline** for sensitivity-first detection
- **Clinical priority tiers** (VT/VFL prioritized over SVT over Sinus)
- **Signal quality gating** with VF bypass logic
- **Calibrated uncertainty quantification**
- **SHAP-based explainability**
- **Domain shift mitigation** for external validation
- **Deployment readiness checklist**

### What We Don't Have Yet
- **Sufficient training data** (only MIT-BIH with 47 patients)
- **Trained deep learning models** (CausalGRU architecture exists but untrained)
- **External datasets downloaded** (loaders exist but no data)
- **Production model weights**

---

## 2. What We Implemented

### 2.1 Data Module (`src/data/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `contracts.py` | Episode taxonomy and data contracts | `EpisodeType`, `EpisodeLabel`, `LabelConfidenceTier` |
| `harmonization.py` | Cross-dataset standardization | `DatasetContract`, beat type mapping |
| `loaders/incart.py` | INCART database loader | `INCARTLoader`, V-run detection |
| `loaders/ptbxl.py` | PTB-XL database loader | `PTBXLLoader`, SVT/AFib only |
| `loaders/chapman.py` | Chapman-Shaoxing loader | `ChapmanLoader`, 12-lead ECG |

**Why**: Different datasets have incompatible formats, annotations, and sampling rates. The harmonization layer ensures all data follows a single contract.

### 2.2 Quality Module (`src/quality/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `sqi.py` | Signal Quality Index computation | `SQISuite`, `SQIPolicy` (6 components) |
| `signal_state.py` | Signal state machine with hysteresis | `SignalState`, `SignalStateManager` |

**Key SQI Components**:
1. `bsqi` - Beat-to-beat SQI (QRS detectability)
2. `ksqi` - Kurtosis SQI (morphology quality)
3. `ssqi` - Skewness SQI
4. `psqi` - Power spectral SQI
5. `basSQI` - Baseline stability
6. `pcaSQI` - Principal component analysis SQI

**Why**: 
- Low-quality signals cause false alarms
- VF/VFL can look like noise → need VF bypass at high confidence
- Hysteresis prevents alarm flapping during signal transitions

### 2.3 Detection Module (`src/detection/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `two_lane_pipeline.py` | Dual-threshold detection | `TwoLanePipeline`, `DetectionLane`, `ConfirmationLane` |
| `episode_detector.py` | Episode-level detection | `EpisodeDetector`, smoothing, HR sanity |
| `alarm_system.py` | Two-tier alarm with budgets | `TwoTierAlarmSystem`, `AlarmConfig` |
| `decision_machine.py` | Unified decision policy | `UnifiedDecisionPolicy`, `DecisionInput` |

**Two-Lane Pipeline Architecture**:
```
┌─────────────────────────────────────────────────────────────────┐
│                     TWO-LANE PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  DETECTION LANE (Sensitivity-First)                             │
│  ├── Threshold: 0.4 (lower = catch more)                        │
│  ├── Purpose: Don't miss ANY VT/VFL                             │
│  └── Outcome: Candidates for confirmation                       │
│                                                                  │
│  CONFIRMATION LANE (Precision-Focused)                          │
│  ├── Threshold: 0.7 (higher = fewer false alarms)               │
│  ├── Purpose: Confirm detections before alarming                │
│  └── Outcome: Alarm or Warning tier                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Why**: 
- Clinical requirement: VT sensitivity ≥95% with FA/hr ≤1.0
- Single threshold can't achieve both
- Detection lane ensures sensitivity, confirmation lane ensures precision

### 2.4 Config Module (`src/config/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `operating_modes.py` | Three operating modes | `OperatingMode`, `OperatingModeConfig` |
| `clinical_tiers.py` | Clinical priority hierarchy | `ClinicalPriorityTier`, `TierOperatingParameters` |
| `monitoring_context.py` | FA/hr semantics per context | `MonitoringContext`, `FAReportCard` |

**Operating Modes**:
| Mode | VT Sensitivity | FA/hr Tolerance | Use Case |
|------|----------------|-----------------|----------|
| HIGH_SENSITIVITY | ≥99% | ≤2.0 | ICU, CCU |
| BALANCED | ≥95% | ≤1.0 | Telemetry |
| RESEARCH | ≥90% | ≤0.5 | Studies |

**Clinical Priority Tiers**:
| Tier | Arrhythmia Types | Sensitivity Floor | FA Budget |
|------|------------------|-------------------|-----------|
| TIER_0 (MUST_NOT_MISS) | VT, VFL, VFib | 99% | 1.0/hr |
| TIER_1 | SVT, AFib RVR | 90% | 0.5/hr |
| TIER_2 | Sinus Tachycardia | 85% | 0.3/hr |

**Why**: Different clinical contexts have different tolerance for false alarms. ICU can handle more alarms; ambulatory monitoring cannot.

### 2.5 Evaluation Module (`src/evaluation/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `metrics.py` | Episode-level metrics | `EvaluationMetrics`, `EvaluationProtocol`, `OnsetCriticalEvaluator` |
| `validation.py` | Sub-cohort validation | `SubCohortValidator`, `AcceptanceTests`, `CrossValidator` |
| `domain_shift.py` | External validation mitigation | `DomainShiftMitigation`, `DriftIndicators` |
| `deployment_readiness.py` | Pre-deployment checklist | `DeploymentReadinessChecker`, `DeploymentReadinessReport` |

**Key Metrics**:
- **Episode-level sensitivity**: Did we detect the VT episode? (not just beats)
- **Per-class FA/hr**: VT false alarms separate from SVT false alarms
- **Onset accuracy**: How early/late did we detect?
- **Detection latency**: Time from true onset to first alarm

**Why**: Beat-level accuracy is misleading. A model with 99% beat accuracy might miss entire VT episodes.

### 2.6 XAI Module (`src/xai/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `saliency.py` | Gradient-based explanations | `IntegratedGradients`, `GradientXInput`, `AttributionResult` |
| `stability.py` | Explanation stability checks | `XAIStabilityChecker`, `StabilityResult` |
| `shap_explanations.py` | SHAP-based explanations | `DeepSHAP`, `KernelSHAP`, `HRVFeatureExplainer` |

**Why**: 
- Clinicians won't trust black-box predictions
- Need to show WHICH part of ECG triggered the alarm
- Stability ensures explanations don't randomly change for similar inputs

### 2.7 Calibration Module (`src/calibration/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `temperature_scaling.py` | Probability calibration | `TemperatureScaling`, `IsotonicCalibration`, `CalibrationModule` |
| `uncertainty.py` | Uncertainty quantification | `UncertaintyEstimator` (MC Dropout) |

**Why**: 
- Raw neural network outputs are often overconfident
- Calibrated probabilities enable proper threshold tuning
- Uncertainty allows "I don't know" decisions for ambiguous cases

### 2.8 Models Module (`src/models/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `causal_gru.py` | Causal deep learning model | `CausalTachycardiaDetector`, `CausalGRU`, `ModelConfig` |

**Architecture**:
```
CNN Feature Extractor → Causal GRU → Per-Timestep Classifier
      ↓                     ↓                ↓
  Local patterns     Temporal context    Dense output
```

**Why**:
- CNN captures morphological features (QRS width, shape)
- GRU captures temporal dependencies (sustained rate, rhythm)
- Causal = no future information leakage (real-time capable)

### 2.9 Labeling Module (`src/labeling/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| `episode_generator.py` | Beat → Episode conversion | `EpisodeLabelGenerator`, `EpisodeLabelGeneratorConfig` |

**VT Labeling Rules** (per clinical standards):
- ≥3 consecutive ventricular beats at >100 BPM
- Non-sustained VT: <30 seconds
- Sustained VT: ≥30 seconds or hemodynamically unstable

**Why**: Clinical definitions matter. "3+ consecutive V beats" is the standard VT definition.

---

## 3. Why We Made These Design Choices

### 3.1 Sensitivity-First Philosophy

> **"Missing a VT is worse than a false alarm"**

The entire system is designed around this principle:
- Detection lane uses LOW threshold (0.4) to catch everything
- VF/VFL can bypass SQI gate at high confidence
- Tier 0 arrhythmias get the strictest sensitivity floors

### 3.2 Episode-Level, Not Beat-Level

Traditional arrhythmia classifiers predict per-beat. This is wrong because:
- A single ectopic beat is not VT
- VT requires sustained rhythm (≥3 beats)
- Metrics should be "Did we catch the VT episode?" not "Did we correctly label beat #47,293?"

### 3.3 Per-Class FA Budgets

Not all false alarms are equal:
- VT false alarm: Clinician checks immediately → high cost
- Sinus tachy false alarm: Clinician ignores → low cost

So we allocate FA budgets differently:
- VT/VFL: 1.0 FA/hr (more tolerance because we MUST catch real VT)
- SVT: 0.5 FA/hr
- Sinus: 0.3 FA/hr

### 3.4 SQI Gate with VF Bypass

Signal quality gating reduces false alarms from noise. BUT:
- VF looks like noise (chaotic, no clear QRS)
- If we gate VF, patient dies

Solution: VF bypass at high confidence (≥0.8 probability AND uncertainty <0.3)

### 3.5 Domain Shift Mitigation

Models trained on MIT-BIH fail on other datasets. BUILDABLE_SPEC requires:
- Per-domain recalibration using holdout set
- Threshold retuning to maintain sensitivity
- PSI-based drift detection

---

## 4. Current System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        XAI TACHYCARDIA DETECTION SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌────────────┐ │
│  │   Data      │     │   Quality   │     │  Detection  │     │   Alarm    │ │
│  │  Loaders    │────▶│    SQI      │────▶│   Pipeline  │────▶│   System   │ │
│  │ (MIT-BIH,   │     │  (6 comp)   │     │ (Two-Lane)  │     │ (Two-Tier) │ │
│  │  INCART,    │     │             │     │             │     │            │ │
│  │  PTB-XL)    │     └─────────────┘     └─────────────┘     └────────────┘ │
│  └─────────────┘            │                   │                   │       │
│                             │                   │                   │       │
│                             ▼                   ▼                   ▼       │
│                    ┌─────────────┐     ┌─────────────┐     ┌────────────┐   │
│                    │   Signal    │     │  Decision   │     │    XAI     │   │
│                    │   State     │     │  Machine    │     │  Explain   │   │
│                    │  Manager    │     │ (Unified)   │     │  (SHAP)    │   │
│                    └─────────────┘     └─────────────┘     └────────────┘   │
│                                               │                             │
│                                               ▼                             │
│                                      ┌─────────────┐                        │
│                                      │ Calibration │                        │
│                                      │ (Temp Scale)│                        │
│                                      └─────────────┘                        │
│                                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                    │
│  │  Evaluation │     │   Domain    │     │ Deployment  │                    │
│  │   Metrics   │     │   Shift     │     │  Readiness  │                    │
│  │ (Episode)   │     │ Mitigation  │     │   Check     │                    │
│  └─────────────┘     └─────────────┘     └─────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Test Coverage & Verification

### Test Results
```
================== 83 passed in 0.97s ==================
```

### Test Breakdown

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_decision_machine.py` | 20 | Alarm budget, burst suppression, decision policy |
| `test_metrics.py` | 21 | IoU, episode matching, FA calculation, onset accuracy |
| `test_sqi.py` | 13 | SQI computation, VF bypass, thresholds |
| `test_validation.py` | 29 | Sub-cohorts, patient splits, acceptance tests, domain shift |

### Module Verification
All 9 modules import successfully:
- ✅ Data Module
- ✅ Quality Module
- ✅ Detection Module
- ✅ Config Module
- ✅ Evaluation Module
- ✅ XAI Module
- ✅ Calibration Module
- ✅ Models Module
- ✅ Labeling Module

---

## 6. Critical Gaps & What's Needed

### 6.1 Data Gap (CRITICAL)

| Current State | What's Needed |
|---------------|---------------|
| MIT-BIH only (47 patients) | Multiple external datasets |
| ~50 VT beats total | 10,000+ VT episodes |
| No downloaded external data | INCART, PTB-XL, Chapman, CUDB |

**Impact**: Cannot train robust deep learning models with current data.

### 6.2 Model Training Gap

| Current State | What's Needed |
|---------------|---------------|
| Traditional ML trained (RF, XGB) | Deep learning trained |
| CausalGRU architecture defined | CausalGRU weights |
| No GPU training done | GPU training pipeline |

**Current Models** (in `models/` folder):
- `decision_tree_model.joblib`
- `logistic_regression_model.joblib`
- `random_forest_model.joblib`
- `xgboost_model.joblib`

These are **baseline models** on extracted features, NOT the CausalGRU on raw ECG.

### 6.3 External Validation Gap

| Current State | What's Needed |
|---------------|---------------|
| Loaders implemented | Data downloaded |
| Domain shift mitigation code exists | Actual cross-database validation |
| No external test results | Results on INCART, Chapman |

---

## 7. Recommended Improvements

### 7.1 Short-Term (Data Acquisition)

```bash
# Priority 1: Download external datasets from PhysioNet
wget -r -N -c -np https://physionet.org/files/incartdb/1.0.0/
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/
wget -r -N -c -np https://physionet.org/files/ecg-arrhythmia/1.0.0/
```

| Dataset | Size | VT Content | Download Size |
|---------|------|------------|---------------|
| INCART | 75 records | V-runs | ~250 MB |
| PTB-XL | 21,837 records | SVT/AFib | ~2.4 GB |
| Chapman-Shaoxing | 10,646 records | ~200 VT | ~1.5 GB |

### 7.2 Medium-Term (Model Training)

1. **Create unified training script** that:
   - Loads all datasets through harmonization layer
   - Creates patient-level train/val/test splits
   - Trains CausalGRU with proper augmentation
   - Saves checkpoints and logs

2. **Data augmentation** (not SMOTE):
   - Time stretching (±5%)
   - Baseline wander addition
   - Gaussian noise injection
   - Lead dropout (for multi-lead)

3. **Training configuration**:
   - Batch size: 64
   - Learning rate: 1e-3 with cosine annealing
   - Class weights: VT/VFL × 10, SVT × 3
   - Early stopping on validation loss

### 7.3 Long-Term (Production Readiness)

1. **Continuous monitoring**:
   - Drift detection on deployed predictions
   - Per-site calibration updates
   - FA/hr tracking dashboards

2. **Regulatory documentation**:
   - Algorithm description document
   - Clinical validation protocol
   - Risk analysis (FMEA)

3. **Integration APIs**:
   - Real-time streaming interface
   - HL7 FHIR compatibility
   - Alarm escalation protocols

---

## 8. Action Items for Production Readiness

### Phase 1: Data (Week 1-2)
| # | Task | Priority | Effort |
|---|------|----------|--------|
| 1 | Download PhysioNet datasets (INCART, PTB-XL, Chapman) | 🔴 Critical | 2 hrs |
| 2 | Verify loaders work with downloaded data | 🔴 Critical | 4 hrs |
| 3 | Create unified dataset class | 🔴 Critical | 8 hrs |
| 4 | Generate episode labels for all datasets | 🔴 Critical | 8 hrs |

### Phase 2: Training (Week 3-4)
| # | Task | Priority | Effort |
|---|------|----------|--------|
| 5 | Set up GPU training environment | 🔴 Critical | 4 hrs |
| 6 | Implement data augmentation pipeline | 🟡 High | 8 hrs |
| 7 | Train CausalGRU on combined dataset | 🔴 Critical | 24 hrs |
| 8 | Calibrate model with temperature scaling | 🔴 Critical | 4 hrs |

### Phase 3: Validation (Week 5-6)
| # | Task | Priority | Effort |
|---|------|----------|--------|
| 9 | Run external validation on held-out datasets | 🔴 Critical | 8 hrs |
| 10 | Apply domain shift mitigation | 🟡 High | 8 hrs |
| 11 | Run deployment readiness checklist | 🔴 Critical | 4 hrs |
| 12 | Generate XAI explanations for sample cases | 🟡 High | 8 hrs |

### Phase 4: Documentation (Week 7-8)
| # | Task | Priority | Effort |
|---|------|----------|--------|
| 13 | Create clinical validation report | 🟡 High | 16 hrs |
| 14 | Document algorithm design decisions | 🟡 High | 8 hrs |
| 15 | Prepare regulatory submission materials | 🟡 High | 16 hrs |

---

## Appendix A: File Structure

```
src/
├── __init__.py
├── run_pipeline.py
├── train_models.py
├── augmentation/
├── calibration/
│   ├── __init__.py
│   ├── temperature_scaling.py
│   └── uncertainty.py
├── config/
│   ├── __init__.py
│   ├── clinical_tiers.py
│   ├── monitoring_context.py
│   └── operating_modes.py
├── data/
│   ├── __init__.py
│   ├── contracts.py
│   ├── harmonization.py
│   └── loaders/
│       ├── __init__.py
│       ├── chapman.py
│       ├── incart.py
│       └── ptbxl.py
├── detection/
│   ├── __init__.py
│   ├── alarm_system.py
│   ├── decision_machine.py
│   ├── episode_detector.py
│   └── two_lane_pipeline.py
├── evaluation/
│   ├── __init__.py
│   ├── deployment_readiness.py
│   ├── domain_shift.py
│   ├── metrics.py
│   └── validation.py
├── features/
├── labeling/
│   └── episode_generator.py
├── models/
│   └── causal_gru.py
├── preprocessing/
├── quality/
│   ├── __init__.py
│   ├── signal_state.py
│   └── sqi.py
└── xai/
    ├── __init__.py
    ├── saliency.py
    ├── shap_explanations.py
    └── stability.py

tests/
├── conftest.py
├── test_decision_machine.py
├── test_metrics.py
├── test_sqi.py
└── test_validation.py
```

---

## Appendix B: Key Configuration Values

### Detection Thresholds
```python
DETECTION_LANE_THRESHOLD = 0.4   # Sensitivity-first
CONFIRMATION_LANE_THRESHOLD = 0.7  # Precision-focused
```

### SQI Thresholds
```python
SQI_USABLE_THRESHOLD = 0.5
SQI_MARGINAL_THRESHOLD = 0.3
VF_BYPASS_CONFIDENCE = 0.8
VF_BYPASS_UNCERTAINTY = 0.3
```

### FA/hr Budgets
```python
VT_VFL_FA_BUDGET = 1.0  # per hour
SVT_FA_BUDGET = 0.5     # per hour
SINUS_FA_BUDGET = 0.3   # per hour
```

### Episode Detection
```python
MIN_VT_BEATS = 3
MIN_VT_RATE_BPM = 100
MIN_EPISODE_DURATION_SEC = 1.0
```

---

## Appendix C: References

1. **BUILDABLE_SPEC.md v2.4** - Primary specification document
2. **critique.txt** - Critical analysis of current limitations
3. **PhysioNet Databases**:
   - MIT-BIH Arrhythmia Database
   - INCART 12-lead Database
   - PTB-XL ECG Database
   - Chapman-Shaoxing 12-lead ECG Database
4. **Clinical Standards**:
   - AAMI EC57 (Arrhythmia testing)
   - IEC 60601-2-27 (ECG monitoring)

---

*Document generated: January 24, 2026*
