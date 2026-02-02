# 🫀 XAI Tachycardia Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active_Development-blue?style=for-the-badge)

**An Explainable AI Framework for Real-time Tachycardia Detection from ECG Signals**

[Features](#-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Architecture](#-architecture) •
[Documentation](#-documentation)

</div>

---

## 📋 Overview

This project implements a **comprehensive explainable AI (XAI) framework** for detecting tachycardia episodes from ECG signals. Unlike traditional black-box approaches, our system provides **clinically interpretable explanations** for every detection, enabling healthcare professionals to understand and trust AI-assisted diagnoses.

### 🎯 Key Objectives

- **High Sensitivity**: Prioritize catching all life-threatening arrhythmias (VT/VFL)
- **Low False Alarm Rate**: Reduce alarm fatigue in clinical settings
- **Explainability**: SHAP-based feature importance for every prediction
- **Calibrated Uncertainty**: Know when the model is uncertain
- **Clinical Priority Tiers**: VT/VFL > SVT > Sinus Tachycardia

---

## ✨ Features

### 🔬 Core Detection Pipeline

| Feature | Description |
|---------|-------------|
| **Two-Lane Detection** | Sensitivity-first detection lane + precision-focused confirmation lane |
| **Episode-Level Analysis** | Detects complete tachycardia episodes, not just individual beats |
| **Signal Quality Gating** | 6-component SQI system with VF bypass logic |
| **Multi-Model Ensemble** | Random Forest, XGBoost, Logistic Regression, Decision Tree |

### 🧠 Explainability (XAI)

- **SHAP Analysis**: Feature importance for every prediction
- **LIME Integration**: Local interpretable model explanations
- **Clinical Feature Mapping**: Maps AI features to medical concepts
- **Uncertainty Quantification**: Temperature-scaled calibration

### 📊 Supported Datasets

| Dataset | Source | Patients | Features |
|---------|--------|----------|----------|
| **MIT-BIH** | PhysioNet | 47 | Gold standard arrhythmia annotations |
| **INCART** | PhysioNet | 75 | 12-lead ECG with V-run detection |
| **PTB-XL** | PhysioNet | 21,837 | Large-scale diagnostic ECG |
| **Chapman-Shaoxing** | PhysioNet | 10,646 | 12-lead, rhythm annotations |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Quick Install

```bash
# Clone the repository
git clone https://github.com/ORION2809/XAI_Tachycardia-.git
cd XAI_Tachycardia-

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Quick Start

### Run the Demo

```bash
# Run end-to-end demonstration
python demo_e2e.py
```

### Train Models

```bash
# Train all models on MIT-BIH data
python src/train_models.py
```

### Run Full Pipeline

```bash
# Execute complete detection pipeline
python src/run_pipeline.py
```

### Verify Implementation

```bash
# Run verification tests
python verify_implementation.py
```

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        XAI TACHYCARDIA SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Data       │───▶│  Quality     │───▶│  Feature     │               │
│  │   Loading    │    │  Assessment  │    │  Extraction  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────────────────────────────────────────────┐               │
│  │              TWO-LANE DETECTION PIPELINE              │               │
│  ├──────────────────────────────────────────────────────┤               │
│  │  Detection Lane (Sensitivity)  │  Confirmation Lane   │               │
│  │  Threshold: 0.4                │  Threshold: 0.7      │               │
│  └──────────────────────────────────────────────────────┘               │
│                          │                                               │
│                          ▼                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Episode    │───▶│   XAI        │───▶│   Alarm      │               │
│  │   Detection  │    │   Explain    │    │   System     │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
XAI_Tachycardia/
├── 📂 src/                          # Source code
│   ├── 📂 data/                     # Data loading & harmonization
│   ├── 📂 preprocessing/            # Signal preprocessing
│   ├── 📂 features/                 # Feature extraction
│   ├── 📂 quality/                  # Signal quality assessment
│   ├── 📂 detection/                # Detection pipeline
│   ├── 📂 models/                   # ML model definitions
│   ├── 📂 xai/                      # Explainability modules
│   ├── 📂 calibration/              # Uncertainty calibration
│   ├── 📂 evaluation/               # Performance metrics
│   └── 📂 augmentation/             # Data augmentation
├── 📂 tests/                        # Unit tests (83+ passing)
├── 📂 models/                       # Trained model files
├── 📂 data/                         # Processed features
├── 📂 mitbih_database/              # MIT-BIH ECG data
├── 📂 results/                      # Output results
├── 📄 requirements.txt              # Dependencies
├── 📄 demo_e2e.py                   # End-to-end demo
└── 📄 IMPLEMENTATION_SUMMARY.md     # Detailed documentation
```

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Complete implementation details |
| [BUILDABLE_SPEC.md](BUILDABLE_SPEC.md) | System specifications |
| [XAI_TACHYCARDIA_IMPLEMENTATION_PLAN.md](XAI_TACHYCARDIA_IMPLEMENTATION_PLAN.md) | Development roadmap |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

**Current Test Status**: ✅ 83 tests passing

---

## 📊 Performance Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Sensitivity (VT/VFL) | ≥95% | In development |
| PPV | ≥40% | In development |
| False Alarm Rate | <10% | In development |
| ECE (Calibration) | <0.05 | In development |

---

## 🔮 Future Roadmap

- [ ] Deep learning models (CausalGRU, Transformer)
- [ ] Real-time streaming inference
- [ ] Multi-lead ECG support
- [ ] Clinical validation study
- [ ] FHIR/HL7 integration
- [ ] Mobile/edge deployment

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [PhysioNet](https://physionet.org/) for providing ECG databases
- MIT-BIH Arrhythmia Database contributors
- The open-source ML/XAI community

---

<div align="center">

**Made with ❤️ for better cardiac care**

⭐ Star this repository if you find it useful!

</div>
