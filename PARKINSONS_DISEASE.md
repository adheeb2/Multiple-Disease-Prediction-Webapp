# Parkinson's Disease Prediction Documentation

## Overview

The Parkinson's Disease Prediction system uses an SVC (Support Vector Classifier) to predict Parkinson's disease based on 22 voice and motor feature measurements. The model analyzes voice samples and motor coordination tests to detect patterns indicative of Parkinson's disease.

**Location**: `Frontend/models/parkinsons_model.sav`

---

## Files

| File | Description |
|------|-------------|
| `models/parkinsons_model.sav` | Trained SVC model |
| `app.py` (lines 274-349) | Prediction UI and logic |

---

## Input Features

Based on UCI Parkinson's Disease Dataset (Motor & Voice Measurements):

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| MDVP:Fo(Hz) | Average fundamental frequency | 88-260 | Hz |
| MDVP:Fhi(Hz) | Maximum fundamental frequency | 102-592 | Hz |
| MDVP:Flo(Hz) | Minimum fundamental frequency | 65-299 | Hz |
| MDVP:Jitter(%) | Frequency variation | 0.001-0.05 | % |
| MDVP:Jitter(Abs) | Absolute jitter | 0.00001-0.0001 | - |
| MDVP:RAP | Relative amplitude perturbation | 0.001-0.03 | - |
| MDVP:PPQ | Pitch period perturbation quotient | 0.001-0.03 | - |
| Jitter:DDP | Jitter difference | 0.001-0.09 | - |
| MDVP:Shimmer | Amplitude variation | 0.01-0.15 | - |
| MDVP:Shimmer(dB) | Shimmer in dB | 0.09-1.0 | dB |
| Shimmer:APQ3 | 3-point amplitude perturbation quotient | 0.01-0.12 | - |
| Shimmer:APQ5 | 5-point amplitude perturbation quotient | 0.01-0.14 | - |
| MDVP:APQ | Amplitude perturbation quotient | 0.01-0.14 | - |
| Shimmer:DDA | Differential asymmetry | 0.03-0.45 | - |
| NHR | Noise-to-Harmonics Ratio | 0.01-0.35 | - |
| HNR | Harmonics-to-Noise Ratio | 8-40 | dB |
| RPDE | Recurrence Period Density Entropy | 0.1-0.7 | - |
| DFA | Detrended Fluctuation Analysis | 0.5-0.9 | - |
| spread1 | Signal irregularity measure | -2.5 to -6.5 | - |
| spread2 | Signal irregularity measure | 0.1-0.6 | - |
| D2 | Correlation dimension | 1-3.5 | - |
| PPE | Pitch period entropy | 0.05-0.45 | - |

### Feature Categories

**Frequency Features** (MDVP:Fo, Fhi, Flo):
- Measure fundamental voice frequency
- Parkinson's patients show reduced vocal frequency variation

**Jitter Features** (MDVP:Jitter, RAP, PPQ, DDP):
- Measure frequency variation/perturbation
- Higher values indicate voice instability

**Shimmer Features** (MDVP:Shimmer, APQ, DDA):
- Measure amplitude variation
- Higher values indicate voice instability

**Noise Ratios** (NHR, HNR):
- NHR: Higher in Parkinson's (more noise)
- HNR: Lower in Parkinson's (less harmonics)

**Nonlinear Dynamics** (RPDE, DFA, D2, PPE):
- Measure complexity of voice signal
- Parkinson's affects signal complexity

---

## Algorithm: SVC (Support Vector Classifier)

SVC is used because:
- Works well with high-dimensional data (22 features)
- Effective with relatively small datasets
- Good generalization capability
- Robust to outliers
- Handles non-linear relationships via kernel trick

---

## How It Works

### Model Loading

```python
import pickle

parkinsons_model = pickle.load(open('models/parkinsons_model.sav', 'rb'))
```

### Prediction in App

```python
# Collect user inputs (voice measurements)
fo = st.number_input('MDVP:Fo(Hz) - Average fundamental frequency', 88.0, 260.0, 150.0)
fhi = st.number_input('MDVP:Fhi(Hz) - Maximum fundamental frequency', 102.0, 592.0, 200.0)
flo = st.number_input('MDVP:Flo(Hz) - Minimum fundamental frequency', 65.0, 299.0, 100.0)
jitter = st.number_input('MDVP:Jitter(%) - Frequency variation', 0.001, 0.05, 0.01)
jitter_abs = st.number_input('MDVP:Jitter(Abs) - Absolute jitter', 0.00001, 0.0001, 0.00002)
rap = st.number_input('MDVP:RAP - Relative amplitude perturbation', 0.001, 0.03, 0.005)
ppq = st.number_input('MDVP:PPQ - Pitch period perturbation quotient', 0.001, 0.03, 0.005)
ddp = st.number_input('Jitter:DDP - Jitter difference', 0.001, 0.09, 0.01)
shimmer = st.number_input('MDVP:Shimmer - Amplitude variation', 0.01, 0.15, 0.03)
shimmer_db = st.number_input('MDVP:Shimmer(dB) - Shimmer in dB', 0.09, 1.0, 0.2)
apq3 = st.number_input('Shimmer:APQ3 - 3-point APQ', 0.01, 0.12, 0.02)
apq5 = st.number_input('Shimmer:APQ5 - 5-point APQ', 0.01, 0.14, 0.03)
apq = st.number_input('MDVP:APQ - Amplitude perturbation quotient', 0.01, 0.14, 0.03)
dda = st.number_input('Shimmer:DDA - Differential asymmetry', 0.03, 0.45, 0.1)
nhr = st.number_input('NHR - Noise-to-Harmonics Ratio', 0.01, 0.35, 0.1)
hnr = st.number_input('HNR - Harmonics-to-Noise Ratio', 8.0, 40.0, 20.0)
rpde = st.number_input('RPDE - Recurrence Period Density Entropy', 0.1, 0.7, 0.4)
dfa = st.number_input('DFA - Detrended Fluctuation Analysis', 0.5, 0.9, 0.7)
spread1 = st.number_input('spread1 - Signal irregularity (1)', -2.5, -6.5, -4.0)
spread2 = st.number_input('spread2 - Signal irregularity (2)', 0.1, 0.6, 0.3)
d2 = st.number_input('D2 - Correlation dimension', 1.0, 3.5, 2.0)
ppe = st.number_input('PPE - Pitch period entropy', 0.05, 0.45, 0.2)

# Create feature array
parkinsons_features = [[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp, shimmer,
                        shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa,
                        spread1, spread2, d2, ppe]]

# Predict
if st.button("Predict Parkinson's Disease"):
    prediction = parkinsons_model.predict(parkinsons_features)
    if prediction[0] == 1:
        st.error("Parkinson's Disease Detected")
    else:
        st.success("No Parkinson's Disease Detected")
```

---

## Prediction Flow

```
User inputs voice/motor measurements
        ↓
Feature normalization (if needed)
        ↓
parkinsons_model.predict() → 0 or 1
        ↓
Display prediction result
```

---

## Clinical Significance

Parkinson's disease affects:
- **Voice**: Causes vocal cord dysfunction, leading to frequency and amplitude variations
- **Motor Control**: Affects coordination, causing irregular movement patterns
- **Speech**: Results in softer, more monotone speech

The model uses voice analysis because:
- Non-invasive and easy to collect
- Early indicator of Parkinson's
- Can detect subtle changes before motor symptoms appear

---

## References

- Dataset: [UCI Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Parkinson+Disease+Voice+and+Motor+Measurements)
- Original research: Little M. et al., "Exploiting Nonlinear Recurrence and Density and Nonlinear Dimensionality Reduction for Detecting Parkinson's Disease" (IEEE 2009)
