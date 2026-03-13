# Multiple Disease Prediction Webapp - Documentation

## Project Overview

A Streamlit-based web application that predicts multiple diseases using machine learning. Users can input symptoms or medical test results to get disease predictions.

---

## Project Structure

```
Multiple-Disease-Prediction-Webapp/
├── Frontend/
│   ├── app.py                    # Main Streamlit app
│   ├── code/
│   │   ├── DiseaseModel.py       # General disease prediction (XGBoost)
│   │   ├── helper.py             # Symptom array preparation
│   │   └── train.py              # Training script
│   ├── model/
│   │   └── xgboost_model.json    # XGBoost model (41 diseases)
│   ├── models/                   # Pre-trained .sav files
│   │   ├── diabetes_model.sav
│   │   ├── heart_disease_model.sav
│   │   ├── parkinsons_model.sav
│   │   ├── liver_model.sav
│   │   ├── hepatitis_model.sav
│   │   └── lung_cancer_model.sav
│   └── data/                     # Datasets
│       ├── dataset.csv
│       ├── clean_dataset.tsv
│       ├── symptom_Description.csv
│       ├── symptom_precaution.csv
│       └── lung_cancer.csv
├── code/
│   └── PIMA/                     # Diabetes ML pipeline
│       ├── training.py
│       ├── models.py
│       ├── config.yml
│       └── artifacts/
├── DOCUMENTATION.md              
├── DIABETES_PIPELINE.md         # Diabetes ML pipeline docs
└── GENERAL_DISEASE_PREDICTION.md # General disease prediction docs
```

---

## Disease Predictions Available

| # | Disease | Input Type | Model File | Algorithm |
|---|---------|------------|------------|-----------|
| 1 | **General Disease** | Symptoms (multi-select) | `model/xgboost_model.json` | XGBoost |
| 2 | **Diabetes** | 8 medical features | `models/diabetes_model.sav` | SVC |
| 3 | **Heart Disease** | 13 medical features | `models/heart_disease_model.sav` | Logistic Regression |
| 4 | **Parkinson's** | 22 voice/motor features | `models/parkinsons_model.sav` | SVC |
| 5 | **Liver Disease** | 10 liver function tests | `models/liver_model.sav` | Logistic Regression |
| 6 | **Hepatitis** | 12 liver function tests | `models/hepititisc_model.sav` | Random Forest |
| 7 | **Lung Cancer** | 15 lifestyle/symptoms | `models/lung_cancer_model.sav` | Logistic Regression |

---

## Input Features by Disease

### 1. General Disease (XGBoost)
- **Input**: Select symptoms from 133 possible symptoms
- **Predicts**: 41 different diseases based on symptoms
- **Code**: `Frontend/code/DiseaseModel.py`, `Frontend/code/helper.py`

### 2. Diabetes
| Feature | Description |
|---------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-Hour serum insulin (mu U/ml) |
| BMI | Body mass index (weight in kg / height in m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |

**Code**: `Frontend/app.py` lines 95-144

### 3. Heart Disease
| Feature | Description |
|---------|-------------|
| age | Patient age |
| sex | Gender (1=Male, 0=Female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar >120mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia |

**Code**: `Frontend/app.py` lines 151-268

### 4. Parkinson's Disease
Voice/motor measurements (22 features from UCI Parkinson's dataset):
- MDVP:Fo(Hz) - Average fundamental frequency
- MDVP:Fhi(Hz) - Maximum fundamental frequency
- MDVP:Flo(Hz) - Minimum fundamental frequency
- MDVP:Jitter(%) - Frequency variation
- MDVP:Jitter(Abs) - Absolute jitter
- MDVP:RAP - Relative amplitude perturbation
- MDVP:PPQ - Pitch period perturbation quotient
- Jitter:DDP - Jitter difference
- MDVP:Shimmer - Amplitude variation
- MDVP:Shimmer(dB) - Shimmer in dB
- Shimmer:APQ3 - 3-point amplitude perturbation quotient
- Shimmer:APQ5 - 5-point amplitude perturbation quotient
- MDVP:APQ - Amplitude perturbation quotient
- Shimmer:DDA - Differential asymmetry
- NHR - Noise-to-Harmonics Ratio
- HNR - Harmonics-to-Noise Ratio
- RPDE - Recurrence Period Density Entropy
- DFA - Detrended Fluctuation Analysis
- spread1, spread2 - Non-linear measures
- D2 - Correlation dimension
- PPE - Pitch period entropy

**Code**: `Frontend/app.py` lines 274-349

### 5. Liver Disease
| Feature | Description |
|---------|-------------|
| Sex | Gender (0=Male, 1=Female) |
| age | Patient age |
| Total_Bilirubin | Total bilirubin (mg/dl) |
| Direct_Bilirubin | Direct/conjugated bilirubin |
| Alkaline_Phosphotase | Liver enzyme (IU/L) |
| Alamine_Aminotransferase | ALT enzyme (IU/L) |
| Aspartate_Aminotransferase | AST enzyme (IU/L) |
| Total_Protiens | Total protein (g/dl) |
| Albumin | Liver protein (g/dl) |
| Albumin_and_Globulin_Ratio | A/G ratio |

**Code**: `Frontend/app.py` lines 457-513

### 6. Hepatitis
| Feature | Description |
|---------|-------------|
| Age | Patient age |
| Sex | Gender (1=Male, 2=Female) |
| Total_Bilirubin | Total bilirubin |
| Direct_Bilirubin | Direct bilirubin |
| Alkaline_Phosphatase | ALP enzyme |
| Alamine_Aminotransferase | ALT enzyme |
| Aspartate_Aminotransferase | AST enzyme |
| Total_Proteins | Total proteins |
| Albumin | Albumin |
| Albumin_and_Globulin_Ratio | A/G ratio |
| GGT | Gamma-Glutamyl Transferase |
| PROT | Total Proteins |

**Code**: `Frontend/app.py` lines 520-594

### 7. Lung Cancer
| Feature | Description |
|---------|-------------|
| Gender | Male/Female |
| Age | Patient age |
| Smoking | Yes/No |
| Yellow_Fingers | Yes/No |
| Anxiety | Yes/No |
| Peer_Pressure | Yes/No |
| Chronic_Disease | Yes/No |
| Fatigue | Yes/No |
| Allergy | Yes/No |
| Wheezing | Yes/No |
| Alcohol_Consuming | Yes/No |
| Coughing | Yes/No |
| Shortness_of_Breath | Yes/No |
| Swallowing_Difficulty | Yes/No |
| Chest_Pain | Yes/No |

**Code**: `Frontend/app.py` lines 359-452

---

## Algorithms Used Summary

| Disease | Algorithm | Notes |
|---------|-----------|-------|
| General Disease | XGBoost | 41 diseases, symptom-based |
| Diabetes | SVC | See DIABETES_PIPELINE.md for details |
| Heart Disease | Logistic Regression | Linearly separable data |
| Parkinson's | SVC | 
| Liver Disease | Logistic Regression | Linearly separable data |
| Hepatitis | Random Forest | Ensemble of decision trees |
| Lung Cancer | Logistic Regression | Linearly separable data |

---


