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

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| Pregnancies | Number of pregnancies | 0-17 | count |
| Glucose | Plasma glucose concentration | 0-199 | mg/dL |
| BloodPressure | Diastolic blood pressure | 0-122 | mm Hg |
| SkinThickness | Triceps skin fold thickness | 0-99 | mm |
| Insulin | 2-Hour serum insulin | 0-846 | mu U/ml |
| BMI | Body mass index | 0-67.1 | kg/m² |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0.08-2.42 | - |
| Age | Age in years | 21-81 | years |

**Code**: `Frontend/app.py` lines 95-144

### 3. Heart Disease

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| age | Patient age | 29-77 | years |
| sex | Gender | 0=Female, 1=Male | categorical |
| cp | Chest pain type | 0-3 | categorical |
| trestbps | Resting blood pressure | 94-200 | mmHg |
| chol | Serum cholesterol | 126-564 | mg/dl |
| fbs | Fasting blood sugar >120mg/dl | 0 or 1 | boolean |
| restecg | Resting ECG results | 0-2 | categorical |
| thalach | Maximum heart rate achieved | 40-202 | bpm | 
| exang | Exercise-induced angina | 0 or 1 | boolean |
| oldpeak | ST depression induced by exercise | 0-6.2 | mm |
| slope | Peak exercise ST segment | 0-2 | categorical |
| ca | Major vessels colored by fluoroscopy | 0-3 | count |
| thal | Thalassemia | 0-2 | categorical |

**Chest Pain Types (cp)**:
- 0 = typical angina
- 1 = atypical angina
- 2 = non-anginal pain
- 3 = asymptomatic

**Resting ECG (restecg)**:
- 0 = normal
- 1 = ST-T wave abnormality
- 2 = left ventricular hypertrophy

**ST Segment Slope**:
- 0 = upsloping
- 1 = flat
- 2 = downsloping

**Thalassemia (thal)**:
- 0 = normal
- 1 = fixed defect
- 2 = reversible defect

**Code**: `Frontend/app.py` lines 151-268

### 4. Parkinson's Disease

Voice/motor measurements (22 features from UCI Parkinson's dataset):

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

**Code**: `Frontend/app.py` lines 274-349

### 5. Liver Disease

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| Sex | Gender | 0=Male, 1=Female | categorical |
| age | Patient age | 4-90 | years |
| Total_Bilirubin | Total bilirubin | 0.1-75 | mg/dl |
| Direct_Bilirubin | Direct/conjugated bilirubin | 0.1-30 | mg/dl |
| Alkaline_Phosphotase | Liver enzyme | 63-211 | IU/L |
| Alamine_Aminotransferase | ALT enzyme (SGPT) | 5-100 | IU/L |
| Aspartate_Aminotransferase | AST enzyme (SGOT) | 10-100 | IU/L |
| Total_Protiens | Total protein | 3.0-9.6 | g/dl |
| Albumin | Liver protein | 1.8-5.5 | g/dl |
| Albumin_and_Globulin_Ratio | A/G ratio | 0.3-2.5 | ratio |

**Normal Reference Ranges**:
- Total Bilirubin: 0.1-1.2 mg/dl
- Direct Bilirubin: 0.0-0.3 mg/dl
- Alkaline Phosphatase: 44-147 IU/L
- ALT: 7-56 IU/L
- AST: 10-40 IU/L
- Total Protein: 6.0-8.3 g/dl
- Albumin: 3.5-5.0 g/dl
- A/G Ratio: 0.8-2.0

**Code**: `Frontend/app.py` lines 457-513

### 6. Hepatitis

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| Age | Patient age | 7-78 | years |
| Sex | Gender | 1=Male, 2=Female | categorical |
| Total_Bilirubin | Total bilirubin | 0.4-30 | mg/dl |
| Direct_Bilirubin | Direct bilirubin | 0.1-15 | mg/dl |
| Alkaline_Phosphatase | ALP enzyme | 52-280 | IU/L |
| Alamine_Aminotransferase | ALT enzyme (SGPT) | 10-200 | IU/L |
| Aspartate_Aminotransferase | AST enzyme (SGOT) | 15-250 | IU/L |
| Total_Proteins | Total proteins | 3.5-9.0 | g/dl |
| Albumin | Albumin | 2.0-5.5 | g/dl |
| Albumin_and_Globulin_Ratio | A/G ratio | 0.5-2.5 | ratio |
| GGT | Gamma-Glutamyl Transferase | 10-120 | IU/L |
| PROT | Total Proteins | 3.5-9.0 | g/dl |

**Normal Reference Ranges**:
- Total Bilirubin: 0.1-1.2 mg/dl
- Direct Bilirubin: 0.0-0.3 mg/dl
- Alkaline Phosphatase: 44-147 IU/L
- ALT: 7-56 IU/L
- AST: 10-40 IU/L
- GGT: 9-48 IU/L
- Total Protein: 6.0-8.3 g/dl
- Albumin: 3.5-5.0 g/dl
- A/G Ratio: 0.8-2.0

**Code**: `Frontend/app.py` lines 520-594

### 7. Lung Cancer

| Feature | Description | Range | Unit |
|---------|-------------|-------|------|
| Gender | Gender | Male/Female | categorical |
| Age | Patient age | 15-80 | years |
| Smoking | Smoking status | YES/NO | categorical |
| Yellow_Fingers | Yellow fingers | YES/NO | categorical |
| Anxiety | Anxiety | YES/NO | categorical |
| Peer_Pressure | Peer pressure | YES/NO | categorical |
| Chronic_Disease | Chronic disease | YES/NO | categorical |
| Fatigue | Fatigue | YES/NO | categorical |
| Allergy | Allergy | YES/NO | categorical |
| Wheezing | Wheezing | YES/NO | categorical |
| Alcohol_Consuming | Alcohol consuming | YES/NO | categorical |
| Coughing | Coughing | YES/NO | categorical |
| Shortness_of_Breath | Shortness of breath | YES/NO | categorical |
| Swallowing_Difficulty | Swallowing difficulty | YES/NO | categorical |
| Chest_Pain | Chest pain | YES/NO | categorical |

**Code**: `Frontend/app.py` lines 359-452

---

## Algorithms Used Summary

| Disease | Algorithm | Notes |
|---------|-----------|-------|
| General Disease | XGBoost | 41 diseases, symptom-based |
| Diabetes | SVC | See DIABETES_PIPELINE.md for details |
| Heart Disease | Logistic Regression | Linearly separable data |
| Parkinson's | SVC | Works well with voice features |
| Liver Disease | Logistic Regression | Linearly separable data |
| Hepatitis | Random Forest | Ensemble of decision trees |
| Lung Cancer | Logistic Regression | Linearly separable data |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
cd Frontend
streamlit run app.py
```

---

## Related Documentation

- [Diabetes ML Pipeline](DIABETES_PIPELINE.md)
- [General Disease Prediction](GENERAL_DISEASE_PREDICTION.md)
