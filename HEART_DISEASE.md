# Heart Disease Prediction Documentation

## Overview

The Heart Disease Prediction system uses a Logistic Regression classifier to predict the presence of heart disease based on 13 clinical features. It evaluates various cardiovascular indicators to provide a binary prediction (positive/negative for heart disease).

**Location**: `Frontend/models/heart_disease_model.sav`

---

## Files

| File | Description |
|------|-------------|
| `models/heart_disease_model.sav` | Trained Logistic Regression model |
| `app.py` (lines 151-268) | Prediction UI and logic |

---

## Input Features

Based on the Cleveland Heart Disease dataset:

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

### Feature Value Mappings

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

---

## Algorithm: Logistic Regression

Logistic Regression is used because:
- Works well with linearly separable data (heart disease indicators)
- Provides probability estimates alongside predictions
- Fast inference time
- Interpretable coefficients for medical use
- Handles both categorical and continuous features well

---

## How It Works

### Model Loading

```python
import pickle

heart_model = pickle.load(open('models/heart_disease_model.sav', 'rb'))
```

### Prediction in App

```python
# Collect user inputs
age = st.number_input('Age', 29, 77, 50)
sex = st.selectbox('Sex', ['Female', 'Male'])
cp = st.selectbox('Chest Pain Type', ['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic'])
trestbps = st.number_input('Resting Blood Pressure (mmHg)', 94, 200, 120)
chol = st.number_input('Serum Cholesterol (mg/dl)', 126, 564, 200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
restecg = st.selectbox('Resting ECG Results', ['normal', 'ST-T wave abnormality', 'left ventricular hypertrophy'])
thalach = st.number_input('Maximum Heart Rate Achieved', 40, 202, 150)
exang = st.selectbox('Exercise-Induced Angina', ['No', 'Yes'])
oldpeak = st.number_input('ST Depression', 0.0, 6.2, 1.0)
slope = st.selectbox('Peak Exercise ST Segment', ['upsloping', 'flat', 'downsloping'])
ca = st.slider('Major Vessels Colored', 0, 3, 0)
thal = st.selectbox('Thalassemia', ['normal', 'fixed defect', 'reversible defect'])

# Encode categorical values
sex = 1 if sex == 'Male' else 0
# ... (other encodings)

# Create feature array
heart_features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

# Predict
if st.button('Predict Heart Disease'):
    prediction = heart_model.predict(heart_features)
    if prediction[0] == 1:
        st.error('Heart Disease Detected')
    else:
        st.success('No Heart Disease Detected')
```

---

## Prediction Flow

```
User inputs clinical features
        ↓
Encode categorical values
        ↓
heart_model.predict() → 0 or 1
        ↓
Display prediction result
```

---

## References

- Dataset: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Original research: Heart Disease Prediction using Machine Learning (IEEE 2019)
