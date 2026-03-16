# Liver Disease Prediction Documentation

## Overview

The Liver Disease Prediction system uses a Logistic Regression classifier to predict liver disease based on 10 liver function test results. It evaluates various biochemical markers to determine if a patient has liver disease.

**Location**: `Frontend/models/liver_model.sav`

---

## Files

| File | Description |
|------|-------------|
| `models/liver_model.sav` | Trained Logistic Regression model |
| `app.py` (lines 457-513) | Prediction UI and logic |

---

## Input Features

Based on Indian Liver Patient Dataset:

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

### Normal Reference Ranges

| Test | Normal Range |
|------|-------------|
| Total Bilirubin | 0.1-1.2 mg/dl |
| Direct Bilirubin | 0.0-0.3 mg/dl |
| Alkaline Phosphatase | 44-147 IU/L |
| ALT (SGPT) | 7-56 IU/L |
| AST (SGOT) | 10-40 IU/L |
| Total Protein | 6.0-8.3 g/dl |
| Albumin | 3.5-5.0 g/dl |
| A/G Ratio | 0.8-2.0 |

---

## Algorithm: Logistic Regression

Logistic Regression is used because:
- Works well with linearly separable data (liver function tests)
- Provides probability estimates alongside predictions
- Fast inference time
- Interpretable coefficients for medical interpretation
- Good performance on binary classification (liver disease vs normal)

---

## How It Works

### Model Loading

```python
import pickle

liver_model = pickle.load(open('models/liver_model.sav', 'rb'))
```

### Prediction in App

```python
# Collect user inputs
sex = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', 4, 90, 40)
Total_Bilirubin = st.number_input('Total Bilirubin (mg/dl)', 0.1, 75.0, 1.0)
Direct_Bilirubin = st.number_input('Direct Bilirubin (mg/dl)', 0.1, 30.0, 0.3)
Alkaline_Phosphotase = st.number_input('Alkaline Phosphotase (IU/L)', 63, 211, 100)
Alamine_Aminotransferase = st.number_input('Alamine Aminotransferase (IU/L)', 5, 100, 30)
Aspartate_Aminotransferase = st.number_input('Aspartate Aminotransferase (IU/L)', 10, 100, 35)
Total_Protiens = st.number_input('Total Protiens (g/dl)', 3.0, 9.6, 6.5)
Albumin = st.number_input('Albumin (g/dl)', 1.8, 5.5, 3.5)
Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio', 0.3, 2.5, 1.0)

# Encode categorical values
sex = 1 if sex == 'Female' else 0

# Create feature array
liver_features = [[sex, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                   Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                   Albumin, Albumin_and_Globulin_Ratio]]

# Predict
if st.button('Predict Liver Disease'):
    prediction = liver_model.predict(liver_features)
    if prediction[0] == 1:
        st.error('Liver Disease Detected')
    else:
        st.success('No Liver Disease Detected')
```

---

## Prediction Flow

```
User inputs liver function test results
        ↓
Encode categorical values (sex)
        ↓
liver_model.predict() → 0 or 1
        ↓
Display prediction result
```

---

## Key Liver Function Markers

- **Bilirubin**: Elevated levels indicate liver dysfunction or bile duct obstruction
- **Alkaline Phosphatase (ALP)**: Elevated in cholestatic liver disease
- **ALT (Alamine Aminotransferase)**: Elevated in liver cell damage
- **AST (Aspartate Aminotransferase)**: Elevated in liver and heart disease
- **Albumin**: Low levels indicate chronic liver disease
- **A/G Ratio**: Decreased ratio suggests chronic liver disease

---

## References

- Dataset: [UCI ILPD (Indian Liver Patient Dataset)](https://archive.ics.uci.edu/ml/datasets/ILPD+(Indian+Liver+Patient+Dataset))
- Original research: Liver Disease Prediction using Machine Learning approaches
