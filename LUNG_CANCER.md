# Lung Cancer Prediction Documentation

## Overview

The Lung Cancer Prediction system uses a Logistic Regression classifier to predict lung cancer based on 15 lifestyle and symptom features. It evaluates various risk factors and symptoms to determine the likelihood of lung cancer.

**Location**: `Frontend/models/lung_cancer_model.sav`

---

## Files

| File | Description |
|------|-------------|
| `models/lung_cancer_model.sav` | Trained Logistic Regression model |
| `data/lung_cancer.csv` | Training dataset |
| `app.py` (lines 359-452) | Prediction UI and logic |

---

## Input Features

| Feature | Description | Values |
|---------|-------------|--------|
| Gender | Patient gender | Male/Female |
| Age | Patient age | 15-80 | years |
| Smoking | Smoking status | YES/NO |
| Yellow_Fingers | Yellow fingers (sign of smoking) | YES/NO |
| Anxiety | Anxiety disorder | YES/NO |
| Peer_Pressure | Peer pressure influence | YES/NO |
| Chronic_Disease | Chronic disease history | YES/NO |
| Fatigue | Chronic fatigue | YES/NO |
| Allergy | Allergies | YES/NO |
| Wheezing | Wheezing symptoms | YES/NO |
| Alcohol_Consuming | Alcohol consumption | YES/NO |
| Coughing | Chronic coughing | YES/NO |
| Shortness_of_Breath | Breathing difficulty | YES/NO |
| Swallowing_Difficulty | Dysphagia | YES/NO |
| Chest_Pain | Chest pain | YES/NO |

### Feature Encoding

All categorical features (except Age and Gender) are encoded as:
- YES = 1
- NO = 0

Gender is encoded as:
- Male = 1
- Female = 0

---

## Algorithm: Logistic Regression

Logistic Regression is used because:
- Works well with linearly separable risk factor data
- Provides probability estimates alongside predictions
- Fast inference time
- Interpretable coefficients for medical interpretation
- Good performance on binary classification (cancer vs no cancer)

---

## How It Works

### Model Loading

```python
import pickle

lung_cancer_model = pickle.load(open('models/lung_cancer_model.sav', 'rb'))
```

### Prediction in App

```python
# Collect user inputs
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', 15, 80, 50)
smoking = st.selectbox('Smoking', ['NO', 'YES'])
yellow_fingers = st.selectbox('Yellow Fingers', ['NO', 'YES'])
anxiety = st.selectbox('Anxiety', ['NO', 'YES'])
peer_pressure = st.selectbox('Peer Pressure', ['NO', 'YES'])
chronic_disease = st.selectbox('Chronic Disease', ['NO', 'YES'])
fatigue = st.selectbox('Fatigue', ['NO', 'YES'])
allergy = st.selectbox('Allergy', ['NO', 'YES'])
wheezing = st.selectbox('Wheezing', ['NO', 'YES'])
alcohol = st.selectbox('Alcohol Consuming', ['NO', 'YES'])
coughing = st.selectbox('Coughing', ['NO', 'YES'])
shortness_breath = st.selectbox('Shortness of Breath', ['NO', 'YES'])
swallowing_difficulty = st.selectbox('Swallowing Difficulty', ['NO', 'YES'])
chest_pain = st.selectbox('Chest Pain', ['NO', 'YES'])

# Encode categorical values
gender = 1 if gender == 'Male' else 0
smoking = 1 if smoking == 'YES' else 0
# ... (other encodings)

# Create feature array
lung_features = [[gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
                  chronic_disease, fatigue, allergy, wheezing, alcohol, coughing,
                  shortness_breath, swallowing_difficulty, chest_pain]]

# Predict
if st.button('Predict Lung Cancer'):
    prediction = lung_cancer_model.predict(lung_features)
    if prediction[0] == 1:
        st.error('Lung Cancer Detected')
    else:
        st.success('No Lung Cancer Detected')
```

---

## Prediction Flow

```
User inputs risk factors and symptoms
        ↓
Encode categorical values (YES/NO → 1/0)
        ↓
lung_cancer_model.predict() → 0 or 1
        ↓
Display prediction result
```

---

## Risk Factors Evaluated

### Major Risk Factors
- **Smoking**: Primary risk factor for lung cancer
- **Yellow Fingers**: Indicator of long-term smoking
- **Alcohol Consuming**: Combined with smoking increases risk

### Symptoms
- **Chronic Coughing**: Persistent cough lasting >2 weeks
- **Wheezing**: Whistling sound during breathing
- **Shortness of Breath**: Difficulty breathing
- **Chest Pain**: Pain in chest area
- **Swallowing Difficulty**: Dysphagia

### Other Factors
- **Chronic Disease**: Underlying health conditions
- **Anxiety**: Psychological factor
- **Allergy**: Environmental triggers
- **Peer Pressure**: Social influences

---

## References

- Dataset: [Lung Cancer Dataset (Kaggle)](https://www.kaggle.com/datasets/mysarahmadbhat/lung-cancer)
- Original research: Lung Cancer Prediction using Machine Learning approaches
