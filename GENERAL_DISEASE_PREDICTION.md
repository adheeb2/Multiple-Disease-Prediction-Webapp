# General Disease Prediction Documentation

## Overview

The General Disease Prediction feature uses an XGBoost classifier to predict 41 different diseases based on user-selected symptoms. It provides disease descriptions and precautionary measures after prediction.

**Location**: `Frontend/code/`

---

## Files

| File | Description |
|------|-------------|
| `DiseaseModel.py` | Main model class for disease prediction |
| `helper.py` | Symptom array preparation utility |
| `train.py` | Training script to build the model |


---

## Supported Diseases (41 Diseases)

The model can predict these diseases based on symptoms:

1. Acne
2. Allergic Rhinitis
3. Arthritis
4. Asthma
5. Bronchial Asthma
6. Cervical Spondylosis
7. Chickenpox
8. Chronic Cholestasis
9. Common Cold
10. Dengue
11. Diabetes
12. Dimorphic Hemorrhoids
13. Drug Reaction
14. Fatigue
15. Fungal Infection
16. Gastritis
17. GERD
18. Headache
19. Hepatitis B
20. Hepatitis C
21. Hypertension
22. Hyperthyroidism
23. Hypoglycemia
24. Hypothyroidism
25. Impetigo
26. Jaundice
27. Malaria
28. Migraine
29. Osteoarthritis
30. Paroxysmal Positional Vertigo
31. Peptic Ulcer
32. Pneumonia
33. Psoriasis
34. Renal Disease
35. Rheumatoid Arthritis
36. Ringworm
37. Scarlet Fever
38. Tuberculosis
39. Typhoid
40. Urinary Tract Infection
41. Varicose Veins

---

## Input: Symptoms

Users select from **133 possible symptoms** in the UI. Some examples include:

- itching
- skin_rash
- nodal_skin_eruptions
- continuous_sneezing
- shivering
- chills
- joint_pain
- stomach_pain
- acidity
- ulcers_on_tongue
- muscle_wasting
- vomiting
- burning_micturition
- spotting_urination
- fatigue
- weight_loss
- cold_hands_and_feets
- mood_swings
- weight_gain
- anxiety
- fast_heart_rate
- thyroid
- trembling
- chest_pain
- cough
- high_fever
- ...and many more

The complete list of symptoms is stored in the training dataset: `Frontend/data/clean_dataset.tsv`

---

## How It Works

### 1. Data Preprocessing (`train.py`)

```python
# Load raw dataset
dataset_df = pd.read_csv('data/dataset.csv')

# Clean data - strip whitespace
dataset_df = dataset_df.apply(lambda col: col.str.strip())

# One-hot encode symptoms
test = pd.get_dummies(dataset_df.filter(regex='Symptom'), prefix='', prefix_sep='')

# Group duplicate symptom columns
test = test.groupby(test.columns, axis=1).agg(np.max)

# Merge with disease labels
clean_df = pd.merge(test, dataset_df['Disease'], left_index=True, right_index=True)

# Save cleaned dataset
clean_df.to_csv('data/clean_dataset.tsv', sep='\t', index=False)
```

### 2. Symptom Encoding (`helper.py`)

When user selects symptoms, convert to binary array:

```python
def prepare_symptoms_array(symptoms):
    symptoms_array = np.zeros((1, 133))
    df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
    
    for symptom in symptoms:
        symptom_idx = df.columns.get_loc(symptom)
        symptoms_array[0, symptom_idx] = 1
    
    return symptoms_array
```

### 3. Model Training (`train.py`)

```python
# Load cleaned data
clean_df = pd.read_csv('data/clean_dataset.tsv', sep='\t')

# Split features and target
X_data = clean_df.iloc[:, :-1]
y_data = clean_df.iloc[:, -1]

# Convert target to categorical
y_data = y_data.astype('category')

# Label encoding for target
le = preprocessing.LabelEncoder()
le.fit(y_data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2)

# Convert labels to numbers
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Train XGBoost
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Save model
model.save_model("model/xgboost_model.json")
```

---

## Prediction Flow

```
User selects symptoms
        ↓
prepare_symptoms_array() → Binary array (133 features)
        ↓
DiseaseModel.predict() → Disease index
        ↓
Get disease name from categories
        ↓
Get description & precautions from CSV files
```

### Code in `app.py`:

```python
# Initialize model
disease_model = DiseaseModel()
disease_model.load_xgboost('model/xgboost_model.json')

# Get user input
symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)
X = prepare_symptoms_array(symptoms)

# Predict
if st.button('Predict'):
    prediction, prob = disease_model.predict(X)
    st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')
```

---

## Data Files

| File | Description |
|------|-------------|
| `data/dataset.csv` | Raw dataset with symptoms and diseases |
| `data/clean_dataset.tsv` | Preprocessed one-hot encoded dataset |
| `data/symptom_Description.csv` | Disease descriptions |
| `data/symptom_precaution.csv` | Disease precautions (4 per disease) |
| `data/Symptom-severity.csv` | Symptom severity scores |
| `model/xgboost_model.json` | Trained XGBoost model |

---

## DiseaseModel Class Methods

```python
class DiseaseModel:
    def load_xgboost(self, model_path):
        """Load pre-trained XGBoost model"""
        
    def predict(self, X):
        """Predict disease and return (disease_name, probability)"""
        
    def describe_disease(self, disease_name):
        """Get description for a specific disease"""
        
    def describe_predicted_disease(self):
        """Get description for predicted disease"""
        
    def disease_precautions(self, disease_name):
        """Get precautions for a specific disease"""
        
    def predicted_disease_precautions(self):
        """Get precautions for predicted disease"""
        
    def disease_list(self, kaggle_dataset):
        """Get list of all diseases the model can predict"""
```

---

## Example Output

After prediction, the app displays:

1. **Disease Name**: e.g., "Common Cold with 85.50% probability"
2. **Description Tab**: Information about the disease
3. **Precautions Tab**: 4 precautionary measures

---

## Algorithm: XGBoost

**XGBoost (eXtreme Gradient Boosting)** is used because:
- Handles high-dimensional sparse data well (133 symptom features)
- Fast training and prediction
- Good performance on multi-class classification (41 classes)
- Built-in regularization to prevent overfitting

---

## Training Configuration

- **Test size**: 20%
- **Algorithm**: XGBoost Classifier
- **Target encoding**: Label encoding for 41 disease categories
- **Input features**: 133 binary features (one per symptom)

---

## References

- Dataset: Symptom disease dataset (Kaggle)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Research: Multiple Disease Prediction using ML approaches (JETIR Paper)
