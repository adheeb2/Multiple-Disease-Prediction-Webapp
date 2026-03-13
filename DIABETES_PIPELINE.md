# Diabetes ML Pipeline Documentation

## Overview

The diabetes prediction system uses a comprehensive machine learning pipeline that evaluates multiple algorithms and combines the best ones using ensemble techniques.

**Location**: `code/PIMA/`

---

## Pipeline Architecture

```
Raw Data → Feature Engineering → Preprocessing → Model Selection → Ensemble → Final Model
```

---

## Files in Pipeline

| File | Description |
|------|-------------|
| `training.py` | Main pipeline orchestration |
| `models.py` | Model factory with 10+ algorithms |
| `config.yml` | Configuration parameters |
| `data_prep.py` | Data preprocessing functions |
| `feature_engineer.py` | Feature engineering |
| `evaluation.py` | Model evaluation and reporting |
| `config_utils.py` | Configuration utilities |
| `pima_diabetes.csv` | PIMA Indians Diabetes dataset |
| `artifacts/` | Saved models and reports |

---

## Configuration (`config.yml`)

```yaml
data_path: "pima_diabetes.csv"
target: "Outcome"
test_size: 0.2
random_state: 42
cv_folds: 5
nested_cv_folds: 3
top_n: 4
n_trials: 30
artifact_dir: "artifacts"
use_gpu: false
save_pickle: true
report_pdf_name: "model_report.pdf"
```

---

## Input Features

From the PIMA Indians Diabetes Dataset:

| Feature | Description | Range |
|---------|-------------|-------|
| Pregnancies | Number of pregnancies | 0-17 |
| Glucose | Plasma glucose concentration | 0-199 mg/dL |
| BloodPressure | Diastolic blood pressure | 0-122 mm Hg |
| SkinThickness | Triceps skin fold thickness | 0-99 mm |
| Insulin | 2-Hour serum insulin | 0-846 mu U/ml |
| BMI | Body mass index | 0-67.1 |
| DiabetesPedigreeFunction | Diabetes pedigree function | 0.08-2.42 |
| Age | Age in years | 21-81 |
| **Outcome** (target) | Class label (0=No diabetes, 1=Diabetes) | 0/1 |

---

## Algorithms Tested (`models.py`)

The pipeline tests 10+ algorithms:

| Algorithm | Type | Notes |
|-----------|------|-------|
| RandomForest | Ensemble | 300 estimators |
| ExtraTrees | Ensemble | 300 estimators |
| HistGradientBoosting | Ensemble | Histogram-based |
| GradientBoosting | Ensemble | 300 estimators |
| SVC | Support Vector | RBF kernel, probability=True |
| KNN | Distance-based | k=7 neighbors |
| MLP | Neural Network | (64,32) hidden layers |
| GaussianNB | Probabilistic | Naive Bayes |
| DecisionTree | Tree-based | Single tree |
| XGBoost | Gradient Boosting | 300 estimators, lr=0.05 |
| CatBoost | Gradient Boosting | 300 iterations |

---

## Pipeline Steps (`training.py`)

### 1. Data Loading
```python
df = pd.read_csv(cfg['data_path'])
```

### 2. Feature Engineering (`feature_engineer.py`)
- Applies transformations to raw features
- Creates derived features if needed

### 3. Train-Test Split
```python
X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

### 4. Target Encoding
- For categorical features: K-Fold target encoding
- Prevents data leakage during cross-validation

### 5. Numeric Preprocessing (`data_prep.py`)
- **Imputation**: Handles missing values
- **Scaling**: StandardScaler normalization

### 6. Feature Selection
- Uses `SelectKBest` with ANOVA F-value
- Selects top 20 features to reduce noise and overfitting

### 7. Resampling (SMOTETomek)
- Addresses class imbalance using hybrid technique:
  - **SMOTE**: Synthetic Minority Over-sampling
  - **Tomek Links**: Removes majority class near boundaries

### 8. Cross-Validation Ranking
- 5-fold stratified CV on resampled data
- Evaluates all 10+ algorithms
- Ranks by accuracy

### 9. Top Model Selection
- Selects top 4 models for ensemble

### 10. Probability Calibration
- Uses `CalibratedClassifierCV` with sigmoid method
- Improves probability estimates

### 11. Ensemble Methods

#### Voting Classifier (Soft Voting)
- Combines predictions using optimized weights
- Weights determined via numerical optimization (SLSQP)
- Uses soft voting (probability averaging)

#### Stacking Classifier
- Meta-learner: Logistic Regression
- Stack method: predict_proba
- 5-fold CV for stacking

### 12. Final Model Selection
- Evaluates both Voting and Stacking on holdout set
- Selects the better performing one

---

## Ensemble Optimization

### Weight Optimization
```python
def ensemble_obj(w):
    w = w / (w.sum() + 1e-9)
    vc = VotingClassifier(estimators=estimators, voting='soft', weights=list(w))
    sc = cross_val_score(vc, X_res, y_res, cv=3, scoring='accuracy').mean()
    return -sc
```

Minimized using:
- **Method**: SLSQP (Sequential Least Squares Programming)
- **Bounds**: [0, 1] for each weight
- **Constraint**: Weights sum to 1
- **Max iterations**: 200

---

## Output Artifacts

Saved in `code/PIMA/artifacts/`:

| File | Description |
|------|-------------|
| `final_model.joblib` | Final trained ensemble model |
| `final_model.sav` | Pickle version of final model |
| `preproc.joblib` | Preprocessing objects (imputer, scaler, selector) |
| `cv_scores.json` | Cross-validation scores and top models |
| `model_report.pdf` | PDF report with metrics and plots |

---

## Model Selection in App

The trained model is loaded in `Frontend/app.py`:

```python
diabetes_model = joblib.load("models/diabetes_model.sav")
```

**Prediction**:
```python
diabetes_prediction = diabetes_model.predict([[
    Pregnancies, Glucose, BloodPressure, SkinThickness, 
    Insulin, BMI, DiabetesPedigreefunction, Age
]])
```






## References

- Dataset: [PIMA Indians Diabetes Database](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
- Original research on diabetes prediction using ML approaches (IEEE 2019)
