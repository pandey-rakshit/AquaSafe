# 🌊 AquaSafe: Water Quality Classification

A machine learning system to classify water quality based on physico-chemical and biological parameters, using data from the Maharashtra Pollution Control Board (MPCB) National Water Monitoring Programme.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Executive Summary](#-executive-summary)
- [Dataset Overview](#-dataset-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Pipeline Workflow](#-pipeline-workflow)
- [Feature Description](#-feature-description)
- [Key Observations](#-key-observations)
- [Model Performance](#-model-performance)
- [Results](#-results)
- [Known Limitations](#-known-limitations)
- [Recommendations](#-recommendations)
- [Usage](#-usage)
- [License](#-license)

---

## 🎯 Problem Statement

Classify water bodies into regulatory quality classes based on their suitability for different uses:

| Class | Description | Use Case |
|-------|-------------|----------|
| **A** | Drinking water source without conventional treatment but after disinfection | Highest quality |
| **B** | Outdoor bathing (Organized) | Recreational use |
| **C** | Drinking water source with conventional treatment | Potable after treatment |
| **E** | Irrigation, industrial cooling, controlled waste disposal | Agricultural/Industrial |

---

## 📊 Executive Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 175 (after cleaning) |
| **Train / Test Split** | 140 / 35 (80/20) |
| **Features** | 92 (after encoding) |
| **Target Classes** | 4 (A, B, C, E) |
| **Best Model** | Logistic Regression |
| **Test Accuracy** | 94.29% |
| **Test F1 (Macro)** | 90.83% |
| **Prediction Confidence** | 91.28% (mean) |

---

## 📁 Dataset Overview

**Source:** Maharashtra Pollution Control Board (MPCB) - National Water Monitoring Programme (August 2025)

### Class Distribution

| Class | Train Samples | Test Samples | Percentage |
|-------|---------------|--------------|------------|
| A | 116 | 29 | 82.9% |
| B | 4 | 1 | 2.9% |
| C | 4 | 1 | 2.9% |
| E | 16 | 4 | 11.4% |

**Note:** Significant class imbalance present (A dominates at ~83%)

### Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Chemical Parameters | 20+ | pH, conductivity, dissolved O₂, BOD, COD |
| Biological Indicators | 5 | Fecal coliform, total coliform, fecal streptococci |
| Physical Properties | 10+ | Temperature, turbidity, total suspended solids |
| Contextual Features | 15+ | Water body type, human activities, flow, depth |
| BDL Flags | 17 | Binary indicators for "Below Detection Limit" |

---

## 📂 Project Structure

```
AquaSafe/
├── data/
│   ├── raw/
│   │   └── NWMP_August2025_MPCB_0.csv
│   └── processed/
│       ├── csv/
│       │   ├── cleaned_water_quality.csv
│       │   ├── train.csv
│       │   └── test.csv
│       ├── parquet/
│       │   ├── train.parquet
│       │   └── test.parquet
│       └── feature_registry.json
│
├── models/
│   ├── logisticregression_pipeline.pkl
│   ├── randomforest_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   ├── label_encoder.pkl
│   ├── numeric_imputer.pkl
│   ├── onehot_encoder.pkl
│   ├── feature_names.pkl
│   └── best_model_name.txt
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── src/
│   └── data_preprocessing/
│       └── create_dataframe.py
│
├── utils/
│   └── config.py
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/aquasafe.git
cd aquasafe

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
streamlit>=1.28.0
```

---

## 🔄 Pipeline Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AquaSafe ML Pipeline                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   01_EDA.ipynb   │───▶│ 02_Cleaning.ipynb│───▶│ 03_Features.ipynb│
│                  │    │                  │    │                  │
│ • Distributions  │    │ • Parse BDL      │    │ • SPLIT FIRST    │
│ • Missing values │    │ • Convert coords │    │ • Impute (train) │
│ • Correlations   │    │ • Map target     │    │ • Encode (train) │
│ • Leakage check  │    │ • Drop columns   │    │ • Export splits  │
└──────────────────┘    │ • Export w/ NaN  │    └────────┬─────────┘
                        └──────────────────┘             │
                                                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│05_Evaluation.ipynb│◀──│04_Training.ipynb │◀───│   train.csv      │
│                  │    │                  │    │   test.csv       │
│ • Confusion mat  │    │ • Load splits    │    └──────────────────┘
│ • Per-class F1   │    │ • 5-Fold CV      │
│ • Confidence     │    │ • Train models   │
│ • Feature import │    │ • Save pipelines │
└──────────────────┘    └──────────────────┘
```

### Data Leakage Prevention

This pipeline follows industry best practices to prevent data leakage:

| Step | Location | Fit On |
|------|----------|--------|
| Train-Test Split | Notebook 03 | N/A (before any transformation) |
| Numeric Imputation (median) | Notebook 03 | Train only |
| Categorical Imputation (mode) | Notebook 03 | Train only |
| One-Hot Encoding | Notebook 03 | Train only |
| Scaling (RobustScaler) | Notebook 04 | Train only (in pipeline) |

---

## 📊 Feature Description

### Top 10 Most Important Features

| Rank | Feature | Importance | Category |
|------|---------|------------|----------|
| 1 | approx_depth_Less than 50cm | 0.5715 | Physical |
| 2 | human_activities_Others | 0.4596 | Contextual |
| 3 | approx_depth_Greater than 100cm | 0.3763 | Physical |
| 4 | phosphate | 0.3421 | Chemical |
| 5 | approx_depth_50-100cm | 0.3044 | Physical |
| 6 | conductivity | 0.3044 | Chemical |
| 7 | turbidity_is_bdl | 0.3022 | BDL Flag |
| 8 | flow | 0.3011 | Physical |
| 9 | human_activities_Bathing,Washing,Fishing | 0.2908 | Contextual |
| 10 | flouride | 0.2797 | Chemical |

### Feature Categories

**Chemical Parameters:**
- pH, conductivity, dissolved oxygen, BOD, COD
- Nitrates, phosphates, chlorides, sulphates
- Heavy metals: boron, flouride, potassium, sodium

**Biological Indicators:**
- Fecal coliform, total coliform
- Fecal streptococci
- Total Kjeldahl nitrogen

**Physical Properties:**
- Temperature, turbidity, flow
- Total suspended solids, total dissolved solids
- Approximate depth, water body type

**BDL (Below Detection Limit) Flags:**
- 17 binary features indicating when measurements fall below lab detection thresholds
- Preserved as they carry important signal about water purity

---

## 🔍 Key Observations

### 1. Severe Class Imbalance
- Class A dominates with ~83% of samples
- Classes B and C have very few samples (1 each in test set)
- Addressed using `class_weight='balanced'` in models

### 2. Water Depth is Highly Predictive
- All three depth categories appear in top 10 features
- Shallow waters (<50cm) most strongly associated with quality class

### 3. Human Activity Impact
- Human activities significantly influence water quality classification
- "Others" and "Bathing,Washing,Fishing" are strong predictors

### 4. Chemical Indicators
- Phosphate and conductivity are key chemical predictors
- Fluoride levels contribute to classification decisions

### 5. BDL Flags Carry Signal
- `turbidity_is_bdl` in top 10 features
- Below-detection-limit values indicate cleaner water

---

## 🏆 Model Performance

### Model Comparison (Test Set)

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| **Logistic Regression** | **0.9429** | **0.9083** | **0.9839** | **0.8750** |
| XGBoost | 0.9143 | 0.7583 | 0.8589 | 0.8125 |
| Random Forest | 0.8571 | 0.4802 | 0.4632 | 0.5000 |

**Winner: Logistic Regression** 🏆

### Per-Class Performance (Best Model)

| Class | Precision | Recall | F1-Score | Support | Status |
|-------|-----------|--------|----------|---------|--------|
| A | 0.935 | 1.000 | 0.967 | 29 | ✅ Excellent |
| B | 1.000 | 1.000 | 1.000 | 1 | ✅ Excellent |
| C | 1.000 | 1.000 | 1.000 | 1 | ✅ Excellent |
| E | 1.000 | 0.500 | 0.667 | 4 | ⚠️ Fair |

### Confusion Matrix

```
              Predicted
              A    B    C    E
Actual  A  [ 29    0    0    0 ]  100% recall
        B  [  0    1    0    0 ]  100% recall
        C  [  0    0    1    0 ]  100% recall
        E  [  2    0    0    2 ]   50% recall  ← Weakness
```

### Misclassification Analysis

- **Total misclassified:** 2 samples (5.7%)
- **Pattern:** Both errors are E → A (Class E predicted as Class A)
- **Confidence on errors:** Low (~0.55), indicating model uncertainty

---

## 📈 Results

### Key Achievements

1. **94.29% Test Accuracy** — Strong overall classification performance
2. **90.83% Macro F1** — Balanced performance across imbalanced classes
3. **91.28% Mean Confidence** — Model predictions are well-calibrated
4. **Zero False Positives for B, C, E** — Perfect precision on minority classes
5. **Interpretable Model** — Logistic Regression provides clear feature coefficients

### Prediction Confidence

| Metric | Value |
|--------|-------|
| Mean confidence (all predictions) | 91.28% |
| Mean confidence (correct predictions) | 93.46% |
| Mean confidence (incorrect predictions) | 55.42% |

The model shows appropriate uncertainty on difficult cases.

---

## 🚨 Critical Safety Limitation

> **⚠️ NOT RECOMMENDED FOR PRODUCTION USE WITHOUT ADDRESSING CLASS E RECALL**

The model misclassifies **Class E (industrial/waste disposal) as Class A (drinking water)** — this is a **dangerous false negative** in a real-world scenario:

| Actual | Predicted | Risk Level |
|--------|-----------|------------|
| E (Waste disposal) | A (Drinking water) | 🔴 **CRITICAL** — Could approve contaminated water for drinking |

### Why This Happens

This is **NOT classic overfitting or underfitting** — it's a **class imbalance + feature overlap problem**:

| Issue | Evidence |
|-------|----------|
| **Severe Imbalance** | Class A = 83%, Class E = 11% — model biased toward majority |
| **Feature Similarity** | E→A errors have ~55% confidence — model sees ambiguous signal |
| **Insufficient E Samples** | Only 16 training samples for Class E |

### Recommended Fixes Before Deployment

1. **Collect 50+ more Class E samples** — current data insufficient for reliable boundary
2. **Adjust classification threshold** — lower threshold for Class E (e.g., 0.3 instead of 0.5)
3. **Cost-sensitive learning** — penalize E→A errors 10x more than other errors
4. **Add a "flagged for manual review" category** — when confidence < 70%, don't auto-approve
5. **Ensemble with rule-based checks** — if key pollutants exceed thresholds, override to E

**Until fixed, this model should only be used for preliminary screening, not final classification decisions.**

---

## ⚠️ Known Limitations

### 1. Small Dataset
- Only 175 samples after cleaning
- Test set has only 35 samples
- Classes B and C have minimal representation

### 2. Class E Recall Issue
- Model struggles to identify Class E (50% recall)
- 2 out of 4 Class E samples misclassified as Class A
- Likely due to similar characteristics between degraded Class A and Class E water

### 3. Geographic Scope
- Data only from Maharashtra, India
- May not generalize to other regions

### 4. Temporal Limitations
- Single month snapshot (August 2025)
- Seasonal variations not captured

---

## 💡 Recommendations

### Immediate Improvements

1. **Collect More Class E Samples**
   - Current recall is only 50%
   - More training examples would help distinguish E from A

2. **Hyperparameter Tuning**
   - Use GridSearchCV or Optuna for optimization
   - Focus on regularization strength for Logistic Regression

3. **Threshold Adjustment**
   - Current model may be too conservative for Class E
   - Consider class-specific probability thresholds

### Future Enhancements

1. **Ensemble Methods**
   - Combine Logistic Regression with XGBoost via stacking
   - May improve Class E recall

2. **SHAP Analysis**
   - Add SHAP values for better interpretability
   - Understand per-prediction feature contributions

3. **Temporal Features**
   - Include seasonal indicators when more data available
   - Water quality varies significantly by season

4. **Geographic Expansion**
   - Collect data from other states/regions
   - Build regional models or add location features

---

## 🚀 Usage

### Running the Notebooks

```bash
# Execute notebooks in order
jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_data_cleaning.ipynb
jupyter notebook notebooks/03_feature_engineering.ipynb
jupyter notebook notebooks/04_model_training.ipynb
jupyter notebook notebooks/05_model_evaluation.ipynb
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('models/logisticregression_pipeline.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
feature_names = joblib.load('models/feature_names.pkl')

# Prepare your data (must have same features as training)
# X_new = pd.DataFrame(...)  # Your new data

# Predict
predictions = model.predict(X_new)
predicted_classes = label_encoder.inverse_transform(predictions)
probabilities = model.predict_proba(X_new)

print(f"Predicted class: {predicted_classes[0]}")
print(f"Confidence: {probabilities.max():.2%}")
```

### Running Streamlit App

```bash
streamlit run app/streamlit_app.py
```

---

## 📄 License

This project is for educational and research purposes. Data sourced from Maharashtra Pollution Control Board under open government data initiatives.

---

## 👥 Contributors

- **Project:** AquaSafe Water Quality Classification
- **Data Source:** MPCB National Water Monitoring Programme
- **Framework:** Scikit-learn, XGBoost, Pandas

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

*Last Updated: February 2026*