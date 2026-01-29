# AquaSafe – Water Quality ML Classification System


![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning solution for predicting water quality classifications based on physicochemical and biological parameters from Maharashtra's water monitoring program.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Water Quality Classes](#water-quality-classes)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

**AquaSafe** is an AI-powered water quality classification system designed to predict water use categories (A, B, C, E) based on physicochemical and biological test parameters. Built using data from the Maharashtra Pollution Control Board (MPCB), this system provides:

- **Real-time predictions** for water quality classification
- **Interactive data exploration** through EDA dashboards
- **Model comparison** and performance analysis
- **Deployment-ready** Streamlit web application

### Problem Statement

How do physicochemical and biological parameters influence water quality classification and public health safety across monitoring stations in Maharashtra?

### Solution

A complete machine learning pipeline that:
1. Cleans and preprocesses raw water quality data
2. Engineers relevant features for classification
3. Trains multiple ML models (Logistic Regression, Random Forest, XGBoost)
4. Deploys an interactive web application for stakeholders

---

## ✨ Features

### 🔮 Prediction System
- Real-time water quality classification
- Multi-model support (LogReg, Random Forest, XGBoost)
- Confidence scores and probability distributions
- Input validation against water quality standards

### 📊 Data Analysis
- Comprehensive exploratory data analysis
- Interactive visualizations (Plotly)
- Parameter distribution analysis
- Correlation and relationship exploration

### 📈 Model Insights
- Performance comparison across models
- Confusion matrices and classification reports
- Feature importance analysis
- Cross-validation results

### 🚀 Deployment
- Professional Streamlit web application
- Modular, maintainable codebase
- Responsive UI with custom styling
- Easy deployment to Streamlit Cloud

---

## 📊 Dataset

**Source:** Maharashtra Pollution Control Board (MPCB)  
**File:** `NWMP_August2025_MPCB_0.csv`  
**Records:** 222 monitoring stations  
**Features:** 54 parameters  

### Parameters Included

| Category | Parameters |
|----------|------------|
| **Chemical** | pH, BOD, COD, Dissolved Oxygen, Conductivity, TDS, Alkalinity, Hardness, Chlorides, Nitrates, etc. |
| **Biological** | Total Coliform, Fecal Coliform, Fecal Streptococci |
| **Physical** | Temperature, Turbidity, Color, Odor, Total Suspended Solids |
| **Geographic** | Station Code, Location, District, River Basin |

### Data Quality
- ✅ Clean schemas with mixed data types
- ⚠️ 15-30% missing values (handled via imputation)
- ⚠️ BDL (Below Detection Limit) annotations preserved
- ✅ Geographic coordinates standardized

---

## 📁 Project Structure

```
AquaSafe/
│
├── data/
│   ├── raw/                          # Original dataset
│   │   └── NWMP_August2025_MPCB_0.csv
│   └── processed/                    # Cleaned and processed data
│       ├── csv/
│       │   ├── cleaned_water_quality_data.csv
│       │   └── nwmp_features_v1.csv
│       └── parquet/
│           └── nwmp_features_v1.parquet
│
├── notebooks/                        # Jupyter notebooks (analysis pipeline)
│   ├── 01_eda.ipynb                 # Exploratory Data Analysis
│   ├── 02_data_cleaning.ipynb       # Data cleaning and preprocessing
│   ├── 03_feature_engineering.ipynb # Feature encoding and engineering
│   └── 04_model_training.ipynb      # Model training and evaluation
│
├── models/                           # Trained models and artifacts
│   ├── logisticregression_pipeline.pkl
│   ├── randomforest_pipeline.pkl
│   ├── xgboost_pipeline.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── best_model_name.txt
│
├── app/                              # Streamlit web application
│   ├── app.py                       # Main application entry point
│   ├── config.py                    # Configuration and constants
│   │
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   ├── model_loader.py         # Model loading utilities
│   │   ├── data_loader.py          # Data loading utilities
│   │   └── visualizations.py       # Plotting functions
│   │
│   ├── pages/                       # Application pages
│   │   ├── __init__.py
│   │   ├── home.py                 # Home/landing page
│   │   ├── predict.py              # Prediction interface
│   │   ├── eda.py                  # EDA dashboard
│   │   └── models.py               # Model comparison page
│   │
│   └── requirements.txt             # Streamlit dependencies
│
├── src/                              # Source code modules
│   ├── data_preprocessing/
│   │   ├── create_dataframe.py
│   │   └── data_analysis.py
│   │
│   └── eda/
│       ├── correlation/
│       ├── skewness/
│       └── outliers/
│
├── utils/                            # Project-wide utilities
│   └── config.py                    # Global configuration
│
├── requirements.txt                  # Project dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
└── LICENSE                          # License file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/AquaSafe.git
cd AquaSafe
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate

# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all project dependencies
pip install -r requirements.txt

# Install Streamlit app dependencies
pip install -r app/requirements.txt
```

### Main Dependencies

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
streamlit>=1.28.0
plotly>=5.17.0
joblib>=1.3.0
```

---

## 💻 Usage

### Option 1: Run the Complete Pipeline

Execute notebooks in order:

```bash
# 1. Exploratory Data Analysis
jupyter notebook notebooks/01_eda.ipynb

# 2. Data Cleaning
jupyter notebook notebooks/02_data_cleaning.ipynb

# 3. Feature Engineering
jupyter notebook notebooks/03_feature_engineering.ipynb

# 4. Model Training
jupyter notebook notebooks/04_model_training.ipynb
```

### Option 2: Run the Streamlit App

```bash
# From project root
streamlit run app/app.py

# App will open at http://localhost:8501
```

### Option 3: Use Trained Models Directly

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/randomforest_pipeline.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Make prediction
X_new = pd.DataFrame({...})  # Your input data
prediction_encoded = model.predict(X_new)
prediction = label_encoder.inverse_transform(prediction_encoded)

print(f"Water Quality Class: {prediction[0]}")
```

---

## 💧 Water Quality Classes

| Class | Description | Use | Water Quality Standards |
|-------|-------------|-----|------------------------|
| **A** | Drinking water source (post-disinfection) | Potable without conventional treatment | pH: 6.5-8.5, DO: ≥6 mg/L, BOD: ≤2 mg/L |
| **B** | Outdoor bathing (Organized) | Recreational activities | pH: 6.5-8.5, DO: ≥5 mg/L, BOD: ≤3 mg/L |
| **C** | Drinking water source | Potable after conventional treatment | DO: ≥4 mg/L, BOD: ≤3 mg/L |
| **E** | Irrigation, industrial cooling | Non-potable uses | pH: 5.5-9.0, more lenient parameters |

### Classification Criteria

Based on **Bureau of Indian Standards (BIS)** and **WHO guidelines** for:
- Physicochemical parameters (pH, dissolved oxygen, BOD, COD)
- Biological indicators (coliform bacteria)
- Physical characteristics (turbidity, color, odor)

---

## 📈 Model Performance

### Test Set Results

| Model | Accuracy | F1 (Macro) | Precision | Recall |
|-------|----------|------------|-----------|--------|
| **Logistic Regression** | 0.9485 | 0.7122 | 0.7810 | 0.6750 |
| **Random Forest** | 0.9543 | 0.7456 | 0.7923 | 0.7102 |
| **XGBoost** | 0.9571 | 0.7589 | 0.8045 | 0.7234 |

### Best Model: **XGBoost**
- **Test Accuracy:** 95.71%
- **F1 Score (Macro):** 75.89%
- **Cross-Validation:** Stratified 3-fold CV

### Key Insights
- ✅ High overall accuracy across all models
- ⚠️ Class imbalance (82% Class A) addressed via `class_weight='balanced'`
- ✅ XGBoost performs best on minority classes (B, C)
- ✅ Feature importance: DO, pH, BOD, Total Coliform are top predictors

---

## 🔧 Technical Details

### Machine Learning Pipeline

```
Raw Data → Cleaning → Feature Engineering → Model Training → Deployment
```

#### 1. Data Cleaning (`02_data_cleaning.ipynb`)
- **BDL Handling:** Extracted numeric values, preserved flags
- **Missing Values:** Median (numeric), mode (categorical)
- **Type Conversion:** DMS coordinates → decimal degrees
- **Target Mapping:** Verbose labels → compact codes (A/B/C/E)
- **Feature Curation:** Removed 18 leakage/metadata columns

#### 2. Feature Engineering (`03_feature_engineering.ipynb`)
- **Encoding:** One-hot encoding for categorical features
- **Validation:** Input/output contract checks
- **Registry:** JSON manifest for deployment
- **Output:** 179 features ready for modeling

#### 3. Model Training (`04_model_training.ipynb`)
- **Split:** 80/20 train-test (stratified)
- **Cross-Validation:** 3-fold stratified CV
- **Scaling:** RobustScaler (handles outliers)
- **Class Balance:** `class_weight='balanced'`
- **Metrics:** Accuracy, F1 (macro), Precision, Recall

### Algorithms Used

1. **Logistic Regression**
   - Baseline linear model
   - L2 regularization, max_iter=2000
   - Fast inference, interpretable

2. **Random Forest**
   - Ensemble of 300 decision trees
   - Handles non-linearity, robust to outliers
   - Feature importance available

3. **XGBoost**
   - Gradient boosting (best performer)
   - max_depth=4, learning_rate=0.1
   - Handles class imbalance well

### Model Selection Criteria
- **Primary:** F1 Score (Macro) - balanced performance across classes
- **Secondary:** Confusion matrix analysis
- **Deployment:** XGBoost (best overall performance)

---

## 🎨 Streamlit Application

### Features

#### 🏠 Home Page
- Project overview and introduction
- Water quality class descriptions
- Dataset information and statistics
- Quick start guide

#### 🔮 Predict Page
- Interactive input form for water parameters
- Real-time classification predictions
- Confidence scores and probability distribution
- Input validation against BIS standards
- Actionable recommendations

#### 📊 EDA Dashboard
- Target variable distribution
- Parameter distributions by category
- Correlation analysis
- Class-wise parameter comparison
- Interactive filters and selections

#### 📈 Model Comparison
- Performance metrics comparison
- Confusion matrices (all models)
- Per-class performance analysis
- Feature importance visualization
- Model details and documentation

### Technologies Used
- **Frontend:** Streamlit
- **Visualizations:** Plotly, Matplotlib, Seaborn
- **Backend:** Scikit-learn, XGBoost
- **Data Processing:** Pandas, NumPy

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Models Not Found
```bash
# Error: Model files not found
# Solution: Run model training notebook
jupyter notebook notebooks/04_model_training.ipynb
```

#### 2. Data Files Missing
```bash
# Error: Data files not found
# Solution: Ensure data files exist in data/processed/
# Run cleaning and feature engineering notebooks
```

#### 3. Import Errors
```bash
# Error: Module not found
# Solution: Install all dependencies
pip install -r requirements.txt
pip install -r app/requirements.txt
```

#### 4. Streamlit Won't Start
```bash
# Error: Streamlit command not found
# Solution: Ensure virtual environment is activated
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Then reinstall streamlit
pip install streamlit
```

### Getting Help
- Check notebook outputs for detailed error messages
- Review config.py for correct file paths
- Ensure all notebooks are run in sequence
- Open an issue on GitHub for persistent problems

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Update README.md for new features
- Test thoroughly before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Data Source:** Maharashtra Pollution Control Board (MPCB)
- **Standards:** Bureau of Indian Standards (BIS), WHO Water Quality Guidelines
- **Frameworks:** Scikit-learn, XGBoost, Streamlit
- **Community:** Open-source ML and data science community

---

## 📧 Contact

**Project Maintainer:** Rakshit Pandey

**Email:** inbox.rakshitpandey@gmail.com  
**GitHub:** [@pandey-rakshit](https://github.com/pandey-rakshit)  
**LinkedIn:** [pandey-rakshit](https://linkedin.com/in/pandey-rakshit)

---

## 🔮 Future Enhancements

- [ ] Deploy to Streamlit Cloud / Heroku
- [ ] Add SHAP values for model explainability
- [ ] Implement ensemble model (voting classifier)
- [ ] Add time-series analysis for temporal patterns
- [ ] Create API endpoint for programmatic access
- [ ] Add batch prediction functionality
- [ ] Implement A/B testing for model comparison
- [ ] Add user authentication and data persistence
- [ ] Create mobile-responsive design
- [ ] Add multilingual support (Hindi, Marathi)

---

## 📊 Project Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **EDA** | Days 1-2 | 10+ visualizations, data quality report |
| **Cleaning** | Days 3-4 | Cleaned dataset, preprocessing pipeline |
| **Modeling** | Days 3-4 | 3 trained models, evaluation metrics |
| **Deployment** | Days 5-6 | Streamlit app, documentation |
| **Presentation** | Day 7 | Slides, demo video, technical report |

---

## 📚 References

1. Bureau of Indian Standards (BIS). (2012). *Drinking Water Specifications* (IS 10500:2012)
2. World Health Organization. (2017). *Guidelines for Drinking-water Quality*
3. Maharashtra Pollution Control Board. (2025). *National Water Monitoring Programme Dataset*
4. Scikit-learn Documentation: https://scikit-learn.org/
5. XGBoost Documentation: https://xgboost.readthedocs.io/
6. Streamlit Documentation: https://docs.streamlit.io/

---

## 🌟 Star History

If you find this project helpful, please consider giving it a ⭐ on GitHub!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/AquaSafe&type=Date)](https://star-history.com/#yourusername/AquaSafe&Date)

---

**Built with ❤️ for cleaner water and safer communities**

*Last Updated: January 2026*