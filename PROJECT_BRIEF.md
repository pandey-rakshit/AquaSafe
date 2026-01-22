# AquaSafe – Water Quality ML Classification System

## 📌 Project Overview

This project is an **end-to-end machine learning application** that analyzes and predicts **water quality classes** using physicochemical, biological, and geographic data collected from monitoring stations across **Maharashtra, India**.

The project was completed as a **1-week intensive data science challenge** and covers the full lifecycle of a real-world data science project — from **exploratory data analysis (EDA)** to **model deployment using Streamlit**.

---

## 🎯 Problem Statement

How do chemical, biological, and physical parameters influence water quality and public health safety?

The objective is to:

* Analyze water quality data
* Build predictive machine learning models
* Deploy an interactive application for stakeholders to assess water quality

---

## 📊 Dataset Information

* **Source:** Maharashtra Pollution Control Board (MPCB)
* **File:** `NWMP_August2025_MPCB_0.csv`
* **Records:** Water quality measurements from multiple monitoring stations
* **Features:** 54 columns including chemical, biological, physical, and geographic attributes

### Water Quality Classes

| Class | Description                                |
| ----- | ------------------------------------------ |
| A     | Drinking water source (after disinfection) |
| B     | Outdoor bathing                            |
| C     | Drinking water (with treatment)            |
| E     | Irrigation and industrial cooling          |

### Key Parameters

* **Chemical:** pH, BOD, COD, Dissolved Oxygen, Conductivity, TDS
* **Biological:** Fecal Coliform, Total Coliform
* **Physical:** Temperature, Turbidity, Color
* **Geographic:** Station, District, River Basin

---

## 🧠 Project Workflow

### 1️⃣ Exploratory Data Analysis (EDA)

* Data quality and missing value analysis
* Distribution analysis and correlation heatmaps
* Statistical summaries
* Geographic and district-level insights

### 2️⃣ Data Preprocessing & Modeling

* Handling missing and invalid values (BDL / ND / NA)
* Feature engineering and encoding
* Scaling numerical features
* Handling class imbalance
* Training multiple models:

  * Logistic Regression (baseline)
  * Random Forest
  * XGBoost / SVM
* Model evaluation using Accuracy, F1-Score, Confusion Matrix, and Cross-Validation

### 3️⃣ Streamlit Application

* Interactive prediction interface
* EDA dashboards with visual insights
* Model performance comparison
* Clean, user-friendly UI

---

## 🚀 Streamlit App Features

* **Home:** Project overview and usage instructions
* **Predict:** Enter water parameters to predict quality class
* **EDA Dashboard:** Interactive charts and data exploration
* **Models:** Comparison of model performance and metrics

---

## 🗂️ Repository Structure

```
water-quality-classification-ml/
│
├── data/          # Dataset files
├── notebooks/     # EDA and modeling notebooks
├── src/           # Data preprocessing and training scripts
├── models/        # Saved trained models (.pkl)
├── app/           # Streamlit application
├── README.md
└── requirements.txt
```

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Libraries:** pandas, numpy, matplotlib, seaborn
* **Machine Learning:** scikit-learn, XGBoost
* **Deployment:** Streamlit

---

## 📈 Key Learnings

* Real-world data cleaning and preprocessing
* Feature engineering for environmental data
* Model selection and evaluation
* Handling imbalanced datasets
* Building and deploying interactive ML applications
* Writing clean, professional project documentation

---

## 📌 Future Improvements

* Add real-time data ingestion
* Improve geographic visualizations
* Integrate model explainability (SHAP)
* Deploy on cloud with CI/CD

---

## 📄 License

This project is created for educational and portfolio purposes.