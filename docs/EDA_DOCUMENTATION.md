# Exploratory Data Analysis (EDA) - Complete Documentation

## AquaSafe Water Quality Classification Project

**Date:** January 2026

**Notebook:** `notebooks/01_eda.ipynb`

**Purpose:** Understand data structure, quality, patterns, and prepare for modeling

**Status:** ✅ Complete & Production-Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [EDA Phases](#eda-phases)
4. [Key Findings](#key-findings)
5. [Feature Taxonomy](#feature-taxonomy)
6. [Data Quality Issues & Observations](#data-quality-issues--observations)
7. [Code Documentation Standards](#code-documentation-standards)

---

## Overview

### Objectives

* ✅ Assess data completeness and quality baseline
* ✅ Understand distributions, outliers, and correlations in numeric features
* ✅ Identify leakage risks, metadata, and non-predictive columns

### Data Source

* **File:** `data/NWMP_August2025_MPCB_0.csv`
* **Domain:** Water quality monitoring stations (Maharashtra, India)
* **Records:** ~300+ samples with 100+ features
* **Target:** Water use-based classification (4 categories: A, B, C, E)

---

## Dataset Description

### Target Variable: `use_based_class`

| Code  | Classification                             | Use Case                     | Quality Standard |
| ----- | ------------------------------------------ | ---------------------------- | ---------------- |
| **A** | Drinking (no treatment, disinfection only) | High-grade potable supply    | Excellent        |
| **B** | Outdoor bathing (organized)                | Recreational use             | Good             |
| **C** | Drinking water source                      | Municipal treatment required | Acceptable       |
| **E** | Irrigation/Industrial/Waste                | Non-potable controlled use   | Regulated        |

**Distribution:** Moderate class imbalance (observed via class frequency)

---

### Feature Categories

#### 🔢 Numeric Features (~40–50)

**Physicochemical Parameters**

* Dissolved oxygen (DO)
* pH, Conductivity, Total Dissolved Solids (TDS)
* Turbidity
* Temperature

**Chemical Contaminants**

* Boron
* Nitrogen compounds
* Sodium, Chlorides, Sulphate
* Hardness, Alkalinity

**Biological Indicators**

* Fecal coliform
* Total coliform
* Fecal streptococci

---

#### 🏷️ Categorical Features (~30–40)

**Domain-Relevant**

* `color`, `odor`
* `human_activities`

**Metadata (Non-Predictive)**

* `sampling_date`, `sampling_time`, `month`
* `state_name`, `mon_agency`, `frequency`

**Identifiers**

* `stn_code`, `stn_name`, `name_of_water_body`
* `latitude`, `longitude`

**Data Quality Observations**

* `remark` (~98% missing)
* `use_of_water_in_down_stream` (~95% missing)
* `major_polluting_sources` — **leakage risk**

---

## EDA Phases

### Phase 1: Data Load & Inspection

**Cells:** Load, examine shape, columns, dtypes

```python
✓ Loaded CSV file → df
✓ Examined df.shape, df.info(), df.head()
✓ Identified numeric vs categorical columns
✓ Initial missingness assessment
```

---

### Phase 2: Numeric Feature Analysis

**Analyses Performed**

#### 1. Skewness Detection

* Method: Fisher–Pearson coefficient
* Observation: ~30–40% numeric features exhibit strong skewness

#### 2. Outlier Detection

* Method: IQR-based bounds
* Observation: 10–15% values flagged as outliers
* Interpretation: Values appear domain-valid rather than erroneous

#### 3. Correlation Analysis

* Method: Pearson correlation
* Observation: Strong correlation blocks among solids, hardness, conductivity

---

### Phase 3: Categorical Feature Analysis

**Methods**

* Cardinality analysis using `nunique`
* Value distribution inspection
* Target relationship using `pd.crosstab(..., normalize="index")`

**Key Observations**

| Category         | Observation                                |
| ---------------- | ------------------------------------------ |
| High cardinality | Identifiers and numeric-as-string fields   |
| Low cardinality  | True categorical signals                   |
| Target coupling  | Near-deterministic mapping in some columns |

**Leakage Detection**

* `major_polluting_sources` shows quasi-deterministic relationship with target
* Indicates encoding of labeling logic rather than independent measurement

---

## Key Findings

### ✅ Data Quality (Observed)

| Aspect          | Observation                               |
| --------------- | ----------------------------------------- |
| Missingness     | Non-uniform; some near-empty columns      |
| Consistency     | Mixed numeric/string encodings            |
| Domain Validity | Majority of values within expected ranges |

---

### ✅ Feature Signal (Observed)

| Feature Group | Insight                                  |
| ------------- | ---------------------------------------- |
| Numeric       | Skewed distributions common              |
| Categorical   | Some high-cardinality, low-signal fields |
| Temporal      | Administrative context only              |
| Geospatial    | Location identifiers, not predictors     |

---

## Feature Taxonomy (Observed)

### Candidate Predictive Signals

* Numeric physicochemical and biological measurements
* Low-cardinality environmental descriptors (`color`, `odor`)

### Non-Predictive / Risk-Prone

* Identifiers (`stn_code`, `stn_name`)
* Metadata (`sampling_date`, `mon_agency`)
* Leakage-prone (`major_polluting_sources`)
* Sparse (`remark`, `use_of_water_in_down_stream`)

---

## Data Quality Issues & Observations

### Numeric Values Encoded as Strings

* Chemical measurements contain annotations (e.g., `BDL`)
* Prevents direct statistical analysis

### Geographic Coordinates Format

* Stored in non-decimal representation
* Not directly usable for spatial analysis

### Column Naming Inconsistency

* Mixed casing, units embedded in names
* Reduces readability and tooling compatibility

---

## Modeling Considerations (Derived from EDA)

* Multi-class classification problem (A, B, C, E)
* Class imbalance present
* Leakage risk identified and must be handled
* Outliers likely domain-valid
* Feature redundancy expected among correlated measures

---

## Code Documentation Standards

### Principle 1: Explain WHY, Not WHAT

```python
# ❌ BAD
df = df[df["use_based_class"].notna()]

# ✅ GOOD
# Rows without target labels provide no learning signal
```

### Principle 2: Use Section Headers

```python
# ============================================================================
# SECTION NAME
# ============================================================================
```

### Principle 3: Meaningful Variable Names

```python
LEAKAGE_PRONE_COLS = [...]
METADATA_COLS = [...]
IDENTIFIER_COLS = [...]
```

---

## Summary Checklist

✅ **EDA Completed**

* [x] Data inspected
* [x] Missingness assessed
* [x] Numeric distributions analyzed
* [x] Outliers flagged (not removed)
* [x] Correlations inspected
* [x] Categorical cardinality evaluated
* [x] Leakage risks identified
* [x] Metadata and identifiers recognized

❌ **Not performed in EDA**

* Cleaning
* Imputation
* Encoding
* Feature removal
* Modeling

---

## Next Steps (Planned)

1. **Data Cleaning** (`02_data_cleaning.ipynb`)
2. **Feature Engineering** (`03_feature_engineering.ipynb`)
3. **Model Training**
4. **Evaluation & Deployment**

---

**Document Version:** 1.0
**Last Updated:** January 2026
**Author:** [pandey-rakshit](https://www.github.com/pandey-rakshit/)
**Status:** EDA Complete ✅