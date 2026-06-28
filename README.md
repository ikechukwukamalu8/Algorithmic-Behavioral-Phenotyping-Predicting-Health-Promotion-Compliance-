# Algorithmic Behavioral Phenotyping: Predicting Health-Promotion Compliance via Multi-Domain Classification Pipelines

An end-to-end Machine Learning preprocessing and ensemble classification pipeline engineered to uncover, map, and predict multi-domain behavioral phenotypes governing health-promotion lifestyle compliance from sub-clinical community survey matrices.

## 🚀 Architectural Overview
This framework automates features transformation and predictive validation boundaries using an isolated, leak-proof `Scikit-Learn Pipeline`. It strips string metric variances, standardizes numerical scaling boundaries, handles categorical multi-collinearity via dropping dummy traps, and trains a `RandomForestClassifier` ensemble to identify structural behavioral lifestyle predictors.

## 📁 Repository Blueprint
```text
├── data/
│   └── behavioural_phenotypes_data.xlsx      # Structured sub-clinical input workbook
├── outputs/
│   ├── behavioral_confusion_matrix.png       # Generated validation boundary map
│   ├── behavioral_roc_curve.png              # Receiver Operating Curve graphic
│   └── behavioral_feature_importances.png     # Gini Impurity feature importance profile
├── model_pipeline.py                         # Production-grade standalone Python script
├── requirements.txt                          # Project environmental requirements
└── README.md                                 # Project documentation 
```
## 📊 Pipeline Validation Performance Metrics

The workflow splits and evaluates out-of-sample data points using a stratified target strategy to ensure balanced phenotype representation across compliance classes.

### Classification Boundary Report

```
==================================================
CLASSIFICATION BOUNDARY REPORT
==================================================

                 precision    recall  f1-score   support

 Low Compliance       0.62      0.75      0.68        57
High Compliance       0.66      0.51      0.57        53

       accuracy                           0.64       110
      macro avg       0.64      0.63      0.63       110
   weighted avg       0.64      0.64      0.63       110
```

### ROC-AUC Performance

**Receiver Operating Characteristic (ROC-AUC) Score:** `0.6984`

---

## 🔬 Core Algorithmic Discoveries

By tracking relative Gini impurity reductions, the model reveals that health-promotion compliance is not random but represents a behavioral phenotype strongly associated with environmental and academic factors.

| Predictor | Relative Importance |
|------------|-------------------|
| Academic Level (Year of Study) | **29.26%** |
| Information Access Point Uncertainty ("Somehow") | **9.21%** |
| Hostel Co-habitation (Reside on Campus = Yes) | **8.72%** |

### Key Findings

- **Academic Level (Year of Study)** dictates **29.26%** of overall predictive compliance variance.
- **Information Access Point Uncertainty ("Somehow")** contributes **9.21%** of decision splits.
- **Hostel Co-habitation (Reside on Campus = Yes)** drives **8.72%** of model importance weight.

These findings suggest that educational progression, information-seeking behavior, and living environment collectively shape health-promotion compliance patterns among undergraduate students.

---

## ⚙️ Local Execution Instructions

### Clone the Repository

```bash
git clone https://github.com/ikechukwukamalu8/algorithmic-behavioral-phenotyping.git
cd algorithmic-behavioral-phenotyping
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Complete Pipeline

```bash
python model_pipeline.py
```

---

## 📈 Project Summary

This project investigates whether health-promotion compliance can be algorithmically characterized through behavioral phenotyping. Using supervised machine learning and interpretable classification pipelines, the framework identifies key demographic, academic, and environmental determinants associated with compliance outcomes.

The resulting model achieves:

- **Accuracy:** 64%
- **Macro F1-Score:** 0.63
- **ROC-AUC:** 0.6984

These results demonstrate that meaningful predictive signals exist within multidomain behavioral data and support the development of data-driven interventions for improving health-promotion practices among university students.
