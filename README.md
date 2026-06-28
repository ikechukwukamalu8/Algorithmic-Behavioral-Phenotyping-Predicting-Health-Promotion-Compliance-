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
