# Behavioral Phenotyping of Undergraduate Health Compliance Using Machine Learning

An end-to-end Machine Learning pipeline that classifies university undergraduate health compliance phenotypes (**Low Compliance** vs. **High Compliance**) using multi-dimensional survey data spanning demographics, lifestyle behaviours, environmental factors, and health awareness.

The pipeline implements automated preprocessing, model benchmarking, explainable artificial intelligence (XAI), uncertainty estimation, and reproducible evaluation using a leak-proof Scikit-Learn workflow.

---

# Key Highlights

- **439 undergraduate participants**
- **41 engineered behavioural features**
- Automated preprocessing pipeline
- 10-Fold Stratified Cross-Validation
- Multi-model benchmarking
- Explainable AI using Permutation Importance and SHAP
- Bootstrap Confidence Intervals
- Production-ready serialized pipeline

---

# Champion Model Performance

| Metric | Value |
|---------|------:|
| Champion Model | Random Forest Classifier |
| Number of Trees | 300 |
| Maximum Depth | 6 |
| Cross-Validation Accuracy | **75.1%** |
| Cross-Validation ROC-AUC | **0.815** |
| Cross-Validation F1-Score | **0.717** |
| Holdout Accuracy | **80.0%** |
| Holdout ROC-AUC | **0.846** |
| 95% Bootstrap CI | **0.767 – 0.912** |
| Brier Score | **0.171** |

---

# Cross-Validation Benchmark Results

Models were evaluated using **10-Fold Stratified Cross-Validation** on the training dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|---------:|----------:|-------:|---------:|---------:|
| **Random Forest** | **0.751** | **0.774** | **0.680** | **0.717** | **0.815** |
| Gradient Boosting | 0.742 | 0.742 | 0.718 | 0.726 | 0.787 |
| XGBoost | 0.723 | 0.722 | 0.705 | 0.708 | 0.790 |
| LightGBM | 0.711 | 0.709 | 0.686 | 0.693 | 0.783 |
| Logistic Regression | 0.690 | 0.694 | 0.667 | 0.676 | 0.747 |

---

# Holdout Test Classification Report

Performance on previously unseen participants.

```text
                 precision    recall  f1-score   support

 Low Compliance       0.78      0.86      0.82        57
High Compliance       0.83      0.74      0.78        53

       accuracy                           0.80       110
      macro avg       0.80      0.80      0.80       110
   weighted avg       0.80      0.80      0.80       110
```

---

# Key Behavioural Drivers

Permutation Importance identified the strongest predictors of undergraduate health compliance.

| Feature | Relative Importance |
|---------|--------------------:|
| Physical Exercise Routine | 0.150 |
| Sexual Health Awareness | 0.020 |
| Sexual Habits (Abstinence) | 0.020 |
| Alcohol Avoidance Behaviour | 0.018 |
| Dentistry Programme Enrollment | 0.018 |

These findings indicate that health-promotion compliance emerges from behavioural, educational, and lifestyle interactions rather than isolated demographic variables.

---

# Repository Structure

```text
behavioral-phenotyping-ml/
│
├── data/
│   └── behavioural_phenotypes_data.xlsx
│
├── outputs/
│   ├── behavioral_phenotyping_pipeline.pkl
│   ├── model_benchmark_results.csv
│   ├── permutation_importances.csv
│   ├── misclassified_cases.csv
│   ├── confusion_matrix.png
│   ├── roc_pr_curves.png
│   └── shap_summary_plot.png
│   ├── permutation_importances.png
│
├── train_phenotyping_model.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository.

```bash
git clone https://github.com/YOUR_USERNAME/behavioral-phenotyping-ml.git
```

Move into the project directory.

```bash
cd behavioral-phenotyping-ml
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Running the Pipeline

Execute the complete workflow.

```bash
python train_phenotyping_model.py --data_path data/behavioural_phenotypes_data.xlsx
```

The script automatically performs:

- Data loading
- Data cleaning
- Feature engineering
- Missing value imputation
- One-hot encoding
- Feature scaling
- Model benchmarking
- Cross-validation
- Holdout evaluation
- Explainability analysis
- Model serialization

---

# Data Processing Pipeline

## Target Variable

Survey responses were converted into a binary classification problem.

| Survey Response | Encoded Class |
|-----------------|--------------:|
| Often | 1 |
| Routinely | 1 |
| Never | 0 |
| Sometimes | 0 |

---

## Categorical Variables

Categorical variables are processed using

- Constant-value imputation ("Unknown")
- One-Hot Encoding
- Unknown-category protection (`handle_unknown="ignore"`)

---

## Ordinal Variables

Frequency responses are mapped onto ordinal scales.

| Response | Numeric Value |
|-----------|--------------:|
| Never | 0 |
| Sometimes | 1 |
| Often | 2 |
| Routinely | 3 |

All ordinal features are standardized using **StandardScaler**.

---

# Validation Strategy

The workflow uses:

- Stratified 75/25 Train-Test Split
- 10-Fold Stratified Cross-Validation
- ROC-AUC Evaluation
- Precision
- Recall
- F1-score
- Brier Score
- 1,000 Bootstrap Confidence Interval Estimation

This prevents information leakage while providing stable model estimates.

---

# Explainable Artificial Intelligence (XAI)

Model interpretation is provided using two complementary techniques.

## Permutation Importance

Measures the reduction in predictive performance after randomly permuting each feature.

Output:

- `permutation_importances.csv`
- `permutation_importance.png`

---

## SHAP (TreeSHAP)

Explains individual feature contributions to model predictions.

Output:

- `shap_summary_plot.png`

---

# Generated Outputs

After execution, the following artifacts are automatically generated.

| Output | Description |
|---------|-------------|
| behavioral_phenotyping_pipeline.pkl | Serialized production pipeline |
| model_benchmark_results.csv | Cross-validation benchmark |
| permutation_importances.csv | Feature importance values |
| misclassified_cases.csv | Holdout classification errors |
| confusion_matrix.png | Confusion matrix visualization |
| roc_pr_curves.png | ROC and Precision-Recall curves |
| shap_summary_plot.png | SHAP explainability plot |

---

# Dependencies

Major Python libraries include:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib
- xgboost *(optional)*
- lightgbm *(optional)*
- shap *(optional)*

Install all requirements using

```bash
pip install -r requirements.txt
```

---

# Citation

If you use this repository in academic work, please cite it as:

```text
Kamalu, I. O.
Behavioral Phenotyping of Undergraduate Health Compliance Using Machine Learning.
GitHub Repository.
```

---

# License

This project is released under the **MIT License**.

See the `LICENSE` file for details.
