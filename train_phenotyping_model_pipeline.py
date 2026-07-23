"""
Behavioral Phenotyping Machine Learning Pipeline
================================================
A self-contained production pipeline for training, benchmarking, 
and evaluating Machine Learning models on health compliance survey data.

Usage:
    python train_phenotyping_model.py --data_path path/to/behavioural_phenotypes_data.xlsx
"""

import os
import argparse
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.inspection import permutation_importance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


class BehavioralPhenotypingPipeline:
    def __init__(self, data_path: str, output_dir: str = "outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.cat_cols = []
        self.num_cols = []
        self.feature_names_out = []
        self.preprocessor = None
        self.best_model_name = None
        self.best_pipeline = None

    def load_and_preprocess(self):
        """Loads survey data, encodes target, handles missing values, and prepares feature spaces."""
        logging.info(f"Loading raw dataset from '{self.data_path}'...")
        
        if self.data_path.endswith(('.xlsx', '.xls')):
            raw_df = pd.read_excel(self.data_path)
        else:
            raw_df = pd.read_csv(self.data_path)

        logging.info(f"Dataset shape: {raw_df.shape}")

        # Clean headers
        raw_df.columns = [str(c).strip().strip('"').strip("'") for c in raw_df.columns]
        
        # Deduplicate headers if any
        seen = {}
        unique_cols = []
        for col in raw_df.columns:
            if col in seen:
                seen[col] += 1
                unique_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                unique_cols.append(col)
        raw_df.columns = unique_cols
        self.df = raw_df.copy()

        # Target column mapping
        target_candidates = [c for c in self.df.columns if 'How Often do you practice healthy' in c]
        if not target_candidates:
            target_candidates = [c for c in self.df.columns if 'practice healthy' in c.lower()]
        target_col = target_candidates[0]

        self.df['Target_Clean'] = self.df[target_col].astype(str).str.strip().str.strip('"')
        self.df['High_Compliance'] = self.df['Target_Clean'].apply(
            lambda x: 1 if str(x).strip().capitalize() in ['Often', 'Routinely'] else 0
        )

        # Extract numeric Level
        if 'Level' in self.df.columns:
            level_extract = self.df['Level'].astype(str).str.extract(r'(\d+)')[0]
            self.df['Level_Num'] = pd.to_numeric(level_extract, errors='coerce').fillna(100)
        else:
            self.df['Level_Num'] = 100

        # Feature separation
        exclude_cols = [target_col, 'Target_Clean', 'High_Compliance', 'Level', 'Timestamp', 'Unnamed: 0', 'Unnamed: 0.1']
        candidate_cols = [c for c in self.df.columns if c not in exclude_cols and not c.startswith('Level_Num')]

        freq_map = {'never': 0, 'no': 0, 'sometimes': 1, 'somehow': 1, 'often': 2, 'yes': 2, 'routinely': 3}

        num_cols = ['Level_Num']
        cat_cols = []

        for col in candidate_cols:
            col_str_vals = set(self.df[col].dropna().astype(str).str.strip().str.lower().unique())
            
            if col_str_vals.issubset({'never', 'sometimes', 'often', 'routinely', 'yes', 'no', 'somehow', 'nan', 'none', '', 'np.nan'}):
                num_col_name = f"{col[:30]}_Ord"
                self.df[num_col_name] = self.df[col].astype(str).str.strip().str.lower().map(freq_map).fillna(0)
                num_cols.append(num_col_name)
            else:
                self.df[col] = self.df[col].astype(str).str.strip().str.capitalize()
                cat_cols.append(col)

        self.num_cols = sorted(list(set(num_cols)))
        self.cat_cols = sorted(list(set(cat_cols) - set(self.num_cols)))

        X = self.df[self.cat_cols + self.num_cols].copy()
        y = self.df['High_Compliance'].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', cat_transformer, self.cat_cols),
            ('num', num_transformer, self.num_cols)
        ])

        logging.info(f"Ingested {X.shape[1]} features ({len(self.cat_cols)} categorical, {len(self.num_cols)} numerical/ordinal).")
        logging.info(f"Training set: {self.X_train.shape[0]} samples | Test set: {self.X_test.shape[0]} samples")

    def run_benchmark(self):
        """Cross-validates available classification algorithms."""
        logging.info("Starting 10-Fold Stratified Cross-Validation Benchmark...")

        candidate_models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=42)
        }

        if HAS_XGBOOST:
            candidate_models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
        if HAS_LIGHTGBM:
            candidate_models["LightGBM"] = LGBMClassifier(random_state=42, verbosity=-1)

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        results = []
        best_auc = -1.0

        for name, clf in candidate_models.items():
            pipe = Pipeline([('preprocessor', self.preprocessor), ('classifier', clf)])
            scores = cross_validate(
                pipe, self.X_train, self.y_train, cv=cv,
                scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            )

            mean_auc = scores['test_roc_auc'].mean()
            results.append({
                "Model": name,
                "Accuracy": scores['test_accuracy'].mean(),
                "Precision": scores['test_precision'].mean(),
                "Recall": scores['test_recall'].mean(),
                "F1-Score": scores['test_f1'].mean(),
                "ROC-AUC": mean_auc
            })

            if mean_auc > best_auc:
                best_auc = mean_auc
                self.best_model_name = name
                self.best_pipeline = pipe

        bench_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
        bench_df.to_csv(os.path.join(self.output_dir, "model_benchmark_results.csv"), index=False)
        
        print("\n" + "="*65)
        print("MODEL BENCHMARK SUMMARY")
        print("="*65)
        print(bench_df.to_string(index=False))
        print("="*65 + "\n")

        logging.info(f"Fitting Champion Model ({self.best_model_name}) on full training dataset...")
        self.best_pipeline.fit(self.X_train, self.y_train)

        cat_encoder = self.best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_cat_features = list(cat_encoder.get_feature_names_out(self.cat_cols))
        self.feature_names_out = encoded_cat_features + self.num_cols

    def evaluate_champion(self):
        """Generates holdout test performance metrics and plots."""
        preds = self.best_pipeline.predict(self.X_test)
        probs = self.best_pipeline.predict_proba(self.X_test)[:, 1]

        print("\n" + "="*65)
        print(f"CHAMPION MODEL TEST REPORT ({self.best_model_name})")
        print("="*65)
        print(classification_report(self.y_test, preds, target_names=['Low Compliance', 'High Compliance']))

        auc_val = roc_auc_score(self.y_test, probs)
        brier = brier_score_loss(self.y_test, probs)
        ci_lower, ci_upper = self._bootstrap_ci(self.X_test, self.y_test)

        print(f"ROC-AUC Score: {auc_val:.3f} (95% CI: [{ci_lower:.3f} - {ci_upper:.3f}])")
        print(f"Brier Score:   {brier:.3f}")
        print("="*65 + "\n")

        # Save confusion matrix plot
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Low Compliance', 'High Compliance'],
                    yticklabels=['Low Compliance', 'High Compliance'])
        plt.title(f'Confusion Matrix: {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()

        # Save ROC and PR curves
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        axes[0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {auc_val:.3f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        axes[0].set_title('ROC Curve')
        axes[0].legend(loc="lower right")

        prec, rec, _ = precision_recall_curve(self.y_test, probs)
        ap_val = average_precision_score(self.y_test, probs)
        axes[1].plot(rec, prec, color='teal', lw=2.5, label=f'PR (AP = {ap_val:.3f})')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].legend(loc="lower left")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_pr_curves.png'), dpi=300)
        plt.close()

        # Export misclassified cases
        errors = self.X_test.copy()
        errors["True_Target"] = self.y_test
        errors["Predicted"] = preds
        errors["Predicted_Probability"] = probs
        misclassified = errors[errors["True_Target"] != errors["Predicted"]]
        misclassified.to_csv(os.path.join(self.output_dir, "misclassified_cases.csv"), index=False)

    def explainability(self):
        """Calculates Permutation Importance and SHAP values."""
        X_test_transformed = self.best_pipeline.named_steps['preprocessor'].transform(self.X_test)
        classifier = self.best_pipeline.named_steps['classifier']

        # Permutation Importance
        perm_importance = permutation_importance(
            classifier, X_test_transformed, self.y_test, n_repeats=30, random_state=42
        )

        perm_df = pd.DataFrame({
            'Feature': self.feature_names_out,
            'Importance_Mean': perm_importance.importances_mean,
            'Importance_Std': perm_importance.importances_std
        }).sort_values(by='Importance_Mean', ascending=False)
        
        perm_df.to_csv(os.path.join(self.output_dir, "permutation_importances.csv"), index=False)

        plt.figure(figsize=(8, 4.5))
        sns.barplot(x='Importance_Mean', y='Feature', data=perm_df.head(10), color='steelblue')
        plt.title('Top 10 Permutation Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'permutation_importance.png'), dpi=300)
        plt.close()

        # SHAP
        if HAS_SHAP:
            try:
                explainer = shap.Explainer(classifier, X_test_transformed)
                shap_values = explainer(X_test_transformed)

                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test_transformed, feature_names=self.feature_names_out, show=False)
                plt.title(f"SHAP Summary Profile: {self.best_model_name}", pad=15)
                plt.savefig(os.path.join(self.output_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("Generated SHAP explainability plot.")
            except Exception as e:
                logging.warning(f"SHAP plot generation skipped: {e}")

    def save_model(self):
        model_path = os.path.join(self.output_dir, "behavioral_phenotyping_pipeline.pkl")
        joblib.dump(self.best_pipeline, model_path)
        logging.info(f"Model saved to '{model_path}'")

    def _bootstrap_ci(self, X_eval, y_eval, n_bootstraps=1000):
        scores = []
        y_eval_np = np.array(y_eval)
        for i in range(n_bootstraps):
            X_b, y_b = resample(X_eval, y_eval_np, random_state=i, stratify=y_eval_np)
            probs = self.best_pipeline.predict_proba(X_b)[:, 1]
            scores.append(roc_auc_score(y_b, probs))
        return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Behavioral Phenotyping ML Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw dataset (.xlsx or .csv)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save artifacts")
    
    args = parser.parse_args()

    engine = BehavioralPhenotypingPipeline(data_path=args.data_path, output_dir=args.output_dir)
    engine.load_and_preprocess()
    engine.run_benchmark()
    engine.evaluate_champion()
    engine.explainability()
    engine.save_model()
