"""
========================================================================================
Project: Algorithmic Behavioral Phenotyping: Predicting Health-Promotion Compliance 
         via Multi-Domain Classification Pipelines
Author: Ikechukwu Okechi Kamalu
Environment Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl
========================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Configure publication-grade styling properties
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
sns.set_context("paper", font_scale=1.1)

class AlgorithmicPhenotypingPipeline:
    def __init__(self, data_path, output_dir="outputs"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.cat_cols = []
        self.num_cols = []
        
        # Automatically establish output matrix folder if absent
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_and_preprocess_data(self):
        """Loads data matrix from disk and isolates multi-domain survey attributes."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"[ERROR] Target dataset matrix not detected at: {self.data_path}")
            
        # Natively parse Excel file formats to bypass text decoding boundaries
        if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
            self.df = pd.read_excel(self.data_path)
            print("[INFO] Successfully parsed data file as binary Excel format.")
        else:
            self.df = pd.read_csv(self.data_path)
            print("[INFO] Parsed data file as CSV matrix format.")
        
        # 1. Clean and Binarize Behavioral Target Response Scale
        target_col = 'How Often do you practice healthy lifestyle behaviours?'
        self.df['Target_Clean'] = self.df[target_col].astype(str).str.strip()
        self.df['High_Compliance'] = self.df['Target_Clean'].apply(
            lambda x: 1 if x in ['Often', 'Routinely'] else 0
        )
        
        # 2. Harmonize Text Variance Metrics
        self.df['Age_Clean'] = self.df['Age'].astype(str).str.replace(' ', '').str.strip()
        self.df['Gender_Clean'] = self.df['Gender'].astype(str).str.strip().str.capitalize()
        
        # 3. Categorical and Numerical Feature Allocation Mapping
        self.cat_cols = [
            'Age_Clean', 'Gender_Clean', 'Do you reside on campus?', 
            'Are you aware of various health promotion activities offered on campus?',
            'Do you Know where to access information about health promotion activities?',
            'Can you say that the University environment is very healthy and supportive for your undergraduate studies?'
        ]
        
        for col in self.cat_cols:
            self.df[col] = self.df[col].fillna('Unknown').astype(str)
            
        self.num_cols = ['Level']
        self.df['Level'] = pd.to_numeric(self.df['Level'], errors='coerce').fillna(100)

        X = self.df[self.cat_cols + self.num_cols]
        y = self.df['High_Compliance']
        
        # Stratified Split to Guarantee Uniform Distribution of Target Phenotypes
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        print(f"[SUCCESS] Multi-domain feature matrix prepared. Training Shape: {self.X_train.shape}")

    def engineer_and_train_pipeline(self):
        """Constructs an isolated, leak-proof column transformation pipeline framework."""
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, self.cat_cols),
                ('num', numeric_transformer, self.num_cols)
            ]
        )
        
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42))
        ])
        
        print("[INFO] Executing algorithmic workflow training...")
        self.pipeline.fit(self.X_train, self.y_train)
        print("[SUCCESS] Classification engine training completed.")

    def generate_metrics_and_plots(self):
        """Generates statistical performance validations and exports high-res graphic plots."""
        preds = self.pipeline.predict(self.X_test)
        probs = self.pipeline.predict_proba(self.X_test)[:, 1]
        
        print("\n" + "="*50 + "\nCLASSIFICATION BOUNDARY REPORT\n" + "="*50)
        print(classification_report(self.y_test, preds, target_names=['Low Compliance', 'High Compliance']))
        
        # ---- VISUALIZATION 1: CONFUSION MATRIX HEATMAP ----
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(self.y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Low Compliance', 'High Compliance'],
                    yticklabels=['Low Compliance', 'High Compliance'],
                    annot_kws={"size": 11, "weight": "bold"})
        plt.title('Confusion Matrix:\nClassifier Validation Boundary', fontsize=11, fontweight='bold', pad=15)
        plt.xlabel('Predicted Structural Label', fontsize=10)
        plt.ylabel('True Phenotypic Label', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'behavioral_confusion_matrix.png'), dpi=300)
        plt.close()
        
        # ---- VISUALIZATION 2: RECEIVER OPERATING CHARACTERISTIC (ROC) ----
        fpr, tpr, _ = roc_curve(self.y_test, probs)
        auc_val = roc_auc_score(self.y_test, probs)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Model Pipeline (AUC = {auc_val:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Receiver Operating Characteristic\n(ROC) Curve', fontsize=11, fontweight='bold', pad=15)
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10)
        plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'behavioral_roc_curve.png'), dpi=300)
        plt.close()
        
        # ---- VISUALIZATION 3: FEATURE IMPORTANCE PROFILE ----
        cat_encoder = self.pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
        encoded_features = list(cat_encoder.get_feature_names_out(self.cat_cols))
        feature_names = encoded_features + self.num_cols
        
        importances = self.pipeline.named_steps['classifier'].feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)
        
        plt.figure(figsize=(9, 5))
        sns.barplot(x='Importance', y='Feature', data=feat_df, color='steelblue')
        plt.title('Feature Gini Importance Profile:\nKey Compliance Phenotype Drivers', fontsize=11, fontweight='bold', pad=15)
        plt.xlabel('Relative Impurity Reduction Weight', fontsize=10)
        plt.ylabel('Engineered Input Covariates', fontsize=10)
        plt.tick_params(axis='y', labelsize=9)  # Makes y-axis feature words easier to read
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'behavioral_feature_importances.png'), dpi=300)
        plt.close()
        
        print(f"\n[SUCCESS] Pipeline execution complete. Graphics saved directly to execution path: '{self.output_dir}/'")

if __name__ == "__main__":
    # Point directly to the structured Excel data location
    data_input_path = os.path.join("data", "behavioural_phenotypes_data.xlsx")
    
    engine = AlgorithmicPhenotypingPipeline(data_path=data_input_path)
    engine.load_and_preprocess_data()
    engine.engineer_and_train_pipeline()
    engine.generate_metrics_and_plots()
