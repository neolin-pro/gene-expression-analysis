import boto3
import pandas as pd
import joblib
import psycopg2
import os
from datetime import datetime
import numpy as np
from io import StringIO
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.functions import StepFunction

# --- Custom log2 transformer ---
class Log2Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log2(X + 1)

# --- LDA transformer ---
class LDASelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=1):
        self.n_components = n_components
    def fit(self, X, y):
        self.model = LinearDiscriminantAnalysis(n_components=self.n_components)
        self.model.fit(X, y)
        return self
    def transform(self, X):
        return self.model.transform(X)

def model_handler(event=None, context=None):
    # === Load data from S3 ===
    s3 = boto3.client('s3')
    bucket = 'dana-4830-data'
    key_assay = 'data/assay_df_final.csv'
    key_meta = 'data/metadata_df.csv'
    key_deg_tumor = 'data/DEG_Tumor_vs_Normal.csv'
    key_survival = 'data/survival_df.csv'

    assay_obj = s3.get_object(Bucket=bucket, Key=key_assay)
    metadata_obj = s3.get_object(Bucket=bucket, Key=key_meta)
    deg_tumor_obj = s3.get_object(Bucket=bucket, Key=key_deg_tumor)
    deg_survival_obj = s3.get_object(Bucket=bucket, Key=key_survival)

    assay_df = pd.read_csv(assay_obj['Body'], index_col=0)
    metadata_df = pd.read_csv(metadata_obj['Body'])
    deg_df_sample = pd.read_csv(deg_tumor_obj['Body'], index_col=0)
    survival_df = pd.read_csv(deg_survival_obj['Body'], index_col=0)
    survival_df = survival_df.reset_index()
    print("Loaded data from S3:", assay_df.shape, metadata_df.shape, deg_df_sample.shape, survival_df.shape)

    # 1. Set index of survival_df to sample IDs
    survival_df = survival_df.set_index("samples_submitter_id")

    # 2. Select only required survival columns
    survival_cols = survival_df[["vital_status", "overall_survival"]]

    # 3. Merge into metadata_df
    metadata_df = survival_df

    # --- Step 0: Align the data ---
    common_ids = assay_df.columns.intersection(metadata_df.index)

    # Subset both DataFrames to common sample IDs
    assay_df_final_survival = assay_df[common_ids]
    metadata_df_survival = metadata_df.loc[common_ids]

    # Drop rows with missing survival info
    metadata_df_survival = metadata_df_survival.dropna(subset=["vital_status", "overall_survival"])
    print(metadata_df_survival.shape)
    # Drop rows with missing or negative survival time
    metadata_df_survival = metadata_df_survival[
        metadata_df_survival["overall_survival"].notna() &
        (metadata_df_survival["overall_survival"] >= 0)
    ]

    # Re-align assay data after dropping from metadata
    assay_df_final_survival = assay_df_final_survival[metadata_df_survival.index]

    # --- Step 1: Transformation and Subsetting ---
    log2_transformer = Log2Transformer() # Log2 transform
    X_log2 = log2_transformer.fit_transform(assay_df_final_survival.T)

    # Create DataFrame with sample alignment
    X_surv_full = pd.DataFrame(X_log2, index=assay_df_final_survival.columns, columns=assay_df_final_survival.index)

    # --- Step 4: DEG-based subset ---
    top_genes = deg_df_sample[deg_df_sample['padj'] < 0.10].index

    # Subset to top genes
    X_surv = X_surv_full[top_genes]

    # --- Step 2: Metadata Realignment ---
    common_idx = X_surv.index.intersection(metadata_df_survival.index)
    X_surv = X_surv.loc[common_idx]
    metadata_df_survival = metadata_df_survival.loc[common_idx]

    # --- Step 3: Create survival labels ---
    y_surv = Surv.from_arrays(
        event=metadata_df_survival["vital_status"] == "Dead",
        time=metadata_df_survival["overall_survival"]
    )

    # --- Step 4: Train-test split with stratification ---
    X_surv_train, X_surv_test, y_surv_train, y_surv_test = train_test_split(
        X_surv, y_surv,
        test_size=0.2,
        random_state=42,
        stratify=metadata_df_survival["vital_status"]
    )

    # --- Step 5: Define custom C-index scorer ---
    def cindex_scorer(estimator, X, y):
        pred = estimator.predict(X)
        return concordance_index_censored(y["event"], y["time"], pred)[0]

    # --- Step 6: Define RSF model and parameter grid ---
    rsf_base = RandomSurvivalForest(n_jobs=1, random_state=42)

    param_grid = {
        "n_estimators": [100],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [5, 10],
        "max_features": ["sqrt", "log2"],
        "max_depth": [10, 20]
    }

    # --- Step 7: GridSearchCV ---
    grid = GridSearchCV(
        estimator=rsf_base,
        param_grid=param_grid,
        cv=3,
        scoring=cindex_scorer,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_surv_train, y_surv_train)  # This fits and tunes the model
    rsf = grid.best_estimator_            # Save the best estimator to reuse

    # --- Step 8: Evaluate performance on test set ---
    risk_scores = rsf.predict(X_surv_test)

    # Evaluation
    
    risk_scores = rsf.predict(X_surv_test)
    c_index = concordance_index_censored(y_surv_test["event"], y_surv_test["time"], risk_scores)[0]
    tp = fp = fn = tn = 0
    recall = 0
    print(f"C-index (test set): {c_index:.4f}")

    # Save model to S3
    model_path = '/home/ec2-user/dana-4830/scripts/rsf_model.pkl'
    joblib.dump(rsf, model_path)
    s3.upload_file(model_path, bucket, 'models/rsf_model.pkl')

    # Log to PostgreSQL
    conn = psycopg2.connect(
        dbname="gene_db",
        user="postgres",
        password="Dana4830$",
        host="genedb.cbqht9ikryuf.ca-central-1.rds.amazonaws.com",
        port='5432'
    )
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_results (
            id SERIAL PRIMARY KEY,
            model_name TEXT,
            run_time TIMESTAMP,
            accuracy FLOAT,
            recall FLOAT,
            tp INTEGER,
            fp INTEGER,
            fn INTEGER,
            tn INTEGER,
            notes TEXT
        )
    """)

    cur.execute("""
        INSERT INTO model_results (
            model_name, run_time, accuracy, recall, tp, fp, fn, tn, notes
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        "RSF + DEG",
        datetime.now(),
        float(c_index),
        float(recall),
        tp, fp, fn, tn,
        "RSF model on DEG-filtered gene expression"
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {
        'statusCode': 200,
        'body': f"RSF trained with C-index: {c_index:.4f}, model saved to S3"
    }

# === Entry point for local execution ===
if __name__ == "__main__":
    model_handler()