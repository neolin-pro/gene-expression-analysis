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
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# --- Custom log2 transformer ---
class Log2Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return np.log2(X + 1)

# --- NCA feature selector ---
class NCASelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.model = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def transform(self, X):
        return self.model.transform(X)

def model_handler(event=None, context=None):
    # --- Load CSV files from S3 ---
    s3 = boto3.client('s3')
    bucket = 'dana-4830-data'
    key_assay = 'data/assay_df_final.csv'
    key_meta = 'data/metadata_df.csv'
    key_deg_tumor = 'data/DEG_Tumor_vs_Normal.csv'
    assay_obj = s3.get_object(Bucket=bucket, Key=key_assay)
    metadata_obj = s3.get_object(Bucket=bucket, Key=key_meta)
    deg_tumor_obj = s3.get_object(Bucket=bucket, Key=key_deg_tumor)

    assay_df = pd.read_csv(assay_obj['Body'], index_col=0)
    metadata_df = pd.read_csv(metadata_obj['Body'])
    deg_df_sample = pd.read_csv(deg_tumor_obj['Body'], index_col=0)

    print("Loaded data from S3:", assay_df.shape, metadata_df.shape)

    # --- Step 1: Prepare data ---
    X = assay_df.T
    y = metadata_df["Sample"]

    # --- Step 2: Encode labels for all classifiers ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- Step 3: Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42)

    # --- Step 4: DEG-based subset ---
    top_genes = deg_df_sample[deg_df_sample['padj'] < 0.10].index
    # X_train_subset = X_train[top_genes]
    # X_test_subset = X_test[top_genes]

    # --- Model and FS Configs ---
    models = {
        "XGB": (
            XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            {
                'clf__n_estimators': [100, 200],  
                'clf__max_depth': [3, 6, 10],    
                'clf__learning_rate': [0.05, 0.1, 0.2]
            }
        )
    }

    fs_methods = {
        "NCA": NCASelector(n_components=10),
    }

    # --- Pipeline ---
    # 1. Set up the exact identifiers
    fs_name = "NCA"
    fs_transformer = fs_methods[fs_name]

    clf_name = "XGB"
    clf, param_grid = models[clf_name]

    # 2. Re-run the pipeline (or refit just this combo)
    pipeline = Pipeline([
        ('log2', Log2Transformer()),
        ('fs', fs_transformer),
        ('clf', clf)
    ])

    # Fit the GridSearchCV on DEG-filtered gene set
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        n_jobs=1,
        scoring='accuracy',
        verbose=0
    )
    grid.fit(X_train[top_genes], y_train)

    # Get best model and predictions
    tumor_normal_model = grid.best_estimator_
    y_pred = tumor_normal_model.predict(X_test[top_genes])

    # --- Evaluate ---
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')

    if len(set(y_test)) == 2:
        tp, fp, fn, tn = map(int, confusion_matrix(y_test, y_pred).ravel())
    else:
        tn = fp = fn = tp = 0

    print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # --- Save model ---
    model_path = '/home/ec2-user/dana-4830/scripts/nca_xgb_model.pkl'
    joblib.dump(tumor_normal_model, model_path)
    s3.upload_file(model_path, bucket, 'models/nca_xgb_model.pkl')

    # --- Log to PostgreSQL ---
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
        "NCA + XGBoost + DEG",
        datetime.now(),
        float(accuracy),
        float(recall),
        tp, fp, fn, tn,
        "NCA + XGB model on DEG-filtered gene expression"
    ))

    conn.commit()
    cur.close()
    conn.close()

    return {
        'statusCode': 200,
        'body': f"NCA + XGB trained with accuracy={accuracy:.4f}, recall={recall:.4f}, model saved to S3"
    }

# === Entry point for local execution ===
if __name__ == "__main__":
    model_handler(event=None, context=None)