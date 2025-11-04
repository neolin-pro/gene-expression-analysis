import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import psycopg2
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sksurv.util import Surv
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.functions import StepFunction
import tempfile
import boto3

from sqlalchemy import create_engine, text

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
    
# --- Page Config ---
st.set_page_config(page_title="BRCA – TCGA Gene Analysis", layout="wide")

# --- Custom CSS / Header ---
st.markdown("""
<style>
    .block-container { padding: 0rem 1rem; }
</style>
""", unsafe_allow_html=True)

# --- Top Bar with Title & Download Link ---
st.markdown("""
<div style='background-color:#1f3c88; padding:50px 30px; border-radius:5px; margin-bottom:10px; width:100%;'>
    <h2 style='color:white; display:inline;'>🔬 BRCA – TCGA Gene Expression Analysis</h2>
    <a href='#' download style='float:right; color:white; font-weight:bold; text-decoration:none;'>⬇️ Download Report</a>
</div>
""", unsafe_allow_html=True)

# --- Load DEG from PostgreSQL ---
@st.cache_resource
def load_deg_from_postgres():
    engine = create_engine("") # Add your PostgreSQL connection string here
    
    with engine.connect() as conn:
        # Query for upregulated genes
        up_result = conn.execute(text("""
            SELECT "geneName", "log2FoldChange", "padj" 
            FROM deg_tumor_normal 
            WHERE "log2FoldChange" > 1 AND padj < 0.10 
            ORDER BY padj DESC 
            LIMIT 20
        """))
        up = pd.DataFrame(up_result.fetchall(), columns=up_result.keys())

        # Query for downregulated genes
        down_result = conn.execute(text("""
            SELECT "geneName", "log2FoldChange", "padj"  
            FROM deg_tumor_normal 
            WHERE "log2FoldChange" < -1 AND padj < 0.10 
            ORDER BY padj DESC 
            LIMIT 20
        """))
        down = pd.DataFrame(down_result.fetchall(), columns=down_result.keys())

    return up, down

# --- Load Models ---
@st.cache_resource
def load_models():
    s3 = boto3.client("s3")
    bucket = "dana-4830-data"
    model_keys = {
        "tumor_model": "models/nca_xgb_model.pkl",
        "stage_model": "models/lda_xgb_model.pkl",
        "rsf_model": "models/rsf_model.pkl"
    }

    models = {}
    for name, key in model_keys.items():
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3.download_file(bucket, key, tmp.name)
            models[name] = joblib.load(tmp.name)

    return models["tumor_model"], models["stage_model"], models["rsf_model"]

# --- Classification Prediction ---
def predict_class(sample_df, model, label_map):
    pred = model.predict(sample_df)[0]
    prob = model.predict_proba(sample_df)[0].max()
    return label_map.get(pred, pred), round(prob, 2)

# --- Survival Prediction + Plot ---
def plot_survival(rsf, sample_df, sample_id):
    fn = rsf.predict_survival_function(sample_df)[0]
    median_days = int(np.interp(0.5, fn.y[::-1], fn.x[::-1]))
    median_years = round(median_days / 365.25, 1)

    # ❗ Skip C-index if only one sample
    if sample_df.shape[0] < 2:
        c_index = None
    else:
        c_index = round(concordance_index_censored(
            np.ones(sample_df.shape[0], dtype=bool),
            np.repeat(median_days, sample_df.shape[0]),
            fn.x[np.argmax(fn.y <= 0.5)]
        )[0], 3)

    # Plot survival function
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.step(fn.x / 365.25, fn.y, where="post", label="Predicted Survival")
    ax.axhline(0.5, linestyle="--", color="gray")
    ax.axvline(median_years, linestyle="--", color="red", label=f"Median: {median_years} yrs")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Survival Probability")
    ax.set_title(f"Predicted Survival")
    ax.legend()
    st.pyplot(fig)

    return c_index, median_days, median_years

# --- Sidebar Upload and Prediction ---
col_upload, col_spacer = st.columns([1, 3], gap="large")

with col_upload:
    st.markdown("### Sample File Upload")
    uploaded_file = st.file_uploader("Drag file to upload or choose file", type=["csv", "tsv"])
    predict_clicked = st.button("🔍 PREDICT")

if not predict_clicked:
    with col_spacer:
        st.markdown("### 👋 Welcome to the Gene Analysis Tool")
        st.info("""
        Please upload your gene expression file in **CSV or TSV** format to begin analysis.  
        Once uploaded, click the **🔍 PREDICT** button to generate insights including 
        **top 20 upregulated** and **downregulated** genes.
        """)

# --- Run Full Prediction After Button Click ---
if predict_clicked and uploaded_file:
    ext = uploaded_file.name.split('.')[-1]
    df = pd.read_csv(uploaded_file, sep="\t" if ext == "tsv" else ",", index_col=0)

    # Load the user's raw sample (genes as columns)
    user_sample = df

    s3 = boto3.client('s3')
    bucket = 'dana-4830-data'
    key_deg_tumor = 'data/DEG_Tumor_vs_Normal.csv'
    key_deg_early = 'data/DEG_Early_vs_Late_Stage.csv'

    deg_tumor_obj = s3.get_object(Bucket=bucket, Key=key_deg_tumor)
    deg_early_obj = s3.get_object(Bucket=bucket, Key=key_deg_early)
    deg_df_sample = pd.read_csv(deg_tumor_obj['Body'])
    deg_df_sample2 = pd.read_csv(deg_early_obj['Body'], index_col=0)
    
    print("Loaded data from S3:", deg_df_sample.shape)

    top_genes = deg_df_sample.sort_values("padj").index
    top_genes_stage = deg_df_sample2[deg_df_sample2['padj'] < 0.10].index

    # Step 1: Intersect the genes
    available_genes = [gene for gene in top_genes if gene in user_sample.columns]

    # Step 2: Subset and reorder the columns to match top_genes order
    sample_df = user_sample[available_genes]

    # Step 3: Reindex to match top_genes (fill missing ones if needed)
    sample_df = sample_df.reindex(columns=top_genes, fill_value=0)  # Fill missing genes with 0s (or np.nan)

    # Step 4: RawCount update
    #update_rawcount_postgres(sample_df)

    # Load resources
    up_deg, down_deg = load_deg_from_postgres()
    genes = list(up_deg["geneName"]) + list(down_deg["geneName"])
    tumor_model, stage_model, rsf_model = load_models()

    # Predictions
    tumor_pred, tumor_conf = predict_class(sample_df, tumor_model, {0: "Normal", 1: "Tumor"})

    sample_stage_2_df = user_sample[top_genes_stage]

    predicted_class_stage_2 = stage_model.predict(sample_stage_2_df)[0]
    predicted_prob_stage_2 = stage_model.predict_proba(sample_stage_2_df)[0]
    confidence_stage_2 = np.max(predicted_prob_stage_2)
    stage_pred, stage_conf = ("Early" if predicted_class_stage_2 == 0 else "Late", confidence_stage_2)
    # stage_pred, stage_conf = predict_class(sample_df, stage_model, {0: "Early", 1: "Late"})
    # tumor_pred, tumor_conf, stage_pred, stage_conf = predict_two_classes(
    # sample_df,
    # tumor_model, {0: "Normal", 1: "Tumor"},
    # stage_model, {0: "Early", 1: "Late"}

    #stage_pred, stage_conf = "Stage I", 0.95

    with col_spacer:
        st.markdown("### Upregulated and Downregulated Genes (Top 20)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🔺 Upregulated Genes")
            st.dataframe(up_deg, use_container_width=True)

        with c2:
            st.markdown("#### 🔻 Downregulated Genes")
            st.dataframe(down_deg, use_container_width=True)

        st.markdown("### Prediction Summary")
        st.markdown(f"""
        <div style='background-color:#f5f8fa; padding:15px 30px; border-left: 5px solid #1f3c88; border-radius:10px; margin-top:10px;'>
            <p><b>🧬 <span style='color:#000000;'>Tumor Status:</span></b> <span style='color:#d62728;'>{tumor_pred}</span> <span style='color:#000000;'>(Confidence: {tumor_conf:.2f})</span></p>
            <p><b>📈 <span style='color:#000000;'>Stage:&nbsp;&nbsp;&nbsp;&nbsp;</span></b> <span style='color:#2ca02c;'>{stage_pred}</span> <span style='color:#000000;'>(Confidence: {stage_conf:.2f})</span></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='border: 1px solid #e0e0e0; padding: 15px 25px 10px 25px; border-radius: 10px; background-color: #f8f9fa; margin-top: 20px;'>
            <h3 style='color: #1f3c88;'>📊 Survival Analysis</h3>
            <hr style='margin: 5px 0 10px;'/>
        """, unsafe_allow_html=True)

        col_left, col_right = st.columns([3, 2], gap="large")

        with col_left:
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
            c_index, median_days, median_years = plot_survival(rsf_model, sample_df, sample_df.index[0])
            

        with col_right:
            st.markdown(f"""
                <div style='background-color:#1f3c88; padding:18px; border-radius:10px; margin-top:20px;'>
                    <p style='color:white; font-size:15px; margin:0;'>
                        <b>Median Survival:</b> {median_days} days (~{median_years} yrs)<br><br>
                        <i>📅 “A median survival time of {median_days} days suggests that 50% of similar patients
  are expected to survive longer than approximately {median_years} years”</i>
                    </p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)