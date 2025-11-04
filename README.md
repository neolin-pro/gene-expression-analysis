# 🧬 BRCA Gene Expression Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20RDS-orange)
![Airflow](https://img.shields.io/badge/Apache-Airflow-017CEE)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B)

A comprehensive machine learning pipeline for breast cancer (BRCA) gene expression analysis using TCGA data. This project combines differential gene expression (DEG) analysis, multiple ML classification models, and survival analysis to predict tumor status, cancer stage, and patient survival outcomes.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Dashboard](#dashboard)
- [Technologies](#technologies)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for analyzing breast cancer gene expression data from The Cancer Genome Atlas (TCGA). It performs:

1. **Differential Gene Expression (DEG) Analysis** - Identifies upregulated and downregulated genes
2. **Tumor Classification** - Distinguishes between normal and tumor samples
3. **Stage Prediction** - Classifies samples into early vs. late stage cancer
4. **Survival Analysis** - Predicts patient survival probability using Random Survival Forests

The pipeline is automated using Apache Airflow, deployed on AWS infrastructure (EC2, S3, RDS), and features an interactive Streamlit dashboard for real-time predictions.

## ✨ Features

- **Multi-Model Classification Pipeline**: XGBoost with NCA/LDA feature selection
- **Differential Gene Expression**: Identifies top 20 upregulated/downregulated genes with adjusted p-values
- **Survival Analysis**: Random Survival Forest for time-to-event predictions
- **Automated Training**: Apache Airflow DAG triggers daily model retraining on EC2
- **Cloud Infrastructure**: AWS S3 for data storage, RDS PostgreSQL for results logging
- **Interactive Dashboard**: Streamlit app for gene expression file upload and instant predictions
- **Model Versioning**: Models saved to S3 with timestamp-based tracking
- **Performance Logging**: PostgreSQL database stores accuracy, recall, confusion matrix for each run

## 🏗️ Architecture

```
┌─────────────────┐
│  TCGA Data      │
│  (S3 Bucket)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Airflow DAG (EC2)                 │
│   - Daily Schedule (@daily)         │
│   - Triggers model training via SSM │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│   Training Scripts (EC2)            │
│   - train_model_nca_xgb_deg.py      │
│   - train_model_lda_xgb_deg.py      │
│   - train_model_rsf_deg.py          │
└────────┬────────────────────────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
    ┌────────┐      ┌──────────┐      ┌──────────┐
    │   S3   │      │   RDS    │      │Streamlit │
    │ Models │      │PostgreSQL│      │Dashboard │
    └────────┘      └──────────┘      └──────────┘
```

## 🤖 Machine Learning Models

### 1. Tumor Classification (NCA + XGBoost)
- **Purpose**: Classify samples as Normal vs. Tumor
- **Feature Selection**: Neighborhood Components Analysis (NCA) with 10 components
- **Classifier**: XGBoost with GridSearchCV hyperparameter tuning
- **Features**: DEG-filtered genes (padj < 0.10) from Tumor vs. Normal comparison

### 2. Stage Prediction (LDA + XGBoost)
- **Purpose**: Predict Early vs. Late stage cancer
- **Feature Selection**: Linear Discriminant Analysis (LDA) with 1 component
- **Classifier**: XGBoost with optimized hyperparameters
- **Features**: DEG-filtered genes from Early vs. Late stage comparison

### 3. Survival Analysis (Random Survival Forest)
- **Purpose**: Predict patient survival probability over time
- **Algorithm**: Random Survival Forest (RSF) from scikit-survival
- **Output**: Survival function S(t), median survival time, concordance index
- **Evaluation**: Time-dependent C-index for censored data

### Model Pipeline

```python
Pipeline([
    ('log2', Log2Transformer()),           # Log2(x + 1) transformation
    ('fs', NCASelector(n_components=10)),  # Feature selection
    ('clf', XGBClassifier())               # Classification
])
```

### Hyperparameter Grid

```python
{
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [3, 6, 10],
    'clf__learning_rate': [0.05, 0.1, 0.2]
}
```

## 📁 Project Structure

```
gene-expression-analysis/
├── DAGs/
│   ├── trigger_train.py          # Airflow DAG for automated training
│   └── requirements.txt          # Airflow dependencies
├── scripts/
│   ├── train_model_nca_xgb_deg.py   # NCA + XGBoost (Tumor classification)
│   ├── train_model_lda_xgb_deg.py   # LDA + XGBoost (Stage prediction)
│   └── train_model_rsf_deg.py       # Random Survival Forest
├── dashboard/
│   ├── streamlit_app.py          # Interactive prediction dashboard
│   └── test.ipynb                # Exploratory analysis notebook
├── data/
│   ├── assay_df_final.csv        # Gene expression matrix (samples × genes)
│   ├── metadata_df.csv           # Sample metadata (stage, patient info)
│   ├── survival_df.csv           # Survival time and event data
│   ├── DEG_Tumor_vs_Normal.csv   # Differential expression results
│   ├── DEG_Early_vs_Late_Stage.csv
│   └── DEG Stage_I vs Stage_IV.csv
├── LICENSE
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- AWS account with EC2, S3, and RDS access
- Apache Airflow (for automated training)
- PostgreSQL database

### Step 1: Clone the Repository

```bash
git clone https://github.com/nayzawlin/gene-expression-analysis.git
cd gene-expression-analysis
```

### Step 2: Install Dependencies

```bash
# Training scripts dependencies
pip install boto3 pandas numpy scikit-learn xgboost scikit-survival joblib psycopg2-binary

# Dashboard dependencies
pip install streamlit matplotlib sqlalchemy

# Airflow dependencies (if using automated training)
cd DAGs
pip install -r requirements.txt
```

### Step 3: Configure AWS Credentials

```bash
aws configure
# Enter your AWS Access Key ID, Secret Access Key, and Region (e.g., ca-central-1)
```

### Step 4: Set Up Database

Create PostgreSQL database and table:

```sql
CREATE DATABASE gene_db;

CREATE TABLE model_results (
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
);

-- DEG results table
CREATE TABLE deg_tumor_normal (
    "geneName" TEXT PRIMARY KEY,
    "log2FoldChange" FLOAT,
    padj FLOAT,
    "rawCount" FLOAT
);
```

### Step 5: Upload Data to S3

```bash
aws s3 cp data/ s3://dana-4830-data/data/ --recursive
```

## 💻 Usage

### Training Models Locally

Run individual training scripts:

```bash
# Train tumor classification model (NCA + XGBoost)
cd scripts
python train_model_nca_xgb_deg.py

# Train stage prediction model (LDA + XGBoost)
python train_model_lda_xgb_deg.py

# Train survival model (Random Survival Forest)
python train_model_rsf_deg.py
```

### Automated Training with Airflow

The Airflow DAG automatically triggers model training daily at midnight:

```python
@dag(
    start_date=datetime.datetime(2021, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["training", "ssm", "ec2"]
)
def trigger_model_training():
    # Sends SSM command to EC2 instance
    # Executes: python3 train_model_nca_xgb_deg.py
```

### Running the Dashboard

Launch the Streamlit dashboard for interactive predictions:

```bash
cd dashboard
streamlit run streamlit_app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Data Pipeline

### 1. Data Preprocessing

- **Input**: Raw gene expression counts (RNA-seq) from TCGA
- **Transformation**: Log2(x + 1) normalization
- **Filtering**: Remove genes with all-zero expression
- **Train-Test Split**: 80-20 stratified split by cancer stage

### 2. Differential Gene Expression

- **Method**: DESeq2-like analysis (performed separately)
- **Criteria**: 
  - Upregulated: log2FoldChange > 1 and padj < 0.10
  - Downregulated: log2FoldChange < -1 and padj < 0.10
- **Output**: Top 20 genes in each direction stored in PostgreSQL

### 3. Feature Selection

- **NCA (Tumor Classification)**: Projects genes into 10-dimensional space maximizing class separation
- **LDA (Stage Prediction)**: Single linear discriminant for binary classification
- **DEG Filtering**: Uses only statistically significant genes (padj < 0.10)

### 4. Model Training

- **Cross-Validation**: 5-fold stratified CV during GridSearchCV
- **Evaluation Metrics**: Accuracy, macro-averaged recall, confusion matrix
- **Model Storage**: Serialized with joblib and uploaded to S3
- **Results Logging**: Metrics stored in PostgreSQL with timestamp

### 5. Prediction Workflow

```
User Upload CSV → Load Models from S3 → 
Log2 Transform → Feature Selection → 
Classification (Tumor/Stage) → Survival Analysis → 
Display Results + Visualizations
```

## 🖥️ Dashboard

### Features

1. **File Upload**: Supports CSV/TSV gene expression files
2. **DEG Visualization**: Displays top 20 upregulated/downregulated genes
3. **Tumor Prediction**: Shows Normal/Tumor classification with confidence
4. **Stage Prediction**: Displays Early/Late stage with probability
5. **Survival Curve**: Plots predicted survival function with median survival time
6. **Downloadable Report**: Export analysis results (feature in progress)

### Sample Output

```
🧬 Tumor Status: Tumor (Confidence: 0.94)
📈 Stage: Late (Confidence: 0.87)
📊 Median Survival: 1825 days (~5.0 yrs)
```

## 🛠️ Technologies

### Core ML/Data Science
- **scikit-learn**: Classification, feature selection, preprocessing
- **XGBoost**: Gradient boosting classifier
- **scikit-survival**: Survival analysis (Random Survival Forest)
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Cloud & Infrastructure
- **AWS S3**: Data and model storage
- **AWS EC2**: Model training compute
- **AWS RDS**: PostgreSQL database
- **AWS SSM**: Remote command execution
- **boto3**: AWS SDK for Python

### Orchestration & Deployment
- **Apache Airflow**: Workflow automation
- **Streamlit**: Interactive dashboard
- **joblib**: Model serialization
- **psycopg2**: PostgreSQL adapter
- **SQLAlchemy**: Database ORM

## 📈 Results

### Model Performance (Example Run)

| Model | Accuracy | Recall (Macro) | Dataset |
|-------|----------|----------------|---------|
| NCA + XGBoost (Tumor) | 0.9623 | 0.9615 | Test (20%) |
| LDA + XGBoost (Stage) | 0.8571 | 0.8456 | Test (20%) |
| Random Survival Forest | C-index: 0.745 | - | Full cohort |

### Key Findings

- **Top Upregulated Genes**: Identified genes with log2FC > 1 and padj < 0.10
- **Top Downregulated Genes**: Genes with log2FC < -1 and significant p-values
- **Survival Prediction**: Median survival estimates range from 2-7 years depending on stage
- **Feature Importance**: DEG-based gene filtering improves model accuracy by ~15%

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Nay Zaw Lin**

- GitHub: [@neolin-pro](https://github.com/neolin-pro)
- Email: nayzawlin07@gmail.com

## 🙏 Acknowledgments

- **The Cancer Genome Atlas (TCGA)** for providing the BRCA gene expression dataset
- **scikit-survival** team for the excellent survival analysis library
- **AWS** for cloud infrastructure support
- **Apache Airflow** community for workflow orchestration tools

## 📚 References

- Koboldt, D. C., et al. (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418), 61-70.
- Ishwaran, H., et al. (2008). Random survival forests. *The Annals of Applied Statistics*, 2(3), 841-860.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

---

⭐ If you find this project helpful, please consider giving it a star!

**Built with ❤️ using Python and Streamlit **
